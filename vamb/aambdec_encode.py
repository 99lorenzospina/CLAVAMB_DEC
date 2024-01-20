"""Adversarial autoencoders (AAE) for metagenomics binning, this files contains the implementation of the AAE_DEC"""


import numpy as np
from math import log
import time
from torch.utils.data.dataset import TensorDataset as TensorDataset
from torch.autograd import Variable
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import random
import math

from torch.utils.data import DataLoader as _DataLoader
from sklearn.cluster import KMeans

from typing import Optional
from argparse import Namespace

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################################################# MODEL ###########################################################
class AAEDEC(nn.Module):
    def __init__(
        self,
        ntnf: int, #103
        nsamples: int,
        nhiddens: int, #but it should be a list!
        nlatent_y: int, #careful: nlatent_y should be the number of estimated clusters
        sl: float,
        slr: float,
        lr: float,
        cri_lr: float,
        dis_lr: float,
        alpha: Optional[float],
        _cuda: bool,
        k: int = 4,
        contrast: bool = False,
        optimizer_E = None,
        optimizer_D = None,
        optimizer_G = None,
        optimizer_C = None,
        optimizer_clusters = None,
        degrees: int = 1,
        n_enc_1: int = 500,
        n_enc_2: int = 10,
    ):
        if nsamples is None:
            raise ValueError(
                f"Number of samples  should be provided to define the encoder input layer as well as the categorical latent dimension, not {nsamples}"
            )

        super(AAEDEC, self).__init__()
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        self.nsamples = nsamples
        self.ntnf = ntnf
        self.k = k
        self.h_n = nhiddens
        self.y_len = nlatent_y
        self.input_len = int(self.ntnf + self.nsamples)
        self.sl = sl
        self.slr = slr
        self.alpha = alpha
        self.usecuda = _cuda
        self.contrast = contrast
        self.lr = lr
        self.cri_lr = cri_lr
        self.dis_lr = dis_lr
        self.degrees = degrees
        self.optimizer_E = optimizer_E
        self.optimizer_D = optimizer_D
        self.optimizer_C = optimizer_C
        self.optimizer_G = optimizer_G
        self.n_enc_1 = n_enc_1
        self.n_enc_2 = n_enc_2

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_len, self.n_enc_1),
            nn.BatchNorm1d(self.n_enc_1),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_1, self.n_enc_2),
            nn.BatchNorm1d(self.n_enc_2),
            nn.LeakyReLU(),
        )
        # latent layers
        self.mu = nn.Linear(self.n_enc_2, self.h_n)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.h_n, self.n_enc_2),
            nn.BatchNorm1d(self.n_enc_2),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_2, self.n_enc_1),
            nn.BatchNorm1d(self.n_enc_1),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_1, self.input_len),
        )

        # discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_length, self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, int(self.h_n / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.h_n / 2), 1),
            nn.Sigmoid(),
        )


        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.input_length, self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, int(self.h_n / 2)),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size = int(self.h_n / 2)),
            nn.Identity(),
        )

        self.cluster_layer = Parameter(torch.Tensor(self.y_len, self.h_n))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        if optimizer_clusters == None:
            self.optimizer_clusters = torch.optim.Adam([self.cluster_layer], lr=self.lr)

        if _cuda:
            self.cuda()

    def _critic(self, c):
        return self.critic(c)
    
    ## Reparametrisation trick
    def _reparameterization(self, mu, logvar):

        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor

        std = torch.exp(logvar / 2)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), self.ld))))

        if self.usecuda:
            sampled_z = sampled_z.cuda()
        z = sampled_z * std + mu

        return z

    ## Encoder
    def _encode(self, depths, tnfs):
        _input = torch.cat((depths, tnfs), 1)
        x = self.encoder(_input)
        mu = self.mu(x)

        return mu

    def y_length(self):
        return self.y_len

    def z_length(self):
        return self.ld

    def samples_num(self):
        return self.nsamples

    ## Decoder
    def _decode(self, z):
        reconstruction = self.decoder(z)

        _depths_out, tnf_out = (
            reconstruction[:, : self.nsamples],
            reconstruction[:, self.nsamples :],
        )

        depths_out = F.softmax(_depths_out, dim=1)

        return depths_out, tnf_out

    @classmethod
    def load(cls, path, cuda=False, evaluate=True, c=False):
      """Instantiates a AAE from a model file.
      Inputs:
          path: Path to model file as created by functions AAE.save or
                AAE.trainmodel.
          cuda: If network should work on GPU [False]
          evaluate: Return network in evaluation mode [True]
      Output: AAE with weights and parameters matching the saved network.
      """

      # Forcably load to CPU even if model was saves as GPU model
      # dictionary = torch.load(path, map_location=lambda storage, loc: storage)
      dictionary = torch.load(path)
      try:
            ntnf = dictionary["ntnf"]
      except KeyError:
            ntnf = 103
      nsamples = dictionary['nsamples']
      alpha = dictionary['alpha']
      nhiddens = dictionary['nhiddens']
      nlatent_l = dictionary['nlatent_l']
      nlatent_y = dictionary['nlatent_y']
      sl = dictionary['sl']
      slr = dictionary['slr']
      try:
            k = dictionary['k']
      except KeyError:
            k = 4
      try:
            optimizer_E = dictionary["optimizer_E"]
      except KeyError:
            optimizer_E = None
      try:
            optimizer_D = dictionary["optimizer_D"]
      except KeyError:
            optimizer_D = None
      try:
            optimizer_D_z = dictionary["optimizer_D_z"]
      except KeyError:
            optimizer_D_z = None
      try:
            optimizer_D_y = dictionary["optimizer_D_y"]
      except KeyError:
            optimizer_D_y = None
      try:
            optimizer_awl = dictionary["optimizer_awl"]
      except KeyError:
            optimizer_awl = None


      state = dictionary['state']

      aae = cls(ntnf, nsamples, nhiddens, nlatent_l, nlatent_y, alpha, sl, slr, cuda, k=k, contrast=c, optimizer_E=optimizer_E,
                optimizer_D=optimizer_D, optimizer_D_z=optimizer_D_z, optimizer_D_y=optimizer_D_y, optimizer_awl=optimizer_awl)
      aae.load_state_dict(state)

      if cuda:
          aae.cuda()

      if evaluate:
          aae.eval()

      return aae

    def save(self, filehandle):
        """Saves the AAE to a path or binary opened file. Load with AAE.load
        Input: Path or binary opened filehandle
        Output: None
        """
        state = {
                 'ntnf': self.ntnf,
                 'nsamples': self.nsamples,
                 'alpha': self.alpha,
                 'nhiddens': self.h_n,
                 'nlatent_l': self.ld,
                 'nlatent_y': self.y_len,
                 'sl' : self.sl,
                 'slr': self.slr,
                 'k' : self.k,
                 "optimizer_E": self.optimizer_E.state_dict(),
                 "optimizer_D": self.optimizer_D.state_dict(),
                 "optimizer_D_z": self.optimizer_D_z.state_dict(),
                 "optimizer_D_y": self.optimizer_D_y.state_dict(),
                 **({"optimizer_awl": self.optimizer_awl.state_dict()} if self.optimizer_awl is not None else {}),
                 'state': self.state_dict(),
                }

        torch.save(state, filehandle)
    
    def calc_loss(self, depths_in, depths_out, tnf_in, tnf_out):
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = -((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
            ce_weight = (1 - self.alpha) / log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1).mean()
            ce_weight = 1 - self.alpha
        sse = (tnf_out - tnf_in).pow(2).sum(dim=1).mean()
        sse_weight = self.alpha / (tnf_in.shape[1] * 2)
        loss = ce * ce_weight + sse * sse_weight
        return loss, ce, sse

    def forward(self, depths_in, tnfs_in):
        mu = self._encode(depths_in, tnfs_in)
        depths_out, tnfs_out = self._decode(mu)

        return mu, depths_out, tnfs_out
    
    def get_q(self, depths_in, tnfs_in):
        z, depths_out, tnfs_out = self(depths_in, tnfs_in)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q, depths_out, tnfs_out

    def pretrain(self, dataloader, max_iter, logfile):
        lr = self.lr
        cri_lr = self.cri_lr
        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
        depthstensor, tnftensor = dataloader.dataset.tensors
        ncontigs, nsamples = depthstensor.shape

        # Initialize generator and critic

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tAlpha:", self.alpha, file=logfile)
            print("\tN of clusters:", self.y_len, file=logfile)
            print("\n\tTraining properties:", file=logfile)
            print("\tN epochs:", max_iter, file=logfile)
            print("\tStarting batch size:", data_loader.batch_size, file=logfile)
            print("\tN sequences:", ncontigs, file=logfile)
            print("\tN samples:", self.nsamples, file=logfile, end="\n\n")

        enc_params = []
        dec_params = []
        cri_params = []
        for name, param in self.named_parameters():
            if "critic" in name:
                cri_params.append(param)
            elif "encoder" in name:
                enc_params.append(param)
            elif "decoder" in name:
                dec_params.append(param)

        if (self.optimizer_E == None):
            self.optimizer_E = torch.optim.Adam(enc_params, lr=lr)
            self.optimizer_D = torch.optim.Adam(dec_params, lr=lr)
            self.optimizer_C = torch.optim.Adam(cri_params, lr=cri_lr)

        self.train()
        for iter in range(max_iter):
            time_epoch_0 = time.time()
            self.train()
            (
                crit_loss,
                ed_loss,
            ) = (0,0)

            data_loader = _DataLoader(dataset=dataloader.dataset,
                                    batch_size=dataloader.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
            
            for depths_in, tnfs_in in data_loader:
                depths_in.requires_grad = True
                tnfs_in.requires_grad = True
                a = random.random()
                b = random.random()
                s = random.random()
                z = torch.zeros(self.y_len)
                while(b==a):
                    b = random.random()
                while(s==a or s == b):
                    s = random.random()
    
                random_samples = torch.utils.data.RandomSampler(dataloader.dataset, replacement=False, num_samples=2)
                for d_sample, t_sample in random_samples:
                    mu = self._encode(d_sample, t_sample)
                    z += mu*a
                    a = 1 - a
                
                r_depths_out, r_tnfs_out = self._decode(z)
                x = torch.cat((r_depths_out, r_tnfs_out))
                mu = self._encode(depths_in, tnfs_in)
                depths_out, tnfs_out = self._decode(mu)
                reg_term = self._critic(b*torch.cat((depths_in, tnfs_in)) + (1-b)*torch.cat((depths_out, tnfs_out))).pow(2).sum(dim=1).mean()
                crit_loss = torch.abs(self._critic(x) - torch.tensor(a, torch.float))**2 + reg_term
                ed_loss = (torch.dist(torch.cat((depths_in, tnfs_in)), torch.cat((depths_out, tnfs_out)), 2).pow(2)).sum(dim=1).mean() + s*torch.abs(self._critic(x))**2

                ed_loss.backward()                
                self.optimizer_E.step()
                self.optimizer_D.step()
                crit_loss.backward()
                self.optimizer_C.step()

                ed_loss += ed_loss.item()
                crit_loss += crit_loss.item()
                
                time_epoch_1 = time.time()
                time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)

        self.eval()
        data = torch.Tensor(dataloader.dataset).to(device)
        encoded,_,_ = self._encode(data, [])
        kmeans = KMeans(n_clusters=self.y_len, n_init=20)
        y_pred = kmeans.fit_predict(encoded.cpu().numpy())
        return y_pred

    def discriminator_loss(real_output, fake_output, device):
        real_loss = torch.nn.BCEWithLogitsLoss(real_output, torch.ones_like(real_output, device=device))
        fake_loss = torch.nn.BCEWithLogitsLoss(fake_output, torch.zeros_like(fake_output, device=device))
        total_loss = real_loss + fake_loss
        return total_loss

    def calculate_kld(p, q):
        # Ensure the arrays are probability distributions (sum to 1)
        p = p / np.sum(p)
        q = q / np.sum(q)
    
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-10
    
        # Calculate KLD
        kld = np.sum(p * np.log((p + epsilon) / (q + epsilon)))

        return kld


    #STILL TO DO: deal with y_pred vector (all the contigs? How to do with batches?) and for the breakout option
    def train(self, dataloader, max_iter, aux_iter, max_iter_dis, centers, C, targ_iter, tol, lrate,
              logfile=None, modelfile=None, hparams=None, augmentationpath=None, mask=None):
        #C must be a torch.tensor of centroids (vectors), implicitly labeled by order
        
        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
        device = "cuda" if self.usecuda else "cpu"
        depthstensor, tnftensor = dataloader.dataset.tensors
        ncontigs, nsamples = depthstensor.shape

        # Pretrain discriminators

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tAlpha:", self.alpha, file=logfile)
            print("\tN clusters:", self.y_len, file=logfile)
            print("\n\tTraining properties:", file=logfile)
            print("\tN Training epochs:", max_iter, file=logfile)
            print("\tStarting batch size:", data_loader.batch_size, file=logfile)
            print("\tN sequences:", ncontigs, file=logfile)
            print("\tN samples:", self.nsamples, file=logfile, end="\n\n")

        disc_params = []

        for name, param in self.named_parameters():
            if "discriminator" in name:
                disc_params.append(param)
        if self.optimizer_G==None:
            self.optimizer_G = torch.optim.Adam(disc_params, lr=lrate)   

        for i in range(max_iter_dis):
            (
            G_loss,
            ) = (0)
            time_epoch_0 = time.time()
            self.train()
            data_loader = _DataLoader(dataset=dataloader.dataset,
                                    batch_size=dataloader.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
            
            for depths_in, tnfs_in in data_loader:
            
                depths_in.requires_grad = True
                tnfs_in.requires_grad = True
                self.optimizer_G.zero_grad()
                _, detphs_out, tnfs_out = self(depths_in, tnfs_in)
                loss = self.discriminator_loss(torch.cat((depths_in, tnfs_in), dim=1), torch.cat((detphs_out, tnfs_out), dim=1), device)
                loss.backward()
                self.optimizer_G.step()
                G_loss += loss.item()

                time_epoch_1 = time.time()
                time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)

        #Clustering phase


        data_loader = _DataLoader(dataset=dataloader.dataset,
                                    batch_size=dataloader.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
        self.train()
        for epoch in range(max_iter):
            time_epoch_0 = time.time()
            if epoch%targ_iter == 0:
                
        '''
        for h in range(max_iter):
            time_epoch_0 = time.time()
            self.train()

            (D_loss,
             Cls_loss,
             G_loss,) = (0,0,0)
            
            Q = np.empty((0, self.y_len))
            P = np.empty_like(Q)
            y_pred = np.zeros(dataloader.batch_size)    #save the cluster for each sample
            y_pred_old = y_pred
            for depths_in, tnf_in in data_loader:   #take one batch
                depths_in.requires_grad = True
                tnf_in.requires_grad = True
                if h%targ_iter == 0:
                    mu = self._encode(depths_in, tnfs_in)
                    for i in range(len(mu)):    #consider each sample separately, compute P and Q
                        mu_i = mu[i]
                        q_ij_values = np.empty(self.y_len)
                        denominator = np.sum([self.student_t_distribution(mu_i, c) for _,c in self.optimizer_C.state_dict().items()])
                        for _,c in self.optimizer_C.state_dict().items():
                            numerator = self.student_t_distribution(mu_i, c)
                            q_ij = numerator / denominator
                            q_ij_values.append(q_ij)
                        y_pred_old[i] = y_pred[i]
                        y_pred[i] = np.argmax(q_ij_values)    #cluster assignment for sample i
                        Q = np.vstack(Q, q_ij_values)
                    for col in range(Q.shape[1]):
                        P[:, col] = np.sum(Q[:, col])    #P contains the freq_qj, each column one frequence
                    temp = np.empty_like(P)
                    for i in range(len(mu)):
                        for j in range(self.y_len):
                            temp[i,j] = Q[i,j] ** 2 / P[i,j]
                    for i in range(len(mu)):
                        for j in range(self.y_len):
                            P[i,j] = temp[i,j]/ np.sum(temp[i,:])
                    #now P contains what I wanted
                    del temp
                    if (np.sum(y_pred != y_pred_old)< tol*dataloader.batch_size):    #cannot improve this batch anymore
                        break
                loss_g = self.discriminator_loss(torch.cat((depths_in, tnfs_in), dim=1), torch.cat((detphs_out, tnfs_out), dim=1), device)
                loss_d = torch.nn.MSEloss(torch.cat((depths_in, tnfs_in), dim=1), torch.cat((detphs_out, tnfs_out), dim=1))
                loss_cls = torch.nn.BCEWithLogitsLoss(torch.cat((detphs_out, tnfs_out), dim=1), torch.zeros_like(torch.cat((detphs_out, tnfs_out), dim=1), device=device))
                loss_cls += self.kld(P,Q)
                if (h % aux_iter <= (aux_iter/2)):
                    self.optimizer_D.zero_grad()
                    loss_d.backward()
                    self.optimizer_D.step()                             
                else:
                    self.optimizer_E.zero_grad()
                    loss_cls.backward()
                    C = C - self.lr * loss_cls.grad
                    self.optimizer_E.step() 
                    self.optimizer_C.step()
                    self.optimizer_D.zero_grad()
                    loss_d.backward()
                    self.optimizer_D.step()         
                    optimizer_D_z.zero_grad()
                    loss_g.backward()
                    optimizer_D_z.step()
                
                        
            time_epoch_1 = time.time()
            time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)
        '''    
        return
    
    def student_t_distribution(self, z, c):
        # Calcola la distanza tra i vettori z_i e c_j
        dist_squared = np.sum((z - c) ** 2)
        # Calcola il numeratore della formula q_ij
        numerator = (1 + dist_squared / self.degrees) ** (- (self.degrees + 1) / 2)
        return numerator