"""Adversarial autoencoders (AAE) for metagenomics binning, this files contains the implementation of the AAE_DEC"""


import numpy as np
from collections.abc import Sequence
from torch.utils.data.dataset import TensorDataset as _TensorDataset
import time
from torch.utils.data.dataset import TensorDataset as TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import random

from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
'''
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
'''

import torch.nn.functional as F

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
        ntnf: int,
        nsamples: int,
        nhiddens: int = 10,
        nlatent_y: int = 100, #careful: nlatent_y should be the number of estimated clusters
        lr: float = 1e-3,
        cri_lr: float = 1e-3,
        dis_lr: float = 1e-3,
        _cuda: bool = False,
        optimizer_E = None,
        optimizer_D = None,
        optimizer_G = None,
        optimizer_C = None,
        optimizer_clusters = None,
        degrees: int = 1,
        n_enc_1: int = 500,
        n_enc_2: int = 1000,
        alpha: int = 1,
    ):
        if nsamples is None:
            raise ValueError(
                f"Number of samples  should be provided to define the encoder input layer as well as the categorical latent dimension, not {nsamples}"
            )
        
        if nhiddens is None:
            nhiddens = 10

        super(AAEDEC, self).__init__()

        self.nsamples = nsamples
        self.ntnf = ntnf
        self.h_n = nhiddens
        self.y_len = nlatent_y
        self.input_len = int(self.ntnf + self.nsamples)
        self.usecuda = _cuda
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
        self.alpha = alpha

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_len, self.n_enc_1),
            nn.BatchNorm1d(self.n_enc_1),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_1, self.n_enc_2),
            nn.BatchNorm1d(self.n_enc_2),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_2, self.h_n),
            nn.Identity()
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.h_n, self.n_enc_2),
            nn.BatchNorm1d(self.n_enc_2),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_2, self.n_enc_1),
            nn.BatchNorm1d(self.n_enc_1),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_1, self.input_len),
            nn.Identity()
        )

        # discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_len, self.n_enc_1),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_1, self.n_enc_2),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_2, self.h_n),
            nn.Sigmoid(),
        )


        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.input_len, self.n_enc_1),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_1, self.n_enc_2),
            nn.LeakyReLU(),
            nn.Linear(self.n_enc_2, self.h_n),
            nn.LeakyReLU(),
        )

        self.cluster_layer = Parameter(torch.Tensor(self.y_len, self.h_n))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        if optimizer_clusters == None:
            self.optimizer_clusters = torch.optim.Adam([self.cluster_layer], lr=self.lr)
        else:
            self.optimizer_clusters = optimizer_clusters
        if optimizer_E != None:
            self.optimizer_E = optimizer_E
            self.optimizer_D = optimizer_D
            self.optimizer_G = optimizer_G
            self.optimizer_C = optimizer_C

        self.usecuda = False
        if _cuda:
            self.cuda()
            self.usecuda = True

    def _critic(self, c):
        return self.critic(c)

    ## Encoder
    def _encode(self, depths, tnfs=None, tocpu = False):
        _input = None
        if tnfs != None:
            _input = torch.cat((depths, tnfs), 1)
        else:
            _input = depths
        if self.usecuda and not tocpu:
            _input = _input.cuda()
        x = self.encoder(_input)

        return x

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
      nhiddens = dictionary['nhiddens']
      nlatent_y = dictionary['nlatent_y']
      optimizer_E = dictionary["optimizer_E"]
      optimizer_D = dictionary["optimizer_D"]
      optimizer_G = dictionary["optimizer_G"]
      optimizer_C = dictionary["optimizer_C"]
      optimizer_clusters = dictionary["optimizer_clusters"]
      state = dictionary['state']

      aae = cls(ntnf, nsamples, nhiddens, nlatent_y, cuda, optimizer_E=optimizer_E,
                optimizer_D=optimizer_D, optimizer_G=optimizer_G, optimizer_C=optimizer_C, optimizer_clusters=optimizer_clusters)
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
                 'nhiddens': self.h_n,
                 'nlatent_y': self.y_len,
                 "optimizer_E": self.optimizer_E.state_dict(),
                 "optimizer_D": self.optimizer_D.state_dict(),
                 "optimizer_G": self.optimizer_G.state_dict(),
                 "optimizer_C": self.optimizer_D.state_dict(),
                 "optimizer_clusters": self.optimizer_clusters.state_dict(),
                 "n_enc_1": self.n_enc_1,
                 "n_enc_2": self.n_enc_2,
                 'state': self.state_dict(),
                }

        torch.save(state, filehandle)

    def forward(self, depths_in, tnfs_in=None, tocpu=False):
        mu = self._encode(depths_in, tnfs_in, tocpu)
        depths_out, tnfs_out = self._decode(mu)

        return mu, depths_out, tnfs_out
    
    def get_q(self, depths_in, tnfs_in=None, tocpu=False):
        if tocpu and self.usecuda:  #should be in cuda, but it's unfeasible
            self.cpu()
            depths_in.to("cpu")
            if tnfs_in != None:
                tnfs_in.to("cpu")
        mu, depths_out, tnfs_out = self(depths_in, tnfs_in, tocpu)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(mu.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        if tocpu and self.usecuda:
            self.cuda()
        return q, mu, depths_out, tnfs_out

    def pretrain(self, dataloader, max_iter, logfile):
        lr = self.lr
        cri_lr = self.cri_lr
        depthstensor = dataloader.dataset.tensors[0]
        ncontigs, _ = depthstensor.shape

        # Initialize generator and critic

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tAlpha:", self.alpha, file=logfile)
            print("\tN of clusters:", self.y_len, file=logfile)
            print("\n\tPretraining generator and critic properties:", file=logfile)
            print("\tN epochs:", max_iter, file=logfile)
            print("\tStarting batch size:", dataloader.batch_size, file=logfile)
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
        begintime = time.time()/60
        for iter in range(max_iter):
            time_epoch_0 = time.time()
            (
                CRIT_LOSS,
                ED_LOSS,
            ) = (0,0)

            data_loader = _DataLoader(dataset=dataloader.dataset,
                                    batch_size=dataloader.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
            
            for depths_in, tnfs_in, _, _ in data_loader:
                depths_in.requires_grad = True
                tnfs_in.requires_grad = True
                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnfs_in = tnfs_in.cuda()
                a = random.random()
                b = random.random()
                s = random.random()
                while(b==a):
                    b = random.random()
                while(s==a or s == b):
                    s = random.random()
    
                random_indices = list(torch.utils.data.RandomSampler(dataloader.dataset, replacement=False, num_samples=2))
                depth1, depth2 = [dataloader.dataset[i][0] for i in random_indices]
                tnf1, tnf2 = [dataloader.dataset[i][1] for i in random_indices]

                del random_indices

                depth1 = depth1.unsqueeze(0)
                depth2 = depth2.unsqueeze(0)
                tnf1 = tnf1.unsqueeze(0)
                tnf2 = tnf2.unsqueeze(0)
                depths = torch.cat([depth1, depth2], dim=0)
                tnfs = torch.cat([tnf1, tnf2], dim=0)

                if self.usecuda:
                    depths.cuda()
                    tnfs.cuda()

                # Process the two random samples
                factor1 = torch.tensor(a)
                factor2 = 1 - factor1
                if self.usecuda:
                    factor1.cuda()
                    factor2.cuda()
                

                ###TRAINING Autoencoder
                self.optimizer_E.zero_grad()
                self.optimizer_D.zero_grad()
                mu = self._encode(depths_in, tnfs_in)
                depths_out, tnfs_out = self._decode(mu)

                mu = self._encode(depths, tnfs)
                mu[0] *=factor1
                mu[1] *=factor2
                z = mu[0] + mu[1]
                z = z.unsqueeze(0)
                z = torch.cat((z, z), dim = 0)
                r_depths_out, r_tnfs_out = self._decode(z)
                x = torch.cat((r_depths_out, r_tnfs_out), 1)
                ins = torch.cat((depths_in, tnfs_in), dim=1)
                outs = torch.cat((depths_out, tnfs_out), dim=1)

                ed_loss = (ins - outs).pow(2).sum(dim=1).mean()
                ed_loss += s*(self._critic(x)[0].pow(2).mean())
                ed_loss.backward()                
                self.optimizer_E.step()
                self.optimizer_D.step()


                ###TRAINING Critic
                self.optimizer_C.zero_grad()
                mu = self._encode(depths_in, tnfs_in)
                depths_out, tnfs_out = self._decode(mu)
                outs = torch.cat((depths_out, tnfs_out), dim=1)

                mu = self._encode(depths, tnfs)
                mu[0] *=factor1
                mu[1] *=factor2
                z = mu[0] + mu[1]
                z = z.unsqueeze(0)
                z = torch.cat((z, z), dim = 0)
                r_depths_out, r_tnfs_out = self._decode(z)
                x = torch.cat((r_depths_out, r_tnfs_out), 1)

                reg_term = self._critic(b*ins + (1-b)*outs).pow(2).sum(dim=1).mean()
                crit_loss = (self._critic(x)[0] - factor1).pow(2).mean() + reg_term
                crit_loss.backward()
                self.optimizer_C.step()



                ED_LOSS += ed_loss.item()
                CRIT_LOSS += crit_loss.item()
                
                time_epoch_1 = time.time()
                time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)
            
            total_batches_inthis_epoch = len(data_loader)
            if logfile is not None:
                print(
                    "\tEpoch: {}\t Loss Enc/Dec: {:.6f}\t Cri. loss: {:.4f}\t Batchsize: {}\t Epoch time(min): {: .4}".format(
                          iter + 1,
                          ED_LOSS / total_batches_inthis_epoch,
                          CRIT_LOSS / total_batches_inthis_epoch,
                          data_loader.batch_size,
                          time_e,
                    ), file=logfile)

                logfile.flush()
        timepoint_gernerate_input=time.time()/60
        time_generating_input= round(timepoint_gernerate_input-begintime,2)   
        if logfile is not None:
            print(f"\nPretraining in {time_generating_input} minutes", file=logfile)   
        return

    def discriminator_loss(self, real_output, fake_output, device):
        real_output = torch.mean(self.discriminator(real_output), dim=1, keepdim=True)
        fake_output = torch.mean(self.discriminator(fake_output), dim=1, keepdim=True)
        loss = torch.nn.BCEWithLogitsLoss()
        real_loss = loss(real_output, torch.ones_like(real_output, device=device))
        fake_loss = loss(fake_output, torch.zeros_like(fake_output, device=device))
        total_loss = real_loss + fake_loss
        return total_loss, fake_loss


    def student_t_distribution(self, q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
    
    def trainmodel(self, dataloader, max_iter, aux_iter, max_iter_dis, targ_iter, tol, lrate, modelfile=None,
              logfile=None):
        #C must be a torch.tensor of centroids (vectors), implicitly labeled by order
        
        device = "cuda" if self.usecuda else "cpu"

        # Pretrain discriminators

        if logfile is not None:
            print("\n\tPretraining discriminators properties:", file=logfile)
            print("\tN Pretraining epochs:", max_iter_dis, file=logfile)
            print("\tStarting batch size:", dataloader.batch_size, file=logfile, end="\n\n")

        disc_params = []

        for name, param in self.named_parameters():
            if "discriminator" in name:
                disc_params.append(param)
        if self.optimizer_G==None:
            self.optimizer_G = torch.optim.Adam(disc_params, lr=lrate)  
        self.optimizer_E.param_groups[0]['lr'] = lrate 
        self.optimizer_D.param_groups[0]['lr'] = self.dis_lr
        self.optimizer_clusters.param_groups[0]['lr'] = lrate

        
        begintime = time.time()/60
        for i in range(max_iter_dis):
            G_loss = 0
            time_epoch_0 = time.time()
            self.train()
            data_loader = _DataLoader(dataset=dataloader.dataset,
                                    batch_size=dataloader.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
            
            for depths_in, tnfs_in, _, _ in data_loader:
            
                depths_in.requires_grad = True
                tnfs_in.requires_grad = True
                self.optimizer_G.zero_grad()
                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnfs_in = tnfs_in.cuda()
                _, detphs_out, tnfs_out = self(depths_in, tnfs_in)
                loss,_ = self.discriminator_loss(torch.cat((depths_in, tnfs_in), dim=1), torch.cat((detphs_out, tnfs_out), dim=1), device)
                loss.backward()
                self.optimizer_G.step()
                G_loss += loss.item()

                time_epoch_1 = time.time()
                time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)
            
            total_batches_inthis_epoch = len(data_loader)
            if logfile is not None:
                print(
                    "\tEpoch: {}\t Loss Discriminator: {:.6f}\t Batchsize: {}\t Epoch time(min): {: .4}".format(
                          i + 1,
                          G_loss / total_batches_inthis_epoch,
                          data_loader.batch_size,
                          time_e,
                    ), file=logfile)

                logfile.flush()
        timepoint_gernerate_input=time.time()/60
        time_generating_input= round(timepoint_gernerate_input-begintime,2)   
        if logfile is not None:
            print(f"\nPretraining discriminator and initializing clusters in {time_generating_input} minutes", file=logfile)   
        #Clustering phase

        if logfile is not None:
            print("\n Training properties:", file=logfile)
            print("\tN Auxiliary epochs:", aux_iter, file=logfile)
            print("\tN Training epochs:", max_iter, file=logfile)
            print("\tN Target iters:", targ_iter, file=logfile)
            print("\tN Tolerance:", tol, file=logfile)
            print("\tStarting batch size:", dataloader.batch_size, file=logfile, end="\n\n")

        class MyDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                depths_in, tnfs_in = self.data[idx]
                return depths_in, tnfs_in, idx
        depthstensor, tnftensor, _, _ = dataloader.dataset.tensors
        data_loader = _DataLoader(dataset=MyDataset(_TensorDataset(depthstensor, tnftensor)),
                                    batch_size=dataloader.batch_size,
                                    shuffle=False,  #If it's true, I think it's a problem as I don't know who I am clustering exactly
                                    drop_last=False,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
        
        data = torch.cat((depthstensor.clone().detach().to(torch.float), tnftensor.clone().detach().to(torch.float)), dim=1).to(device)
        encoded = self._encode(data)
        '''
        initial_centers = kmeans_plusplus_initializer(encoded, self.y_len).initialize()
        y_pred = kmeans(encoded, initial_centers, ccore=False).process().get_clusters()
        '''
        begintime = time.time()/60
        kmeans = KMeans(n_clusters=self.y_len, n_init=20)
        y_pred = kmeans.fit_predict(encoded.data.cpu().numpy())
        timepoint_gernerate_input=time.time()/60
        time_generating_input= round(timepoint_gernerate_input-begintime,2)   
        if logfile is not None:
            print(f"\nCentroids initialized in {time_generating_input} minutes", file=logfile)     
        
        y_pred_old = y_pred
        self.cluster_layer.data = torch.Tensor(kmeans.cluster_centers_).to(device)
        self.optimizer_D.param_groups[0]['lr'] = lrate

        self.train()
        begintime = time.time()/60
        for epoch in range(max_iter):
            time_epoch_0 = time.time()
            (D_loss,
             Cls_loss,
             G_loss,) = (0,0,0)
            if epoch%targ_iter == 0:
                #Compute q and p
                self.cpu()
                tmp_q,_,_,_ = self.get_q(data.cpu(), tocpu=True)
                tmp_q = tmp_q.data
                p = self.student_t_distribution(tmp_q)
                
                #New assignment
                y_pred = tmp_q.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_old).astype(
                    np.float32) / y_pred.shape[0]
                y_pred_old = y_pred
                data.to(device)
                if epoch > 0 and delta_label < tol:
                    if logfile is not None:
                        print(f"\nReached tolerance threshold. Stopping training. Epoch is {epoch} ", file=logfile)
                    break   #cannot improve anymore
            for depths_in, tnfs_in, idx in data_loader:
                
                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnfs_in = tnfs_in.cuda()
                    self.cuda()
                    #note: p.cuda() is not feasible
                depths_in.requires_grad = True
                tnfs_in.requires_grad = True
                #self.cluster_layer.requires_grad = True

                if (epoch % aux_iter <= (aux_iter/2)):
                    self.optimizer_D.zero_grad()
                    q, _, depths_out, tnfs_out = self.get_q(depths_in, tnfs_in)   #if usecuda, self is back on cuda()!
                    loss_d = torch.nn.MSELoss()(torch.cat((depths_in, tnfs_in), dim=1), torch.cat((depths_out, tnfs_out), dim=1))
                    loss_d.backward()
                    self.optimizer_D.step()
                    D_loss += loss_d.item()                             
                else:
                    self.optimizer_D.zero_grad()
                    q, _, depths_out, tnfs_out = self.get_q(depths_in, tnfs_in)   #if usecuda, self is back on cuda()!
                    loss_d = torch.nn.MSELoss()(torch.cat((depths_in, tnfs_in), dim=1), torch.cat((depths_out, tnfs_out), dim=1))
                    loss_d.backward()
                    self.optimizer_D.step() 
                    D_loss += loss_d.item()  

                    self.optimizer_E.zero_grad()
                    #self.cluster_layer.grad.zero_()
                    self.optimizer_clusters.zero_grad()
                    q, _, depths_out, tnfs_out = self.get_q(depths_in, tnfs_in)
                    _, fake_loss = self.discriminator_loss(torch.cat((depths_in, tnfs_in), dim=1), torch.cat((depths_out, tnfs_out), dim=1), device)
                    loss_cls = fake_loss + F.kl_div(q.cpu().log(), p[idx])
                    loss_cls.backward()
                    self.optimizer_E.step() 
                    self.optimizer_clusters.step()
                    #with torch.no_grad():
                    #    self.cluster_layer.data -= lrate * self.cluster_layer.grad        
                    Cls_loss += loss_cls.item()

                    self.optimizer_G.zero_grad()
                    q, _, depths_out, tnfs_out = self.get_q(depths_in, tnfs_in)
                    loss_g, _ = self.discriminator_loss(torch.cat((depths_in, tnfs_in), dim=1), torch.cat((depths_out, tnfs_out), dim=1), device)
                    loss_g.backward()
                    self.optimizer_G.step()
                    G_loss += loss_g.item()

                time_epoch_1 = time.time()
                time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)
            
            total_batches_inthis_epoch = len(data_loader)
            if logfile is not None:
                print(
                    "\tEpoch: {}\t Loss Enc: {:.6f}\t Loss Dec: {:.6f}\t Discriminator. loss: {:.4f}\t Batchsize: {}\t Epoch time(min): {: .4}".format(
                          epoch + 1,
                          Cls_loss / total_batches_inthis_epoch,
                          D_loss / total_batches_inthis_epoch,
                          G_loss / total_batches_inthis_epoch,
                          data_loader.batch_size,
                          time_e,
                    ), file=logfile)

                logfile.flush()
        timepoint_gernerate_input=time.time()/60
        time_generating_input= round(timepoint_gernerate_input-begintime,2)   
        if logfile is not None:
            print(f"Training and clustering in {time_generating_input} minutes", file=logfile)   
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass
        return y_pred   #return the clustering

    def get_dict(
        self, contignames: Sequence[str], y_pred
    ) -> tuple[dict[str, set[str]]]:
        """Retrieve the categorical latent representation (y) of the inputs

        Inputs:
            dataloader
            contignames

        Output:
            y_clusters_dict ({clust_id : [contigs]})
        """
        self.eval()
        index_contigname = 0
        clust_y_dict: dict[str, set[str]] = dict()
        with torch.no_grad():
                for _y in y_pred:
                    contig_name = contignames[index_contigname]
                    contig_cluster = _y + 1
                    contig_cluster_str = str(contig_cluster)

                    if contig_cluster_str not in clust_y_dict:
                        clust_y_dict[contig_cluster_str] = set()

                    clust_y_dict[contig_cluster_str].add(contig_name)

                    index_contigname += 1
                del y_pred
        return clust_y_dict