"""Adversarial autoencoders (AAE) for metagenomics binning, this files contains the implementation of the AAE_DEC"""


import numpy as np
from math import log
import time
from torch.utils.data.dataset import TensorDataset as TensorDataset
from torch.autograd import Variable
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import math

from torch.utils.data import DataLoader as _DataLoader

from typing import Optional
from argparse import Namespace

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

############################################################################# MODEL ###########################################################
class AAEDEC(nn.Module):
    def __init__(
        self,
        ntnf: int, #103
        nsamples: int,
        nhiddens: int, #but it should be a list!
        nlatent_y, #careful: nlatent_y should be the number of estimated clusters
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
        optimizer_awl = None,
        degrees: int = 1
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
        self.optimizer_awl = optimizer_awl

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_len, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
        )
        # latent layers
        self.mu = nn.Linear(self.h_n, self.y_len)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.y_len, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.input_len),
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
        logvar = self.logvar(x)
        _y = self.y_vector(x)
        y = F.softmax(_y, dim=1)

        return mu, logvar, y

    def y_length(self):
        return self.y_len

    def z_length(self):
        return self.ld

    def samples_num(self):
        return self.nsamples

    ## Decoder
    def _decode(self, z, y):
        z_y = torch.cat((z, y), 1)

        reconstruction = self.decoder(z_y)

        _depths_out, tnf_out = (
            reconstruction[:, : self.nsamples],
            reconstruction[:, self.nsamples :],
        )

        depths_out = F.softmax(_depths_out, dim=1)

        return depths_out, tnf_out

    ## Discriminator Z space (continuous latent space defined by mu and sigma layers)
    def _discriminator_z(self, z):
        return self.discriminator_z(z)

    ## Discriminator Y space (categorical latent space defined by Y layer)

    def _discriminator_y(self, y):
        return self.discriminator_y(y)

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

    def forward(self, depths_in, tnfs_in, z_prior, y_prior):
        mu, logvar, y_latent = self._encode(depths_in, tnfs_in)
        z_latent = self._reparameterization(mu, logvar)
        depths_out, tnfs_out = self._decode(z_latent, y_latent)
        d_z_latent = self._discriminator_z(z_latent)
        d_y_latent = self._discriminator_y(y_latent)

        return mu, logvar, depths_out, tnfs_out, z_latent, y_latent, d_z_latent, d_y_latent

    # ----------
    #  Training
    # ----------

    def trainepoch(self, epoch_i, data_loader, logfile, hparams, Tensor, T, adversarial_loss, awl=None):
        self.train()
        (
            ED_loss_e,
            loss_e,
            D_z_loss_e,
            D_y_loss_e,
            V_loss_e,
            CE_e,
            SSE_e,
        ) = (0, 0, 0, 0, 0, 0, 0)

        total_batches_inthis_epoch = len(data_loader)
        time_epoch_0 = time.time()

        #AAMB
        if hparams == Namespace():
          for depths_in, tnfs_in, _ in data_loader:
                
                nrows, _ = depths_in.shape

                # Adversarial ground truths

                labels_prior = Variable(
                    Tensor(nrows, 1).fill_(1.0), requires_grad=False
                )
                labels_latent = Variable(
                    Tensor(nrows, 1).fill_(0.0), requires_grad=False
                )


                depths_in.requires_grad = True
                tnfs_in.requires_grad = True


                if self.usecuda:
                  z_prior = torch.cuda.FloatTensor(nrows, self.ld).normal_()
                  z_prior.cuda()
                  ohc = RelaxedOneHotCategorical(
                      torch.tensor([T], device="cuda"),
                      torch.ones([nrows, self.y_len], device="cuda"),
                  )
                  y_prior = ohc.sample()
                  y_prior = y_prior.cuda()

                else:
                    z_prior = Variable(
                        Tensor(np.random.normal(0, 1, (nrows, self.ld)))
                    )
                    ohc = RelaxedOneHotCategorical(
                        T, torch.ones([nrows, self.y_len])
                    )
                    y_prior = ohc.sample()

                del ohc

                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnfs_in = tnfs_in.cuda()

                self.optimizer_E.zero_grad()
                self.optimizer_D.zero_grad()

                (
                  mu,
                  logvar,
                  depths_out,
                  tnfs_out,
                  z_latent,
                  y_latent,
                  d_z_latent,
                  d_y_latent,
                  ) = self(depths_in, tnfs_in, z_prior, y_prior)

                vae_loss, ce, sse = self.calc_loss(
                  depths_in, depths_out, tnfs_in, tnfs_out
                )
                g_loss_adv_z = adversarial_loss(
                    self._discriminator_z(z_latent), labels_prior
                )
                g_loss_adv_y = adversarial_loss(
                    self._discriminator_y(y_latent), labels_prior
                )

                #Loss function L
                ed_loss = (
                    (1 - self.sl) * vae_loss
                    + (self.sl * self.slr) * g_loss_adv_z
                    + (self.sl * (1 - self.slr)) * g_loss_adv_y
                )

                ed_loss.backward()
                self.optimizer_E.step()
                self.optimizer_D.step()

                # ----------------------
                #  Train Discriminator z
                # ----------------------

                self.optimizer_D_z.zero_grad()
                mu, logvar = self._encode(depths_in, tnfs_in)[:2]
                z_latent = self._reparameterization(mu, logvar)

                d_z_loss_prior = adversarial_loss(
                    self._discriminator_z(z_prior), labels_prior
                )
                d_z_loss_latent = adversarial_loss(
                    self._discriminator_z(z_latent), labels_latent
                )
                d_z_loss = 0.5 * (d_z_loss_prior + d_z_loss_latent)

                d_z_loss.backward()
                self.optimizer_D_z.step()

                # ----------------------
                #  Train Discriminator y
                # ----------------------

                self.optimizer_D_y.zero_grad()
                y_latent = self._encode(depths_in, tnfs_in)[2]
                d_y_loss_prior = adversarial_loss(
                    self._discriminator_y(y_prior), labels_prior
                )
                d_y_loss_latent = adversarial_loss(
                    self._discriminator_y(y_latent), labels_latent
                )
                d_y_loss = 0.5 * (d_y_loss_prior + d_y_loss_latent)

                d_y_loss.backward()
                self.optimizer_D_y.step()

                ED_loss_e += float(ed_loss.item())
                V_loss_e += float(vae_loss.item())
                D_z_loss_e += float(d_z_loss.item())
                D_y_loss_e += float(d_y_loss.item())
                CE_e += float(ce.item())
                SSE_e += float(sse.item())

                time_epoch_1 = time.time()
                time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)
          if logfile is not None:
                print(
                    "\tEpoch: {}\t Loss Enc/Dec: {:.6f}\t Rec. loss: {:.4f}\t CE: {:.4f}\tSSE: {:.4f}\t Dz loss: {:.7f}\t Dy loss: {:.6f}\t Batchsize: {}\t Epoch time(min): {: .4}".format(
                          epoch_i + 1,
                          ED_loss_e / total_batches_inthis_epoch,
                          V_loss_e / total_batches_inthis_epoch,
                          CE_e / total_batches_inthis_epoch,
                          SSE_e / total_batches_inthis_epoch,
                          D_z_loss_e / total_batches_inthis_epoch,
                          D_y_loss_e / total_batches_inthis_epoch,
                          data_loader.batch_size,
                          time_e,
                    ), file=logfile)

                logfile.flush()
        #CLAMB
        else:
            for depths_in, tnfs_in, tnf_aug1, tnf_aug2 in  data_loader:  # weights currently unused here
                nrows, _ = depths_in.shape

                depths_in.requires_grad = True
                tnfs_in.requires_grad = True
                tnf_aug1.requires_grad = True
                tnf_aug2.requires_grad = True

                # Adversarial ground truths

                labels_prior = Variable(
                    Tensor(nrows, 1).fill_(1.0), requires_grad=False
                )
                labels_latent = Variable(
                    Tensor(nrows, 1).fill_(0.0), requires_grad=False
                )

                # Sample noise as discriminator Z,Y ground truth

                if self.usecuda:
                    z_prior = torch.cuda.FloatTensor(nrows, self.ld).normal_()
                    z_prior.cuda()
                    ohc = RelaxedOneHotCategorical(
                        torch.tensor([T], device="cuda"),
                        torch.ones([nrows, self.y_len], device="cuda"),
                    )
                    y_prior = ohc.sample()
                    y_prior = y_prior.cuda()

                else:
                    z_prior = Variable(
                        Tensor(np.random.normal(0, 1, (nrows, self.ld)))
                    )
                    ohc = RelaxedOneHotCategorical(
                        T, torch.ones([nrows, self.y_len])
                    )
                    y_prior = ohc.sample()

                del ohc

                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnfs_in = tnfs_in.cuda()
                    tnf_aug1 = tnf_aug1.cuda()
                    tnf_aug2 = tnf_aug2.cuda()

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_E.zero_grad()
                self.optimizer_D.zero_grad()
                self.optimizer_awl.zero_grad()

                # Forward pass
                (
                    mu,
                    logvar,
                    depths_out,
                    tnfs_out,
                    z_latent,
                    y_latent,
                    d_z_latent,
                    d_y_latent,
                ) = self(depths_in, tnfs_in, z_prior, y_prior)

                mu1, logvar1, depths_out1, tnf_out_aug1, z_latent1, y_latent1, d_z_latent1, d_y_latent1 = self(depths_in, tnf_aug1, z_prior, y_prior)
                mu2, logvar2, depths_out2, tnf_out_aug2, z_latent2, y_latent2, d_z_latent2, d_y_latent2 = self(depths_in, tnf_aug2, z_prior, y_prior)

                loss_contrast1 = self.nt_xent_loss(tnf_out_aug1, tnf_out_aug2, temperature=hparams.temperature) #shouldn't be self.nt_xent_loss(torch.cat((depths_out1, tnf_out1), 1), torch.cat((depths_out2, tnf_out2), 1), temperature=hparams.temperature)?
                loss_contrast2 = self.nt_xent_loss(tnf_out_aug2, tnfs_out, temperature=hparams.temperature)
                loss_contrast3 = self.nt_xent_loss(tnfs_out, tnf_out_aug1, temperature=hparams.temperature)

                vae_loss, ce, sse = self.calc_loss(
                    depths_in, depths_out, tnfs_in, tnfs_out
                )
                g_loss_adv_z = adversarial_loss(
                    self._discriminator_z(z_latent), labels_prior
                )
                g_loss_adv_y = adversarial_loss(
                    self._discriminator_y(y_latent), labels_prior
                )

                #Loss function L
                ed_loss = (
                    (1 - self.sl) * vae_loss
                    + (self.sl * self.slr) * g_loss_adv_z
                    + (self.sl * (1 - self.slr)) * g_loss_adv_y
                )

                #Final loss function with contrastive learning
                # NOTE: Add weight to avoid gradient disappearance
                loss = awl(hparams.sigma*loss_contrast1, hparams.sigma*loss_contrast2, hparams.sigma*loss_contrast3) + 10000*ed_loss

                loss.backward()
                self.optimizer_E.step()
                self.optimizer_D.step()

                self.optimizer_awl.step()

                # ----------------------
                #  Train Discriminator z
                # ----------------------

                self.optimizer_D_z.zero_grad()
                mu, logvar = self._encode(depths_in, tnfs_in)[:2]
                z_latent = self._reparameterization(mu, logvar)

                d_z_loss_prior = adversarial_loss(
                    self._discriminator_z(z_prior), labels_prior
                )
                d_z_loss_latent = adversarial_loss(
                    self._discriminator_z(z_latent), labels_latent
                )
                d_z_loss = 0.5 * (d_z_loss_prior + d_z_loss_latent)

                d_z_loss.backward()
                self.optimizer_D_z.step()

                # ----------------------
                #  Train Discriminator y
                # ----------------------

                self.optimizer_D_y.zero_grad()
                y_latent = self._encode(depths_in, tnfs_in)[2]
                d_y_loss_prior = adversarial_loss(
                    self._discriminator_y(y_prior), labels_prior
                )
                d_y_loss_latent = adversarial_loss(
                    self._discriminator_y(y_latent), labels_latent
                )
                d_y_loss = 0.5 * (d_y_loss_prior + d_y_loss_latent)

                d_y_loss.backward()
                self.optimizer_D_y.step()

                ED_loss_e += float(ed_loss.item())
                loss_e += float(loss.item())
                V_loss_e += float(vae_loss.item())
                D_z_loss_e += float(d_z_loss.item())
                D_y_loss_e += float(d_y_loss.item())
                CE_e += float(ce.item())
                SSE_e += float(sse.item())

                time_epoch_1 = time.time()
                time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)

                if logfile is not None:
                    print(
                        "\tEpoch: {}\t Loss: {:.6f}\t Loss Enc/Dec: {:.6f}\t Rec. loss: {:.4f}\t CE: {:.4f}\tSSE: {:.4f}\t Dz loss: {:.7f}\t Dy loss: {:.6f}\t Batchsize: {}\t Epoch time(min): {: .4}".format(
                            epoch_i + 1,
                            loss_e / total_batches_inthis_epoch,
                            ED_loss_e / total_batches_inthis_epoch,
                            V_loss_e / total_batches_inthis_epoch,
                            CE_e / total_batches_inthis_epoch,
                            SSE_e / total_batches_inthis_epoch,
                            D_z_loss_e / total_batches_inthis_epoch,
                            D_y_loss_e / total_batches_inthis_epoch,
                            data_loader.batch_size,
                            time_e,
                        ),
                        file=logfile,
                    )
                    logfile.flush()

        return None
    def get_latents(self, contignames, data_loader, last_epoch=True):
        """Retrieve the categorical latent representation (y) and the contiouous latents (l) of the inputs

        Inputs:
            dataloader
            contignames
            last_epoch


        Output:
            y_clusters_dict ({clust_id : [contigs]})
            l_latents array""" #to be clustered with k-medoids
        self.eval()

        new_data_loader = _DataLoader(
            dataset=data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
            pin_memory=data_loader.pin_memory,
        )

        depths_array, _, _ = data_loader.dataset.tensors

        length = len(depths_array)
        latent = np.empty((length, self.ld), dtype=np.float32)
        index_contigname = 0
        row = 0
        clust_y_dict = dict()
        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
        with torch.no_grad():

            for depths_in, tnfs_in, _ in new_data_loader:
                nrows, _ = depths_in.shape
                I = torch.cat((depths_in, tnfs_in), dim=1)

                if self.usecuda:
                    z_prior = torch.cuda.FloatTensor(nrows, self.ld).normal_()
                    z_prior.cuda()
                    ohc = RelaxedOneHotCategorical(
                        torch.tensor([0.15], device="cuda"),
                        torch.ones([nrows, self.y_len], device="cuda"),
                    )
                    y_prior = ohc.sample()
                    y_prior = y_prior.cuda()

                else:
                    z_prior = Variable(
                        Tensor(np.random.normal(0, 1, (nrows, self.ld)))
                    )
                    ohc = RelaxedOneHotCategorical(
                        0.15, torch.ones([nrows, self.y_len])
                    )
                    y_prior = ohc.sample()

                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnfs_in = tnfs_in.cuda()

                if last_epoch:
                    mu, _, _, _, _, y_sample = self(depths_in, tnfs_in, z_prior, y_prior)[
                        0:6
                    ]
                else:
                    y_sample = self(depths_in, tnfs_in, z_prior, y_prior)[5]

                if self.usecuda:
                    Ys = y_sample.cpu().detach().numpy()
                    if last_epoch:
                        mu = mu.cpu().detach().numpy()
                        latent[row : row + len(mu)] = mu
                        row += len(mu)
                else:
                    Ys = y_sample.detach().numpy()
                    if last_epoch:
                        mu = mu.detach().numpy()
                        latent[row : row + len(mu)] = mu
                        row += len(mu)
                del y_sample

                for _y in Ys:
                    contig_name = contignames[index_contigname]
                    contig_cluster = np.argmax(_y) + 1
                    contig_cluster_str = str(contig_cluster)

                    if contig_cluster_str not in clust_y_dict:
                        clust_y_dict[contig_cluster_str] = set()

                    clust_y_dict[contig_cluster_str].add(contig_name)

                    index_contigname += 1
                del Ys

            if last_epoch:
                return clust_y_dict, latent
            else:
                return clust_y_dict

    def nt_xent_loss(self, out_1, out_2, temperature=2, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)

        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # sum by dim, we set dim=1 since our data are sequences
        # [2 * batch_size, 2 * batch_size * world_size] or 1
        # L2_norm = _torch.mm(_torch.sum(_torch.pow(out,2),dim=1,keepdim=True), _torch.sum(_torch.pow(out_dist.t().contiguous(),2),dim=0,keepdim=True))
        # L2_norm = _torch.clamp(L2_norm, min=eps)
        L2_norm = 1.

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.div(torch.mm(out, out_dist.t().contiguous()), L2_norm)
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()
        #print('out',out,cov,sim,neg,row_sub,pos)

        return loss
    

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

        optimizer_E = torch.optim.Adam(enc_params, lr=lr)
        optimizer_D = torch.optim.Adam(dec_params, lr=lr)
        optimizer_C = torch.optim.Adam(cri_params, lr=cri_lr)

        
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
                optimizer_E.step()
                optimizer_D.step()
                crit_loss.backward()
                optimizer_C.step()

                ed_loss += ed_loss.item()
                crit_loss += crit_loss.item()
                
                time_epoch_1 = time.time()
                time_e = np.round((time_epoch_1 - time_epoch_0) / 60, 3)
                
        return optimizer_E, optimizer_D

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
    def train(self, dataloader, max_iter, aux_iter, max_iter_dis, C, targ_iter, tol, lrate,
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

        optimizer_D_z = torch.optim.Adam(disc_params, lr=lrate)   

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
                optimizer_D_z.zero_grad()
                _, detphs_out, tnfs_out = self(depths_in, tnfs_in)
                loss = self.discriminator_loss(torch.cat((depths_in, tnfs_in), dim=1), torch.cat((detphs_out, tnfs_out), dim=1), device)
                loss.backward()
                optimizer_D_z.step()
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
            for depths_in, tnf_in in data_loader:
                depths_in.requires_grad = True
                tnf_in.requires_grad = True
                if h%targ_iter == 0:
                    mu = self._encode(depths_in, tnfs_in)
                    for i in range(len(mu)):    #consider each sample separately
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
            
        return
    
    def student_t_distribution(self, z, c):
        # Calcola la distanza tra i vettori z_i e c_j
        dist_squared = np.sum((z - c) ** 2)
        # Calcola il numeratore della formula q_ij
        numerator = (1 + dist_squared / self.degrees) ** (- (self.degrees + 1) / 2)
        return numerator