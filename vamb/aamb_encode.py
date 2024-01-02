"""Adversarial autoencoders (AAE) for metagenomics binning, this files contains the implementation of the AAE"""


import numpy as np
from math import log
import time
from torch.utils.data.dataset import TensorDataset as TensorDataset
from torch.autograd import Variable
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
import torch.nn as nn
import torch.nn.functional as F
import torch
import math as _math
from glob import glob
import warnings
import random

from torch.utils.data import DataLoader as _DataLoader
from argparse import Namespace as _Namespace

from typing import Optional
from vamb.encode import AutomaticWeightedLoss
import vamb.vambtools

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

############################################################################# MODEL ###########################################################
class AAE(nn.Module):
    def __init__(
        self,
        ntnf: int, #103
        nsamples: int,
        nhiddens: int, #but it should be a list!
        nlatent_l: int,
        nlatent_y, #careful: nlatent_y should be the number of extimated clusters
        sl: float,
        slr: float,
        alpha: Optional[float],
        _cuda: bool,
        k: int = 4,
        contrast: bool = False,
    ):
        if nsamples is None:
            raise ValueError(
                f"Number of samples  should be provided to define the encoder input layer as well as the categorical latent dimension, not {nsamples}"
            )

        super(AAE, self).__init__()
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        self.nsamples = nsamples
        self.ntnf = ntnf
        self.k = k
        self.h_n = nhiddens
        self.ld = nlatent_l
        self.y_len = nlatent_y
        self.input_len = int(self.ntnf + self.nsamples)
        self.sl = sl
        self.slr = slr
        self.alpha = alpha
        self.usecuda = _cuda
        self.contrast = contrast

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
        self.mu = nn.Linear(self.h_n, self.ld)
        self.logvar = nn.Linear(self.h_n, self.ld)
        self.y_vector = nn.Linear(self.h_n, self.y_len)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(int(self.ld + self.y_len), self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.h_n),
            nn.BatchNorm1d(self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, self.input_len),
        )

        # discriminator z
        self.discriminator_z = nn.Sequential(
            nn.Linear(self.ld, self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, int(self.h_n / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.h_n / 2), 1),
            nn.Sigmoid(),
        )

        # discriminator Y
        self.discriminator_y = nn.Sequential(
            nn.Linear(self.y_len, self.h_n),
            nn.LeakyReLU(),
            nn.Linear(self.h_n, int(self.h_n / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.h_n / 2), 1),
            nn.Sigmoid(),
        )

        if _cuda:
            self.cuda()

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
            k = dictionary["k"]
      except KeyError:
            k = 4
      state = dictionary['state']

      aae = cls(ntnf, nsamples, nhiddens, nlatent_l, nlatent_y, alpha, sl, slr, cuda, k=k, c=c)
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
                 'ntfs': self.ntnf,
                 'nsamples': self.nsamples,
                 'alpha': self.alpha,
                 'nhiddens': self.h_n,
                 'nlatent_l': self.ld,
                 'nlatent_y': self.y_len,
                 'sl' : self.sl,
                 'slr': self.slr,
                 'k' : self.k,
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

    def trainepoch(self, epoch_i, data_loader, logfile, hparams, optimizer_E, optimizer_D, optimizer_D_y, optimizer_D_z, Tensor, T, modelfile, adversarial_loss, awl=None, optimizer_awl=None):
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
        if hparams == _Namespace():
          for depths_in, tnf_in in data_loader:
                depths_in.requires_grad = True
                tnf_in.requires_grad = True


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
                    tnf_in = tnf_in.cuda()

                optimizer_E.zero_grad()
                optimizer_D.zero_grad()

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
                optimizer_E.step()
                optimizer_D.step()

                # ----------------------
                #  Train Discriminator z
                # ----------------------

                optimizer_D_z.zero_grad()
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
                optimizer_D_z.step()

                # ----------------------
                #  Train Discriminator y
                # ----------------------

                optimizer_D_y.zero_grad()
                y_latent = self._encode(depths_in, tnfs_in)[2]
                d_y_loss_prior = adversarial_loss(
                    self._discriminator_y(y_prior), labels_prior
                )
                d_y_loss_latent = adversarial_loss(
                    self._discriminator_y(y_latent), labels_latent
                )
                d_y_loss = 0.5 * (d_y_loss_prior + d_y_loss_latent)

                d_y_loss.backward()
                optimizer_D_y.step()

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
              optimizer_E.zero_grad()
              optimizer_D.zero_grad()
              optimizer_awl.zero_grad()

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
              optimizer_E.step()
              optimizer_D.step()

              optimizer_awl.step()

              # ----------------------
              #  Train Discriminator z
              # ----------------------

              optimizer_D_z.zero_grad()
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
              optimizer_D_z.step()

              # ----------------------
              #  Train Discriminator y
              # ----------------------

              optimizer_D_y.zero_grad()
              y_latent = self._encode(depths_in, tnfs_in)[2]
              d_y_loss_prior = adversarial_loss(
                  self._discriminator_y(y_prior), labels_prior
              )
              d_y_loss_latent = adversarial_loss(
                  self._discriminator_y(y_latent), labels_latent
              )
              d_y_loss = 0.5 * (d_y_loss_prior + d_y_loss_latent)

              d_y_loss.backward()
              optimizer_D_y.step()

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

        # save model
        if modelfile is not None:
            try:
                checkpoint = {
                    "state": self.state_dict(),
                    "optimizer_E": optimizer_E.state_dict(),
                    "optimizer_D": optimizer_D.state_dict(),
                    "optimizer_D_z": optimizer_D_z.state_dict(),
                    "optimizer_D_y": optimizer_D_y.state_dict(),
                    "nsamples": self.num_samples,
                    "alpha": self.alpha,
                    "nhiddens": self.h_n,
                    "nlatent_l": self.ld,
                    "nlatent_y": self.y_len,
                    "sl": self.sl,
                    "slr": self.slr,
                    "temp": self.T,
                }
                torch.save(checkpoint, modelfile)

            except:
                pass

        return None
    
    def trainmodel(
        self, dataloader, T, nepochs=320, lrate=1e-3,
                   batchsteps=[25, 75, 150, 300], logfile=None, modelfile=None, hparams=None, augmentationpath=None, mask=None
    ):

        Tensor = torch.cuda.FloatTensor if self.usecuda else torch.FloatTensor
        batchsteps_set = set(batchsteps)
        # Get number of features
        depthstensor, tnftensor = dataloader.dataset.tensors
        ncontigs, nsamples = depthstensor.shape

        # Initialize generator and discriminator

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tAlpha:", self.alpha, file=logfile)
            print("\tY length:", self.y_len, file=logfile)
            print("\tZ length:", self.ld, file=logfile)
            print("\n\tTraining properties:", file=logfile)
            print("\tN epochs:", nepochs, file=logfile)
            print("\tStarting batch size:", data_loader.batch_size, file=logfile)
            batchsteps_string = (
                ", ".join(map(str, sorted(batchsteps))) if batchsteps_set else "None"
            )
            print("\tBatchsteps:", batchsteps_string, file=logfile)
            print("\tN sequences:", ncontigs, file=logfile)
            print("\tN samples:", self.nsamples, file=logfile, end="\n\n")

        # we need to separate the paramters due to the adversarial training

        disc_z_params = []
        disc_y_params = []
        enc_params = []
        dec_params = []
        for name, param in self.named_parameters():
            if "discriminator_z" in name:
                disc_z_params.append(param)
            elif "discriminator_y" in name:
                disc_y_params.append(param)
            elif "encoder" in name:
                enc_params.append(param)
            else:
                dec_params.append(param)

        # Define adversarial loss for the discriminators
        adversarial_loss = torch.nn.BCELoss()
        if self.usecuda:
            adversarial_loss.cuda()

        #### Optimizers
        optimizer_E = torch.optim.Adam(enc_params, lr=lrate)
        optimizer_D = torch.optim.Adam(dec_params, lr=lrate)

        optimizer_D_z = torch.optim.Adam(disc_z_params, lr=lrate)
        optimizer_D_y = torch.optim.Adam(disc_y_params, lr=lrate)

        #Contrastive Learning
        if self.contrast:
          awl = AutomaticWeightedLoss(3)
          optimizer_awl = torch.optim.Adam(awl.parameters(), lr=lrate)


          '''Read augmentation data from indexed files. Note that, CLMB can't guarantee an order training with augmented data if the outdir exists.'''
          aug_all_method = ['GaussianNoise','Transition','Transversion','Mutation','AllAugmentation']
          augmentation_count_number = [0, 0]
          augmentation_count_number[0] = len(glob(rf'{augmentationpath+os.sep}pool0*k{self.k}*')) if hparams.augmode[0] == -1 else len(glob(rf'{augmentationpath+os.sep}pool0*k{self.k}*_{aug_all_method[hparams.augmode[0]]}_*'))
          augmentation_count_number[1] = len(glob(rf'{augmentationpath+os.sep}pool1*k{self.k}*')) if hparams.augmode[0] == -1 else len(glob(rf'{augmentationpath+os.sep}pool1*k{self.k}*_{aug_all_method[hparams.augmode[1]]}_*'))

          if augmentation_count_number[0] > _math.ceil(_math.sqrt(nepochs)) or augmentation_count_number[1] > _math.ceil(_math.sqrt(nepochs)):
              warnings.warn('Too many augmented data, augmented data might not be trained enough. CLAMB do not know how this influence the performance', FutureWarning)
          elif augmentation_count_number[0] < _math.ceil(_math.sqrt(nepochs)) or augmentation_count_number[1] > _math.ceil(_math.sqrt(nepochs)):
              raise RuntimeError('Shortage of augmented data. Please regenerate enough augmented data using fasta files, or do not specify the --contrastive option to run VAMB')


          '''Function for shuffling the augmented data (if needed)'''
          def aug_file_shuffle(_count, _augmentationpath, _augdatashuffle=False):
              _shuffle_file1 = random.randrange(0, sum(_count) - 1)
              if _augdatashuffle:
                  _aug_archive1_file = None if _shuffle_file1 < _count[0] else (glob(rf'{_augmentationpath+os.sep}pool0*k{self.k}_*index{_shuffle_file1 % _count[0]}_*') if hparams.augmode[0] == -1 \
                          else glob(rf'{augmentationpath+os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_{aug_all_method[hparams.augmode[0]]}_*'))
              else:
                  _aug_archive1_file = glob(rf'{_augmentationpath+os.sep}pool0*k{self.k}_*index{_shuffle_file1 % _count[0]}_*') if hparams.augmode[0] == -1 \
                          else glob(rf'{augmentationpath+os.sep}pool0*k{self.k}_*index{_shuffle_file1 % _count[0]}_{aug_all_method[hparams.augmode[0]]}_*')
              _shuffle_file2 = random.randrange(0, sum(_count) - 1)
              if _augdatashuffle:
                  _aug_archive2_file = None if _shuffle_file2 < _count[1] else (glob(rf'{_augmentationpath+_os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_*') if hparams.augmode[1] == -1 \
                          else glob(rf'{augmentationpath+os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_{aug_all_method[hparams.augmode[1]]}_*'))
              else:
                  _aug_archive2_file = glob(rf'{_augmentationpath+os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_*') if hparams.augmode[1] == -1 \
                          else glob(rf'{augmentationpath+os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_{aug_all_method[hparams.augmode[1]]}_*')
              return _aug_archive1_file, _aug_archive2_file


          for epoch_i in range(nepochs):

              aug_archive1_file = glob(rf'{augmentationpath+os.sep}pool0*k{self.k}_*index{epoch_i // augmentation_count_number[0]}_*') if hparams.augmode[0] == -1 \
                            else glob(rf'{augmentationpath+os.sep}pool0*k{self.k}_*index{epoch_i // augmentation_count_number[0]}_{aug_all_method[hparams.augmode[0]]}_*')
              aug_archive2_file = glob(rf'{augmentationpath+os.sep}pool1*k{self.k}_*index{epoch_i % augmentation_count_number[1]}_*') if hparams.augmode[1] == -1 \
                      else glob(rf'{augmentationpath+os.sep}pool1*k{self.k}_*index{epoch_i % augmentation_count_number[1]}_{aug_all_method[hparams.augmode[1]]}_*')

              '''If augdatashuffle in on, read augmentation data from shuffled-indexed files'''
              if hparams.augdatashuffle:
                  shuffle_file1, shuffle_file2 = aug_file_shuffle(augmentation_count_number, augmentationpath, hparams.augdatashuffle)
                  aug_archive1_file, aug_archive2_file = aug_archive1_file if shuffle_file1 is None else shuffle_file1, aug_archive2_file if shuffle_file2 is None else shuffle_file2

              '''Avoid training 2 same augmentation data'''
              aug_tensor1, aug_tensor2 = 0, 0
              while(torch.sum(torch.sub(aug_tensor1, aug_tensor2))==0):
                  aug_arr1, aug_arr2 = vamb.vambtools.read_npz(aug_archive1_file[0]), vamb.vambtools.read_npz(aug_archive2_file[0])
                  '''Mutate rpkm and tnf array in-place instead of making a copy.'''
                  aug_arr1 = vamb.vambtools.numpy_inplace_maskarray(aug_arr1, mask)
                  aug_arr2 = vamb.vambtools.numpy_inplace_maskarray(aug_arr2, mask)
                  '''Zscore for augmentation data (same as the depth and tnf)'''
                  vamb.vambtools.zscore(aug_arr1, axis=0, inplace=True)
                  vamb.vambtools.zscore(aug_arr2, axis=0, inplace=True)
                  aug_tensor1, aug_tensor2 = torch.from_numpy(aug_arr1), torch.from_numpy(aug_arr2)
                  # print('augtensor', torch.sum(aug_tensor1 ** 2), _torch.sum(aug_tensor2 ** 2), aug_archive1_file, aug_archive2_file, _np.sum(aug_arr1 ** 2), _np.sum(aug_arr2 ** 2))
                  # if aug_tensor1 == aug_tensor2, reloop
                  shuffle_file1, shuffle_file2 = aug_file_shuffle(augmentation_count_number, augmentationpath)
                  aug_archive1_file, aug_archive2_file = aug_archive1_file if shuffle_file1 is None else shuffle_file1, aug_archive2_file if shuffle_file2 is None else shuffle_file2
                
              if epoch_i in batchsteps_set:
                  data_loader = _DataLoader(
                      dataset=TensorDataset(depthstensor, tnftensor, aug_tensor1, aug_tensor2),
                      batch_size=data_loader.batch_size * 2,
                      shuffle=True,
                      drop_last=False, #or True?
                      num_workers=data_loader.num_workers,
                      pin_memory=data_loader.pin_memory,
                  )
              else:
                  data_loader = _DataLoader(dataset=TensorDataset(depthstensor, tnftensor, aug_tensor1, aug_tensor2),
                      batch_size=data_loader.batch_size if epoch_i == 0 else data_loader.batch_size,
                      shuffle=True, drop_last=False, num_workers=data_loader.num_workers, pin_memory=data_loader.pin_memory)
              self.trainepoch(epoch, data_loader, logfile, hparams, optimizer_E, optimizer_D, optimizer_D_y, optimizer_D_z, Tensor, T, modelfile, adversarial_loss, awl, optimizer_awl)
        
        #Non contrastive learning
        else:
            data_loader = _DataLoader(dataset=dataloader.dataset,
                                    batch_size=dataloader.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
            for epoch in range(nepochs):
                if epoch in batchsteps_set:
                    data_loader = _DataLoader(dataset=data_loader.dataset,
                                        batch_size=data_loader.batch_size * 2,
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=data_loader.num_workers,
                                        pin_memory=data_loader.pin_memory)
                self.trainepoch(epoch, data_loader, logfile, _Namespace(), optimizer_E, optimizer_D, optimizer_D_y, optimizer_D_z, Tensor, T, modelfile, adversarial_loss)
            

    ########### function that retrieves the clusters from Y latents
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
                        0:5
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
        out_1 = _F.normalize(out_1, dim=1)
        out_2 = _F.normalize(out_2, dim=1)

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
        row_sub = torch.Tensor(neg.shape).fill_(_math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()
        #print('out',out,cov,sim,neg,row_sub,pos)

        return loss