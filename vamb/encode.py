from typing import Optional, IO, Union
import vamb.vambtools as _vambtools
from torch.utils.data.dataset import TensorDataset as _TensorDataset
from torch.utils.data import DataLoader as _DataLoader
from torch.nn.functional import softmax as _softmax
from torch.optim import Adam as _Adam
from torch import Tensor
from torch import nn as _nn
from math import log as _log

__doc__ = """Encode a depths matrix and a tnf matrix to latent representation.

Creates a variational autoencoder in PyTorch and tries to represent the depths
and tnf in the latent space under gaussian noise.

Usage:
>>> vae = VAE(nsamples=6)
>>> dataloader, mask = make_dataloader(depths, tnf, lengths)
>>> vae.trainmodel(dataloader)
>>> latent = vae.encode(dataloader) # Encode to latent representation
>>> latent.shape
(183882, 32)
"""

__cmd_doc__ = """Encode depths and TNF using a VAE to latent representation"""

import numpy as _np
import torch as _torch

_torch.manual_seed(0)


def make_dataloader(
    rpkm: _np.ndarray,
    tnf: _np.ndarray,
    lengths: _np.ndarray,
    batchsize: int = 256,
    destroy: bool = False,
    cuda: bool = False,
) -> tuple[_DataLoader[tuple[Tensor, Tensor, Tensor]], _np.ndarray]:
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x N_TNF)
        lengths: matrix of lengths of each contig
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """

    if not isinstance(rpkm, _np.ndarray) or not isinstance(tnf, _np.ndarray):
        raise ValueError("TNF and RPKM must be Numpy arrays")

    if batchsize < 1:
        raise ValueError(f"Batch size must be minimum 1, not {batchsize}")

    if len(rpkm) != len(tnf) or len(tnf) != len(lengths):
        raise ValueError("Lengths of RPKM, TNF and lengths arrays must be the same")

    if not (rpkm.dtype == tnf.dtype == _np.float32):
        raise ValueError("TNF and RPKM must be Numpy arrays of dtype float32")

    ### Copy arrays and mask them ###
    # Copy if not destroy - this way we can have all following operations in-place
    # for simplicity
    if not destroy:
        rpkm = rpkm.copy()
        tnf = tnf.copy()

    # Normalize samples to have same depth
    sample_depths_sum = rpkm.sum(axis=0)
    if _np.any(sample_depths_sum == 0):
        raise ValueError("One or more samples have zero depth in all sequences, so cannot be depth normalized")
    rpkm *= 1_000_000 / sample_depths_sum

    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    mask = tnf.sum(axis=1) != 0
    depthssum = None
    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        mask &= depthssum != 0
        depthssum = depthssum[mask]
        assert isinstance(depthssum, _np.ndarray)

    if mask.sum() < batchsize:
        raise ValueError(
            "Fewer sequences left after filtering than the batch size. " +
            "This probably means you try to run on a too small dataset (below ~10k sequences), " + 
            "or that nearly all sequences were filtered away. Check the log file, " + 
            "and verify BAM file content is sensible."
            )

    _vambtools.numpy_inplace_maskarray(rpkm, mask)
    _vambtools.numpy_inplace_maskarray(tnf, mask)

    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1:
        assert depthssum is not None  # we set it so just above
        rpkm /= depthssum.reshape((-1, 1))
    else:
        _vambtools.zscore(rpkm, axis=0, inplace=True)

    # Normalize TNF
    _vambtools.zscore(tnf, axis=0, inplace=True)

    # Create weights
    lengths = (lengths[mask]).astype(_np.float32)
    weights = _np.log(lengths).astype(_np.float32) - 5.0
    weights[weights < 2.0] = 2.0
    weights *= len(weights) / weights.sum()
    weights.shape = (len(weights), 1)

    ### Create final tensors and dataloader ###
    depthstensor = _torch.from_numpy(rpkm)  # this is a no-copy operation
    tnftensor = _torch.from_numpy(tnf)
    weightstensor = _torch.from_numpy(weights)
    n_workers = 4 if cuda else 1
    dataset = _TensorDataset(depthstensor, tnftensor, weightstensor)
    dataloader = _DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=True,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
    )

    return dataloader, mask


class VAE(_nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nhiddens: list of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]

    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None

    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.

    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        ntf: int, #103
        nsamples: int,
        k: int = 4,
        nhiddens: Optional[list[int]] = None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        cuda: bool = False,
        c: bool = False,
    ):
        if nlatent < 1:
            raise ValueError(f"Minimum 1 latent neuron, not {nlatent}")

        if nsamples < 1:
            raise ValueError(f"nsamples must be > 0, not {nsamples}")

        # If only 1 sample, we weigh alpha and nhiddens differently
        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if any(i < 1 for i in nhiddens):
            raise ValueError(f"Minimum 1 neuron per layer, not {min(nhiddens)}")

        if beta <= 0:
            raise ValueError(f"beta must be > 0, not {beta}")

        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be 0 < alpha < 1, not {alpha}")

        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be 0 <= dropout < 1, not {dropout}")

        super(VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = ntnf
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout
        self.contrast = c

        # Initialize lists for holding hidden layers
        self.encoderlayers = _nn.ModuleList()
        self.encodernorms = _nn.ModuleList()
        self.decoderlayers = _nn.ModuleList()
        self.decodernorms = _nn.ModuleList()

        # Add all other hidden layers
        for nin, nout in zip(
            [self.nsamples + self.ntnf] + self.nhiddens, self.nhiddens
        ):
            self.encoderlayers.append(_nn.Linear(nin, nout))
            self.encodernorms.append(_nn.BatchNorm1d(nout))

        # Latent layers
        self.mu = _nn.Linear(self.nhiddens[-1], self.nlatent)
        self.logsigma = _nn.Linear(self.nhiddens[-1], self.nlatent)

        # Add first decoding layer
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(_nn.Linear(nin, nout))
            self.decodernorms.append(_nn.BatchNorm1d(nout))

        # Reconstruction (output) layer
        self.outputlayer = _nn.Linear(self.nhiddens[0], self.nsamples + self.ntnf)

        # Activation functions
        self.relu = _nn.LeakyReLU()
        self.softplus = _nn.Softplus()
        self.dropoutlayer = _nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()

    def _encode(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        tensors = list()

        # Hidden layers
        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        # Latent layers
        mu = self.mu(tensor)

        # Note: This softplus constrains logsigma to positive. As reconstruction loss pushes
        # logsigma as low as possible, and KLD pushes it towards 0, the optimizer will
        # always push this to 0, meaning that the logsigma layer will be pushed towards
        # negative infinity. This creates a nasty numerical instability in VAMB. Luckily,
        # the gradient also disappears as it decreases towards negative infinity, avoiding
        # NaN poisoning in most cases. We tried to remove the softplus layer, but this
        # necessitates a new round of hyperparameter optimization, and there is no way in
        # hell I am going to do that at the moment of writing.
        # Also remove needless factor 2 in definition of latent in reparameterize function.
        logsigma = self.softplus(self.logsigma(tensor))

        return mu, logsigma

    # sample with gaussian noise
    def reparameterize(self, mu: Tensor, logsigma: Tensor) -> Tensor:
        epsilon = _torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        # See comment above regarding softplus
        latent = mu + epsilon * _torch.exp(logsigma / 2)

        return latent

    def _decode(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        depths_out = reconstruction.narrow(1, 0, self.nsamples)
        tnf_out = reconstruction.narrow(1, self.nsamples, self.ntnf)

        # If multiple samples, apply softmax
        if self.nsamples > 1:
            depths_out = _softmax(depths_out, dim=1)

        return depths_out, tnf_out

    def forward(
        self, depths: Tensor, tnf: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tensor = _torch.cat((depths, tnf), 1)
        mu, logsigma = self._encode(tensor)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self._decode(latent)

        return depths_out, tnf_out, mu, logsigma

    def calc_loss(
        self,
        depths_in: Tensor,
        depths_out: Tensor,
        tnf_in: Tensor,
        tnf_out: Tensor,
        mu: Tensor,
        logsigma: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # If multiple samples, use cross entropy, else use SSE for abundance
        if self.nsamples > 1:
            # Add 1e-9 to depths_out to avoid numerical instability.
            ce = -((depths_out + 1e-9).log() * depths_in).sum(dim=1)
            ce_weight = (1 - self.alpha) / _log(self.nsamples)
        else:
            ce = (depths_out - depths_in).pow(2).sum(dim=1)
            ce_weight = 1 - self.alpha

        sse = (tnf_out - tnf_in).pow(2).sum(dim=1)
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1)
        sse_weight = self.alpha / self.ntnf
        kld_weight = 1 / (self.nlatent * self.beta)
        reconstruction_loss = ce * ce_weight + sse * sse_weight
        kld_loss = kld * kld_weight
        loss = (reconstruction_loss + kld_loss) * weights

        return loss.mean(), ce.mean(), sse.mean(), kld.mean()

    def trainepoch(
        self,
        data_loader: _DataLoader,
        epoch: int,
        optimizer,
        batchsteps: list[int],
        logfile,
        awl = None,
    ) -> _DataLoader[tuple[Tensor, Tensor, Tensor]]:
        self.train()
        #VAMB
        if hparams == argparse.Namespace():
            epoch_loss = 0.0
            epoch_kldloss = 0.0
            epoch_sseloss = 0.0
            epoch_celoss = 0.0
     
            for depths_in, tnf_in, weights in data_loader:
                depths_in.requires_grad = True
                tnf_in.requires_grad = True
    
                if self.usecuda:
                    depths_in = depths_in.cuda()
                    tnf_in = tnf_in.cuda()
                    weights = weights.cuda()
    
                optimizer.zero_grad()
    
                depths_out, tnf_out, mu, logsigma = self(depths_in, tnf_in)
    
                loss, ce, sse, kld = self.calc_loss(
                    depths_in, depths_out, tnf_in, tnf_out, mu, logsigma, weights
                )
    
                loss.backward()
                optimizer.step()
    
                epoch_loss += loss.data.item()
                epoch_kldloss += kld.data.item()
                epoch_sseloss += sse.data.item()
                epoch_celoss += ce.data.item()
    
            if logfile is not None:
                print(
                    "\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}".format(
                        epoch + 1,
                        epoch_loss / len(data_loader),
                        epoch_celoss / len(data_loader),
                        epoch_sseloss / len(data_loader),
                        epoch_kldloss / len(data_loader),
                        data_loader.batch_size,
                    ),
                    file=logfile,
                )
    
                logfile.flush()
        #CLMB
        else:
            epoch_loss = 0
            epoch_kldloss = 0
            epoch_cesseloss = 0
            epoch_clloss = 0
            # grad_block.clear()
            for depths, tnf_in, tnf_aug1, tnf_aug2 in data_loader:
                # print(_torch.sum(tnf_in1),tnf_in1.shape, file=logfile)
                # depths_in1, tnf_in1, depths_in2, tnf_in2 = depths_in1[0], tnf_in1[0], depths_in2[0], tnf_in2[0]
                depths.requires_grad = True
                tnf_in.requires_grad = True
                tnf_aug1.requires_grad = True
                tnf_aug2.requires_grad = True

                if self.usecuda:
                    depths = depths.cuda()
                    tnf_in = tnf_in.cuda()
                    tnf_aug1 = tnf_aug1.cuda()
                    tnf_aug2 = tnf_aug2.cuda()

                optimizer.zero_grad()

                depths_out, tnf_out, mu, logsigma = self(depths, tnf_in)
                depths_out1, tnf_out_aug1, mu1, logsigma1 = self(depths, tnf_aug1)
                depths_out2, tnf_out_aug2, mu2, logsigma2 = self(depths, tnf_aug2)

                #loss3 = self.nt_xent_loss(_torch.cat((depths_out1, tnf_out1), 1), _torch.cat((depths_out2, tnf_out2), 1), temperature=hparams.temperature)
                loss_contrast1 = self.nt_xent_loss(tnf_out_aug1, tnf_out_aug2, temperature=hparams.temperature)
                loss_contrast2 = self.nt_xent_loss(tnf_out_aug2, tnf_out, temperature=hparams.temperature)
                loss_contrast3 = self.nt_xent_loss(tnf_out, tnf_out_aug1, temperature=hparams.temperature)
                # _torch.concat((depths_out, tnf_out), 1)
                # _torch.concat((depths_out1, tnf_out_aug1), 1)
                # _torch.concat((depths_out2, tnf_out_aug2), 1)
                loss1, ce1, sse1, kld1 = self.calc_loss(depths, depths_out, tnf_in, tnf_out, mu, logsigma)
                # loss2, ce2, sse2, kld2 = self.calc_loss(depths, depths_out, tnf_in, tnf_out, mu, logsigma)
                # loss3, ce3, sse3, kld3 = self.calc_loss(depths, depths_out, tnf_in, tnf_out, mu, logsigma)

                # NOTE: Add weight to avoid gradient disappearance
                # loss = awl(800*loss_contrast1, 800*loss_contrast2, 800*loss_contrast3) + 10000*loss1 + 2000*loss2 + 2000*loss3
                loss = awl(hparams.sigma*loss_contrast1, hparams.sigma*loss_contrast2, hparams.sigma*loss_contrast3) + 10000*loss1
                # loss = awl(awl_c(800*loss_contrast1, 800*loss_contrast2, 800*loss_contrast3), 10000*loss1, 2000*loss2, 2000*loss3)
                loss.backward()

                optimizer.step()
                print('loss', loss-10000*loss1,loss1,loss_contrast1,loss_contrast2,loss_contrast3,file=logfile)

                epoch_loss += loss.data.item()
                epoch_kldloss += (kld1).data.item()
                epoch_cesseloss += (ce1).data.item()
                epoch_clloss += (sse1).data.item()

            #Gradient monitor using hook (require extra memory and time cost)
            #for i in range(len(grad_block)):
            #     print('grad', grad_block[i], file=logfile, end='\t\t')

            if logfile is not None:
                print('\tEpoch: {}\tLoss: {:.6f}\tCL: {:.7f}\tCE SSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
                    epoch + 1,
                    epoch_loss / len(data_loader),
                    epoch_clloss / len(data_loader),
                    epoch_cesseloss / len(data_loader),
                    epoch_kldloss / len(data_loader),
                    data_loader.batch_size,
                    ), file=logfile)

                logfile.flush()
                
        return None

    def encode(self, data_loader) -> _np.ndarray:
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

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

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent = _np.empty((length, self.nlatent), dtype=_np.float32)

        row = 0
        with _torch.no_grad():
            for depths, tnf, weights in new_data_loader:
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()

                # Evaluate
                _, _, mu, _ = self(depths, tnf)

                if self.usecuda:
                    mu = mu.cpu()

                latent[row : row + len(mu)] = mu
                row += len(mu)

        assert row == length
        return latent

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {
            "nsamples": self.nsamples,
            "alpha": self.alpha,
            "beta": self.beta,
            "dropout": self.dropout,
            "nhiddens": self.nhiddens,
            "nlatent": self.nlatent,
            "state": self.state_dict(),
        }

        _torch.save(state, filehandle)

    @classmethod
    def load(
        cls, path: Union[IO[bytes], str], cuda: bool = False, evaluate: bool = True, c: bool = False
    ):
        """Instantiates a VAE from a model file.

        Inputs:
            path: Path to model file as created by functions VAE.save or
                  VAE.trainmodel.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]

        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        dictionary = _torch.load(path, map_location=lambda storage, loc: storage)

        nsamples = dictionary["nsamples"]
        alpha = dictionary["alpha"]
        beta = dictionary["beta"]
        dropout = dictionary["dropout"]
        nhiddens = dictionary["nhiddens"]
        nlatent = dictionary["nlatent"]
        state = dictionary["state"]

        vae = cls(nsamples, nhiddens, nlatent, alpha, beta, dropout, cuda, c=c)
        vae.load_state_dict(state)

        if cuda:
            vae.cuda()

        if evaluate:
            vae.eval()

        return vae

    def trainmodel(
        self,
        dataloader: _DataLoader[tuple[Tensor, Tensor, Tensor]],
        nepochs: int = 500,
        lrate: float = 1e-3,
        batchsteps: Optional[list[int]] = [25, 75, 150, 300],
        logfile: Optional[IO[str]] = None,
        modelfile: Union[None, str, IO[bytes]] = None,
        hparams = None,
        augmentationpath = None,
        mask = None
    ):
        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]
            hparams: CLMB only. Set the batchsize, augmode, temperature for contrastive learning. See the function (trainvae) in (__main.py__) for value setting. [None]
            augmentationpath: CLMB only. Path to find the augmented data [None]
            mask: CLMB only. Mask the augmented data to keep nonzero tnfs and rpkm [None]

        Output: None
        """

        if lrate < 0:
            raise ValueError(f"Learning rate must be positive, not {lrate}")

        if nepochs < 1:
            raise ValueError("Minimum 1 epoch, not {nepochs}")

        if batchsteps is None:
            batchsteps_set: set[int] = set()
        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError("All elements of batchsteps must be integers")
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError("Max batchsteps must not equal or exceed nepochs")
            last_batchsize = dataloader.batch_size * 2 ** len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:  # type: ignore
                raise ValueError(
                    f"Last batch size of {last_batchsize} exceeds dataset length "
                    f"of {len(dataloader.dataset)}. "  # type: ignore
                    "This means you have too few contigs left after filtering to train. "
                    "It is not adviced to run Vamb with fewer than 10,000 sequences "
                    "after filtering. "
                    "Please check the Vamb log file to see where the sequences were "
                    "filtered away, and verify BAM files has sensible content."
                )
            batchsteps_set = set(batchsteps)

        # Get number of features
        # Following line is un-inferrable due to typing problems with DataLoader
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape  # type: ignore
        depthstensor, tnftensor = dataloader.dataset.tensors

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tAlpha:", self.alpha, file=logfile)
            print("\tBeta:", self.beta, file=logfile)
            print("\tDropout:", self.dropout, file=logfile)
            print("\tN hidden:", ", ".join(map(str, self.nhiddens)), file=logfile)
            print("\tN latent:", self.nlatent, file=logfile)
            print("\n\tTraining properties:", file=logfile)
            print("\tN epochs:", nepochs, file=logfile)
            print("\tStarting batch size:", dataloader.batch_size, file=logfile)
            batchsteps_string = (
                ", ".join(map(str, sorted(batchsteps_set)))
                if batchsteps_set
                else "None"
            )
            print("\tBatchsteps:", batchsteps_string, file=logfile)
            print("\tLearning rate:", lrate, file=logfile)
            print("\tN sequences:", ncontigs, file=logfile)
            print("\tN samples:", nsamples, file=logfile, end="\n\n")

        # Train
        # CLMB
        if self.contrast:
            '''Optimizer setting'''
            awl = AutomaticWeightedLoss(3)
            optimizer = _torch.optim.Adam([{'params':self.parameters(), 'lr':lrate}, {'params': awl.parameters(), 'lr':0.111, 'weight_decay': 0, 'eps': 1e-7}])
            # for param in awl.parameters():
            #     print('awl',type(param), param.size())
            #Other optimizer options (not complemented)
            # optimizer.add_param_group({'params': awl.parameters(),'lr':0.1, 'weight_decay': 0})
            # print('optimizer',optimizer.param_groups)

            '''Read augmentation data from indexed files. Note that, CLMB can't guarantee an order training with augmented data if the outdir exists.'''
            aug_all_method = ['GaussianNoise','Transition','Transversion','Mutation','AllAugmentation']
            augmentation_count_number = [0, 0]
            augmentation_count_number[0] = len(glob(rf'{augmentationpath+_os.sep}pool0*k{self.k}*')) if hparams.augmode[0] == -1 else len(_glob(rf'{augmentationpath+_os.sep}pool0*k{self.k}*_{aug_all_method[hparams.augmode[0]]}_*'))
            augmentation_count_number[1] = len(glob(rf'{augmentationpath+_os.sep}pool1*k{self.k}*')) if hparams.augmode[0] == -1 else len(_glob(rf'{augmentationpath+_os.sep}pool1*k{self.k}*_{aug_all_method[hparams.augmode[1]]}_*'))

            if augmentation_count_number[0] > math.ceil(math.sqrt(nepochs)) or augmentation_count_number[1] > math.ceil(math.sqrt(nepochs)):
                warnings.warn('Too many augmented data, augmented data might not be trained enough. CLMB do not know how this influence the performance', FutureWarning)
            elif augmentation_count_number[0] < math.ceil(math.sqrt(nepochs)) or augmentation_count_number[1] > math.ceil(math.sqrt(nepochs)):
                raise RuntimeError('Shortage of augmented data. Please regenerate enough augmented data using fasta files, or do not specify the --contrastive option to run VAMB')

            '''Function for shuffling the augmented data (if needed)'''
            def aug_file_shuffle(_count, _augmentationpath, _augdatashuffle=False):
                _shuffle_file1 = random.randrange(0, sum(_count) - 1)
                if _augdatashuffle:
                    _aug_archive1_file = None if _shuffle_file1 < _count[0] else (glob(rf'{_augmentationpath+_os.sep}pool0*k{self.k}_*index{_shuffle_file1 % _count[0]}_*') if hparams.augmode[0] == -1 \
                            else glob(rf'{augmentationpath+_os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_{aug_all_method[hparams.augmode[0]]}_*'))
                else:
                    _aug_archive1_file = glob(rf'{_augmentationpath+_os.sep}pool0*k{self.k}_*index{_shuffle_file1 % _count[0]}_*') if hparams.augmode[0] == -1 \
                            else glob(rf'{augmentationpath+_os.sep}pool0*k{self.k}_*index{_shuffle_file1 % _count[0]}_{aug_all_method[hparams.augmode[0]]}_*')
                _shuffle_file2 = random.randrange(0, sum(_count) - 1)
                if _augdatashuffle:
                    _aug_archive2_file = None if _shuffle_file2 < _count[1] else (glob(rf'{_augmentationpath+_os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_*') if hparams.augmode[1] == -1 \
                            else glob(rf'{augmentationpath+_os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_{aug_all_method[hparams.augmode[1]]}_*'))
                else:
                    _aug_archive2_file = glob(rf'{_augmentationpath+_os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_*') if hparams.augmode[1] == -1 \
                            else glob(rf'{augmentationpath+_os.sep}pool1*k{self.k}_*index{_shuffle_file2 % _count[1]}_{aug_all_method[hparams.augmode[1]]}_*')
                return _aug_archive1_file, _aug_archive2_file


            for epoch in range(nepochs):
                aug_archive1_file = glob(rf'{augmentationpath+_os.sep}pool0*k{self.k}_*index{epoch // augmentation_count_number[0]}_*') if hparams.augmode[0] == -1 \
                        else glob(rf'{augmentationpath+_os.sep}pool0*k{self.k}_*index{epoch // augmentation_count_number[0]}_{aug_all_method[hparams.augmode[0]]}_*')
                aug_archive2_file = glob(rf'{augmentationpath+_os.sep}pool1*k{self.k}_*index{epoch % augmentation_count_number[1]}_*') if hparams.augmode[1] == -1 \
                        else glob(rf'{augmentationpath+_os.sep}pool1*k{self.k}_*index{epoch % augmentation_count_number[1]}_{aug_all_method[hparams.augmode[1]]}_*')

                '''If augdatashuffle in on, read augmentation data from shuffled-indexed files'''
                if hparams.augdatashuffle:
                    shuffle_file1, shuffle_file2 = aug_file_shuffle(augmentation_count_number, augmentationpath, hparams.augdatashuffle)
                    aug_archive1_file, aug_archive2_file = aug_archive1_file if shuffle_file1 is None else shuffle_file1, aug_archive2_file if shuffle_file2 is None else shuffle_file2

                '''Avoid training 2 same augmentation data'''
                aug_tensor1, aug_tensor2 = 0, 0
                while(_torch.sum(_torch.sub(aug_tensor1, aug_tensor2))==0):
                    aug_arr1, aug_arr2 = _vambtools.read_npz(aug_archive1_file[0]), _vambtools.read_npz(aug_archive2_file[0])
                    '''Mutate rpkm and tnf array in-place instead of making a copy.'''
                    aug_arr1 = _vambtools.numpy_inplace_maskarray(aug_arr1, mask)
                    aug_arr2 = _vambtools.numpy_inplace_maskarray(aug_arr2, mask)
                    '''Zscore for augmentation data (same as the depth and tnf)'''
                    _vambtools.zscore(aug_arr1, axis=0, inplace=True)
                    _vambtools.zscore(aug_arr2, axis=0, inplace=True)
                    aug_tensor1, aug_tensor2 = _torch.from_numpy(aug_arr1), _torch.from_numpy(aug_arr2)
                    # print('augtensor', _torch.sum(aug_tensor1 ** 2), _torch.sum(aug_tensor2 ** 2), aug_archive1_file, aug_archive2_file, _np.sum(aug_arr1 ** 2), _np.sum(aug_arr2 ** 2))
                    # if aug_tensor1 == aug_tensor2, reloop
                    shuffle_file1, shuffle_file2 = _aug_file_shuffle(augmentation_count_number, augmentationpath)
                    aug_archive1_file, aug_archive2_file = aug_archive1_file if shuffle_file1 is None else shuffle_file1, aug_archive2_file if shuffle_file2 is None else shuffle_file2
                # print('difference',_torch.sum(_torch.sub(aug_tensor1, aug_tensor2), axis=1), _torch.sum(_torch.sub(aug_tensor1, aug_tensor2)))

                '''Double the batchsize and decrease the learning rate by 0.8 for each batchstep'''
                if epoch in batchsteps:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= hparams.lrate_decent
                        #if param_group['eps']==1e-7:
                        #    param_group['lr'] *=1
                        #else:
                        #    param_group['lr'] *=1
                    data_loader = _DataLoader(dataset=_TensorDataset(depthstensor, tnftensor, aug_tensor1, aug_tensor2),
                                        batch_size=dataloader.batch_size if epoch == 0 else data_loader.batch_size * 2,
                                        shuffle=True, drop_last=False, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory)
                else:
                    data_loader = _DataLoader(dataset=_TensorDataset(depthstensor, tnftensor, aug_tensor1, aug_tensor2),
                                        batch_size=dataloader.batch_size if epoch == 0 else data_loader.batch_size,
                                        shuffle=True, drop_last=False, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory)
                self.trainepoch(data_loader, epoch, optimizer, batchsteps_set, logfile, hparams, awl)

        # vamb
        else:
            optimizer = _torch.optim.Adam(self.parameters(), lr=lrate)
            data_loader = _DataLoader(dataset=dataloader.dataset,
                                    batch_size=dataloader.batch_size,
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=dataloader.num_workers,
                                    pin_memory=dataloader.pin_memory)
            for epoch in range(nepochs):
                if epoch in batchsteps:
                    data_loader = _DataLoader(dataset=data_loader.dataset,
                                        batch_size=data_loader.batch_size * 2,
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=data_loader.num_workers,
                                        pin_memory=data_loader.pin_memory)
                self.trainepoch(data_loader, epoch, optimizer, batchsteps_set, logfile, argparse.Namespace())

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None
