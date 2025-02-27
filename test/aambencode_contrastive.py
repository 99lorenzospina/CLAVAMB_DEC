import sys
import os
import numpy as np
import torch
from argparse import Namespace

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
import vamb

# Test making the dataloader
tnf = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_tnf.npz'))
rpkm = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_rpkm.npz'))
lengths = np.ones(tnf.shape[0])
lengths = np.exp((lengths + 5.0).astype(np.float32))
dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, lengths, batchsize=64)

assert np.all(mask == np.array([False, False, False, False,  True,  True, False,  True, False,
        True,  True, False,  True,  True, False,  True,  True, False,
       False, False, False, False,  True,  True,  True, False, False,
       False,  True, False,  True, False, False,  True,  True,  True,
       False,  True,  True,  True, False, False,  True, False,  True,
        True, False, False,  True,  True,  True,  True,  True, False,
       False,  True, False, False, False,  True,  True,  True, False,
       False, False, False, False, False,  True, False, False,  True,
        True,  True,  True,  True,  True,  True,  True, False,  True,
       False,  True,  True,  True, False, False, False, False, False,
       False,  True,  True, False,  True,  True,  True,  True,  True,
        True,  True,  True,  True,  True,  True, False, False,  True,
        True,  True, False,  True,  True,  True, False,  True, False,
       False, False, False, False, False, False, False, False, False,
        True,  True, False,  True,  True, False, False,  True, False,
        True, False,  True, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False, False, False]))

assert len(dataloader) == sum(mask) // dataloader.batch_size

# Dataloader fails with too large batchsize
try:
    dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, lengths, batchsize=128)
except ValueError as e:
    assert 'Fewer sequences left after filtering than the batch size.' in str(e)
else:
    raise AssertionError('Should have raised ArgumentError when instantiating dataloader')

try:
    dataloader, mask = vamb.encode.make_dataloader(rpkm.flatten(), tnf, lengths, batchsize=128)
except ValueError as e:
    pass
    #assert e.args == ('Lengths of RPKM and TNF must be the same',)
else:
    raise AssertionError('Should have raised ArgumentError when instantiating dataloader')

# Normalization
assert not np.all(np.abs(np.mean(tnf, axis=0)) < 1e-4) # not normalized
assert not np.all(np.abs(np.sum(rpkm, axis=1) - 1) < 1e-5) # not normalized

assert np.all(np.abs(np.mean(dataloader.dataset.tensors[1].numpy(), axis=0)) < 1e-4) # normalized
assert np.all(np.abs(np.sum(dataloader.dataset.tensors[0].numpy(), axis=1) - 1) < 1e-5) # normalized

dataloader2, mask2 = vamb.encode.make_dataloader(rpkm, tnf, lengths, batchsize=64, destroy=True)

assert np.all(mask == mask2)
assert np.all(np.mean(tnf, axis=0) < 1e-4) # normalized
assert np.all(np.abs(np.sum(rpkm, axis=1) - 1) < 1e-5) # normalized


# Training model works in general

temp = vamb.parsecontigs.Composition.load('./data/joined_composition.npz')
tnf = temp.matrix
contignames = temp.metadata.identifiers
contiglengths = temp.metadata.lengths
print("tnf length: ", tnf.shape)
print("contignames length: ", contignames.shape)
print("contiglengths length: ", contiglengths.shape)
rpkm = np.ones((contiglengths.shape[0],3), dtype=np.float32)
lengths = np.ones(contiglengths.shape[0])
lengths = np.exp((lengths + 5.0).astype(np.float32))
dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, lengths, batchsize=2)

# Can instantiate the AAE
aae = vamb.aamb_encode.AAE(tnf.shape[1], nsamples=3, contrast=True)

hparams = Namespace(
        validation_size=4096,   # Debug only. Validation size for training.
        visualize_size=25600,   # Debug only. Visualization (pca) size for training.
        temperature=1,        # The parameter for contrastive loss
        augmode=[-1, -1],        # Augmentation method choices (in aug_all_method)
        sigma = 4000,           # Add weight on the contrastive loss to avoid gradient disappearance
        lrate_decent = 0.8,     # Decrease the learning rate by lrate_decent for each batchstep
        augdatashuffle = False     # Shuffle the augmented data for training to introduce more noise. Setting True is not recommended. [False]
    )

f = open('./data/test_log_aamb_contrastive.txt', 'w')
aae.trainmodel(dataloader, batchsteps=[5, 10], logfile= f, nepochs=15, hparams=hparams, augmentationpath="./data/", mask=mask)
aae.trainmodel(dataloader, batchsteps=None, nepochs=15, hparams=hparams, augmentationpath="./data/", mask=mask)


# Training model fails with weird batch steps
try:
    aae.trainmodel(dataloader, batchsteps=[5, 10, 15, 20], nepochs=25, hparams=hparams, augmentationpath="./data/", mask=mask)
except ValueError as e:
    assert 'Last batch size of' in str(e)
    pass

try:
    aae.trainmodel(dataloader, batchsteps=[5, 10], nepochs=10, hparams=hparams, augmentationpath="./data/", mask=mask)
except ValueError as e:
    assert e.args == ('Max batchsteps must not equal or exceed nepochs',)


aae.save('./data/saved_aae_contrastive_pc.pt')
aae = vamb.aamb_encode.AAE.load('./data/saved_aae_contrastive_pc.pt')

target_latent = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_latent.npz'))

clust, latent = aae.get_latents(contignames, dataloader)

#assert np.all(np.abs(latent - target_latent) < 1e-4)

# Encoding also withs with a minibatch of one
inputs = dataloader.dataset.tensors
inputs = inputs[0][:65], inputs[1][:65], inputs[2][:65]
ds = torch.utils.data.dataset.TensorDataset(inputs[0], inputs[1], inputs[2])
new_dataloader = torch.utils.data.DataLoader(dataset=ds, batch_size=64, shuffle=False, num_workers=1,)

clust, latent = aae.get_latents(contignames, dataloader)
#assert np.all(np.abs(latent - target_latent[:65]) < 1e-4)
