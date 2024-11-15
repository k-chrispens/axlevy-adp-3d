# ADP-3D: Solving Inverse Problems in Protein Space Using Diffusion-Based Priors

ADP-3D (Atomic Denoising Priors for 3D Reconstruction) is a method to leverage a pre-trained diffusion model in protein space as a prior to solve 3D reconstruction problems.

![method](images/method_white.png)

[Link to the paper](https://arxiv.org/abs/2406.04239)

[Link to the website](https://axel-levy.github.io/adp-3d/)

Here, we provide instructions to use [Chroma](https://generatebiomedicines.com/chroma) as the pretrained model.

## Installation

```
conda create --prefix adp-3d-env python=3.9
conda acticvate adp-3d-env
pip install -r requirements.txt
```

## Chroma API Key

Follow [this link](https://chroma-weights.generatebiomedicines.com/) to request your Chroma API key.

```
python register.py --key YOUR_API_KEY
```

## Structure Completion

We provide an example with [PDB:8ok3](https://www.rcsb.org/structure/8OK3).

```
python structure_completion.py -o /path/to/output/directory -c data/cifs/8ok3.cif --fix-every 4
```

## Distances to Structure

```
python distances_to_structure.py -o /path/to/output/directory -c data/cifs/8ok3.cif --n-distances 500 --lr-distance 0.4
```

## Model Refinement

We provide an example with [PDB:7PZT](https://www.rcsb.org/structure/7PZT). We generate the density map in [ChimeraX](https://www.cgl.ucsf.edu/chimerax/):

```
molmap #1 2.0 gridSpacing 0.5 edgePadding 50
```

An incomplete model was obtained with [ModelAngelo](https://github.com/3dem/model-angelo), using the default parameters.

```
python model_refinement.py -o /path/to/output/directory --mrc data/mrcs/7pzt_2.0A.mrc --ma-cif data/cifs/7pzt_MA_2.0A.cif --pdb data/pdbs/7pzt.pdb --unpad-len 85
```

[plot RMSD]

## Troubleshooting

*Failed to resolve 'chroma-weights.generatebiomedicines.com'*

You may not have access to the internet. If you are able to instantiate a Chroma model in a different configuration (e.g., on the login node of a cluster), you will see that the weights of the model (backbone model and design model) are stored in a temporary location (`/tmp/...` ending with `.pt`). To bypass the need for connecting to the internet, you can copy/paste these `.pt` files to a stable location and point to them with the following additional arguments:
```
--weights-backbone /path/to/weights/of/backbone/model --weights-design /path/to/weights/of/design/model
```
