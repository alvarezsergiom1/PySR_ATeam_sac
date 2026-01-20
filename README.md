## Examples to learn the use of PySR and symbolic expression generation

By Alfonso Torres-Rua, 2025, using PySR code by Miles Crammer

This repository aims to provide be a living code where learning and improvements to perform regression and classification tasks using information mainly from Earth Science (remote sensing) and ground information.

Examples:

1. Green Biomass estimation (LAI) for vineyards using drone data.
2. UAV temperature correction using meteorological and ground measurements data

</div>

If you find PySR useful, please cite the original paper [arXiv:2305.01582](https://arxiv.org/abs/2305.01582).


## Installation notes

It is better to create a new environment e.g. in anaconda terminal:

```bash
conda create --name pysr'
```
then activate the environment

```bash
conda activate pysr'
```

### Pip

You can install PySR with pip:

```bash
pip install pysr
```

Julia dependencies will be installed at first import.

### Conda

Similarly, with conda:

```bash
conda install -c conda-forge pysr
```
### EXTRA STEPS

After conda or pip installation, in the same environment 

```bash
conda install matplotlib
pip install skillmetrics
conda install jinja2
```
Then, in the terminal:
```bash
python -c "import pysr"
```


this will complete additional libraries installation (in Julia language)




