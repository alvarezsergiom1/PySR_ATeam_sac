## Examples to learn the use of PySR and symbolic expression generation

By Alfonso Torres-Rua, 2025, using PySR code by Miles Crammer

This repository aims to provide be a living code where learning and improvements to perform regression and classification tasks using information mainly from Earth Science (remote sensing) and ground information.

Examples:
Green Biomass estimation (LAI) for vineyards using drone data.
UAV temperature correction using meteorological and ground measurements data

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
```
Then, in the terminal:
```bash
python -c "import pysr"
```
this will complete additional libraries installation (in Julia language)


`warm_start=True`.
This will cause problems if significant changes are made to the search parameters (like changing the operators). You can run `model.reset()` to reset the state.

You will notice that PySR will save two files:
`hall_of_fame...csv` and `hall_of_fame...pkl`.
The csv file is a list of equations and their losses, and the pkl file is a saved state of the model.
You may load the model from the `pkl` file with:

```python
model = PySRRegressor.from_file("hall_of_fame.2022-08-10_100832.281.pkl")
```

There are several other useful features such as denoising (e.g., `denoise=True`),
feature selection (e.g., `select_k_features=3`).
For examples of these and other features, see the [examples page](https://ai.damtp.cam.ac.uk/pysr/examples).
For a detailed look at more options, see the [options page](https://ai.damtp.cam.ac.uk/pysr/options).
You can also see the full API at [this page](https://ai.damtp.cam.ac.uk/pysr/api).
There are also tips for tuning PySR on [this page](https://ai.damtp.cam.ac.uk/pysr/tuning).

### Detailed Example

The following code makes use of as many PySR features as possible.
Note that is just a demonstration of features and you should not use this example as-is.
For details on what each parameter does, check out the [API page](https://ai.damtp.cam.ac.uk/pysr/api/).

```python
model = PySRRegressor(
    populations=8,
    # ^ Assuming we have 4 cores, this means 2 populations per core, so one is always running.
    population_size=50,
    # ^ Slightly larger populations, for greater diversity.
    ncycles_per_iteration=500,
    # ^ Generations between migrations.
    niterations=10000000,  # Run forever
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 24,
    # ^ Alternatively, stop after 24 hours have passed.
    maxsize=50,
    # ^ Allow greater complexity.
    maxdepth=10,
    # ^ But, avoid deep nesting.
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["square", "cube", "exp", "cos2(x)=cos(x)^2"],
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    # ^ Limit the complexity within each argument.
    # "inv": (-1, 9) states that the numerator has no constraint,
    # but the denominator has a max complexity of 9.
    # "exp": 9 simply states that `exp` can only have
    # an expression of complexity 9 as input.
    nested_constraints={
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
    },
    # ^ Nesting constraints on operators. For example,
    # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
    complexity_of_operators={"/": 2, "exp": 3},
    # ^ Custom complexity of particular operators.
    complexity_of_constants=2,
    # ^ Punish constants more than variables
    select_k_features=4,
    # ^ Train on only the 4 most important features
    progress=True,
    # ^ Can set to false if printing to a file.
    weight_randomize=0.1,
    # ^ Randomize the tree much more frequently
    cluster_manager=None,
    # ^ Can be set to, e.g., "slurm", to run a slurm
    # cluster. Just launch one script from the head node.
    precision=64,
    # ^ Higher precision calculations.
    warm_start=True,
    # ^ Start from where left off.
    turbo=True,
    # ^ Faster evaluation (experimental)
    extra_sympy_mappings={"cos2": lambda x: sympy.cos(x)**2},
    # extra_torch_mappings={sympy.cos: torch.cos},
    # ^ Not needed as cos already defined, but this
    # is how you define custom torch operators.
    # extra_jax_mappings={sympy.cos: "jnp.cos"},
    # ^ For JAX, one passes a string.
)
```

