

## Examples to learn the use of PySR and symbolic expression generation

By Alfonso Torres-Rua, 2025, using PySR code by Miles Crammer

</div>

If you find PySR useful, please cite the original paper [arXiv:2305.01582](https://arxiv.org/abs/2305.01582).


If you've finished a project with PySR, please submit a PR to showcase your work on the [PySR research showcase page](https://ai.damtp.cam.ac.uk/pysr/papers)!

**Contents**:

- [Why PySR?](#why-pysr)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [→ Documentation](https://ai.damtp.cam.ac.uk/pysr)
- [Contributors](#contributors-)




## Why PySR?

PySR is an open-source tool for *Symbolic Regression*: a machine learning
task where the goal is to find an interpretable symbolic expression that optimizes some objective.

Over a period of several years, PySR has been engineered from the ground up
to be (1) as high-performance as possible,
(2) as configurable as possible, and (3) easy to use.
PySR is developed alongside the Julia library [SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl),
which forms the powerful search engine of PySR.
The details of these algorithms are described in the [PySR paper](https://arxiv.org/abs/2305.01582).

Symbolic regression works best on low-dimensional datasets, but
one can also extend these approaches to higher-dimensional
spaces by using "*Symbolic Distillation*" of Neural Networks, as explained in
[2006.11287](https://arxiv.org/abs/2006.11287), where we apply
it to N-body problems. Here, one essentially uses
symbolic regression to convert a neural net
to an analytic equation. Thus, these tools simultaneously present
an explicit and powerful way to interpret deep neural networks.

## Installation notes by Alfonso

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
### EXTRA STEPS NOTES

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


## Quickstart

You might wish to try the interactive tutorial [here](https://colab.research.google.com/github/MilesCranmer/PySR/blob/master/examples/pysr_demo.ipynb), which uses the notebook in `examples/pysr_demo.ipynb`.

In practice, I highly recommend using IPython rather than Jupyter, as the printing is much nicer.
Below is a quick demo here which you can paste into a Python runtime.
First, let's import numpy to generate some test data:

```python
import numpy as np

X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
```

We have created a dataset with 100 datapoints, with 5 features each.
The relation we wish to model is $2.5382 \cos(x_3) + x_0^2 - 0.5$.

Now, let's create a PySR model and train it.
PySR's main interface is in the style of scikit-learn:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)
```

This will set up the model for 40 iterations of the search code, which contains hundreds of thousands of mutations and equation evaluations.

Let's train this model on our dataset:

```python
model.fit(X, y)
```

Internally, this launches a Julia process which will do a multithreaded search for equations to fit the dataset.

Equations will be printed during training, and once you are satisfied, you may
quit early by hitting 'q' and then \<enter\>.

After the model has been fit, you can run `model.predict(X)`
to see the predictions on a given dataset using the automatically-selected expression,
or, for example, `model.predict(X, 3)` to see the predictions of the 3rd equation.

You may run:

```python
print(model)
```

to print the learned equations:

```python
PySRRegressor.equations_ = [
	   pick     score                                           equation       loss  complexity
	0        0.000000                                          4.4324794  42.354317           1
	1        1.255691                                          (x0 * x0)   3.437307           3
	2        0.011629                          ((x0 * x0) + -0.28087974)   3.358285           5
	3        0.897855                              ((x0 * x0) + cos(x3))   1.368308           6
	4        0.857018                ((x0 * x0) + (cos(x3) * 2.4566472))   0.246483           8
	5  >>>>       inf  (((cos(x3) + -0.19699033) * 2.5382123) + (x0 *...   0.000000          10
]
```

This arrow in the `pick` column indicates which equation is currently selected by your
`model_selection` strategy for prediction.
(You may change `model_selection` after `.fit(X, y)` as well.)

`model.equations_` is a pandas DataFrame containing all equations, including callable format
(`lambda_format`),
SymPy format (`sympy_format` - which you can also get with `model.sympy()`), and even JAX and PyTorch format
(both of which are differentiable - which you can get with `model.jax()` and `model.pytorch()`).

Note that `PySRRegressor` stores the state of the last search, and will restart from where you left off the next time you call `.fit()`, assuming you have set `warm_start=True`.
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

