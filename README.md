# CombFormer

CombFormer is a learning-based separation algorithm for Strengthened Comb Inequalities (SCIs) for the Capacitated Vehicle Routing Problem (CVRP).

This repository contains the code, test CVRP instances, and pretrained model used for the paper:

**A Learning-Based Separation Algorithm for the Strengthened Comb Inequalities**

## Repository Layout

```text
.
├── code/
│   ├── GATRL.py                         # CombFormer neural policy
│   ├── data_prep_rl_50_100_test.py      # graph construction and SCI search
│   ├── graph_shrinking.py               # LP support graph shrinking heuristic
│   ├── teethsep_valid.py                # deterministic teeth construction
│   ├── cvrpsep_cuts.jl                  # Julia wrapper for CVRPSEP cuts
│   ├── run_fixed_lp_sep_cutdump.py      # fixed-LP offline SCI separation experiment
│   └── run_root_iterlimit_union_cutdump.py
├── data/
│   ├── cvrp_instances/                  # test instances
│   ├── experimental_results/appendix/   # full CSV tables for appendix results
│   └── ub_results/                      # upper-bound reference results
├── models/
│   └── train_smaller_2_tan5.pt          # pretrained CombFormer model
└── results/                             # created when experiments are run
```

## Requirements

The release code uses Python, Julia, Gurobi, PyTorch, PyTorch Geometric, and CVRPSEP.jl.

The dependency versions below are based on the original author's `sci` conda environment. Packages unrelated to the released main script are omitted.

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.9.18 | Runtime environment |
| gurobipy | 11.0.1 | LP solver for root-node relaxations |
| networkx | 3.1 | Graph construction |
| numpy | 1.26.3 | Numerical computation |
| torch | 2.2.0 | CombFormer neural network inference |
| julia | 0.6.2 | PyJulia bridge to Julia/CVRPSEP |
| numba | 0.60.0 | JIT compilation for graph shrinking and teeth separation |
| scipy | 1.12.0 | Numerical routines used by the NN/data pipeline |

The model code also imports `torch-geometric` and `torch-scatter`; install versions compatible with your PyTorch and CUDA/CPU build.

You also need:

- a working Gurobi license
- Julia 1.10.5
- CVRPSEP.jl, installed from GitHub(https://github.com/chkwon/CVRPSEP.jl)
- PyCall.jl 1.96.4

## Installation

Create and activate the Python environment:

```bash
conda create -n sci python=3.9.18
conda activate sci
which python
python --version
python -m pip install --upgrade pip
```

After activation, `which python` should point inside the `sci` conda environment and `python --version` should print Python 3.9.18. If `python` is not found or pip prints "Defaulting to user installation", the conda environment is not active.

Install the core Python requirements:

```bash
python -m pip install -r requirements.txt
```

Install PyTorch Geometric dependencies matching PyTorch 2.2.0 and your CUDA or CPU build. For example, for CUDA 12.1:

```bash
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
python -m pip install torch-geometric
```

For a CPU-only PyTorch installation, use the corresponding CPU wheel index from PyTorch Geometric.

Install the Julia packages:

```bash
export PYTHON=$(which python)
julia -e 'using Pkg; Pkg.add(url="https://github.com/chkwon/CVRPSEP.jl"); Pkg.add(name="PyCall", version="1.96.4"); Pkg.build("PyCall")'
```

If PyJulia cannot find your Julia installation, initialize it once from Python:

```bash
python -c "import julia; julia.install()"
```

## Running the Main Experiment (Union)

The main release entry point is:

```bash
python code/run_root_iterlimit_union_cutdump.py --sizes 50 --count 1
```

By default, this script reads:

- instances from `data/cvrp_instances`
- the pretrained model from `models/train_smaller_2_tan5.pt`
- the Julia CVRPSEP wrapper from `code/cvrpsep_cuts.jl`

Results are written to:

```text
results/root_iterlimit_union_cutdump/
```

Useful options:

```bash
python code/run_root_iterlimit_union_cutdump.py \
  --sizes 50 100 200 \
  --base-seed 10000000 \
  --count 10 \
  --capacity 500 \
  --n-rolls 8 \
  --max-time 7200
```

## Running Fixed-LP Separation Evaluation

The fixed-LP separation script evaluates CVRPSEP and CombFormer offline on the same RCI-only root LP trajectories. SCIs are generated and dumped for separation-strength analysis, but they are not added to the LP.

```bash
python code/run_fixed_lp_sep_cutdump.py --sizes 50 --count 1 --n-rolls 1 --max-time 10
```

By default, results are written to:

```text
results/fixed_lp_sep_cutdump/
```

## Data

The provided CVRP instances follow the naming convention:

```text
data/cvrp_instances/cvrp_n{n}_s{seed}.json
```

Each JSON file contains:

- `n`: number of customers
- `seed`: generation seed
- `demands`: depot demand followed by customer demands
- `coords`: depot coordinate followed by customer coordinates

The provided test sizes are `50, 100, 200, 300, 400, 500, 750, 1000`, with 10 seeds per size.

Complete instance-level results underlying Tables 3--6 in the paper are provided in machine-readable CSV format:

```text
data/experimental_results/appendix/root_time_limited.csv
data/experimental_results/appendix/root_iteration_limited.csv
```

Both files contain one row per instance-method pair with the following columns:

- `instance`: CVRP instance identifier
- `size`: number of customers
- `method`: separator configuration
- `best_known`: best known upper bound used for gap computation
- `root_lower_bound`: final root-node lower bound
- `gap_percent`: root gap in percent
- `time_seconds`: elapsed wall-clock time
- `iterations`: number of root cutting-plane iterations
- `cuts`: number of added cuts reported for the run
- `combformer_cut_percent`: percentage of cuts contributed by CombFormer for union runs
