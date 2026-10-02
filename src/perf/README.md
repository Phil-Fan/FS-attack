# README

## Installation

Targets Python 3.11+. [`uv`](https://docs.astral.sh/uv/) is recommended.

```bash
uv sync                 # or: python -m venv .venv && pip install -r requirements.txt
```

The optional time-series backbones used by `probe/` (MiniRocket / InceptionTime)
are heavy and not installed by default:

```bash
pip install torch tsai "sktime>=0.9.0"
```

## Quick start

```bash
export MODEL_PATH=/path/to/models     # directory containing the checkpoints
export MODEL_NAME=Qwen3-8B            # checkpoint folder name
export RUN_OUTPUT_DIR=$PWD/output/quickstart

./serve.sh                            # or ./serve-sglang.sh
# See ./serve.sh --help for GPU_MEMORY, MAX_SEQS, MAX_MODEL_LEN, TENSOR_PARALLEL_SIZE, ...

python attack_main.py --sweep <sweep_id>
./post.sh "$RUN_OUTPUT_DIR"
./summary.sh
```

## Configuration

Config is layered YAML under `config/`: `base.yaml` (defaults), `servers.yaml`
(endpoints), `datasets.yaml` (catalogue), `sweeps.yaml` (experiment matrices).
Any string value supports `${VAR}` / `${VAR:-default}`, which keeps machine-local
paths out of the repo (e.g. `prober.model_dir: "${PROBER_MODEL_DIR:-probe/data/model}"`).

**User types** are mixed via `users.ratios`:

| Type | Description |
| --- | --- |
| `normal` | Benign traffic (takes the remaining ratio) |
| `baseline` | Fixed-schedule attack baselines (loop / engorgio) |
| `fs_attack` | Adaptive Fill-and-Squeeze attack users |
| `prober` | Probe users that estimate KV-cache pressure |
| `warmup` | Warmup users run before measurement |

## Output and visualization

```bash
python visualize/cli.py all --root output/benchmark   # output-, SLO- and system-side figures
```

Each run writes a self-contained directory:

```
output/<timestamp>__<name>/
├── config.yaml          # full configuration snapshot
├── results.csv          # per-request metrics
├── itl.csv              # inter-token latency traces
├── logs/                # experiment + server logs
└── figures/             # generated figures (after post-processing)
```

Other entry points (`probe/cli.py`, `visualize/cli.py`, `attack_main.py`) accept
`--help`.
