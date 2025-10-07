# Anonymous Code for ACL Submission

This repository contains the core source code and example experiment scripts used in our ACL submission. The project implements a general-purpose framework for multi-hop retrieval over knowledge graphs and compares it with several widely used baselines.

## Overview

We propose a matrix-based retrieval framework called `LogosKG` for computing k-hop neighborhoods over knowledge graphs. The system supports multiple sparse computation backends (SciPy, Numba, Torch), and is benchmarked against existing CPU-based graph libraries.

## File Structure

```
.
├── LogosKG/
│   └── LogosKG.py              # Core LogosKG matrix-based retrieval module
│
├── utils/
│   └── baselines_cpu.py        # Baseline methods using NetworkX, igraph, graph-tool, SNAP, GraphBLAS
│
├── LogosKG_large.py            # Partitioned large-scale version with caching (optional)
├── main.py                     # Entry script for evaluating Jaccard similarity on toy graph
├── README.md                   # This file
```

## Installation

This project requires Python 3.8 or higher. Install dependencies using pip:

```bash
pip install numpy scipy numba torch networkx
```

Note: Additional libraries such as `igraph`, `graph-tool`, `snap`, and `graphblas` are used for baseline comparisons. These are optional and will be skipped if unavailable.

## Running the Toy Experiment

The toy experiment compares LogosKG against several baseline methods on a small synthetic knowledge graph. To run the experiment:

```bash
python main.py
```

The script will print Jaccard similarity scores between LogosKG variants and each baseline at different hop distances.

## Example Output

```
=== LogosKG vs Baselines (Jaccard) ===

[Hops = 1]
                 graphblas       igraph     networkx     graphtool          snap
LogosKG-SciPy       1.0000       1.0000       1.0000       1.0000       1.0000
LogosKG-Numba       1.0000       1.0000       1.0000       1.0000       1.0000
LogosKG-Torch       1.0000       1.0000       1.0000       1.0000       1.0000
```

## Notes

- Missing backends will be automatically skipped with a warning message.
- The toy knowledge graph is hardcoded in `main.py`.
- `LogosKG_large.py` is provided for completeness but is not needed to reproduce the results in `main.py`.

## License

This repository is intended solely for anonymous academic review. Please refer to the accompanying paper for more information.
