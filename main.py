import sys
import pathlib
from typing import List, Dict
from utils.baselines_cpu import KnowledgeGraphRetrievalBaselinesCPU
from LogosKG.LogosKG import LogosKG

# Toy KG and test seeds
triplets = [
    ("A", "r1", "B"), ("A", "r1", "C"), ("B", "r1", "D"), ("C", "r1", "E"),
    ("D", "r2", "F"), ("E", "r2", "F"), ("F", "r2", "G"), ("G", "r3", "C"),
    ("H", "r1", "I"), ("I", "r2", "J"), ("K", "r3", "L"), ("L", "r3", "K"),
    ("M", "r2", "N"), ("O", "r1", "O"),
]
test_samples = [{"A"}, {"H"}, {"K"}, {"M"}, {"O"}, {"A", "H"}]
HOPS_RANGE = range(1, 6)

# Register baselines
baseline_backends = ["graphblas", "igraph", "networkx", "graphtool", "snap"]
baseline_methods = {}

for backend in baseline_backends:
    try:
        runner = KnowledgeGraphRetrievalBaselinesCPU(triplets, backend=backend)
        baseline_methods[backend] = lambda seeds, h, obj=runner, b=backend: getattr(obj, f"{b}_khop")(seeds, h)
    except Exception as e:
        print(f"[skip] {backend} unavailable: {e}")

# Register LogosKG variants
logoskg_variants = {
    "LogosKG-SciPy": LogosKG(triplets, backend="scipy"),
    "LogosKG-Numba": LogosKG(triplets, backend="numba"),
    "LogosKG-Torch": LogosKG(triplets, backend="torch", device="cpu"),
}
logoskg_methods = {
    name: lambda seeds, h, obj=kg: obj.retrieve_at_k_hop(seeds, hops=h, shortest_path=True)
    for name, kg in logoskg_variants.items()
}

# Jaccard function
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0

# LogosKG vs. baselines
def compare_logoskg(samples: List[set], hops: int) -> Dict[str, Dict[str, float]]:
    logos = list(logoskg_methods)
    baselines = list(baseline_methods)

    scores = {l: {b: [] for b in baselines} for l in logos}
    for s in samples:
        logos_out = {l: set(logoskg_methods[l](s, hops)) for l in logos}
        base_out = {b: set(baseline_methods[b](s, hops)) for b in baselines}
        for l in logos:
            for b in baselines:
                scores[l][b].append(jaccard(logos_out[l], base_out[b]))
    return {
        l: {b: sum(v) / len(v) if v else 0.0 for b, v in row.items()}
        for l, row in scores.items()
    }

# -------- Run LogosKG vs Baselines --------
print("\n=== LogosKG vs Baselines (Jaccard) ===")
for h in HOPS_RANGE:
    print(f"\n[Hops = {h}]")
    if baseline_methods:
        jac2 = compare_logoskg(test_samples, h)
        bnames = list(baseline_methods)
        print("".ljust(18) + "".join(b.rjust(14) for b in bnames))
        for l in logoskg_methods:
            print(l.ljust(18) + "".join(f"{jac2[l][b]:.4f}".rjust(14) for b in bnames))
    else:
        print("[No baselines available]")
