from __future__ import annotations
import numpy as np

def persistent_homology_cycles_ripser(points, maxdim: int = 1):
    """Compute persistent homology and cycles using ripser (optional)."""
    try:
        from ripser import ripser
    except Exception:
        raise ImportError("Ripser not installed. Use: pip install tda-toolkit[ripser]")
    result = ripser(points, maxdim=maxdim, do_cocycles=False)
    diagrams = result['dgms']
    cycles = result.get('cycles', None)
    return diagrams, cycles


def plot_most_persistent_cycle(points, diagrams, cycles) -> None:
    import matplotlib.pyplot as plt
    if cycles is None or len(cycles) < 2 or len(cycles[1]) == 0:
        print("No 1D cycles to plot.")
        return
    import numpy as np
    gen_idx = int(np.argmax(diagrams[1][:, 1] - diagrams[1][:, 0]))
    gen = cycles[1][gen_idx]
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.5, label='Points')
    for edge in gen:
        pt0, pt1 = points[edge[0]], points[edge[1]]
        plt.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], lw=2)
    plt.title("Most Persistent 1D Cycle")
    plt.show()
