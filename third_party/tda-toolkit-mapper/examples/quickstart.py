import numpy as np
from tda_toolkit.persistence import compute_rips_persistence, plot_persistence_diagram

def main():
    X = np.random.rand(100, 2)
    st = compute_rips_persistence(X, max_dim=1)
    plot_persistence_diagram(st, dimension=1, title="H1")

if __name__ == "__main__":
    main()
