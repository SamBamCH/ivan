import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def plot_lineup_metrics(
    own_avg: np.ndarray,
    proj_norm: np.ndarray,
    stack_labels: list[str],
    final_inds: np.ndarray,
    bin_edges: np.ndarray | None = None,
    own_bin_probs: np.ndarray | None = None,
    proj_bin_probs: np.ndarray | None = None,
    stack_dist: dict[str, float] | None = None,
    figsize: tuple[int, int] = (6, 4)
) -> None:
    """
    Plot lineup metrics. If target distributions are provided (own_bin_probs,
    proj_bin_probs, stack_dist), performs diagnostics comparing target vs sampled.
    Otherwise, plots sampled histograms only.

    Parameters:
    - own_avg: per-lineup average ownerships
    - proj_norm: per-lineup normalized projection sums
    - stack_labels: stack-type label per lineup (e.g., "3-1")
    - final_inds: indices of lineups to include in the plots
    - bin_edges: bin edges for histograms/diagnostics (optional for histograms)
    - own_bin_probs: target ownership histogram probabilities
    - proj_bin_probs: target projection histogram probabilities
    - stack_dist: target stack-type probabilities
    - figsize: figure size
    """
    # Convert to numpy for indexing
    own_avg = np.asarray(own_avg)
    proj_norm = np.asarray(proj_norm)
    final_inds = np.asarray(final_inds)

    # Determine mode: diagnostic if full targets provided
    diagnostic_mode = (
        own_bin_probs is not None
        and proj_bin_probs is not None
        and stack_dist is not None
        and bin_edges is not None
    )

    # Common centers and widths if diagnostic
    if diagnostic_mode:
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        widths = bin_edges[1:] - bin_edges[:-1]
        dens_own = own_bin_probs / widths
        dens_proj = proj_bin_probs / widths

        # Ownership diagnostic
        plt.figure(figsize=figsize)
        plt.step(centers, dens_own, where='mid', label='Target Density (Own)')
        plt.hist(own_avg[final_inds], bins=bin_edges, density=True, alpha=0.5,
                 label='Sampled (Own)')
        plt.title("Ownership: Target vs Sampled")
        plt.xlabel("Avg Ownership")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Projection diagnostic
        plt.figure(figsize=figsize)
        plt.step(centers, dens_proj, where='mid', label='Target Density (Proj)')
        plt.hist(proj_norm[final_inds], bins=bin_edges, density=True, alpha=0.5,
                 label='Sampled (Proj)')
        plt.title("Projection: Target vs Sampled")
        plt.xlabel("Normalized Projection")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Stack-type diagnostic
        types = sorted(stack_dist.keys())
        y_target = np.array([stack_dist.get(t, 0) for t in types])
        counts = Counter(stack_labels[i] for i in final_inds)
        total = len(final_inds)
        y_actual = np.array([counts.get(t, 0) / total for t in types])

        x = np.arange(len(types))
        width = 0.35
        plt.figure(figsize=(len(types) * 0.4 + 2, figsize[1]))
        plt.bar(x - width/2, y_target, width=width, label='Target')
        plt.bar(x + width/2, y_actual, width=width, label='Sampled')
        plt.xticks(x, types, rotation=45, ha='right')
        plt.title("Stack-type: Target vs Sampled")
        plt.ylabel("Probability")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        # Histogram mode
        # Ownership histogram
        plt.figure(figsize=figsize)
        plt.hist(own_avg[final_inds], bins=bin_edges, density=True, alpha=0.7)
        plt.title("Ownership Distribution (Sampled)")
        plt.xlabel("Avg Ownership")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

        # Projection histogram
        plt.figure(figsize=figsize)
        plt.hist(proj_norm[final_inds], bins=bin_edges, density=True, alpha=0.7)
        plt.title("Projection Distribution (Sampled)")
        plt.xlabel("Normalized Projection")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

        # Stack-type histogram
        types = sorted(set(stack_labels))
        counts = Counter(stack_labels[i] for i in final_inds)
        total = len(final_inds)
        y_actual = np.array([counts.get(t, 0) / total for t in types])

        x = np.arange(len(types))
        plt.figure(figsize=(len(types) * 0.4 + 2, figsize[1]))
        plt.bar(x, y_actual)
        plt.xticks(x, types, rotation=45, ha='right')
        plt.title("Stack-type Distribution (Sampled)")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.show()
