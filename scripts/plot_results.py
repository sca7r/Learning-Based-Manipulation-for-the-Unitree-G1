"""The README plot: v3 vs Diffusion Policy success on the same grid starts, per task (bar: 95% Wilson interval).

  python scripts/plot_results.py collect <v3 grid logs> <diffusion grid logs> media/plot_data.json
  python scripts/plot_results.py plot media/plot_data.json media/
collect counts successes and trials per clip in evaluate.py's logs (one <clip>.log per clip).
"""
import glob, json, os, sys
from math import sqrt

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def collect(v3, dp, out):
    """Write {policy: {clip: [successes, trials]}} from the two folders of evaluate.py logs."""
    count = lambda f: [sum("SUCCESS" in l for l in open(f)), sum(" trial " in l for l in open(f))]
    json.dump({name: {os.path.basename(f)[:-4]: count(f) for f in sorted(glob.glob(f"{d}/*.log"))}
               for name, d in (("v3", v3), ("diffusion", dp))}, open(out, "w"), indent=1)


def wilson(k, n, z=1.96):
    """Success rate k/n with its 95% Wilson interval, which stays sensible near 0% and 100%."""
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def plot(data, outdir):
    """Draw v3_vs_diffusion.png: per task, one bar per policy with its interval and k/n written above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "text.color": INK, "axes.labelcolor": MUTED,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    grid = json.load(open(data))  # {policy: {clip: [successes, trials]}}
    tasks = [("pp_", "Pick-and-place"), ("su_", "Stand-up")]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for k, (name, label, col) in enumerate((("v3", "v3 (ACT)", BLUE), ("diffusion", "Diffusion Policy", ORANGE))):
        for t, (prefix, _) in enumerate(tasks):
            ok, n = (sum(v[j] for c, v in grid[name].items() if c.startswith(prefix)) for j in (0, 1))
            p, lo, hi = wilson(ok, n)
            x = t + (k - 0.5) * 0.36
            ax.bar(x, p, width=0.34, color=col, label=label if t == 0 else None)
            ax.plot([x, x], [lo, hi], color=INK, linewidth=1.2)
            ax.text(x, hi + 0.03, f"{100 * p:.0f}%\n{ok}/{n}", ha="center", va="bottom", fontsize=12, color=INK)
    ax.set_xticks(range(len(tasks)), [t[1] for t in tasks], fontsize=14, color=INK)
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("success rate on the grid starts")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=12)
    ax.set_title("v3 and Diffusion Policy on the same starts", loc="left", fontsize=15, fontweight="bold", pad=14)
    fig.tight_layout(); fig.savefig(f"{outdir}/v3_vs_diffusion.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    {"collect": lambda: collect(*sys.argv[2:5]), "plot": lambda: plot(*sys.argv[2:4])}[sys.argv[1]]()
