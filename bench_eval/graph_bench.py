import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = "\n".join(
    [
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{libertine}",
        r"\usepackage{inconsolata}",
    ]
)

if __name__ == "__main__":
    bench_results = """BMCS 10: Mean +- std dev: 87.6 ms +- 11.6 ms
BMCS 20: Mean +- std dev: 167 ms +- 23 ms
BMCS 30: Mean +- std dev: 285 ms +- 37 ms
BMCS 40: Mean +- std dev: 400 ms +- 40 ms
BMCS 50: Mean +- std dev: 505 ms +- 46 ms
BMCS 60: Mean +- std dev: 603 ms +- 49 ms
BMCS 70: Mean +- std dev: 711 ms +- 46 ms
BMCS 80: Mean +- std dev: 797 ms +- 63 ms
BMCS 90: Mean +- std dev: 919 ms +- 51 ms
BMCS 100: Mean +- std dev: 966 ms +- 114 ms
"""
    bench_results_mcts = """MCTS 10: Mean +- std dev: 4.51 ms +- 0.32 ms
MCTS 20: Mean +- std dev: 16.6 ms +- 0.4 ms
MCTS 30: Mean +- std dev: 37.7 ms +- 3.2 ms
MCTS 40: Mean +- std dev: 63.7 ms +- 5.6 ms
MCTS 50: Mean +- std dev: 94.9 ms +- 5.1 ms
MCTS 60: Mean +- std dev: 130 ms +- 16 ms
MCTS 70: Mean +- std dev: 169 ms +- 27 ms
MCTS 80: Mean +- std dev: 216 ms +- 58 ms
MCTS 90: Mean +- std dev: 269 ms +- 63 ms
MCTS 100: Mean +- std dev: 331 ms +- 88 ms
"""

    fig, ax = plt.subplots()

    for i, br in enumerate([bench_results, bench_results_mcts]):
        res = [
            [
                int(a.strip().split(" ")[1])
                if "BMCS" in a or "MCTS" in a
                else [float(z.strip()[:-3]) for z in a.strip().split("+-")]
                for a in x.split(":")
                if "BMCS" in a or "MCTS" in a or "ms" in a
            ]
            for x in br.split("\n")
            if x
        ]

        xs = [x[0] for x in res]
        ys = [x[1][0] for x in res]  # type: ignore
        errs = [x[1][1] for x in res]  # type: ignore

        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            fmt=".",
            capsize=3,
            elinewidth=0.75,
            label="BMCS" if i == 0 else "MCTS",
            color="#D81B60" if i == 0 else "#1E88E5",
        )

    ax.set_xticks(range(10, 110, 10))

    ax.set_title(
        r"Benchmarking the \texttt{search} function", fontdict={"fontsize": 22}
    )

    ax.tick_params(labelsize=14)

    ax.set_ylabel(r"Time per call / ms", fontdict={"size": 18})
    ax.set_xlabel(r"$\rho$", fontdict={"size": 18})

    ax.legend(prop={"size": 14})

    fig.tight_layout()

    fig.savefig("bench_eval/bench.pdf")

    plt.show()
