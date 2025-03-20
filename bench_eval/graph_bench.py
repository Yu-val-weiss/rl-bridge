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

    res = [
        [
            int(a.strip().split(" ")[1])
            if "BMCS" in a
            else [float(z.strip()[:-3]) for z in a.strip().split("+-")]
            for a in x.split(":")
            if "BMCS" in a or "ms" in a
        ]
        for x in bench_results.split("\n")
        if x
    ]

    xs = [x[0] for x in res]
    ys = [x[1][0] for x in res]  # type: ignore
    errs = [x[1][1] for x in res]  # type: ignore

    fig, ax = plt.subplots()

    ax.errorbar(xs, ys, yerr=errs, fmt=".", capsize=3, elinewidth=0.75)

    ax.set_xticks(range(10, 110, 10))

    ax.set_title(r"Benchmarking the BMCS \texttt{search} function")

    ax.set_ylabel(r"Time per call / ms")
    ax.set_xlabel(r"$\rho = R_{\max} = D_{\max}$")

    fig.savefig("bench_eval/bench.pdf")

    plt.show()
