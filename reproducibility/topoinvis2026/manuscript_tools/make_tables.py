#!/usr/bin/env python3
"""Generate the corrected PD/MT result tables for the manuscript.

Values below are transcribed from the post-submission findings compendium
(snapshot 2 Oct 2026), which itself was produced from
  corrected_pd_mt_method_means.csv  (sha256 0b1743626dfdc1d2...)
  corrected_pd_mt_joined.csv        (sha256 f2355b6d9ca4bdf4...)

If the CSVs are available, pass --means path/to/corrected_pd_mt_method_means.csv
to overwrite the transcribed means with the authoritative file (column names
are configurable below).  The script then

  1. recomputes every derived quantity used in the paper (percent change vs
     UV and vs CNN, ranks, 2^k factorial effects, Pareto fronts) and checks
     them against the values printed in the compendium, and
  2. writes LaTeX table bodies to tables/*.tex, which template.tex \\input{}s.

Run:  python3 tools/make_tables.py [--joined corrected_pd_mt_joined.csv] [--out DIR]

With --joined, every mean, median and win count is recomputed from the frozen
per-field table and checked against the numbers printed in the paper; any
mismatch is reported and the script exits with status 1.
"""
import argparse, csv, itertools, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tables")

# ---------------------------------------------------------------------------
# 1. Method inventory  (id, display label, objective in paper notation)
# ---------------------------------------------------------------------------
METHODS = [
    # (id, display label in paper tables, objective shorthand)
    ("cnn", "Pretrained CNN", r"---"),
    ("gan", "Pretrained GAN", r"---"),
    ("uv", "Reconstruction-only control", r"$L_{uv}$"),
    ("speed_only", "+ Speed", r"$+S$"),
    ("levelset_only", "+ Level-set", r"$+L$"),
    ("speed_levelset", "+ Speed + Level-set", r"$+S+L$"),
    ("grad_only", "+ Gradient", r"$+G$"),
    ("speed_grad", "+ Speed + Gradient", r"$+S+G$"),
    ("grad_levelset", "+ Gradient + Level-set", r"$+G+L$"),
    ("candidate_b", "+ Speed + Gradient + Level-set", r"$+S+G+L$"),
    ("candidate_c", "+ Speed + Gradient + Level-set + Local-max", r"$+S+G+L+M$"),
    ("uv_crit", "+ Local-max", r"$+M$"),
    ("uv_e2", "+ Fixed-pair", r"$+FP$"),
    ("b_e2", "+ Speed + Gradient + Level-set + Fixed-pair", r"$+S+G+L+FP$"),
    ("c_e2", "+ Speed + Gradient + Level-set + Local-max + Fixed-pair", r"$+S+G+L+M+FP$"),
    ("f1_grad_e2", "\\textbf{+ Gradient + Fixed-pair (final)}", r"$+G+FP$"),
    ("f2_grad_levelset_e2", "+ Gradient + Level-set + Fixed-pair", r"$+G+L+FP$"),
    ("f3_grad_crit", "+ Gradient + Local-max", r"$+G+M$"),
]
IDS = [m[0] for m in METHODS]
LABEL = {m[0]: m[1] for m in METHODS}
OBJ = {m[0]: m[2] for m in METHODS}
LEARNED = IDS[2:]

# ---------------------------------------------------------------------------
# 2. Corrected means (compendium Sec. 4) and medians (Sec. 4.1)
#    order: dB, W2inf, W22, MT  -- all lower-is-better
# ---------------------------------------------------------------------------
METRICS = ["dB", "W2inf", "W22", "MT"]
MEAN = {
    "cnn":                 (3.124108, 19.152145, 24.552151, 5.867795),
    "gan":                 (5.058572, 14.676695, 17.458202, 8.348150),
    "uv":                  (3.287864, 21.062447, 27.508675, 6.011866),
    "speed_only":          (3.261689, 21.010241, 27.429427, 5.999595),
    "levelset_only":       (3.270083, 21.028084, 27.456104, 6.007606),
    "speed_levelset":      (3.188429, 20.818179, 27.141192, 5.944105),
    "grad_only":           (2.543778, 15.143539, 18.693469, 6.055993),
    "speed_grad":          (2.681291, 15.861962, 19.678508, 6.290475),
    "grad_levelset":       (2.631998, 15.615840, 19.324099, 6.199585),
    "candidate_b":         (2.698179, 16.008296, 19.865552, 6.161182),
    "candidate_c":         (2.494517, 15.867222, 19.706848, 6.080287),
    "uv_crit":             (2.817442, 20.167454, 26.245141, 5.689883),
    "uv_e2":               (2.236784, 15.168102, 19.099428, 5.593966),
    "b_e2":                (2.169134, 14.477289, 18.042440, 5.677425),
    "c_e2":                (2.151272, 14.702102, 18.369800, 5.662811),
    "f1_grad_e2":          (2.149744, 14.325676, 17.831897, 5.656570),
    "f2_grad_levelset_e2": (2.176493, 14.295293, 17.764074, 5.674230),
    "f3_grad_crit":        (2.376836, 15.050233, 18.559058, 5.984007),
}
MEDIAN = {
    "cnn":                 (2.912511, 18.815892, 24.056543, 5.712190),
    "gan":                 (4.196002, 13.517923, 15.834912, 7.586709),
    "uv":                  (3.188247, 21.289925, 27.841107, 5.971776),
    "speed_only":          (3.188752, 21.259265, 27.866416, 5.991089),
    "levelset_only":       (3.192363, 21.279661, 27.871766, 5.970467),
    "speed_levelset":      (3.140505, 21.100786, 27.582122, 5.913371),
    "grad_only":           (2.343015, 14.324143, 17.630126, 5.914869),
    "speed_grad":          (2.523534, 15.114283, 18.691572, 6.203293),
    "grad_levelset":       (2.473319, 14.877916, 18.330971, 6.057604),
    "candidate_b":         (2.530936, 15.594750, 19.328461, 6.040620),
    "candidate_c":         (2.341908, 15.147728, 18.754786, 5.995775),
    "uv_crit":             (2.619083, 20.355713, 26.612561, 5.660775),
    "uv_e2":               (2.049572, 15.258929, 18.727971, 5.501540),
    "b_e2":                (1.955754, 14.303971, 17.707242, 5.593558),
    "c_e2":                (1.968690, 14.608570, 18.310481, 5.543108),
    "f1_grad_e2":          (1.934064, 14.122292, 17.531404, 5.518639),
    "f2_grad_levelset_e2": (1.978684, 14.136512, 17.479146, 5.546118),
    "f3_grad_crit":        (2.231072, 14.424322, 17.854902, 5.925751),
}

# Sample-wise win counts out of 168 (compendium Sec. 6):
# dB, W2inf, W22, MT, all-3-PD, all-3-PD+MT
WINS_VS_CNN = {
    "gan": (39, 155, 158, 20, 39, 16),
    "uv": (62, 22, 17, 64, 11, 7),
    "speed_only": (65, 20, 17, 61, 11, 6),
    "levelset_only": (62, 22, 18, 64, 11, 5),
    "speed_levelset": (70, 29, 25, 66, 16, 10),
    "grad_only": (133, 161, 162, 66, 131, 61),
    "speed_grad": (122, 158, 160, 42, 119, 34),
    "grad_levelset": (126, 158, 162, 41, 123, 36),
    "candidate_b": (117, 157, 160, 40, 114, 35),
    "candidate_c": (137, 158, 160, 52, 133, 45),
    "uv_crit": (105, 57, 49, 94, 41, 34),
    "uv_e2": (148, 165, 166, 104, 148, 100),
    "b_e2": (148, 166, 166, 90, 147, 86),
    "c_e2": (154, 166, 166, 94, 153, 89),
    "f1_grad_e2": (151, 167, 166, 98, 150, 91),
    "f2_grad_levelset_e2": (148, 167, 166, 102, 147, 94),
    "f3_grad_crit": (139, 161, 161, 61, 138, 55),
}
WINS_VS_UV = {
    "cnn": (102, 146, 151, 104, 93, 67),
    "gan": (43, 167, 168, 21, 43, 16),
    "speed_only": (102, 129, 133, 95, 80, 49),
    "levelset_only": (103, 120, 127, 89, 67, 38),
    "speed_levelset": (129, 168, 168, 116, 129, 89),
    "grad_only": (162, 168, 168, 91, 162, 89),
    "speed_grad": (159, 168, 168, 52, 159, 49),
    "grad_levelset": (161, 168, 168, 63, 161, 60),
    "candidate_b": (162, 168, 168, 67, 162, 64),
    "candidate_c": (161, 168, 168, 76, 161, 73),
    "uv_crit": (151, 168, 168, 147, 151, 133),
    "uv_e2": (156, 168, 168, 136, 156, 132),
    "b_e2": (158, 168, 168, 121, 158, 120),
    "c_e2": (157, 168, 168, 122, 157, 119),
    "f1_grad_e2": (156, 168, 168, 124, 156, 119),
    "f2_grad_levelset_e2": (156, 168, 168, 122, 156, 119),
    "f3_grad_crit": (163, 168, 168, 90, 163, 88),
}

# ---------------------------------------------------------------------------
# 3. Values printed in the compendium that we re-derive as a check
# ---------------------------------------------------------------------------
EXPECT_PCT_VS_UV = {  # Sec. 5.1
    "cnn": (4.98, 9.07, 10.75, 2.40), "gan": (-53.86, 30.32, 36.54, -38.86),
    "speed_only": (0.80, 0.25, 0.29, 0.20), "levelset_only": (0.54, 0.16, 0.19, 0.07),
    "speed_levelset": (3.02, 1.16, 1.34, 1.13), "grad_only": (22.63, 28.10, 32.05, -0.73),
    "speed_grad": (18.45, 24.69, 28.46, -4.63), "grad_levelset": (19.95, 25.86, 29.75, -3.12),
    "candidate_b": (17.94, 24.00, 27.78, -2.48), "candidate_c": (24.13, 24.67, 28.36, -1.14),
    "uv_crit": (14.31, 4.25, 4.59, 5.36), "uv_e2": (31.97, 27.99, 30.57, 6.95),
    "b_e2": (34.03, 31.26, 34.41, 5.56), "c_e2": (34.57, 30.20, 33.22, 5.81),
    "f1_grad_e2": (34.62, 31.98, 35.18, 5.91), "f2_grad_levelset_e2": (33.80, 32.13, 35.42, 5.62),
    "f3_grad_crit": (27.71, 28.54, 32.53, 0.46),
}
EXPECT_PCT_VS_CNN = {  # Sec. 5.2 (subset)
    "uv": (-5.24, -9.97, -12.04, -2.46), "gan": (-61.92, 23.37, 28.89, -42.27),
    "grad_only": (18.58, 20.93, 23.86, -3.21), "uv_e2": (28.40, 20.80, 22.21, 4.67),
    "f1_grad_e2": (31.19, 25.20, 27.37, 3.60), "f2_grad_levelset_e2": (30.33, 25.36, 27.65, 3.30),
}
EXPECT_RANK = {  # Sec. 4 (ranks over all 18 methods)
    "cnn": (13, 13, 13, 7), "gan": (18, 4, 1, 18), "uv": (17, 18, 18, 12),
    "grad_only": (8, 7, 7, 13), "uv_e2": (5, 8, 8, 1), "b_e2": (3, 3, 4, 5),
    "c_e2": (2, 5, 5, 3), "f1_grad_e2": (1, 2, 3, 2), "f2_grad_levelset_e2": (4, 1, 2, 4),
    "f3_grad_crit": (6, 6, 6, 9),
}
EXPECT_FACTORIAL = {
    # Sec. 7.1  speed x gradient x level-set on the B scaffold
    "B": {
        "speed": (-0.0240, -0.2122, -0.2831, -0.0301),
        "gradient": (0.6132, 5.3223, 7.9934, -0.1860),
        "level-set": (-0.0035, -0.0981, -0.1192, 0.0114),
        "speed:gradient": (-0.0779, -0.3432, -0.4802, -0.0680),
        "speed:level-set": (0.0317, 0.1209, 0.1698, 0.0810),
        "gradient:level-set": (-0.0490, -0.2113, -0.2896, -0.0185),
        "speed:gradient:level-set": (0.0040, 0.0421, 0.0520, 0.0554),
    },
    # Sec. 7.2  critical proxy x E2 on the B scaffold
    "CE2": {
        "crit": (0.1108, -0.0419, -0.0843, 0.0478),
        "E2": (0.4361, 1.3481, 1.5801, 0.4506),
        "crit:E2": (-0.0929, -0.1829, -0.2430, -0.0331),
    },
    # Sec. 7.3  level-set x E2 on the gradient scaffold
    "LE2": {
        "level-set": (-0.0575, -0.2210, -0.2814, -0.0806),
        "E2": (0.4248, 1.0692, 1.2108, 0.4624),
        "level-set:E2": (0.0307, 0.2513, 0.3492, 0.0630),
    },
}
EXPECT_PARETO = {  # Sec. 10.3
    ("dB", "MT"): {"f1_grad_e2", "uv_e2"},
    ("W2inf", "MT"): {"f1_grad_e2", "f2_grad_levelset_e2", "uv_e2"},
    ("W22", "MT"): {"f1_grad_e2", "f2_grad_levelset_e2", "gan", "uv_e2"},
    ("dB", "W2inf", "W22", "MT"): {"f1_grad_e2", "f2_grad_levelset_e2", "gan", "uv_e2"},
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def pct(ref, x):
    """percent improvement of x over ref for a lower-is-better distance"""
    return 100.0 * (ref - x) / ref


def ranks(table, col):
    order = sorted(table, key=lambda k: table[k][col])
    return {k: i + 1 for i, k in enumerate(order)}


def factorial(cells, factors):
    """2^k effects with -1/+1 coding on y = -distance.
    cells: dict mapping tuple of 0/1 levels -> method id."""
    k = len(factors)
    effects = {}
    for r in range(1, k + 1):
        for combo in itertools.combinations(range(k), r):
            name = ":".join(factors[i] for i in combo)
            vals = []
            for m in range(4):
                plus, minus = [], []
                for lv, mid in cells.items():
                    sign = 1
                    for i in combo:
                        sign *= 1 if lv[i] else -1
                    (plus if sign > 0 else minus).append(-MEAN[mid][m])
                vals.append(sum(plus) / len(plus) - sum(minus) / len(minus))
            effects[name] = tuple(vals)
    return effects


def pareto(ids, cols):
    front = set()
    for a in ids:
        dominated = False
        for b in ids:
            if a == b:
                continue
            if all(MEAN[b][c] <= MEAN[a][c] for c in cols) and any(MEAN[b][c] < MEAN[a][c] for c in cols):
                dominated = True
                break
        if not dominated:
            front.add(a)
    return front


FAILS = []


def check(name, got, exp, tol):
    for g, e in zip(got, exp):
        if abs(g - e) > tol:
            FAILS.append(f"{name}: got {tuple(round(x, 4) for x in got)} expected {exp}")
            return


def verify():
    for k, exp in EXPECT_PCT_VS_UV.items():
        check(f"pct_vs_uv[{k}]", [pct(MEAN["uv"][m], MEAN[k][m]) for m in range(4)], exp, 0.006)
    for k, exp in EXPECT_PCT_VS_CNN.items():
        check(f"pct_vs_cnn[{k}]", [pct(MEAN["cnn"][m], MEAN[k][m]) for m in range(4)], exp, 0.006)
    R = [ranks(MEAN, m) for m in range(4)]
    for k, exp in EXPECT_RANK.items():
        check(f"rank[{k}]", [R[m][k] for m in range(4)], exp, 0)
    fB = factorial({(0, 0, 0): "uv", (1, 0, 0): "speed_only", (0, 1, 0): "grad_only", (0, 0, 1): "levelset_only",
                    (1, 1, 0): "speed_grad", (1, 0, 1): "speed_levelset", (0, 1, 1): "grad_levelset",
                    (1, 1, 1): "candidate_b"}, ["speed", "gradient", "level-set"])
    fC = factorial({(0, 0): "candidate_b", (1, 0): "candidate_c", (0, 1): "b_e2", (1, 1): "c_e2"}, ["crit", "E2"])
    fL = factorial({(0, 0): "grad_only", (1, 0): "grad_levelset", (0, 1): "f1_grad_e2", (1, 1): "f2_grad_levelset_e2"},
                   ["level-set", "E2"])
    for tag, got in (("B", fB), ("CE2", fC), ("LE2", fL)):
        for name, exp in EXPECT_FACTORIAL[tag].items():
            check(f"factorial[{tag}][{name}]", got[name], exp, 0.00051)
    names = {"dB": 0, "W2inf": 1, "W22": 2, "MT": 3}
    for cols, exp in EXPECT_PARETO.items():
        got = pareto(IDS, [names[c] for c in cols])
        if got != exp:
            FAILS.append(f"pareto{cols}: got {sorted(got)} expected {sorted(exp)}")
    # win-count internal consistency: all-3-PD <= each PD count, all+MT <= all-3 and <= MT
    for tab, nm in ((WINS_VS_CNN, "cnn"), (WINS_VS_UV, "uv")):
        for k, w in tab.items():
            if not (w[4] <= min(w[:3]) and w[5] <= min(w[4], w[3])):
                FAILS.append(f"wins_vs_{nm}[{k}] inconsistent {w}")
    return fB, fC, fL


# ---------------------------------------------------------------------------
# LaTeX writers
# ---------------------------------------------------------------------------
def f(x, d=3):
    return f"{x:.{d}f}"


def sgn(x, d=1):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{d}f}"


def cell_color(p):
    """blue for improvement vs UV, red for degradation, intensity ~ |p|."""
    if abs(p) < 0.05:
        return ""
    lvl = min(int(round(abs(p) / 40.0 * 45)) + 5, 50)
    return rf"\cellcolor{{{'improve' if p > 0 else 'degrade'}!{lvl}}}"


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("% AUTO-GENERATED by tools/make_tables.py -- do not edit by hand\n")
        fh.write(body)


def table_all_candidates():
    R = [ranks(MEAN, m) for m in range(4)]
    best = [min(MEAN, key=lambda k: MEAN[k][m]) for m in range(4)]
    second = [sorted(MEAN, key=lambda k: MEAN[k][m])[1] for m in range(4)]
    rows = []
    groups = [["cnn", "gan"], ["uv"],
              ["speed_only", "levelset_only", "speed_levelset", "grad_only", "speed_grad", "grad_levelset", "candidate_b"],
              ["candidate_c", "uv_crit", "f3_grad_crit"],
              ["uv_e2", "b_e2", "c_e2", "f1_grad_e2", "f2_grad_levelset_e2"]]
    for gi, g in enumerate(groups):
        if gi:
            rows.append(r"\midrule")
        for k in g:
            cells = [LABEL[k]]
            for m in range(4):
                v = f(MEAN[k][m], 3 if m != 1 and m != 2 else 2)
                if k == best[m]:
                    v = rf"\textbf{{{v}}}"
                elif k == second[m]:
                    v = rf"\underline{{{v}}}"
                cells.append(v)
            for m in range(4):
                if k == "uv":
                    cells.append("---")
                else:
                    p = pct(MEAN["uv"][m], MEAN[k][m])
                    cells.append(cell_color(p) + sgn(p))
            cells.append("/".join(str(R[m][k]) for m in range(4)))
            rows.append(" & ".join(cells) + r"\\")
    write("tab_all_candidates.tex", "\n".join(rows) + "\n")


def table_factorial(fB, fC, fL):
    def block(title, eff, order):
        out = [rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}}\\"]
        for nm, disp in order:
            e = eff[nm]
            vals = []
            for m in range(4):
                s = f"{e[m]:+.3f}".replace("+", "$+$").replace("-", "$-$")
                vals.append(s)
            out.append(rf"\quad {disp} & " + " & ".join(vals) + r"\\")
        return out
    rows = []
    rows += block("(a) Speed $\\times$ gradient $\\times$ level-set: all eight combinations", fB, [
        ("speed", "Speed"), ("gradient", "\\textbf{Gradient}"), ("level-set", "Level-set"),
        ("speed:gradient", "Speed$\\times$gradient"), ("speed:level-set", "Speed$\\times$level-set"),
        ("gradient:level-set", "Gradient$\\times$level-set"),
        ("speed:gradient:level-set", "Three-way")])
    rows.append(r"\midrule")
    rows += block("(b) Local-max $\\times$ fixed-pair, added to speed + gradient + level-set", fC, [
        ("crit", "Local-max"), ("E2", "\\textbf{Fixed-pair}"), ("crit:E2", "Local-max$\\times$fixed-pair")])
    rows.append(r"\midrule")
    rows += block("(c) Level-set $\\times$ fixed-pair, added to gradient", fL, [
        ("level-set", "Level-set"), ("E2", "\\textbf{Fixed-pair}"), ("level-set:E2", "Level-set$\\times$fixed-pair")])
    write("tab_factorial.tex", "\n".join(rows) + "\n")


def table_headline():
    """CNN / UV / F1 means plus % change and sample-wise wins."""
    rows = []
    disp = {"dB": r"Bottleneck $d_B$", "W2inf": r"$W_{2,\infty}$", "W22": r"$W_{2,2}$", "MT": r"Merge tree"}
    for m, nm in enumerate(METRICS):
        c, u, t = MEAN["cnn"][m], MEAN["uv"][m], MEAN["f1_grad_e2"][m]
        d = 3 if m in (0, 3) else 2
        rows.append(" & ".join([
            disp[nm], f(c, d), f(u, d), rf"\textbf{{{f(t, d)}}}",
            sgn(pct(c, u)) + r"\%", sgn(pct(c, t)) + r"\%", sgn(pct(u, t)) + r"\%",
            f"{WINS_VS_CNN['f1_grad_e2'][m]}/168", f"{WINS_VS_UV['f1_grad_e2'][m]}/168",
        ]) + r"\\")
    rows.append(r"\midrule")
    rows.append(r"All three PD lower & & & & & & & "
                f"{WINS_VS_CNN['f1_grad_e2'][4]}/168 & {WINS_VS_UV['f1_grad_e2'][4]}/168\\\\")
    rows.append(r"All three PD and MT lower & & & & & & & "
                f"{WINS_VS_CNN['f1_grad_e2'][5]}/168 & {WINS_VS_UV['f1_grad_e2'][5]}/168\\\\")
    write("tab_headline.tex", "\n".join(rows) + "\n")


def table_medians():
    rows = []
    for k in IDS:
        rows.append(" & ".join([LABEL[k]] + [f(MEDIAN[k][m], 3) for m in range(4)]) + r"\\")
    write("tab_medians.tex", "\n".join(rows) + "\n")


def table_wins():
    rows = []
    for k in IDS:
        a = WINS_VS_CNN.get(k)
        b = WINS_VS_UV.get(k)
        ca = [str(x) for x in a] if a else ["---"] * 6
        cb = [str(x) for x in b] if b else ["---"] * 6
        rows.append(" & ".join([LABEL[k]] + ca[:4] + [ca[4], ca[5]] + cb[:4] + [cb[4], cb[5]]) + r"\\")
    write("tab_wins.tex", "\n".join(rows) + "\n")



# Targeted contrasts (compendium Sec. 7.4): (label, from, to, expected % for dB,W2inf,W22,MT)
CONTRASTS = [
    ("Speed, added to control", "uv", "speed_only", (0.80, 0.25, 0.29, 0.20)),
    ("Level-set, added to control", "uv", "levelset_only", (0.54, 0.16, 0.19, 0.07)),
    ("Gradient, added to control", "uv", "grad_only", (22.63, 28.10, 32.05, -0.73)),
    ("Local-max, added to control", "uv", "uv_crit", (14.31, 4.25, 4.59, 5.36)),
    ("Fixed-pair, added to control", "uv", "uv_e2", (31.97, 27.99, 30.57, 6.95)),
    ("Local-max, added to S+G+L", "candidate_b", "candidate_c", (7.55, 0.88, 0.80, 1.31)),
    ("Fixed-pair, added to S+G+L", "candidate_b", "b_e2", (19.61, 9.56, 9.18, 7.85)),
    ("Fixed-pair, added to S+G+L+M", "candidate_c", "c_e2", (13.76, 7.34, 6.78, 6.87)),
    ("Local-max, added to S+G+L+FP", "b_e2", "c_e2", (0.82, -1.55, -1.81, 0.26)),
    ("Local-max, added to G", "grad_only", "f3_grad_crit", (6.56, 0.62, 0.72, 1.19)),
    ("Fixed-pair, added to G (gives final)", "grad_only", "f1_grad_e2", (15.49, 5.40, 4.61, 6.60)),
    ("Level-set, added to final G+FP", "f1_grad_e2", "f2_grad_levelset_e2", (-1.24, 0.21, 0.38, -0.31)),
    ("Gradient, added to FP (gives final)", "uv_e2", "f1_grad_e2", (3.89, 5.55, 6.64, -1.12)),
    ("Remove S and L from S+G+L+FP (gives final)", "b_e2", "f1_grad_e2", (0.89, 1.05, 1.17, 0.37)),
]


def table_contrasts():
    rows = []
    for lab, a, b, exp in CONTRASTS:
        got = [pct(MEAN[a][m], MEAN[b][m]) for m in range(4)]
        check(f"contrast[{lab}]", got, exp, 0.006)
        rows.append(" & ".join([lab] + [sgn(g, 2) + r"\%" for g in got]) + r"\\")
    write("tab_contrasts.tex", "\n".join(rows) + "\n")


def maybe_load_csv(path):
    """Overwrite MEAN/MEDIAN from corrected_pd_mt_method_means.csv
    (columns: method_id, mean_dB, median_dB, ..., mean_MT, median_MT)."""
    with open(path) as fh:
        for row in csv.DictReader(fh):
            k = row["method_id"]
            if k in MEAN:
                MEAN[k] = tuple(float(row[f"mean_{m}"]) for m in METRICS)
                MEDIAN[k] = tuple(float(row[f"median_{m}"]) for m in METRICS)


def load_joined(path):
    """Rebuild MEAN, MEDIAN and both win-count tables from the frozen per-field
    table corrected_pd_mt_joined.csv (columns: sample_idx, method_id, dB, W2inf,
    W22, MT; 18 methods x 168 fields), and check them against the values that
    were transcribed from the compendium (the numbers printed in the paper)."""
    import statistics, math
    vals = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            k = row["method_id"]
            sample = int(row["sample_idx"])
            method_rows = vals.setdefault(k, {})
            if sample in method_rows:
                sys.exit(f"[error] duplicate method/sample pair: {k}/{sample}")
            data = tuple(float(row[m]) for m in METRICS)
            if not all(math.isfinite(v) and v >= 0 for v in data):
                sys.exit(f"[error] invalid distance at {k}/{sample}: {data}")
            method_rows[sample] = data
    missing = sorted(set(IDS) - set(vals))
    extra = sorted(set(vals) - set(IDS))
    if missing or extra:
        sys.exit(f"[error] method ids differ from the paper: missing={missing} extra={extra}")
    samples = sorted(vals["cnn"])
    if samples != list(range(168)):
        sys.exit("[error] expected samples 0..167")
    for k in IDS:
        if sorted(vals[k]) != samples:
            sys.exit(f"[error] {k}: incomplete sample coverage")

    def wins(k, ref):
        w = [0] * 6
        for sidx in samples:
            a, b = vals[k][sidx], vals[ref][sidx]
            lower = [a[m] < b[m] for m in range(4)]
            w[0] += lower[0]; w[1] += lower[1]; w[2] += lower[2]; w[3] += lower[3]
            all3 = lower[0] and lower[1] and lower[2]
            w[4] += all3; w[5] += all3 and lower[3]
        return tuple(w)

    old_mean = dict(MEAN)
    old_median = dict(MEDIAN)
    old_wcnn, old_wuv = dict(WINS_VS_CNN), dict(WINS_VS_UV)
    for k in IDS:
        cols = list(zip(*[vals[k][sidx] for sidx in samples]))
        MEAN[k] = tuple(sum(c) / len(c) for c in cols)
        MEDIAN[k] = tuple(statistics.median(c) for c in cols)
        if k != "cnn":
            WINS_VS_CNN[k] = wins(k, "cnn")
        if k != "uv":
            WINS_VS_UV[k] = wins(k, "uv")
    for k in IDS:
        check(f"mean[{k}]", MEAN[k], old_mean[k], 0.00002)
        check(f"median[{k}]", MEDIAN[k], old_median[k], 0.0006)
        if k in old_wcnn:
            check(f"wins_vs_cnn[{k}]", WINS_VS_CNN[k], old_wcnn[k], 0)
        if k in old_wuv:
            check(f"wins_vs_uv[{k}]", WINS_VS_UV[k], old_wuv[k], 0)
    print(f"loaded {path}: {len(IDS)} methods x {len(samples)} fields")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", help="corrected_pd_mt_joined.csv (per-field; preferred)")
    ap.add_argument("--means", help="corrected_pd_mt_method_means.csv (means/medians only)")
    ap.add_argument("--out", help="output directory for the LaTeX table bodies")
    a = ap.parse_args()
    if a.out:
        OUT = a.out
    if a.joined:
        load_joined(a.joined)
    elif a.means:
        maybe_load_csv(a.means)
    fB, fC, fL = verify()
    table_contrasts()
    if FAILS:
        print("VERIFICATION FAILURES:")
        print("\n".join(FAILS))
        sys.exit(1)
    print("All derived quantities match the values reported in the manuscript "
          f"({len(EXPECT_PCT_VS_UV)} vs-UV rows, {len(EXPECT_RANK)} rank rows, "
          f"{sum(len(v) for v in EXPECT_FACTORIAL.values())} factorial effects, {len(EXPECT_PARETO)} Pareto fronts).")
    table_all_candidates()
    table_factorial(fB, fC, fL)
    table_headline()
    table_medians()
    table_wins()
    print("wrote", sorted(os.listdir(OUT)))
