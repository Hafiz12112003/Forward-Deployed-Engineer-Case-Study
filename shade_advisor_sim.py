"""
Synthetic proof-of-point for the Ceramic Tiles case study.
Point proved: given a print file's per-region ink composition (known from the file) and ONE
calibration round (7 variants), a pooled sensitivity model can recommend a 6-channel correction
that lands inside tolerance in ~1 verification round, versus an eyeballed manual loop.

NOT real data. Printer physics, drift size, operator behaviour are all ASSUMED.
Colour difference = Euclidean dE*ab in CIELAB (simple stand-in for dE2000).
Run: python shade_advisor_sim.py
"""
import numpy as np

rng = np.random.default_rng(7)
K, R = 6, 8                       # 6 ink channels, 8 colour regions of the design
TOL_MEAN, TOL_MAX = 1.0, 1.5      # assumed acceptance
MEAS_NOISE = 0.30                 # lab repeatability (LAB units), assumed
EYE_NOISE = 1.00                  # operator visual judgement noise, assumed
RESID = 0.40                      # drift NOT explainable by ink changes (LAB units), assumed
MAXSTEP = 0.12                    # advisor never changes a channel more than +/-12%

# ink effect on LAB (3 x 6): how a full-strength change in each ink moves L,a,b (assumed)
M_true = np.array([[-18, -12, -10, -6, -14, -8],
                   [  6,  -4,   9, -2,   3,  5],
                   [ 10,  12,  -5,  8,  -9,  4]], float)
W = rng.dirichlet(np.ones(K) * 0.6, size=R)       # region ink composition, from the print file
base = rng.uniform([40, -10, -10], [85, 15, 25], size=(R, 3))

def lab(s, drift=0.0, noise=0.0):
    """s: per-channel scale (1.0 = file as-is). Mild nonlinearity (dot gain)."""
    d = (s ** 1.3 - 1.0)
    out = base + (W * d) @ M_true.T + drift
    return out + rng.normal(0, noise, out.shape) if noise else out

def de(a, b):                      # per-region dE*ab
    return np.linalg.norm(a - b, axis=1)

def passes(m):
    e = de(m, master)
    return e.mean() <= TOL_MEAN and e.max() <= TOL_MAX

# ---- one-time calibration per printer/ink set: 7 variants, +/-10% on random channels ----
def calibrate(n_var=7):
    X, Y = [], []
    ref = lab(np.ones(K), noise=MEAS_NOISE)
    for _ in range(n_var):
        s = np.ones(K) + rng.choice([-.10, 0, .10], K)
        m = lab(s, noise=MEAS_NOISE)
        for r in range(R):                      # pooled: shared 3x6 ink matrix across regions
            X.append(W[r] * (s ** 1.3 - 1.0))
            Y.append(m[r] - ref[r])
    X, Y = np.array(X), np.array(Y)
    return np.linalg.lstsq(X.T @ X + 1e-4 * np.eye(K), X.T @ Y, rcond=None)[0].T   # 3 x 6

M_hat = calibrate()

def advisor_round(s, meas):
    """Return new scale vector from measured shift (linearised, ridge, clamped)."""
    resid = (master - meas)                                     # R x 3, what we still need
    A = np.vstack([(W[r][None, :] * M_hat) for r in range(R)])  # (3R x 6) per unit d(scale^1.3)
    b = resid.reshape(-1)
    dd = np.linalg.solve(A.T @ A + 1e-2 * np.eye(K), A.T @ b)
    dd = np.clip(dd, -MAXSTEP, MAXSTEP)
    return np.clip(s + dd, 0.7, 1.3)

def run_advisor(drift, cap=6):
    s = np.ones(K)
    for rnd in range(1, cap + 1):
        m = lab(s, drift, MEAS_NOISE)            # print+fire ONE verification tile, measure
        if passes(m): return rnd - 1, de(m, master).mean()   # extra rounds beyond first sample
        s = advisor_round(s, m)
    return cap, de(m, master).mean()

def run_manual(drift, cap=8):
    s = np.ones(K)
    for rnd in range(1, cap + 1):
        m0 = lab(s, drift, MEAS_NOISE)
        if passes(m0): return rnd - 1, de(m0, master).mean()
        best, best_seen = None, 1e9
        for _ in range(7):                       # 6-7 variants, "raise brown 5%"-style edits
            v = s.copy()
            for k in rng.choice(K, rng.integers(1, 3), replace=False):
                v[k] *= 1 + rng.choice([-.05, .05])
            seen = de(lab(v, drift), master).mean() + rng.normal(0, EYE_NOISE)
            if seen < best_seen: best, best_seen = v, seen
        s = best
    return cap, de(lab(s, drift, MEAS_NOISE), master).mean()

if __name__ == "__main__":
    N = 500
    res = {"manual": [], "advisor": []}
    for _ in range(N):
        master = lab(np.ones(K))                                  # master tile (noise-free truth)
        # ASSUMPTION (the one the whole approach rests on): rerun drift behaves mostly like an
        # ink-density shift (compensable via the 6 channels) plus unexplained residual.
        g = np.ones(K) + rng.normal(0, 0.06, K)
        drift = lab(g) - lab(np.ones(K)) + rng.normal(0, RESID, (R, 3))
        res["manual"].append(run_manual(drift))
        res["advisor"].append(run_advisor(drift))
    print(f"Calibration error of ink matrix: {np.abs(M_hat - M_true).mean():.2f} (mean abs, true scale ~10)")
    print(f"Noise floor: lab repeatability {MEAS_NOISE} -> advisor cannot beat ~{MEAS_NOISE*1.7:.1f} dE\n")
    for k, v in res.items():
        r = np.array([x[0] for x in v]); e = np.array([x[1] for x in v])
        print(f"{k:8s} rounds after 1st sample: mean {r.mean():.2f} | median {np.median(r):.0f} | "
              f"<=1 round {np.mean(r<=1)*100:.0f}% | hit cap {np.mean(r>=6)*100:.0f}% | final mean dE {e.mean():.2f}")
