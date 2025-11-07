# resa_3plus1_optimized.py
# Optimized Replica-Exchange (Parallel Tempering) for 3+1 neutrino oscillation fit
import numpy as np ; import math
from dataclasses import dataclass, field

PI = math.pi
I = 1j

def deg2rad(d): return d * PI / 180.0

# ----------- RNG: use numpy Generator --------
rng = np.random.default_rng()
def uniform01(): return float(rng.random())
def gaussian(sigma): return float(rng.normal(0.0, sigma))

# ---------- Small matrix utilities ----------
def rotation_ij(i, j, theta, phase=0.0):
    """4x4 complex rotation in plane (i,j) with optional phase on sin terms."""
    # Using numpy arrays; this is tiny and cheap compared to osc loops
    R = np.eye(4, dtype=np.complex128)
    c = math.cos(theta)
    s = math.sin(theta)
    e_minus = np.exp(-I * phase)
    e_plus  = np.exp( I * phase)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = s * e_minus
    R[j, i] = -s * e_plus
    return R

def buildU(angles, delta24):
    """Sequence: R12 * R13 * R14 * R23 * R24(with phase) * R34"""
    seq = [
        (0, 1, angles[0], 0.0),
        (0, 2, angles[1], 0.0),
        (0, 3, angles[2], 0.0),
        (1, 2, angles[3], 0.0),
        (1, 3, angles[4], delta24),
        (2, 3, angles[5], 0.0)
    ]
    U = np.eye(4, dtype=np.complex128)
    for (i, j, th, ph) in seq:
        U = U @ rotation_ij(i, j, th, ph)
    return U

# ---------- Data container with precomputed arrays ----------
@dataclass
class DataSet:
    Es: np.ndarray        # shape (N,)
    Ls: np.ndarray        # shape (N,)
    alphas: np.ndarray    # shape (N,) ints
    betas: np.ndarray     # shape (N,) ints
    obs: np.ndarray       # shape (N,)
    err: np.ndarray       # shape (N,)

    @classmethod
    def from_datapoints(cls, datapoints):
        Es = np.array([d.E for d in datapoints], dtype=float)
        Ls = np.array([d.L for d in datapoints], dtype=float)
        alphas = np.array([d.alpha for d in datapoints], dtype=np.int64)
        betas  = np.array([d.beta  for d in datapoints], dtype=np.int64)
        obs = np.array([d.obs for d in datapoints], dtype=float)
        err = np.maximum(np.array([d.err for d in datapoints], dtype=float), 1e-12)
        return cls(Es, Ls, alphas, betas, obs, err)

# ---------- Oscillation probability (fully vectorized) ----------
def oscProb_vectorized(U, alphas, betas, Es, Ls, dm21, dm31, dm41):
    """
    Vectorized computation of P_{alpha->beta} for arrays Es,Ls,alphas,betas.
    - U : 4x4 complex mixing matrix
    - alphas, betas, Es, Ls : 1D arrays of same length N
    Returns: 1D array of probabilities length N
    """
    # m^2 array
    m2 = np.array([0.0, dm21, dm31, dm41], dtype=float)  # shape (4,)

    # shape (N,4): U_alpha = U[alphas,:], U_beta = U[betas,:]
    U_alpha = U[alphas, :]         # broadcasting fancy index -> (N,4)
    U_beta  = U[betas, :]          # (N,4)
    conjU_beta = np.conjugate(U_beta)

    # M_ij = U_alpha[i,j] * conj(U_beta[i,j])  => shape (N,4)
    M = U_alpha * conjU_beta       # elementwise complex multiply

    # compute phi_j(i) = 1.267 * m2_j * L_i / E_i
    # Ls_over_Es shape (N,)
    L_over_E = Ls / Es             # (N,)
    # phi shape (4,N) -> transpose to (N,4) to match M
    phi = 1.267 * np.outer(L_over_E, m2)  # (N,4)
    exp_phase = np.exp(-1j * phi)         # (N,4) complex

    # amplitude per datapoint: amp_i = sum_j M_ij * exp(-i phi_ij)
    amp = np.sum(M * exp_phase, axis=1)    # (N,) complex
    probs = np.abs(amp)**2
    # clamp for numerical safety
    np.clip(probs, 0.0, 1.0, out=probs)
    return probs

# ---------- DataPoint & chi2 ----------
@dataclass
class DataPoint:
    E: float
    L: float
    alpha: int
    beta: int
    obs: float
    err: float

def chi2_for_params_dataset(dataset: DataSet, angles, delta24, dm21, dm31, dm41):
    """Compute χ² using precomputed dataset arrays (fast)."""
    U = buildU(angles, delta24)
    pred = oscProb_vectorized(U,
                              dataset.alphas, dataset.betas,
                              dataset.Es, dataset.Ls,
                              dm21, dm31, dm41)
    res = dataset.obs - pred
    return float(np.sum((res / dataset.err) ** 2))

# ---------- Params & perturb ----------
@dataclass
class Params:
    dm21: float
    dm31: float
    dm41: float
    angles: np.ndarray  # length 6
    delta24: float

def random_init():
    return Params(
        dm21=7.5e-5,
        dm31=2.5e-3,
        dm41=1.0,
        angles=np.array([
            deg2rad(33.44), deg2rad(8.57), deg2rad(2.0),
            deg2rad(49.2), deg2rad(5.0),  deg2rad(10.0)
        ], dtype=float),
        delta24=0.0
    )

def perturb(p: Params, step_scales):
    """Perturb all params; step_scales is a length-10 sequence."""
    # scalar params
    dm21 = p.dm21 + gaussian(step_scales[0])
    dm31 = p.dm31 + gaussian(step_scales[1])
    dm41 = p.dm41 + gaussian(step_scales[2])
    # angles perturb (vector)
    angle_noise = rng.normal(0.0, scale=step_scales[3:9])
    angles = p.angles + angle_noise
    delta24 = p.delta24 + gaussian(step_scales[9])

    # enforce bounds
    dm21 = max(dm21, 1e-12)
    dm31 = max(dm31, 1e-12)
    dm41 = max(dm41, 1e-12)
    # map angles into [0, pi/2]
    angles = np.abs(angles) % (PI/2)
    # wrap phase to [-pi, pi]
    delta24 = ((delta24 + PI) % (2*PI)) - PI

    return Params(dm21, dm31, dm41, angles, delta24)

# ---------- Replica ----------
@dataclass
class Replica:
    params: Params
    T: float
    chi2: float = field(default=np.inf)

# ---------- Replica exchange (main loop) ----------
def run_replica_exchange(reps, dataset: DataSet, step_scales, nsteps_each_replica, swap_interval, rng_seed=None, save_traces=False):
    global rng
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)

    N = len(reps)
    # diagnostic arrays
    chi2_trace = np.zeros((nsteps_each_replica, N), dtype=float) if save_traces else None
    local_accepts = np.zeros(N, dtype=int)
    local_trials  = np.zeros(N, dtype=int)
    swap_accepts = np.zeros(N-1, dtype=int)
    swap_trials  = np.zeros(N-1, dtype=int)

    # initial chi2
    for r in reps:
        p = r.params
        r.chi2 = chi2_for_params_dataset(dataset, p.angles, p.delta24, p.dm21, p.dm31, p.dm41)

    for step in range(nsteps_each_replica):
        for idx, r in enumerate(reps):
            prop = perturb(r.params, step_scales)
            chi2_prop = chi2_for_params_dataset(dataset, prop.angles, prop.delta24,
                                                prop.dm21, prop.dm31, prop.dm41)
            dE = chi2_prop - r.chi2
            beta = 1.0 / r.T
            local_trials[idx] += 1
            if dE <= 0.0 or uniform01() < math.exp(-beta * dE):
                r.params = prop
                r.chi2 = chi2_prop
                local_accepts[idx] += 1

        # record chi2
        if save_traces:
            for k in range(N):
                chi2_trace[step, k] = reps[k].chi2

        # swaps
        if (step % swap_interval) == 0:
            for i in range(N-1):
                swap_trials[i] += 1
                beta1 = 1.0 / reps[i].T
                beta2 = 1.0 / reps[i+1].T
                chi1 = reps[i].chi2
                chi2 = reps[i+1].chi2
                ex = (beta1 - beta2) * (chi2 - chi1)
                acc = 1.0 if ex >= 0 else math.exp(ex)
                if uniform01() < acc:
                    reps[i].params, reps[i+1].params = reps[i+1].params, reps[i].params
                    reps[i].chi2, reps[i+1].chi2 = reps[i+1].chi2, reps[i].chi2
                    swap_accepts[i] += 1

    diagnostics = {
        "chi2_trace": chi2_trace,
        "local_accepts": local_accepts,
        "local_trials": local_trials,
        "swap_accepts": swap_accepts,
        "swap_trials": swap_trials
    }
    return diagnostics

# --------- Tune Steps ----------
def tune_step_scales(base_params: Params, dataset: DataSet, init_scales, target_accept=0.3,
                     ncycles=500, Nrep=8, Tmin=1.0, Tmax=10.0, rng_seed=None):
    """
    Automatic step-scale tuner for the parallel-tempering sampler.
    Runs short adaptive pilot chains and rescales per-parameter step sizes
    toward a target acceptance fraction (default 0.3).
    Returns tuned step_scales list.
    """
    print(f"Tuning step scales for ~{ncycles} cycles, {Nrep} replicas...")
    global rng
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)

    temps = [Tmin * (Tmax/Tmin)**(i/(Nrep-1)) for i in range(Nrep)]
    reps = [Replica(base_params, T=t) for t in temps]
    step_scales = np.array(init_scales, dtype=float)

    acc = np.zeros_like(step_scales)
    trials = np.zeros_like(step_scales)

    # Adaptation rate
    adapt_rate = 0.5

    for c in range(ncycles):
        for r in reps:
            p0 = r.params
            prop = perturb(p0, step_scales)
            chi2_old = chi2_for_params_dataset(dataset, p0.angles, p0.delta24, p0.dm21, p0.dm31, p0.dm41)
            chi2_new = chi2_for_params_dataset(dataset, prop.angles, prop.delta24, prop.dm21, prop.dm31, prop.dm41)
            dE = chi2_new - chi2_old
            beta = 1.0 / r.T
            accept = dE <= 0.0 or uniform01() < math.exp(-beta * dE)
            # record trial/accept for each parameter
            for i in range(len(step_scales)):
                trials[i] += 1
                if accept:
                    acc[i] += 1
            if accept:
                r.params = prop

        # adapt every 50 steps
        if (c+1) % 50 == 0:
            rates = acc / np.maximum(trials, 1)
            for i in range(len(step_scales)):
                ratio = rates[i] / target_accept
                # limit the adjustment to ±20% per update
                scale_factor = np.clip(ratio ** adapt_rate, 0.8, 1.2)
                step_scales[i] *= scale_factor
            acc[:] = 0
            trials[:] = 0

    print("Tuned step scales:")
    for i, s in enumerate(step_scales):
        print(f"  scale[{i}] = {s:.3e}")
    return step_scales.tolist()

# ---------- Example usage ----------
def main():
    data_points = []
    L = 1.0
    for E in np.arange(0.5, 10.0 + 1e-9, 0.5):
        data_points.append(DataPoint(E=E, L=L, alpha=1, beta=0, obs=0.01 * uniform01(), err=0.02))
    dataset = DataSet.from_datapoints(data_points)

    # --- Initial rough guess for step scales ---
    init_scales = [
        1e-6, 5e-4, 0.1,
        deg2rad(0.5), deg2rad(0.5), deg2rad(0.5),
        deg2rad(0.5), deg2rad(0.5), deg2rad(0.5),
        deg2rad(10.0)
    ]

    # --- Auto-tune ---
    tuned_scales = tune_step_scales(random_init(), dataset, init_scales,
                                    target_accept=0.3, ncycles=500,
                                    Nrep=8, Tmin=1.0, Tmax=10.0, rng_seed=123)

    # --- Use tuned scales for the real run ---
    step_scales = tuned_scales

    print("Tuned step scales:")
    for i, s in enumerate(step_scales):
        print(f"  param {i}: {s:.3e}")

    # --- Define replica ladder (now with tuned scales) ---
    Nrep = 12
    Tmin, Tmax = 1.0, 100.0
    temps = [Tmin * (Tmax/Tmin)**(i/(Nrep-1)) for i in range(Nrep)]
    reps = [Replica(random_init(), T=t) for t in temps]

    ncycles = 20000
    swap_interval = 5
    rng_seed = 12345

    print(f"Running {ncycles} cycles, {Nrep} replicas...")
    diagnostics = run_replica_exchange(reps, dataset, step_scales, ncycles, swap_interval, rng_seed=rng_seed, save_traces=True)

    # --- Find best replica ---
    best = min(reps, key=lambda r: r.chi2)
    p = best.params
    print(f"Best chi2 = {best.chi2:.6g}")
    print(f"dm21 = {p.dm21:.6g} eV^2")
    print(f"dm31 = {p.dm31:.6g} eV^2")
    print(f"dm41 = {p.dm41:.6g} eV^2")
    print("angles (deg):", ", ".join(f"{math.degrees(a):.3f}" for a in p.angles))
    print(f"delta24 (deg) = {math.degrees(p.delta24):.3f}")

    # --- Diagnostics ---
    la = diagnostics["local_accepts"]
    lt = diagnostics["local_trials"]
    print("Local acceptance rates per replica:", (la / np.maximum(lt, 1)).tolist())
    sa = diagnostics["swap_accepts"]
    st = diagnostics["swap_trials"]
    print("Swap acceptance rates between replicas:", (sa / np.maximum(st, 1)).tolist())

    # --- Save results ---
    np.savez("resa_chains.npz",
             chi2_trace=diagnostics["chi2_trace"],
             local_accepts=la, local_trials=lt,
             swap_accepts=sa, swap_trials=st,
             final_params=[r.params for r in reps])

if __name__ == "__main__":
    main()