# amd_chemistry_fast.pyx
"""
Fast AMD chemistry solver — Cython implementation.

Mirrors amd_chemistry_fast.py but compiled ahead-of-time:
  - No JIT warmup cost
  - All helpers are `cdef` (inlined C functions, zero call overhead)
  - prange → OpenMP threads (compile with -fopenmp)
  - Fixed-size C arrays for the Newton Jacobian (no heap allocation per cell)

Compile
-------
Add to your setup.py / pyproject.toml::

    from setuptools import setup
    from Cython.Build import cythonize
    import numpy as np
    from setuptools.extension import Extension

    ext = Extension(
        "amd_chemistry_fast",
        sources=["amd_chemistry_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-fopenmp", "-ffast-math"],
        extra_link_args=["-fopenmp"],
    )
    setup(ext_modules=cythonize([ext], compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "nonecheck": False,
        "language_level": "3",
    }))

Then: python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport (
    log10, sqrt, exp, fmax, fmin, isfinite, pow as cpow, fabs
)

# ── dtype alias ───────────────────────────────────────────────────────────────
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# ── Module-level constants (cdef = pure C, no Python object overhead) ─────────
cdef double K_FERRIC  = 10.0**-9.74 * 1e4 / 60.0
cdef double K_DO      = 10.0**-8.19
cdef double K1_OX     = 1.33e12
cdef double K2_OX     = 2.91e-9
cdef double P02       = 0.21
cdef double KW        = 1.0e-14
cdef double KSP       = 7.762471e4          # 10^4.891
cdef double H_CAP     = 1.0e4


# ═════════════════════════════════════════════════════════════════════════════
# PART A  Analytical helpers  (cdef nogil — callable inside prange)
# ═════════════════════════════════════════════════════════════════════════════

cdef inline void _davies_activity(
        double h_c, double fe3_c, double fe2_c, double so4_c,
        double *gam_h, double *gam_fe3) noexcept nogil:
    """Davies-equation activity coefficients for H⁺ (z=1) and Fe³⁺ (z=3)."""
    cdef double I, sqI, d
    I   = 0.5 * (h_c + 9.0*fe3_c + 4.0*fe2_c + 4.0*so4_c)
    if I > 1.0:
        I = 1.0
    sqI      = sqrt(I)
    d        = sqI / (1.0 + sqI) - 0.3 * I
    gam_h[0]   = cpow(10.0, -0.5 * 1.0 * d)
    gam_fe3[0] = cpow(10.0, -0.5 * 9.0 * d)


cdef inline void _apply_equilibrium_precip(
        double *fe3, double *h, double *fe2, double *so4, 
        double *fe_oh3, double *bedload,
        double vol_safe) noexcept nogil:
    """
    Instantaneous Fe(OH)₃ ↔ Fe³⁺ equilibrium (closed-form).

    Computes the Davies-corrected equilibrium [Fe³⁺] and moves the
    surplus/deficit to/from fe_oh3 (and bedload if needed).
    Modifies fe3, h, fe_oh3, bedload in place via pointers.
    """
    cdef double h_c, fe3_c, gam_h, gam_fe3, fe2_c, so4_c
    cdef double act_h, eq_act, fe3_eq_mol, delta
    cdef double p, need, avail_sus, from_sus, remaining, from_bed

    h_c   = fmax(h[0]   / vol_safe, 1e-14)
    fe3_c = fmax(fe3[0] / vol_safe, 0.0)
    fe2_c = fmax(fe2[0] / vol_safe, 0.0)
    so4_c = fmax(so4[0] / vol_safe, 0.0)

    _davies_activity(h_c, fe3_c, fe2_c, so4_c, &gam_h, &gam_fe3)

    act_h      = gam_h * h_c
    eq_act     = KSP * act_h * act_h * act_h   # KSP * [H+]_act^3
    fe3_eq_mol = (eq_act / gam_fe3) * vol_safe  # equilibrium moles

    delta = fe3[0] - fe3_eq_mol   # > 0 → supersaturated → precipitate

    if delta > 0.0:
        # Precipitation:  Fe³⁺ → Fe(OH)₃  +  3 H⁺
        p        = fmin(delta, fe3[0])
        fe3[0]   -= p
        fe_oh3[0]+= p
        h[0]     += 3.0 * p
    else:
        # Dissolution:  Fe(OH)₃ → Fe³⁺  (consumes 3 H⁺)
        need     = -delta
        avail_sus = fmax(fe_oh3[0], 0.0)
        from_sus  = fmin(need, avail_sus)
        fe_oh3[0]-= from_sus
        fe3[0]   += from_sus
        h[0]      = fmax(h[0] - 3.0*from_sus, 0.0)

        remaining = need - from_sus
        if remaining > 0.0:
            from_bed    = fmin(remaining, fmax(bedload[0], 0.0))
            bedload[0] -= from_bed
            fe3[0]     += from_bed
            h[0]        = fmax(h[0] - 3.0*from_bed, 0.0)


cdef inline void _analytical_fe2_oxidation(
        double *fe2, double *fe3, double *h,
        double vol_safe,
        double dt) noexcept nogil:
    """
    Exact exponential solution for Fe²⁺ → Fe³⁺ oxidation.

    ODE:  d(fe2)/dt = -λ·fe2   →   fe2(t) = fe2₀·exp(-λ·dt)

    No stiffness issues regardless of pH.
    Modifies fe2, fe3, h in place.
    """
    cdef double h_c, oh_c, lam, fe2_new, fe2_ox, max_h, scale

    h_c  = fmax(h[0] / vol_safe, 1e-14)
    oh_c = KW / h_c
    lam  = K2_OX * P02 + K1_OX * (oh_c * oh_c) * P02

    fe2_new = fe2[0] * exp(-lam * dt)
    fe2_new = fmax(fe2_new, 0.0)
    fe2_ox  = fe2[0] - fe2_new

    fe3[0] += fe2_ox
    h[0]    = fmax(h[0] - fe2_ox, 0.0)
    fe2[0]  = fe2_new


# ═════════════════════════════════════════════════════════════════════════════
# PART B  Backward-Euler Newton for pyrite oxidation  (cdef nogil)
# ═════════════════════════════════════════════════════════════════════════════

cdef inline void _pyrite_rhs(
        double fe2, double fe3, double so4, double h,
        double ore, double vol_safe, double h2o, double do_sqrt,
        double *dfe2, double *dfe3, double *dso4, double *dh) noexcept nogil:
    """Pyrite reaction rates (Steps 1 & 2) in mol s⁻¹."""
    cdef double fe3_c, h_c, rate1, rate2

    dfe2[0] = 0.0; dfe3[0] = 0.0; dso4[0] = 0.0; dh[0] = 0.0

    # Step 1: FeS₂ + Fe³⁺
    if fe3 > 0.0 and ore > 0.0:
        fe3_c = fmax(fe3 / vol_safe, 1e-10)
        h_c   = fmax(h   / vol_safe, 1e-7)
        rate1 = K_FERRIC * sqrt(fe3_c) / sqrt(h_c) * ore
        rate1 = fmin(rate1, fmin(fe3, h2o))
        dfe2[0] += rate1 * 1.07
        dfe3[0] -= rate1
        dh[0]   += rate1 * 1.14
        dso4[0] += rate1 * (2.0 / 14.0)

    # Step 2: FeS₂ + O₂
    if ore > 0.0:
        h_c   = fmax(h / vol_safe, 1e-7)
        rate2 = K_DO * do_sqrt / cpow(h_c, 0.11) * ore
        rate2 = fmin(rate2, h2o)
        dfe2[0] += rate2
        dso4[0] += 2.0 * rate2
        dh[0]   += 2.0 * rate2


cdef inline void _gauss4(double A[4][4], double b[4], double x[4]) noexcept nogil:
    """
    In-place Gaussian elimination with partial pivoting for Ax = b, n=4.
    All storage is on the C stack — zero heap allocation.
    """
    cdef double Ab[4][5]
    cdef double pivot, factor, tmp
    cdef int i, j, k, col, row, max_row

    for i in range(4):
        for j in range(4):
            Ab[i][j] = A[i][j]
        Ab[i][4] = b[i]

    for col in range(4):
        # Partial pivot
        max_row = col
        for row in range(col+1, 4):
            if fabs(Ab[row][col]) > fabs(Ab[max_row][col]):
                max_row = row
        if max_row != col:
            for k in range(5):
                tmp           = Ab[col][k]
                Ab[col][k]    = Ab[max_row][k]
                Ab[max_row][k]= tmp
        pivot = Ab[col][col]
        if fabs(pivot) < 1e-30:
            continue
        for row in range(col+1, 4):
            factor = Ab[row][col] / pivot
            for k in range(col, 5):
                Ab[row][k] -= factor * Ab[col][k]

    for row in range(3, -1, -1):
        x[row] = Ab[row][4]
        for col in range(row+1, 4):
            x[row] -= Ab[row][col] * x[col]
        if fabs(Ab[row][row]) > 1e-30:
            x[row] /= Ab[row][row]


cdef inline void _backward_euler_pyrite(
        double *fe2, double *fe3, double *so4, double *h,
        double ore, double vol_safe, double h2o, double do_sqrt,
        double dt, int n_substeps) noexcept nogil:
    """
    Backward-Euler with Newton iteration for the pyrite oxidation ODEs.

    After removing the stiff Fe²⁺-oxidation and precipitation terms, the
    pyrite ODEs are only mildly stiff; n_substeps=1–4 converges in 2–5
    Newton iterations per sub-step.

    Uses only stack-allocated C arrays — no Python objects, no heap.
    """
    cdef double sub_dt = dt / n_substeps
    cdef double fe2_n, fe3_n, so4_n, h_n          # value at start of sub-step
    cdef double fe2_k, fe3_k, so4_k, h_k          # Newton iterate
    cdef double d2, d3, ds, dh                     # RHS values
    cdef double r2, r3, rs, rh                     # residuals
    cdef double norm, eps = 1e-7
    cdef double J[4][4]
    cdef double R[4]
    cdef double delta[4]
    cdef double f2e, f3e, fse, fhe                 # perturbed RHS
    cdef int s, it
    cdef double f2_k, f3_k, fs_k, fh_k # f(y_k) at current iteration

    for s in range(n_substeps):
        fe2_n = fe2[0]; fe3_n = fe3[0]
        so4_n = so4[0]; h_n   = h[0]

        # Initial guess: explicit Euler
        _pyrite_rhs(fe2_n, fe3_n, so4_n, h_n, ore, vol_safe, h2o, do_sqrt,
                    &d2, &d3, &ds, &dh)
        fe2_k = fmax(fe2_n + sub_dt * d2, 0.0)
        fe3_k = fmax(fe3_n + sub_dt * d3, 0.0)
        so4_k = fmax(so4_n + sub_dt * ds, 0.0)
        h_k   = fmax(h_n   + sub_dt * dh, 0.0)

        # Newton iterations
        for it in range(20):
            _pyrite_rhs(fe2_k, fe3_k, so4_k, h_k, ore, vol_safe, h2o, do_sqrt,
                        &f2_k, &f3_k, &fs_k, &fh_k)
            r2 = fe2_k - fe2_n - sub_dt * f2_k
            r3 = fe3_k - fe3_n - sub_dt * f3_k
            rs = so4_k - so4_n - sub_dt * fs_k
            rh = h_k   - h_n   - sub_dt * fh_k

            norm = sqrt(r2*r2 + r3*r3 + rs*rs + rh*rh)
            if norm < 1e-10:
                break

            # ── Build 4×4 Jacobian via forward finite differences ────────────
            # Column 0: perturb fe2
            _pyrite_rhs(fe2_k+eps, fe3_k, so4_k, h_k, ore, vol_safe, h2o, do_sqrt,
                        &f2e, &f3e, &fse, &fhe)
            J[0][0] = 1.0 - sub_dt*(f2e-f2_k)/eps
            J[1][0] =     - sub_dt*(f3e-f3_k)/eps
            J[2][0] =     - sub_dt*(fse-fs_k)/eps
            J[3][0] =     - sub_dt*(fhe-fh_k)/eps

            # Column 1: perturb fe3
            _pyrite_rhs(fe2_k, fe3_k+eps, so4_k, h_k, ore, vol_safe, h2o, do_sqrt,
                        &f2e, &f3e, &fse, &fhe)
            J[0][1] =     - sub_dt*(f2e-f2_k)/eps
            J[1][1] = 1.0 - sub_dt*(f3e-f3_k)/eps
            J[2][1] =     - sub_dt*(fse-fs_k)/eps
            J[3][1] =     - sub_dt*(fhe-fh_k)/eps

            # Column 2: perturb so4
            _pyrite_rhs(fe2_k, fe3_k, so4_k+eps, h_k, ore, vol_safe, h2o, do_sqrt,
                        &f2e, &f3e, &fse, &fhe)
            J[0][2] =     - sub_dt*(f2e-f2_k)/eps
            J[1][2] =     - sub_dt*(f3e-f3_k)/eps
            J[2][2] = 1.0 - sub_dt*(fse-fs_k)/eps
            J[3][2] =     - sub_dt*(fhe-fh_k)/eps

            # Column 3: perturb h
            _pyrite_rhs(fe2_k, fe3_k, so4_k, h_k+eps, ore, vol_safe, h2o, do_sqrt,
                        &f2e, &f3e, &fse, &fhe)
            J[0][3] =     - sub_dt*(f2e-f2_k)/eps
            J[1][3] =     - sub_dt*(f3e-f3_k)/eps
            J[2][3] =     - sub_dt*(fse-fs_k)/eps
            J[3][3] = 1.0 - sub_dt*(fhe-fh_k)/eps

            # Solve J·Δ = −R
            R[0] = -r2; R[1] = -r3; R[2] = -rs; R[3] = -rh
            _gauss4(J, R, delta)

            fe2_k = fmax(fe2_k + delta[0], 0.0)
            fe3_k = fmax(fe3_k + delta[1], 0.0)
            so4_k = fmax(so4_k + delta[2], 0.0)
            h_k   = fmax(h_k   + delta[3], 0.0)

        fe2[0] = fe2_k; fe3[0] = fe3_k
        so4[0] = so4_k; h[0]   = h_k


# ═════════════════════════════════════════════════════════════════════════════
# PART C  Main public function
# ═════════════════════════════════════════════════════════════════════════════

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def process_chemistry(
    double[:, ::1] fe2,
    double[:, ::1] fe3,
    double[:, ::1] so4,
    double[:, ::1] h,
    double[:, ::1] fe_oh3,
    double[:, ::1] bedload_storage,
    float[:, ::1]  ore,
    float[:, ::1]  volume,
    double do_val,
    double buffer_capacity,
    double time_step_seconds,
    Py_ssize_t[::1] valid_rows,
    Py_ssize_t[::1] valid_cols,
    Py_ssize_t num_valid,
    int n_substeps = 2,
):
    """
    Fast AMD chemistry update for large spatial grids.

    Parameters
    ----------
    fe2, fe3, so4, h, fe_oh3, bedload_storage : double[:, ::1]
        Species amounts in moles.  Modified in place.
    ore, volume, : float[:, ::1]
    do_val              : dissolved O₂
    buffer_capacity     : acid-neutralising capacity (mol/L); 0 = none
    time_step_seconds   : model time step (s)
    valid_rows/cols     : active cell indices (contiguous int arrays)
    num_valid           : number of active cells
    n_substeps          : Backward-Euler sub-steps for pyrite ODEs (default 2).
                          Increase to 4–8 only if you observe mass-balance drift
                          at very long time steps or extreme concentrations.

    Compile flags required for full performance
    -------------------------------------------
    -O3 -fopenmp -ffast-math

    Thread count is controlled by the OMP_NUM_THREADS environment variable
    or by calling openmp.omp_set_num_threads() from Cython.
    """
    cdef Py_ssize_t k, r, c
    cdef double fe2_, fe3_, so4_, h_, foh_, bed_
    cdef double vol_, vol_safe, h2o, do_sqrt, h_cap
    cdef double ore_, 
    cdef double h_produced, 
    cdef double h_initial, 
    cdef double max_neutral, 
    cdef double neutralised
    cdef double h_new

    do_sqrt = do_val ** 0.5

    for k in prange(num_valid, nogil=True, schedule="dynamic"):
        r = valid_rows[k]
        c = valid_cols[k]

        vol_  = <double>volume[r, c]
        ore_  = <double>ore[r, c]

        if vol_ <= 0.0 or not isfinite(vol_) or not isfinite(ore_):
            continue

        fe2_ = fe2[r, c]
        fe3_ = fe3[r, c]
        so4_ = so4[r, c]
        h_   = h[r, c]
        foh_ = fe_oh3[r, c]
        bed_ = bedload_storage[r, c]
        h_initial = h_

        if (not isfinite(fe2_) or not isfinite(fe3_) or
                not isfinite(so4_) or not isfinite(h_) or
                not isfinite(foh_)):
            continue

        vol_safe = fmax(vol_, 1e-30)
        h2o      = (0.99704702 * vol_ * 1000.0) / 18.01528

        # ── A: pyrite oxidation  (Backward-Euler Newton, mildly stiff) ───────
        _backward_euler_pyrite(
            &fe2_, &fe3_, &so4_, &h_,
            ore_, vol_safe, h2o, do_sqrt,
            time_step_seconds, n_substeps)

        # ── B: Fe²⁺ → Fe³⁺ oxidation  (exact analytical exponential) ────────
        _analytical_fe2_oxidation(
            &fe2_, &fe3_, &h_,
            vol_safe, time_step_seconds)

        # ── C: Fe(OH)₃ equilibrium  (analytical closed-form) ─────────────────
        _apply_equilibrium_precip(&fe3_, &h_, &fe2_, &so4_, &foh_, &bed_, vol_safe)
        
        h_produced = h_ - h_initial

        if h_produced > 0.0 and buffer_capacity > 0.0:
            max_neutral = buffer_capacity * vol_safe
            neutralised = fmin(h_produced, max_neutral)
            h_new = h_ - neutralised
        else:
            h_new = h_

        # ── Clip & write back ─────────────────────────────────────────────────
        h_cap        = H_CAP * vol_
        fe2[r, c]             = fmax(fe2_, 0.0)
        fe3[r, c]             = fmax(fe3_, 0.0)
        so4[r, c]             = fmax(so4_, 0.0)
        h[r, c]               = fmin(fmax(h_new, 0.0), h_cap)
        fe_oh3[r, c]          = fmax(foh_, 0.0)
        bedload_storage[r, c] = fmax(bed_, 0.0)
