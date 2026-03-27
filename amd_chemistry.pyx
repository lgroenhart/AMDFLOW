# amd_chemistry.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log10, fmax, fmin, isfinite
from libc.math cimport sqrt, pow as cpow

# Declare the dtype once — avoids repetition
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def process_chemistry(
    # Memoryviews — direct pointer to the NumPy buffer, no Python overhead
    double[::1] fe2,
    double[::1] fe3,
    double[::1] so4,
    double[::1] h,
    double[::1] fe_oh3,
    double[::1] ore,
    double[::1] volume,
    double do_val,            # dissolved oxygen (scalar)
    double time_step_seconds  # scalar
):
    """
    Runs all 5 chemistry steps in a single C-speed loop over n cells.
    Modifies arrays in-place — no copies, no return value needed.
    """
    cdef Py_ssize_t n = fe2.shape[0]
    cdef Py_ssize_t i

    # Pre-compute constants outside the loop (hoisted to C scope)
    cdef double k_ferric   = 10.0**-9.74 * 10.0**4 / 60.0
    cdef double k_do       = 10.0**-8.19
    cdef double k1_ox      = 8.0e13 / 60.0
    cdef double k2_ox      = 1.0e-7 / 60.0
    cdef double p02        = 0.21
    cdef double Kw         = 1.0e14
    cdef double do_sqrt    = do_val ** 0.5

    # Per-iteration temporaries — declared once at C scope
    cdef double vol_safe, h2o
    cdef double fe3_conc, h_conc, fe3_safe, h_safe
    cdef double rate, ferric_consumed, max_ferric
    cdef double h_conc_2, h_safe_2, ferrous_amount
    cdef double fe2_conc, h_conc_3, h_safe_3
    cdef double fe2_safe, oh_conc, ox_rate, fe2_oxidised
    cdef double diff, adjustment

    for i in range(n):
        if not isfinite(volume[i]) or not isfinite(ore[i]) or not isfinite(fe2[i]):
            continue
        vol_safe = volume[i] if volume[i] > 0.0 else 1.0
        h2o = (0.99704702 * (volume[i] * 1000.0)) / 18.01528

        # ── Step 1: pyrite oxidation by ferric iron ──────────────────────
        if fe3[i] > 0.0 and ore[i] > 0.0:
            fe3_conc  = fe3[i] / vol_safe
            h_conc    = h[i]   / vol_safe
            fe3_safe  = fe3_conc  if (fe3_conc  > 0.0 and isfinite(fe3_conc))  else 1e-10
            h_safe    = h_conc    if (h_conc    > 0.0 and isfinite(h_conc))    else 1e-7
            rate      = k_ferric * sqrt(fe3_safe) / sqrt(h_safe) 
            ferric_consumed = rate * ore[i] * time_step_seconds
            max_ferric = 1.75 * h2o
            ferric_consumed = fmin(ferric_consumed, fe3[i])
            ferric_consumed = fmin(ferric_consumed, max_ferric)
            fe2[i]  += ferric_consumed * 1.07
            fe3[i]  -= ferric_consumed
            h[i]    += ferric_consumed * 1.14

        # ── Step 2: pyrite oxidation by dissolved O₂ ─────────────────────
        if ore[i] > 0.0:
            h_conc_2  = h[i] / vol_safe
            h_safe_2  = h_conc_2 if (h_conc_2 > 0.0 and isfinite(h_conc_2)) else 1e-7
            rate      = k_do * do_sqrt / (h_safe_2 ** 0.11)
            ferrous_amount = fmin(rate * ore[i] * time_step_seconds, 1.0 * h2o)
            fe2[i]  += ferrous_amount
            so4[i]  += 2.0 * ferrous_amount
            h[i]    += 2.0 * ferrous_amount

        # ── Step 3: Fe²⁺ → Fe³⁺ oxidation (Singer & Stumm) ──────────────
        fe2_conc  = fe2[i] / vol_safe
        h_conc_3  = h[i]   / vol_safe
        fe2_safe  = fe2_conc if (isfinite(fe2_conc) and fe2_conc > 0.0) else 0.0
        h_safe_3  = h_conc_3 if (isfinite(h_conc_3) and h_conc_3 > 0.0) else 1e-7
        h_safe_3  = fmax(h_safe_3, 1e-14)
        oh_conc   = Kw / h_safe_3
        ox_rate   = (k1_ox * p02 + k2_ox * (oh_conc * oh_conc) * p02) * fe2_safe
        fe2_oxidised = fmin(ox_rate * vol_safe * time_step_seconds, fe2[i])
        if not isfinite(fe2_oxidised):
            fe2_oxidised = 0.0
        fe3[i]  += fe2_oxidised
        h[i]    -= fe2_oxidised
        fe2[i]  -= fe2_oxidised

        # ── Step 4: Fe³⁺ ↔ Fe(OH)₃ equilibrium ──────────────────────────
        diff        = fe3[i] - fe_oh3[i]
        adjustment  = 0.5 * diff
        fe3[i]     -= adjustment
        fe_oh3[i]  += adjustment
        h[i]       += adjustment * 3.0

        # ── Step 5: clip negatives ────────────────────────────────────────
        fe2[i]   = fmax(fe2[i],   0.0)
        fe3[i]   = fmax(fe3[i],   0.0)
        h[i]     = fmax(h[i],     0.0)
        so4[i]   = fmax(so4[i],   0.0)
        fe_oh3[i]= fmax(fe_oh3[i],0.0)