# amd_chemistry.pyx
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
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
    # Memoryviews — direct pointer to the NumPy buffer
    double[::1] fe2,
    double[::1] fe3,
    double[::1] so4,
    double[::1] h,
    double[::1] fe_oh3,
    double[::1] bedload_storage,
    double[::1] ore,
    double[::1] volume,
    double[::1] median_vol,
    double do_val,            # dissolved oxygen (scalar)
    double time_step_seconds  # scalar
):
    """
    chemistry steps in a single C-speed loop over n cells.
    """
    cdef Py_ssize_t n = fe2.shape[0]
    cdef Py_ssize_t i

    # pre-compute constants
    cdef double k_ferric = 10.0**-9.74 * 10.0**4 / 60.0
    cdef double k_do = 10.0**-8.19
    cdef double k1_ox = 1.33e12
    cdef double k2_ox = 2.91e-9
    cdef double p02 = 0.21
    cdef double Kw = 1.0e-14
    cdef double do_sqrt = do_val ** 0.5

    # per-iteration temporaries
    cdef double vol_safe, h2o
    cdef double fe3_conc, h_conc, fe3_safe, h_safe
    cdef double rate, ferric_consumed, max_ferric
    cdef double h_conc_2, h_safe_2, ferrous_amount
    cdef double fe2_conc, h_conc_3, h_safe_3, h_safe_4, 
    cdef double fe2_safe, oh_conc, ox_rate, fe2_oxidised
    cdef double ph, so4_conc, I, gamma_h, gamma_fe3, act_fe3, act_h, eq_act, dissolve
    cdef double precip, dissolved_needed, dissolved_sus, dissolve_bed, remaining

    for i in prange(n, nogil = True, schedule = "static"):
        if not isfinite(volume[i]) or not isfinite(ore[i]) or not isfinite(fe2[i]):
            continue
        
        if volume[i] <= 0.0:
            continue
        
        if (not isfinite(fe2[i]) or not isfinite(fe3[i]) or
            not isfinite(so4[i]) or not isfinite(h[i]) or
            not isfinite(fe_oh3[i])):
            continue
        
        vol_safe = fmax(volume[i], median_vol[i])
        h2o = (0.99704702 * (volume[i] * 1000.0)) / 18.01528

        # step 1: pyrite oxidation by ferric iron 
        if fe3[i] > 0.0 and ore[i] > 0.0:
            fe3_conc = fe3[i] / vol_safe
            h_conc = h[i]   / vol_safe
            fe3_safe = fe3_conc if (fe3_conc > 0.0 and isfinite(fe3_conc)) else 1e-10
            h_safe = h_conc    if (h_conc    > 0.0 and isfinite(h_conc)) else 1e-7
            rate = k_ferric * sqrt(fe3_safe) / sqrt(h_safe) 
            ferric_consumed = rate * ore[i] * time_step_seconds
            max_ferric = 1.75 * h2o
            ferric_consumed = fmin(ferric_consumed, fe3[i])
            ferric_consumed = fmin(ferric_consumed, max_ferric)
            fe2[i] += ferric_consumed * 1.07
            fe3[i] -= ferric_consumed
            h[i] += ferric_consumed * 1.14
            so4[i] += ferric_consumed * (2.0 / 14.0)

        fe2[i] = fmax(fe2[i], 0.0)
        fe3[i] = fmax(fe3[i], 0.0)
        so4[i] = fmax(so4[i], 0.0)
        h[i] = fmax(h[i], 0.0)
        fe_oh3[i] = fmax(fe_oh3[i], 0.0)

        # step 2: pyrite oxidation by dissolved O2
        if ore[i] > 0.0:
            h_conc_2 = h[i] / vol_safe
            h_safe_2 = h_conc_2 if (h_conc_2 > 0.0 and isfinite(h_conc_2)) else 1e-7
            rate = k_do * do_sqrt / (h_safe_2 ** 0.11)
            ferrous_amount = fmin(rate * ore[i] * time_step_seconds, 1.0 * h2o)
            fe2[i] += ferrous_amount
            so4[i] += 2.0 * ferrous_amount
            h[i] += 2.0 * ferrous_amount

        fe2[i] = fmax(fe2[i], 0.0)
        fe3[i] = fmax(fe3[i], 0.0)
        so4[i] = fmax(so4[i], 0.0)
        h[i] = fmax(h[i], 0.0)
        fe_oh3[i] = fmax(fe_oh3[i], 0.0)

        # step 3: Fe2+ -> Fe3+ oxidation (Singer & Stumm, & PHREEQC)
        fe2_conc = fe2[i] / vol_safe
        h_conc_3 = h[i] / vol_safe
        fe2_safe = fe2_conc if (isfinite(fe2_conc) and fe2_conc > 0.0) else 0.0
        h_safe_3 = h_conc_3 if (isfinite(h_conc_3) and h_conc_3 > 0.0) else 1e-7
        h_safe_3 = fmax(h_safe_3, 1e-14)
        oh_conc = Kw / h_safe_3

        ox_rate = (k2_ox * p02 + k1_ox * (oh_conc * oh_conc) * p02) * fe2_safe
        
        fe2_oxidised = fmin(ox_rate * vol_safe * time_step_seconds, fe2[i])
        if not isfinite(fe2_oxidised):
            fe2_oxidised = 0.0
        fe3[i] += fe2_oxidised
        h[i] -= fe2_oxidised
        fe2[i] -= fe2_oxidised

        fe2[i] = fmax(fe2[i], 0.0)
        fe3[i] = fmax(fe3[i], 0.0)
        so4[i] = fmax(so4[i], 0.0)
        h[i] = fmax(h[i], 0.0)
        fe_oh3[i] = fmax(fe_oh3[i], 0.0)

        # step 4: Fe3+ <-> Fe(OH)3 hydrolysis and precipitation
        h_safe_4 = fmax(h[i] / vol_safe, 1e-14)
        ph = -log10(h_safe_4)
        fe3_conc = fe3[i] / vol_safe
        fe2_conc = fe2[i] / vol_safe
        so4_conc = so4[i] / vol_safe

        # Fe(OH)3 <-> Fe3+ equilibrium based on activities
        I = 0.5 * ((h_safe_4 * 1**2) + (fe3_conc * 3**2) + (fe2_conc * 2**2) + (so4_conc * 2**2))
        gamma_h = 10**(-0.5 * 1**2 * (sqrt(I) / (1 + sqrt(I)) -0.3 * I))
        gamma_fe3 = 10**(-0.5 * 3**2 * (sqrt(I) / (1 + sqrt(I)) -0.3 * I))
        act_h = gamma_h * h_safe_4
        act_fe3 = gamma_fe3 * fe3_conc
        eq_act = 10**4.891 * act_h**3

        if act_fe3 > eq_act:
            # precipitation
            precip = (fe3_conc - eq_act / gamma_fe3) * vol_safe
            precip = fmin(fmax(precip, 0.0), fe3[i])
            fe3[i] -= precip
            fe_oh3[i] += precip
            h[i] += precip * 3.0

        else:
            # dissolving from suspended flow first
            dissolve_needed = (eq_act / gamma_fe3 - fe3_conc) * vol_safe
            dissolve_needed = fmax(dissolve_needed, 0.0)

            dissolve_sus = fmin(dissolve_needed, fe_oh3[i])
            fe_oh3[i] -= dissolve_sus
            fe3[i] += dissolve_sus
            h[i] = fmax(h[i] - dissolve_sus * 3.0, 0.0)

            # dissolve any remainder needed from bedload if possible
            remaining = dissolve_needed - dissolve_sus
            if remaining > 0.0:
                dissolve_bed = fmin(remaining, bedload_storage[i])
                bedload_storage[i] -= dissolve_bed
                fe3[i] += dissolve_bed
                h[i] = fmax(h[i] - dissolve_bed * 3.0, 0.0)

        # ── Step 5: clip negatives ────────────────────────────────────────
        fe2[i] = fmax(fe2[i], 0.0)
        fe3[i] = fmax(fe3[i], 0.0)
        h[i] = fmax(h[i], 0.0)
        so4[i] = fmax(so4[i], 0.0)
        fe_oh3[i] = fmax(fe_oh3[i],0.0)
        bedload_storage[i] = fmax(bedload_storage[i], 0.0)