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
    double[:, ::1] fe2,
    double[:, ::1] fe3,
    double[:, ::1] so4,
    double[:, ::1] h,
    double[:, ::1] fe_oh3,
    double[:, ::1] bedload_storage,
    float[:, ::1] ore,
    float[:, ::1] volume,
    float[:, ::1] median_vol,
    double do_val,
    double ssa,
    double buffer_capacity,            
    double time_step_seconds,  
    Py_ssize_t[::1] valid_rows,
    Py_ssize_t[::1] valid_cols,
    Py_ssize_t num_valid
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
    cdef double h_val_conc_cap = 1e4

    # per-iteration temporaries
    cdef double fe2_val, fe3_val, so4_val, h_val, fe_oh3_val, bedload_storage_val, ore_val, vol_val, median_vol_val
    cdef double vol_safe, h2o
    cdef double fe3_conc, h_conc, fe3_safe, h_safe
    cdef double rate, ferric_consumed, max_ferric
    cdef double h_conc_2, h_safe_2, ferrous_amount
    cdef double fe2_conc, h_conc_3, h_safe_3, h_safe_4, 
    cdef double fe2_safe, oh_conc, ox_rate, fe2_oxidised
    cdef double ph, so4_conc, I, gamma_h, gamma_fe3, act_fe3, act_h, eq_act, dissolve
    cdef double precip, dissolved_needed, dissolved_sus, dissolve_bed, remaining
    cdef double h_val_cap, pyrite_consumed1, ore_loss1, ore_loss2

    for k in prange(num_valid, nogil = True, schedule = "static"):
        r = valid_rows[k]
        c = valid_cols[k]
        fe2_val = <double>fe2[r, c]
        fe3_val = <double>fe3[r, c]
        so4_val = <double>so4[r, c]
        h_val = <double>h[r, c]
        fe_oh3_val = <double>fe_oh3[r, c]
        bedload_storage_val = <double>bedload_storage[r, c]
        ore_val = <double>ore[r, c]
        vol_val = <double>volume[r, c]
        median_vol_val = <double>median_vol[r, c]
        
        if not isfinite(vol_val) or not isfinite(ore_val) or not isfinite(fe2_val):
            continue
        
        if vol_val <= 0.0:
            continue
        
        if (not isfinite(fe2_val) or not isfinite(fe3_val) or
            not isfinite(so4_val) or not isfinite(h_val) or
            not isfinite(fe_oh3_val)):
            continue
        
        vol_safe = fmax(vol_val, median_vol_val)
        h2o = (0.99704702 * (vol_val * 1000.0)) / 18.01528

        # step 1: pyrite oxidation by ferric iron 
        if fe3_val > 0.0 and ore_val > 0.0:
            fe3_conc = fe3_val / vol_safe
            h_conc = h_val   / vol_safe
            fe3_safe = fe3_conc if (fe3_conc > 0.0 and isfinite(fe3_conc)) else 1e-10
            h_safe = h_conc    if (h_conc    > 0.0 and isfinite(h_conc)) else 1e-7
            rate = k_ferric * sqrt(fe3_safe) / sqrt(h_safe) 
            ferric_consumed = rate * ore_val * time_step_seconds
            max_ferric = 1.75 * h2o
            ferric_consumed = fmin(ferric_consumed, fe3_val)
            ferric_consumed = fmin(ferric_consumed, max_ferric)
            fe2_val += ferric_consumed * 1.07
            fe3_val -= ferric_consumed
            h_val += ferric_consumed * 1.14
            so4_val += ferric_consumed * (2.0 / 14.0)
            pyrite_consumed1 = ferric_consumed / 14.0
            ore_loss1 = pyrite_consumed1 * 119.98 * ssa
            ore_val = fmax(ore_val - ore_loss1, 0.0)

        fe2_val = fmax(fe2_val, 0.0)
        fe3_val = fmax(fe3_val, 0.0)
        so4_val = fmax(so4_val, 0.0)
        h_val = fmax(h_val, 0.0)
        h_val = fmin(h_val, h_val_conc_cap * vol_val)
        fe_oh3_val = fmax(fe_oh3_val, 0.0)

        # step 2: pyrite oxidation by dissolved O2
        if ore_val > 0.0:
            h_conc_2 = h_val / vol_safe
            h_safe_2 = h_conc_2 if (h_conc_2 > 0.0 and isfinite(h_conc_2)) else 1e-7
            rate = k_do * do_sqrt / (h_safe_2 ** 0.11)
            ferrous_amount = fmin(rate * ore_val * time_step_seconds, 1.0 * h2o)
            fe2_val += ferrous_amount
            so4_val += 2.0 * ferrous_amount
            h_val += 2.0 * ferrous_amount
            ore_loss2 = ferrous_amount * 119.98 * ssa
            ore_val = fmax(ore_val - ore_loss2, 0.0)

        fe2_val = fmax(fe2_val, 0.0)
        fe3_val = fmax(fe3_val, 0.0)
        so4_val = fmax(so4_val, 0.0)
        h_val = fmax(h_val, 0.0)
        h_val = fmin(h_val, h_val_conc_cap * vol_val)
        fe_oh3_val = fmax(fe_oh3_val, 0.0)

        # step 3: Fe2+ -> Fe3+ oxidation (Singer & Stumm, & PHREEQC)
        fe2_conc = fe2_val / vol_safe
        h_conc_3 = h_val / vol_safe
        fe2_safe = fe2_conc if (isfinite(fe2_conc) and fe2_conc > 0.0) else 0.0
        h_safe_3 = h_conc_3 if (isfinite(h_conc_3) and h_conc_3 > 0.0) else 1e-7
        h_safe_3 = fmax(h_safe_3, 1e-14)
        oh_conc = Kw / h_safe_3

        ox_rate = (k2_ox * p02 + k1_ox * (oh_conc * oh_conc) * p02) * fe2_safe
        
        fe2_oxidised = fmin(ox_rate * vol_safe * time_step_seconds, fe2_val)

        if not isfinite(fe2_oxidised):
            fe2_oxidised = 0.0
        if buffer_capacity > 0.0 and fe2_oxidised > 0.0:
            h_conc_pre = h_val / vol_safe
            ph_before = -log10(fmax(h_conc_pre, 1e-14))
            h_consumed = fe2_oxidised

            max_h_consumed = buffer_capacity * vol_safe * 1.0

            if h_consumed > max_h_consumed:
                scale = max_h_consumed / fe2_oxidised
                fe2_oxidised = fe2_oxidised * scale

        fe3_val = fe3_val + fe2_oxidised
        h_val = h_val - fe2_oxidised
        fe2_val = fe2_val - fe2_oxidised

        fe2_val = fmax(fe2_val, 0.0)
        fe3_val = fmax(fe3_val, 0.0)
        so4_val = fmax(so4_val, 0.0)
        h_val = fmax(h_val, 0.0)
        h_val = fmin(h_val, h_val_conc_cap * vol_val)
        fe_oh3_val = fmax(fe_oh3_val, 0.0)

        # step 4: Fe3+ <-> Fe(OH)3 hydrolysis and precipitation
        h_safe_4 = fmax(h_val / vol_safe, 1e-14)
        ph = -log10(h_safe_4)
        fe3_conc = fe3_val / vol_safe
        fe2_conc = fe2_val / vol_safe
        so4_conc = so4_val / vol_safe

        # Fe(OH)3 <-> Fe3+ equilibrium based on activities
        I = 0.5 * ((h_safe_4 * 1**2) + (fe3_conc * 3**2) + (fe2_conc * 2**2) + (so4_conc * 2**2))
        I = fmin(I, 1.0) # cap ionic strength
        gamma_h = 10**(-0.5 * 1**2 * (sqrt(I) / (1 + sqrt(I)) -0.3 * I))
        gamma_fe3 = 10**(-0.5 * 3**2 * (sqrt(I) / (1 + sqrt(I)) -0.3 * I))
        act_h = gamma_h * h_safe_4
        act_fe3 = gamma_fe3 * fe3_conc
        eq_act = 10**4.891 * act_h**3

        if act_fe3 > eq_act:
            # precipitation
            precip = (fe3_conc - eq_act / gamma_fe3) * vol_safe
            precip = fmin(fmax(precip, 0.0), fe3_val)
            fe3_val = fe3_val - precip
            fe_oh3_val = fe_oh3_val + precip
            h_val = h_val + precip * 3.0

        else:
            # dissolving from suspended flow first
            dissolve_needed = (eq_act / gamma_fe3 - fe3_conc) * vol_safe
            dissolve_needed = fmax(dissolve_needed, 0.0)

            dissolve_sus = fmin(dissolve_needed, fe_oh3_val)
            fe_oh3_val = fe_oh3_val - dissolve_sus
            fe3_val = fe3_val + dissolve_sus
            h_val = fmax(h_val - dissolve_sus * 3.0, 0.0)

            # dissolve any remainder needed from bedload if possible
            remaining = dissolve_needed - dissolve_sus
            if remaining > 0.0:
                dissolve_bed = fmin(remaining, bedload_storage_val)
                bedload_storage_val = bedload_storage_val - dissolve_bed
                fe3_val = fe3_val + dissolve_bed
                h_val = fmax(h_val - dissolve_bed * 3.0, 0.0)

        # ── Step 5: clip negatives ────────────────────────────────────────
        fe2_val = fmax(fe2_val, 0.0)
        fe3_val = fmax(fe3_val, 0.0)
        h_val = fmax(h_val, 0.0)
        h_val = fmin(h_val, h_val_conc_cap * vol_val)
        so4_val = fmax(so4_val, 0.0)
        fe_oh3_val = fmax(fe_oh3_val, 0.0)
        bedload_storage_val = fmax(bedload_storage_val, 0.0)

        # write back to arrays
        fe2[r, c] = fe2_val
        fe3[r, c] = fe3_val
        h[r, c] = h_val
        so4[r, c] = so4_val
        fe_oh3[r, c] = fe_oh3_val
        bedload_storage[r, c] = bedload_storage_val
        ore[r, c] = <float>ore_val