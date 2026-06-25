# amd_chemistry_fast.pyx
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport (
    log10, sqrt, exp, fmax, fmin, isfinite, pow as cpow, fabs
)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double K_FERRIC = 10.0**-9.74 * 1e4 / 60.0
cdef double K_DO = 10.0**-8.19
cdef double K1_OX = 1.33e12
cdef double K2_OX = 2.91e-9
cdef double P02 = 0.21
cdef double KW = 1.0e-14
cdef double KSP = 7.762471e4 # 10**4.891
cdef double H_CAP = 1.0e4


cdef inline void _davies_activity(
        double h_c, double fe3_c, double fe2_c, double so4_c,
        double *gam_h, double *gam_fe3) noexcept nogil:
    """Davies-equation activity coefficients for solution
    
    Parameters:
    --------------
    h_c : np.ndarray (double)
        molar concentration of hydrons (mol/L)
    fe3_c : np.ndarray (double)
        molar concentration of ferric iron (mol/L)
    fe2_c :np.ndarray (double)
        molar concentration of ferrous iron (mol/L)
    so4_c :np.ndarray (double)
        molar concentration of sulphate (mol/L)
    gam_h : np.ndarray (*double)
        output of function (modified in-place): the activity coefficient of hydrons in the solution
    gam_fe3 : np.ndarray (*double)
        output of function (modified in-place): the activity coefficient of ferric iron in the solution
    """
    cdef double I, sqI, d
    I = 0.5 * (h_c + 9.0*fe3_c + 4.0*fe2_c + 4.0*so4_c)
    if I > 1.0:
        I = 1.0
    sqI = sqrt(I)
    d = sqI / (1.0 + sqI) - 0.3 * I
    gam_h[0] = cpow(10.0, -0.5 * 1.0 * d)
    gam_fe3[0] = cpow(10.0, -0.5 * 9.0 * d)


cdef inline void _apply_equilibrium_precip(
        double *fe3, double *h, double *fe2, double *so4, 
        double *fe_oh3, double *bedload,
        double vol_safe) noexcept nogil:
    """Instant Fe(OH)_3 ↔ Fe**3+ equilibrium.

    Computes the Davies-corrected equilibrium [Fe**3+] and moves the
    surplus/deficit to/from fe_oh3 (and bedload if needed).

    Parameters:
    ---------------------------
    fe3 : np.ndarray (*double)
        ferric iron amount (mol)
    h : np.ndarray (*double)
        hydron amount (mol)
    fe2 : np.ndarray (*double)
        ferrous iron amount (mol)
    so4 : np.ndarray (*double)
        sulphate amount (mol)
    fe_oh3 : np.ndarray (*double)
        suspended ferric oxyhydroxide amount (mol)
    bedload : np.ndarray (*double)
        deposited ferric oxyhydroxide amount (mol)
    vol_safe : np.ndarray (double)
        safe (clipped to 1e-30) volume amount (L)
    """
    cdef double h_c, fe3_c, gam_h, gam_fe3, fe2_c, so4_c
    cdef double act_h, eq_act, fe3_eq_mol, delta
    cdef double p, need, avail_sus, from_sus, remaining, from_bed, max_diss_h, max_diss_h_bed

    h_c = fmax(h[0]   / vol_safe, 1e-14)
    fe3_c = fmax(fe3[0] / vol_safe, 0.0)
    fe2_c = fmax(fe2[0] / vol_safe, 0.0)
    so4_c = fmax(so4[0] / vol_safe, 0.0)

    _davies_activity(h_c, fe3_c, fe2_c, so4_c, &gam_h, &gam_fe3)

    act_h = gam_h * h_c
    eq_act = KSP * act_h * act_h * act_h # KSP * [H+]_act^3
    fe3_eq_mol = (eq_act / gam_fe3) * vol_safe # equilibrium moles

    delta = fe3[0] - fe3_eq_mol # > 0 --> saturated --> precipitate

    if delta > 0.0:
        # precipitation
        p = fmin(delta, fe3[0])
        fe3[0] -= p
        fe_oh3[0]+= p
        h[0] += 3.0 * p
    else:
        # dissolution
        need = -delta
        avail_sus = fmax(fe_oh3[0], 0.0)
        max_diss_h = fmax(h[0] / 3.0, 0.0)
        from_sus = fmin(need, fmin(avail_sus, max_diss_h))
        fe_oh3[0]-= from_sus
        fe3[0] += from_sus
        h[0] = fmax(h[0] - 3.0*from_sus, 0.0)

        remaining = need - from_sus
        if remaining > 0.0:
            max_diss_h_bed = fmax(h[0] / 3.0, 0.0)
            from_bed = fmin(remaining, fmin(fmax(bedload[0], 0.0), max_diss_h_bed))
            bedload[0] -= from_bed
            fe3[0] += from_bed
            h[0] = fmax(h[0] - 3.0*from_bed, 0.0)


cdef inline void _analytical_fe2_oxidation(
        double *fe2, double *fe3, double *h,
        double vol_safe,
        double dt) noexcept nogil:
    """Exponential solution for ferrous iron oxidation.
    
    Parameters:
    ---------------------
    fe2 : np.ndarray (*double)
        ferrous iron amount (mol)
    fe3 : np.ndarray (*double)
        ferric iron amount (mol)
    h : np.ndarray (*double)
        hydron amount (mol)
    vol_safe : np.ndarray (double)
        safe (clipped to 1e-30) volume amount (L)
    dt : float (double)
        timestep (s)
    """
    cdef double h_c, oh_c, lam, fe2_new, fe2_ox, max_h, scale

    h_c = fmax(h[0] / vol_safe, 1e-14)
    oh_c = KW / h_c
    lam = K2_OX * P02 + K1_OX * (oh_c * oh_c) * P02

    fe2_new = fe2[0] * exp(-lam * dt)
    fe2_new = fmax(fe2_new, 0.0)
    fe2_ox = fmin(fe2[0] - fe2_new, h[0])

    fe3[0] += fe2_ox
    h[0] = fmax(h[0] - fe2_ox, 0.0)
    fe2[0] = fe2_new


cdef inline void _pyrite_rhs(
        double fe2, double fe3, double so4, double h,
        double ore, double vol_safe, double h2o, double do_sqrt,
        double *dfe2, double *dfe3, double *dso4, double *dh) noexcept nogil:
    """Pyrite reaction rates (with oxy + with ferric) in mol s**-1

    Parameters:
    ------------------------
    fe2 : np.ndarray (double)
        ferrous iron amount (mol)
    fe3 : np.ndarray (double)
        ferric iron amount (mol)
    so4 : np.ndarray (double)
        sulphate amount (mol)
    h : np.ndarray (double)
        hydron amount (mol)
    ore : np.ndarray (double)
        reactive surface area of pyrite (m**2)
    vol_safe : np.ndarray (double)
        safe (clipped to 1e-30) volume amount (L)
    h2o : np.ndarray (double) 
        available water for reactions (mol) !!!!deprecated!!!!
    do_sqrt : float (double)
        square root of the amount of dissolved oxygen (mol/L**0.5)
        amount of do set in __init__() of AMDModel (classes.py)
    dfe2 : np.ndarray (*double)
        output: rate of fe2 creation (mol/s)
    dfe3 : np.ndarray (*double)
        output: rate fe3 creation (mol/s)
    dso4 : np.ndarray (*double)
        rate of sulphate creation (mol/s)
    dh : np.ndarray (*double)
        rate of hydron creation (mol/s)
    """
    cdef double fe3_c, h_c, rate1, rate2

    dfe2[0] = 0.0; dfe3[0] = 0.0; dso4[0] = 0.0; dh[0] = 0.0

    # FeS_2 + Fe**3+
    if fe3 > 0.0 and ore > 0.0:
        fe3_c = fmax(fe3 / vol_safe, 1e-10)
        h_c = fmax(h   / vol_safe, 1e-7)
        rate1 = K_FERRIC * sqrt(fe3_c) / sqrt(h_c) * ore
        dfe2[0] += rate1 * 15.0
        dfe3[0] -= rate1 * 14.0
        dh[0] += rate1 * 16.0
        dso4[0] += rate1 * 2.0

    # FeS_2 + O_2
    if ore > 0.0:
        h_c = fmax(h / vol_safe, 1e-7)
        rate2 = K_DO * do_sqrt / cpow(h_c, 0.11) * ore
        dfe2[0] += rate2
        dso4[0] += 2.0 * rate2
        dh[0] += 2.0 * rate2


cdef inline void _gauss4(double A[4][4], double b[4], double x[4]) noexcept nogil:
    """In-place Gaussian elimination with partial pivoting for Ax = b, n=4.

    Parameters:
    -----------------------
    A : np.ndarray (double)
        the A matrix, for partial pivoting of [4][4]
    b : np.ndarray (double)
        the b vector for partial pivoting of [4]
    x : np.ndarray (double)
        the x vector for partial pivoting of [4]
    """
    cdef double Ab[4][5]
    cdef double pivot, factor, tmp
    cdef int i, j, k, col, row, max_row

    for i in range(4):
        for j in range(4):
            Ab[i][j] = A[i][j]
        Ab[i][4] = b[i]

    for col in range(4):
        # partial pivot
        max_row = col
        for row in range(col+1, 4):
            if fabs(Ab[row][col]) > fabs(Ab[max_row][col]):
                max_row = row
        if max_row != col:
            for k in range(5):
                tmp = Ab[col][k]
                Ab[col][k] = Ab[max_row][k]
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
    """Backward-Euler with Newton iteration for the pyrite oxidation ODEs.

    Parameters:
    ------------------------
    fe2 : np.ndarray (double)
        ferrous iron amount (mol)
    fe3 : np.ndarray (double)
        ferric iron amount (mol)
    so4 : np.ndarray (double)
        sulphate amount (mol)
    h : np.ndarray (double)
        hydron amount (mol)
    ore : np.ndarray (double)
        reactive surface area of pyrite (m**2)
    vol_safe : np.ndarray (double)
        safe (clipped to 1e-30) volume amount (L)
    h2o : np.ndarray (double)
        available water for reactions (mol)
    do_sqrt : float (double)
        square root of the amount of dissolved oxygen (mol/L)
        amount of do set in __init__() of AMDModel (classes.py)
    dt : float (double)
        timestep time (s)
    n_substeps : int 
        amount of substeps
    """
    cdef double sub_dt = dt / n_substeps
    cdef double fe2_n, fe3_n, so4_n, h_n # value at start of sub-step
    cdef double fe2_k, fe3_k, so4_k, h_k # Newton iterate
    cdef double d2, d3, ds, dh # RHS values
    cdef double r2, r3, rs, rh # residuals
    cdef double norm, eps = 1e-7
    cdef double J[4][4]
    cdef double R[4]
    cdef double delta[4]
    cdef double f2e, f3e, fse, fhe # perturbed RHS
    cdef int s, it
    cdef double f2_k, f3_k, fs_k, fh_k # f(y_k) at current iteration

    for s in range(n_substeps):
        fe2_n = fe2[0]; fe3_n = fe3[0]
        so4_n = so4[0]; h_n = h[0]

        # initial guess: explicit Euler
        _pyrite_rhs(fe2_n, fe3_n, so4_n, h_n, ore, vol_safe, h2o, do_sqrt,
                    &d2, &d3, &ds, &dh)
        fe2_k = fmax(fe2_n + sub_dt * d2, 0.0)
        fe3_k = fmax(fe3_n + sub_dt * d3, 0.0)
        so4_k = fmax(so4_n + sub_dt * ds, 0.0)
        h_k = fmax(h_n   + sub_dt * dh, 0.0)

        # Newton iterations
        for it in range(20):
            _pyrite_rhs(fe2_k, fe3_k, so4_k, h_k, ore, vol_safe, h2o, do_sqrt,
                        &f2_k, &f3_k, &fs_k, &fh_k)
            r2 = fe2_k - fe2_n - sub_dt * f2_k
            r3 = fe3_k - fe3_n - sub_dt * f3_k
            rs = so4_k - so4_n - sub_dt * fs_k
            rh = h_k - h_n   - sub_dt * fh_k

            norm = sqrt(r2*r2 + r3*r3 + rs*rs + rh*rh)
            if norm < 1e-10:
                break

            # 4×4 Jacobian via forward finite difference
            # column 0: perturb fe2
            _pyrite_rhs(fe2_k+eps, fe3_k, so4_k, h_k, ore, vol_safe, h2o, do_sqrt,
                        &f2e, &f3e, &fse, &fhe)
            J[0][0] = 1.0 - sub_dt*(f2e-f2_k)/eps
            J[1][0] = - sub_dt*(f3e-f3_k)/eps
            J[2][0] = - sub_dt*(fse-fs_k)/eps
            J[3][0] = - sub_dt*(fhe-fh_k)/eps

            # column 1: perturb fe3
            _pyrite_rhs(fe2_k, fe3_k+eps, so4_k, h_k, ore, vol_safe, h2o, do_sqrt,
                        &f2e, &f3e, &fse, &fhe)
            J[0][1] = - sub_dt*(f2e-f2_k)/eps
            J[1][1] = 1.0 - sub_dt*(f3e-f3_k)/eps
            J[2][1] = - sub_dt*(fse-fs_k)/eps
            J[3][1] = - sub_dt*(fhe-fh_k)/eps

            # column 2: perturb so4
            _pyrite_rhs(fe2_k, fe3_k, so4_k+eps, h_k, ore, vol_safe, h2o, do_sqrt,
                        &f2e, &f3e, &fse, &fhe)
            J[0][2] = - sub_dt*(f2e-f2_k)/eps
            J[1][2] = - sub_dt*(f3e-f3_k)/eps
            J[2][2] = 1.0 - sub_dt*(fse-fs_k)/eps
            J[3][2] = - sub_dt*(fhe-fh_k)/eps

            # column 3: perturb h
            _pyrite_rhs(fe2_k, fe3_k, so4_k, h_k+eps, ore, vol_safe, h2o, do_sqrt,
                        &f2e, &f3e, &fse, &fhe)
            J[0][3] = - sub_dt*(f2e-f2_k)/eps
            J[1][3] = - sub_dt*(f3e-f3_k)/eps
            J[2][3] = - sub_dt*(fse-fs_k)/eps
            J[3][3] = 1.0 - sub_dt*(fhe-fh_k)/eps

            # solve J·Δ = −R
            R[0] = -r2; R[1] = -r3; R[2] = -rs; R[3] = -rh
            _gauss4(J, R, delta)

            fe2_k = fmax(fe2_k + delta[0], 0.0)
            fe3_k = fmax(fe3_k + delta[1], 0.0)
            so4_k = fmax(so4_k + delta[2], 0.0)
            h_k   = fmax(h_k   + delta[3], 0.0)

        fe2[0] = fe2_k; fe3[0] = fe3_k
        so4[0] = so4_k; h[0]   = h_k

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
    """runs all chemistry steps for the AMDModel class and AMDFLOW model, 
    uses the helper functions from chemistry.pyx: _davies_activity(), _apply_equilibrium_precip(), 
        _analytical_fe2_oxidation(), _pyrite_rhs(), _gauss4(), _backward_euler_pyrite()

    Parameters
    ----------
    fe2, fe3, so4, h, fe_oh3, bedload_storage : np.ndarray (double[:, ::1])
        species amounts in moles
    ore : np.ndarray (float[:, ::1])
        amount of reactive surface area of pyrite ore (m**2)
    volume : np.ndarray (float[:, ::1])
        volume (L)
    do_val : float (double)
        dissolved oxygen (mol/L)
    buffer_capacity : float (double)
        acid-neutralising capacity (mol/L); 0 = none, !!!!deprecated --> always 0!!!!!!!!!!!!!!
    time_step_seconds : float (double)
        model time step (s)
    valid_rows/cols : np.ndarray (Pyssize_t[::1])
        active cell indices (contiguous int arrays)
    num_valid : int (Py_ssize_t)
        number of active cells
    n_substeps : int
        Backward-Euler sub-steps for pyrite ODEs (default 2, 100 default passed from classes.py)
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

        vol_ = <double>volume[r, c]
        ore_ = <double>ore[r, c]

        if vol_ <= 0.0 or not isfinite(vol_) or not isfinite(ore_):
            continue

        fe2_ = fe2[r, c]
        fe3_ = fe3[r, c]
        so4_ = so4[r, c]
        h_ = h[r, c]
        foh_ = fe_oh3[r, c]
        bed_ = bedload_storage[r, c]
        h_initial = h_

        if (not isfinite(fe2_) or not isfinite(fe3_) or
                not isfinite(so4_) or not isfinite(h_) or
                not isfinite(foh_)):
            continue

        vol_safe = fmax(vol_, 1e-30)
        h2o = (0.99704702 * vol_ * 1000.0) / 18.01528

        # pyrite oxidation (Backward-Euler Newton)
        _backward_euler_pyrite(
            &fe2_, &fe3_, &so4_, &h_,
            ore_, vol_safe, h2o, do_sqrt,
            time_step_seconds, n_substeps)

        # Ferrous oxidation (exact analytical exponential)
        _analytical_fe2_oxidation(
            &fe2_, &fe3_, &h_,
            vol_safe, time_step_seconds)

        # Fe(OH)_3 <-> Fe**3+ equilibrium 
        _apply_equilibrium_precip(&fe3_, &h_, &fe2_, &so4_, &foh_, &bed_, vol_safe)
        
        h_produced = h_ - h_initial

        if h_produced > 0.0 and buffer_capacity > 0.0:
            max_neutral = buffer_capacity * vol_safe
            neutralised = fmin(h_produced, max_neutral)
            h_new = h_ - neutralised
        else:
            h_new = h_

        # clip and write back
        h_cap = H_CAP * vol_
        fe2[r, c] = fmax(fe2_, 0.0)
        fe3[r, c] = fmax(fe3_, 0.0)
        so4[r, c] = fmax(so4_, 0.0)
        h[r, c] = fmin(fmax(h_new, 0.0), h_cap)
        fe_oh3[r, c] = fmax(foh_, 0.0)
        bedload_storage[r, c] = fmax(bed_, 0.0)
