# cython: boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True
# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.math cimport exp, fmax, fmin, ceil, sqrt
from libc.stdlib cimport malloc, free

ctypedef np.float64_t F64
ctypedef np.float32_t F32
ctypedef np.int32_t   I32
ctypedef np.int64_t   I64

cdef extern from "math.h":
    double INFINITY
def _transport_cn(
    F32[:, ::1] conc,
    F32[:, ::1] conc_s,
    F32[:, ::1] Q,
    F32[:, ::1] Q_lat,
    F32[:, ::1] C_lat,
    list reaches,
    I32[:] id_to_row,
    I32[:] id_to_col,
    double dx,
    double v,
    double a, 
    double b,
    double c,
    double f,
    double dt,
    double psi,
    double theta,
    double alpha_s,
    double A_s_ratio
    ):

    cdef:
        I64 reach_len, i, j
        I64 r0, c0,
        double Q_i, A_i, V_i, W_i, H_i, D_i
        double hr, Re, F_fric, tau, u_star
        double r, s, p, beta, gamma, kappa
        double C_im1, C_i, C_ip1, C_L_i, q_lin_i
        double ai, bi, ci, di, denom
        double bc_top
        double eps = 1e-30
        double rho = 997.1
        double mu = 0.894e-3
        int MAX_REACH = 50000


    cdef F64[::1] a_arr 
    cdef F64[::1] b_arr
    cdef F64[::1] c_arr 
    cdef F64[::1] d_arr 
    cdef F64[::1] c_prime 
    cdef F64[::1] d_prime 
    cdef F64[::1] x_arr 
    cdef I64[::1] rows 
    cdef I64[::1] cols 
    cdef F64[::1] V_i_arr

    # loop over reaches
    for reach_ids_py in reaches:

        reach_ids = np.asarray(reach_ids_py, dtype = np.int64)
        reach_len = len(reach_ids)
        if reach_len < 2:
            continue

        a_arr = np.empty(reach_len, np.float64)
        b_arr = np.empty(reach_len, np.float64)
        c_arr = np.empty(reach_len, np.float64)
        d_arr = np.empty(reach_len, np.float64)
        c_prime = np.empty(reach_len, np.float64)
        d_prime = np.empty(reach_len, np.float64)
        x_arr = np.empty(reach_len, np.float64)
        rows = np.empty(reach_len, np.int64)
        cols = np.empty(reach_len, np.int64)
        V_i_arr = np.empty(reach_len, np.float64)

        # cache row/col for reach
        for i in range(reach_len):
            rows[i] = id_to_row[reach_ids[i]]
            cols[i] = id_to_col[reach_ids[i]]

        
        # concentration from upstream bound cell
        r0 = rows[0]
        c0 = cols[0]
        Q_i = fmax(Q[r0, c0], eps)
        V_i = fmax(Q_i / v, 1e-6) * dx
        bc_top = conc[r0, c0] / V_i 

        # tri-diagonal coeff
        for i in range(reach_len):
            r0 = rows[i]
            c0 = cols[i]
            Q_i = fmax(Q[r0, c0], eps)
            A_i = fmax(Q_i / v, 1e-6)
            V_i = A_i * dx
            V_i_arr[i] = V_i
        
            # dispersion
            W_i = a * (Q_i ** b)
            H_i = c * (Q_i ** f)
            hr = (H_i * W_i) / (2.0 * H_i + W_i + eps)
            Re = (rho * v * 4.0 * hr) / mu
            f_fric = 64.0 / (Re + eps)
            tau = (f_fric / 8.0) * rho * v * v
            u_star = sqrt(fmax(tau / rho, eps))
            D_i = 5.4 * ((W_i / (H_i + eps)) ** 0.7) * \
                ((v / (u_star + eps)) ** 0.13) * H_i * v

            # dimensionless numbers
            r = D_i * dt / (dx * dx)
            s = v * dt / dx    
            q_lin_i = fmax(Q_lat[r0, c0], 0.0)
            p = q_lin_i * dt / (A_i * dx + eps)

            # current conc
            C_i = conc[r0, c0] / V_i
            C_L_i = C_lat[r0, c0]

            # upstream conc C_(i - 1)
            if i == 0:
                C_im1 = bc_top
            else:
                C_im1 = conc[rows[i - 1], cols[i - 1]] / \
                    fmax(fmax(Q[rows[i - 1], cols[i - 1]], eps) / v * dx, eps)
                
            # downstream conc C_(i + 1)
            if i < reach_len - 1:
                C_ip1 = conc[rows[i + 1], cols[i + 1]] / \
                    fmax(fmax(Q[rows[i + 1], cols[i + 1]], eps) / v * dx, eps)

            else:
                C_ip1 = C_i
            
            # tri-diagonal coeff
            ai = -(theta * r - psi * s)
            bi = 1.0 + theta * (2.0 * r + p) + psi * s
            ci = -theta * r

            # at last cell absorb c into b
            if i == reach_len - 1:
                bi -= ci
                ci = 0.0

            di = ( ((1.0 - theta) * r + (1.0-psi) * s ) * C_im1
                + (1.0 - (1.0 - theta) * (2.0 * r + p)
                - (1.0 - psi) * s) * C_i
                + (1.0 - theta) * r * C_ip1
                + p * C_L_i )

            # storage exchange
            if alpha_s > 0.0:
                beta = alpha_s / A_s_ratio
                kappa = beta * dt / 2.0
                gamma = alpha_s * dt / (1.0 + kappa)

                V_s_i = A_s_ratio * V_i
                C_s_i = conc_s[rows[i], cols[i]] / fmax(V_s_i, eps)

                bi += gamma / 2.0
                di += gamma * C_s_i - (gamma / 2.0) * C_i

            a_arr[i] = ai
            b_arr[i] = bi
            c_arr[i] = ci
            d_arr[i] = di

        # thomas algorithm
        # forward sweep
        c_prime[0] = c_arr[0] / b_arr[0]
        d_prime[0] = d_arr[0] / b_arr[0]

        for i in range(1, reach_len):
            denom = b_arr[i] - a_arr[i] * c_prime[i - 1]
            if abs(denom) < eps:
                denom = eps
            c_prime[i] = c_arr[i] / denom
            d_prime[i] = (d_arr[i] - a_arr[i] * d_prime[i - 1]) / denom

        # back substitution
        x_arr[reach_len - 1] = d_prime[reach_len - 1]
        for i in range(reach_len - 2, -1, -1):
            x_arr[i] = d_prime[i] - c_prime[i] * x_arr[i + 1]
        
        # write back to buffer
        for i in range(reach_len):
            r0 = rows[i]
            c0 = cols[i]

            C_old = conc[r0, c0] / fmax(V_i_arr[i], eps)
            C_new = x_arr[i]

            V_s_i = A_s_ratio * V_i_arr[i]
            C_s_old = conc_s[r0, c0] / fmax(V_s_i, eps)

            beta = alpha_s / A_s_ratio
            kappa = beta * dt / 2.0
            C_s_new = (C_s_old * (1.0 - kappa) + kappa *(C_old + C_new)) / (1.0 + kappa)
            
            Q_i = fmax(Q[r0, c0], eps)
            V_i = fmax(Q_i / v, 1e-6) * dx
            conc[r0, c0] = <float>(fmax(C_new, 0.0)) * V_i_arr[i]
            conc_s[r0, c0] = <float>(fmax(C_s_new, 0.0)) * V_s_i

def _transport_ad_dep(
    F32[:, :, ::1] feoh3_buf,
    F32[:, :, ::1] bedload_storage, # precipitated feoh3 on riverbed
    F32[:, ::1] Q,               # flow at current time step [nlat, nlon]
    I64[:, ::1] ID_grid,           # [nlat, nlon]
    I64[:, ::1] outID_grid,        # [nlat, nlon]
    I32[:] id_to_row,              # mapping from ID to row index (-1 if invalid)
    I32[:] id_to_col,              # mapping from ID to col index
    I32[:] id_to_outid,           # mapping from ID to outID
    long time_idx,                   
    double time_step_seconds,
    double v,
    double dx,
    double a,
    double b,
    double wf, # m/s
    int max_substeps,
    long nlat,
    long nlon,
    I64[:] src_rows,              
    I64[:] src_cols,              
    ):
    """
    Cython kernel for advection-deposition transport of precipitate species,
    along the flow network.
    Buffers are modified in-place.
    """
    cdef:
        I64 n_cells = src_rows.shape[0]
        I64 i, sub, var_idx
        I64 src_r, src_c, dst_r, dst_c
        double Q_val, A_cross, V_cell, C_courant, dt_sub,
        double src_val, moved, src_vol
        double mol_mass_feoh3, kg_per_mol, mass_precip, W, A_bed, ap, aq
        double d_Sos, Qs_mass, DEP_mass, Qs_mol, DEP_mol, u_star, p, tau_zero
        double friction_f, viscosity, hydraulic_dia, hydraulic_rad, H, Re, D
        I64 n_sub, sub_step
        I64[:] dst_rows = np.empty(n_cells, dtype=np.int64)
        I64[:] dst_cols = np.empty(n_cells, dtype=np.int64)
        int[:] valid_cell = np.ones(n_cells, dtype=np.int32)
        I64 valid_count
        I64 current_id, next_id
        double epsilon = 1e-12
        int[:] vol_valid
        int[:] has_next
        I64 max_cells = n_cells

    # determine destination cells for each source
    for i in range(n_cells):
        src_r = src_rows[i]
        src_c = src_cols[i]
        dst_id = outID_grid[src_r, src_c]
        if dst_id >= 0 and dst_id < id_to_row.shape[0]:
            dst_rows[i] = id_to_row[dst_id]
            dst_cols[i] = id_to_col[dst_id]
            if dst_rows[i] < 0 or dst_cols[i] < 0:
                valid_cell[i] = False
        else:
            valid_cell[i] = False

    # filter to valid source‑destination pairs
    valid_count = 0
    for i in range(n_cells):
        if valid_cell[i]:
            src_rows[valid_count] = src_rows[i]
            src_cols[valid_count] = src_cols[i]
            dst_rows[valid_count] = dst_rows[i]
            dst_cols[valid_count] = dst_cols[i]
            valid_count += 1
    if valid_count == 0:
        return

    n_cells = valid_count

    # compute number of substeps (Courant‑like condition)

    cdef double max_C = 0.0
    for i in range(n_cells):
        src_r = src_rows[i]
        src_c = src_cols[i]
        Q_val = Q[src_r, src_c]
        if Q_val > 0:
            A_cross = Q_val / v
            if A_cross < 1e-6:
                A_cross = 1e-6
            V_cell = A_cross * dx
            C_courant = Q_val * time_step_seconds / V_cell
            if C_courant > max_C:
                max_C = C_courant

    n_sub = <I64>ceil(max_C)
    if n_sub < 1:
        n_sub = 1
    elif n_sub > max_substeps:
        n_sub = max_substeps
    dt_sub = time_step_seconds / n_sub

    cdef I64[:] current_src_rows = src_rows
    cdef I64[:] current_src_cols = src_cols
    cdef I64[:] current_dst_rows = dst_rows
    cdef I64[:] current_dst_cols = dst_cols
    cdef I64 current_n = n_cells

    vol_valid = np.empty(max_cells, dtype=np.int32)
    has_next = np.empty(max_cells, dtype=np.int32)

    for sub_step in range(n_sub):
        # merge incoming buffer into resident before each substep except first
        # merge buffers at start of substep (chemicals from previous substep become available)
        if sub_step > 0:
            for i in range(current_n):
                src_r = current_src_rows[i]
                src_c = current_src_cols[i]
                feoh3_buf[0, src_r, src_c] += feoh3_buf[1, src_r, src_c]
                feoh3_buf[1, src_r, src_c] = 0.0

        # compute transport fractions and volume validity for this substep
        for i in range(current_n):
            src_r = current_src_rows[i]
            src_c = current_src_cols[i]
            Q_val = Q[src_r, src_c]
            if Q_val <= 0:
                vol_valid[i] = False
                continue
            
            A_cross = Q_val / v # m2
            if A_cross < 1e-6:
                A_cross = 1e-6

            V_cell = A_cross * dx # m3
            W = a * Q_val**b

            # advection-deposition calcs
            mol_mass_feoh3 = 106.87 
            kg_per_mol = mol_mass_feoh3 / 1000
            mass_precip = feoh3_buf[0, src_r, src_c] * kg_per_mol

            A_bed = W * dx
            aq = Q_val / V_cell
            ap = wf * A_bed / V_cell 
            d_Sos = (1.0 - exp(-(aq + ap) * dt_sub)) * mass_precip

            Qs_mass = (aq / (aq + ap)) * d_Sos # note that the paper states (aq / (aq + aq)) which is strange and resolves to 0.5 always
            DEP_mass = (ap / (aq + ap)) * d_Sos

            Qs_mol = Qs_mass / kg_per_mol
            DEP_mol = DEP_mass / kg_per_mol


            # Fe(OH)3 (iron (III) hydroxide)
            feoh3_buf[0, src_r, src_c] = fmax(feoh3_buf[0, src_r, src_c] - Qs_mol - DEP_mol, 
            0.0)
            feoh3_buf[1, current_dst_rows[i], current_dst_cols[i]] += Qs_mol
            bedload_storage[0, src_r, src_c] += DEP_mol

            src_vol = Q_val * time_step_seconds * 1000.0
            if src_vol <= 0:
                vol_valid[i] = False
            else:
                vol_valid[i] = True

        # filter out invalid cells (zero volume, zero flow, etc.)
        valid_count = 0
        for i in range(current_n):
            if vol_valid[i]:
                current_src_rows[valid_count] = current_src_rows[i]
                current_src_cols[valid_count] = current_src_cols[i]
                current_dst_rows[valid_count] = current_dst_rows[i]
                current_dst_cols[valid_count] = current_dst_cols[i]
                valid_count += 1
        if valid_count == 0:
            break
        current_n = valid_count

        # cascade to next downstream cells
        for i in range(current_n):
            has_next[i] = False
        for i in range(current_n):
            dst_r = current_dst_rows[i]
            dst_c = current_dst_cols[i]
            current_id = ID_grid[dst_r, dst_c]
            if current_id >= 0 and current_id < id_to_outid.shape[0]:
                next_id = id_to_outid[current_id]
                if next_id >= 0 and next_id < id_to_row.shape[0]:
                    next_r = id_to_row[next_id]
                    next_c = id_to_col[next_id]
                    if next_r >= 0 and next_c >= 0:
                        # overwrite the source arrays in‑place for next iteration
                        current_src_rows[i] = dst_r
                        current_src_cols[i] = dst_c
                        current_dst_rows[i] = next_r
                        current_dst_cols[i] = next_c
                        has_next[i] = True

        # compress to only cells that have a valid downstream neighbour
        valid_count = 0
        for i in range(current_n):
            if has_next[i]:
                current_src_rows[valid_count] = current_src_rows[i]
                current_src_cols[valid_count] = current_src_cols[i]
                current_dst_rows[valid_count] = current_dst_rows[i]
                current_dst_cols[valid_count] = current_dst_cols[i]
                valid_count += 1
        if valid_count == 0:
            break
        current_n = valid_count

    # final merge of any remaining incoming material into resident buffer
    for i in range(current_n):
        src_r = current_src_rows[i]
        src_c = current_src_cols[i]
        feoh3_buf[0, src_r, src_c] += feoh3_buf[1, src_r, src_c]
        feoh3_buf[1, src_r, src_c] = 0.0 