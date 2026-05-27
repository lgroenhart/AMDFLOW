# cython: boundscheck=True, wraparound=True, initializedcheck=False, cdivision=True
# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.math cimport exp, fmax, fmin, ceil, sqrt, fabs
from libc.stdlib cimport malloc, free

ctypedef np.float64_t F64
ctypedef np.float32_t F32
ctypedef np.int32_t   I32
ctypedef np.int64_t   I64

cdef extern from "math.h":
    double INFINITY
def _transport_cn(
    double[:, ::1] conc,
    double[:, ::1] conc_s,
    F32[:, ::1] Q,
    F32[:, ::1] Q_lat,
    double[:, ::1] C_lat,
    F32[:, ::1] median_vol,
    list reaches,
    I32[:] id_to_row,
    I32[:] id_to_col,
    double dx,
    double a, 
    double b,
    double c,
    double f,
    double dt,
    double psi,
    double theta,
    double alpha_s,
    double A_s_ratio,
    double v,
    int max_substeps,
    F64[::1] a_arr,        
    F64[::1] b_arr,
    F64[::1] c_arr,
    F64[::1] d_arr,
    F64[::1] c_prime,
    F64[::1] d_prime,
    F64[::1] x_arr,
    I64[::1] rows,
    I64[::1] cols,
    F64[::1] V_i_arr,
    F32[::1] v_arr,
    F32[::1] A_arr,
    F32[::1] D_arr,
    long max_reach_length   
    ):
    cdef:
        I64 reach_len, i
        I64 r0, c0
        double Q_i, A_i, V_i, W_i, H_i, D_i, S_i
        double RH, Re, f_fric, tau, u_star, V_head
        double r, s, p, beta, gamma, kappa, v_i
        double C_im1, C_i, C_ip1, C_L_i, q_lin_i
        double ai, bi, ci, di, denom
        double bc_top
        double eps = 1e-30
        double rho = 997.1
        double mu = 0.894e-3
        double dt_sub , max_C, n_sub_souble, C_cour
        I64 n_sub
        
    for reach_ids_py in reaches:
        reach_ids = np.asarray(reach_ids_py, dtype=np.int64)
        reach_len = len(reach_ids)

        if reach_len > max_reach_length:
            raise ValueError("Reach length exceeds pre-allocated workspace size")

        # cache row/col for reach
        for i in range(reach_len):
            rows[i] = id_to_row[reach_ids[i]]
            cols[i] = id_to_col[reach_ids[i]]
        
        # dynamic substeps based on courant number
        # pre-compute most parameters
        max_C = 0.0
        for i in range(reach_len):
            r0 = rows[i]
            c0 = cols[i]
            Q_i = fmax(Q[r0, c0], eps)
            A_i = fmax(Q_i / v, 1e-6)
            A_arr[i] = A_i
            W_i = a * (Q_i ** b)
            H_i = c * (Q_i ** f)
            RH = (H_i * W_i) / (2.0 * H_i + W_i + eps)
            V_i = A_i * dx
            V_i_arr[i] = V_i
            C_cour = v * dt / dx
            if C_cour > max_C:
                max_C = C_cour
            Re = (rho * v * 4.0 * RH) / mu
            f_fric = 64.0 / (Re + eps)
            tau = (f_fric / 8.0) * rho * v * v
            u_star = sqrt(fmax(tau / rho, eps))
            D_i = 5.4 * ((W_i / (H_i + eps)) ** 0.7) * \
                ((v / (u_star + eps)) ** 0.13) * H_i * v
            D_arr[i] = D_i

        n_sub = <I64>ceil(max_C)
        if n_sub < 1:
            n_sub = 1
        elif n_sub > max_substeps:
            n_sub = max_substeps
        dt_sub = dt / n_sub

        # concentration from upstream boundary cell
        r0 = rows[0]
        c0 = cols[0]
        V_head = V_i_arr[0]
        bc_top = conc[r0, c0] / V_head

        # substep loops
        for sub in range(n_sub):
            bc_top = conc[rows[0], cols[0]] / V_i_arr[0]
            # build tridiagonal coefficients 
            for i in range(reach_len):
                r0 = rows[i]
                c0 = cols[i]
                
                A_i = A_arr[i]
                V_i = V_i_arr[i]
                D_i = D_arr[i]

                
                r = D_i * dt_sub / (dx * dx)
                s_adv = v * dt_sub / dx          
                q_lin_i = fmax(Q_lat[r0, c0], 0.0)
                p = q_lin_i * dt_sub / (V_i + eps)

                C_i = conc[r0, c0] / V_i
                C_L_i = C_lat[r0, c0]

                # upstream concentration
                if i == 0:
                    C_im1 = bc_top
                else:
                    C_im1 = conc[rows[i - 1], cols[i - 1]] / \
                        fmax(fmax(Q[rows[i - 1], cols[i - 1]], eps) / v * dx, eps)

                # downstream concentration
                if i < reach_len - 1:
                    C_ip1 = conc[rows[i + 1], cols[i + 1]] / \
                        fmax(fmax(Q[rows[i + 1], cols[i + 1]], eps) / v * dx, eps)
                else:
                    C_ip1 = C_i

                # coefficients
                ai = -(theta * r - psi * s_adv)
                bi = 1.0 + theta * (2.0 * r + p) + psi * s_adv
                ci = -theta * r

                if i == reach_len - 1:
                    bi += ci
                    ci = 0.0

                di = ( ((1.0 - theta) * r + (1.0 - psi) * s_adv) * C_im1
                    + (1.0 - (1.0 - theta) * (2.0 * r + p)
                    - (1.0 - psi) * s_adv) * C_i
                    + (1.0 - theta) * r * C_ip1
                    + p * C_L_i )
                
                # boundary adjustment
                if i == 0:
                    # inbound
                    bi += ci
                    ai = 0.0
                elif i == reach_len - 1:
                    # outbound
                    bi += ci
                    ci = 0.0
                
                if alpha_s > 0.0:
                    beta = alpha_s / A_s_ratio
                    kappa = beta * dt_sub / 2.0
                    gamma = alpha_s * dt_sub / (1.0 + kappa)

                    V_s_i = A_s_ratio * V_i
                    C_s_i = conc_s[r0, c0] / fmax(V_s_i, eps)

                    bi += gamma / 2.0
                    di += gamma * C_s_i - (gamma / 2.0) * C_i


                a_arr[i] = ai
                b_arr[i] = bi
                c_arr[i] = ci
                d_arr[i] = di

            # Thomas algorithm 
            c_prime[0] = c_arr[0] / b_arr[0]
            d_prime[0] = d_arr[0] / b_arr[0]

            for i in range(1, reach_len):
                denom = b_arr[i] - a_arr[i] * c_prime[i - 1]
                if fabs(denom) < eps:
                    denom = eps
                c_prime[i] = c_arr[i] / denom
                d_prime[i] = (d_arr[i] - a_arr[i] * d_prime[i - 1]) / denom

            x_arr[reach_len - 1] = d_prime[reach_len - 1]
            for i in range(reach_len - 2, -1, -1):
                x_arr[i] = d_prime[i] - c_prime[i] * x_arr[i + 1]

            # write back to buffers 
            for i in range(reach_len):
                r0 = rows[i]
                c0 = cols[i]

                C_old = conc[r0, c0] / fmax(V_i_arr[i], eps)
                C_new = x_arr[i]

                V_s_i = A_s_ratio * V_i_arr[i]
                C_s_old = conc_s[r0, c0] / fmax(V_s_i, eps)

                beta = alpha_s / A_s_ratio
                kappa = beta * dt_sub / 2.0
                C_s_new = (C_s_old * (1.0 - kappa) + kappa * (C_old + C_new)) / (1.0 + kappa)

                conc[r0, c0] = (fmax(C_new, 0.0)) * V_i_arr[i]
                conc_s[r0, c0] = (fmax(C_s_new, 0.0)) * V_s_i


def _transport_ad_dep(
    double[:, :, ::1] feoh3_buf,
    double[:, :, ::1] bedload_storage,
    F32[:, ::1] Q,
    I64[:, ::1] ID_grid,
    I64[:, ::1] outID_grid,
    I32[:] id_to_row,
    I32[:] id_to_col,
    I32[:] id_to_outid,
    long time_idx,
    double time_step_seconds,
    double dx,
    double a,
    double b,
    double c,
    double f,
    double wf,
    double v,
    int max_substeps,
    long nlat,
    long nlon,
    I64[:] src_rows,
    I64[:] src_cols,
    I64[:] dst_rows,
    I64[:] dst_cols,
    int[:] valid_cell,
    int[:] vol_valid,
    int[:] has_next
    ):
    """
    Corrected Cython kernel for advection-deposition transport of precipitate species.
    Operates via fast, isolated heap arrays to preserve grid networks across substeps.
    """
    cdef:
        I64 n_cells = src_rows.shape[0]
        I64 i, sub_step
        I64 src_r, src_c, dst_r, dst_c, dst_id
        double Q_val, A_cross, V_cell, C_courant, dt_sub
        double mass_precip, W, A_bed, ap, aq
        double d_Sos, Qs_mass, DEP_mass, Qs_mol, DEP_mol
        double mol_mass_feoh3 = 106.87
        double kg_per_mol = mol_mass_feoh3 / 1000.0
        double max_C = 0.0
        I64 n_sub, valid_count, current_n
        I64 current_id, next_id, next_r, next_c
        double total_req, mass_scale

    # 1. Map pristine source cells to their respective destinations
    for i in range(n_cells):
        src_r = src_rows[i]
        src_c = src_cols[i]
        dst_id = outID_grid[src_r, src_c]
        if dst_id >= 0 and dst_id < id_to_row.shape[0]:
            dst_rows[i] = id_to_row[dst_id]
            dst_cols[i] = id_to_col[dst_id]
            if dst_rows[i] < 0 or dst_cols[i] < 0:
                valid_cell[i] = 0
            else:
                valid_cell[i] = 1
        else:
            valid_cell[i] = 0

    # 2. Extract valid active network element count
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

    # 3. Compute number of substeps (Courant condition)
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

    # 4. Allocate fast worker heap tracking memory
    cdef I64 *working_src_rows = <I64 *>malloc(n_cells * sizeof(I64))
    cdef I64 *working_src_cols = <I64 *>malloc(n_cells * sizeof(I64))
    cdef I64 *working_dst_rows = <I64 *>malloc(n_cells * sizeof(I64))
    cdef I64 *working_dst_cols = <I64 *>malloc(n_cells * sizeof(I64))

    if not working_src_rows or not working_src_cols or not working_dst_rows or not working_dst_cols:
        if working_src_rows: free(working_src_rows)
        if working_src_cols: free(working_src_cols)
        if working_dst_rows: free(working_dst_rows)
        if working_dst_cols: free(working_dst_cols)
        return

    try:
        for sub_step in range(n_sub):
            current_n = n_cells
            
            # Reset active cascades back to the original filtered state each substep loop
            for i in range(n_cells):
                working_src_rows[i] = src_rows[i]
                working_src_cols[i] = src_cols[i]
                working_dst_rows[i] = dst_rows[i]
                working_dst_cols[i] = dst_cols[i]

            # Merge incoming transport buffers (except step 1)
            if sub_step > 0:
                for i in range(current_n):
                    src_r = working_src_rows[i]
                    src_c = working_src_cols[i]
                    feoh3_buf[0, src_r, src_c] += feoh3_buf[1, src_r, src_c]
                    feoh3_buf[1, src_r, src_c] = 0.0

            # Compute transport equations over active cascade track
            while True:
                for i in range(current_n):
                    src_r = working_src_rows[i]
                    src_c = working_src_cols[i]
                    dst_r = working_dst_rows[i]
                    dst_c = working_dst_cols[i]
                    
                    Q_val = Q[src_r, src_c]
                    if Q_val <= 0 or feoh3_buf[0, src_r, src_c] <= 1e-12:
                        vol_valid[i] = 0
                        continue
                    
                    vol_valid[i] = 1
                    W = a * (Q_val ** b)
                    A_cross = Q_val / v
                    if A_cross < 1e-6: A_cross = 1e-6
                    V_cell = A_cross * dx

                    mass_precip = feoh3_buf[0, src_r, src_c] * kg_per_mol
                    A_bed = W * dx
                    aq = Q_val / V_cell
                    ap = wf * A_bed / V_cell
                    
                    d_Sos = (1.0 - exp(-(aq + ap) * dt_sub)) * mass_precip
                    Qs_mass = (aq / (aq + ap)) * d_Sos
                    DEP_mass = (ap / (aq + ap)) * d_Sos

                    Qs_mol = Qs_mass / kg_per_mol
                    DEP_mol = DEP_mass / kg_per_mol

                    # Mass Balance Safeguard: Prevent negative mass creation
                    total_req = Qs_mol + DEP_mol
                    if total_req > feoh3_buf[0, src_r, src_c]:
                        mass_scale = feoh3_buf[0, src_r, src_c] / total_req
                        Qs_mol *= mass_scale
                        DEP_mol *= mass_scale

                    feoh3_buf[0, src_r, src_c] = fmax(feoh3_buf[0, src_r, src_c] - Qs_mol - DEP_mol, 0.0)
                    feoh3_buf[1, dst_r, dst_c] += Qs_mol
                    bedload_storage[0, src_r, src_c] += DEP_mol

                # Filter out invalid, dry or processed elements inside local variables
                valid_count = 0
                for i in range(current_n):
                    if vol_valid[i]:
                        working_src_rows[valid_count] = working_src_rows[i]
                        working_src_cols[valid_count] = working_src_cols[i]
                        working_dst_rows[valid_count] = working_dst_rows[i]
                        working_dst_cols[valid_count] = working_dst_cols[i]
                        valid_count += 1
                current_n = valid_count
                if current_n == 0:
                    break

                # Advance cascades one step downstream
                for i in range(current_n):
                    has_next[i] = 0
                    dst_r = working_dst_rows[i]
                    dst_c = working_dst_cols[i]
                    current_id = ID_grid[dst_r, dst_c]
                    
                    if current_id >= 0 and current_id < id_to_outid.shape[0]:
                        next_id = id_to_outid[current_id]
                        if next_id >= 0 and next_id < id_to_row.shape[0]:
                            next_r = id_to_row[next_id]
                            next_c = id_to_col[next_id]
                            if next_r >= 0 and next_c >= 0:
                                working_src_rows[i] = dst_r
                                working_src_cols[i] = dst_c
                                working_dst_rows[i] = next_r
                                working_dst_cols[i] = next_c
                                has_next[i] = 1

                # Compress cascade stack to active advancing nodes only
                valid_count = 0
                for i in range(current_n):
                    if has_next[i]:
                        working_src_rows[valid_count] = working_src_rows[i]
                        working_src_cols[valid_count] = working_src_cols[i]
                        working_dst_rows[valid_count] = working_dst_rows[i]
                        working_dst_cols[valid_count] = working_dst_cols[i]
                        valid_count += 1
                current_n = valid_count
                if current_n == 0:
                    break

        # Final terminal merge loop
        for i in range(n_cells):
            src_r = src_rows[i]
            src_c = src_cols[i]
            feoh3_buf[0, src_r, src_c] += feoh3_buf[1, src_r, src_c]
            feoh3_buf[1, src_r, src_c] = 0.0

    finally:
        # 5. Clean up local C allocations to eliminate system memory leaks
        free(working_src_rows)
        free(working_src_cols)
        free(working_dst_rows)
        free(working_dst_cols)