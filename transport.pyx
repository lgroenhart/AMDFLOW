# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
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

def _transport(
    # main channel
    double[:, ::1] conc_fe2,
    double[:, ::1] conc_fe3,
    double[:, ::1] conc_h,
    double[:, ::1] conc_so4,
    double[:, ::1] conc_precip,

    # storage
    double[:, ::1] conc_s_fe2,
    double[:, ::1] conc_s_fe3,
    double[:, ::1] conc_s_h,
    double[:, ::1] conc_s_so4,

    # bedload
    double[:, ::1] bedload_storage,

    # hydrology
    F32[:, ::1] Q,
    F32[:, ::1] Q_lat_fe2,
    F32[:, ::1] Q_lat_fe3,
    F32[:, ::1] Q_lat_h,
    F32[:, ::1] Q_lat_so4,
    F32[:, ::1] Q_lat_precip,

    double[:, ::1] C_lat_fe2,
    double[:, ::1] C_lat_fe3,
    double[:, ::1] C_lat_h,
    double[:, ::1] C_lat_so4,
    double[:, ::1] C_lat_precip,

    # reaches
    list reaches,
    I32[:] id_to_row,
    I32[:] id_to_col,

    # Junction topology – NEW
    I32[::1] tail_r, I32[::1] tail_c,
    I32[::1] dst_r,  I32[::1] dst_c,
    Py_ssize_t n_junctions,

    # hydraulics
    double dx,
    F32[:, ::1] S,
    double a, double b, double c, double f,
    double dt,
    double alpha_s,
    double A_s_ratio,
    double mannings,
    double wf,
    int max_substeps,

    # arrays
    F64[::1] a_arr, F64[::1] b_arr, F64[::1] c_arr, F64[::1] d_arr,
    F64[::1] c_prime, F64[::1] d_prime, F64[::1] x_arr,
    I64[::1] rows, I64[::1] cols,
    F64[::1] V_i_arr, F32[::1] v_arr, F64[::1] A_arr, F64[::1] D_arr,
    F64[::1] H_arr,
    long max_reach_length,
):
    cdef:
        I64 reach_id, i, r0, c0, sub, species_idx
        double Q_i, v_i, A_i, V_i, D_i, H_i, S_i, settling_loss
        double r_coef, s_adv, p, q_lin, alpha_m, alpha_st, gamma
        double denom, C_im1, C_i, C_ip1, C_L_i, ai, bi, ci, di
        double C_new, C_old, C_s_old, C_s_new, settled_moles, lateral_source
        Py_ssize_t reach_len
        I64 n_sub
        double dt_sub, max_C, bc_top, settling_term
        double W_i, RH, Re, f_fric, tau, u_star
        bint has_settling, has_storage
        I32[:] reach_ids

        # Local memoryview variables bound per species
        double[:, ::1] conc
        double[:, ::1] conc_s
        F32[:, ::1] Q_lat
        double[:, ::1] C_lat

        # Buffer to store old concentrations for storage/settling calculations
        double[:] C_old_arr = np.zeros(max_reach_length, dtype=np.float64)

        # Junction mass outflow accumulators
        double outflow_sum[5]
        bint last_is_tail
        Py_ssize_t dst_idx
        double v_tail, A_tail, mass_flux_sub

    # Build mapping: tail cell position -> list of destination (dr, dc)
    tail_to_dest = {}
    cdef Py_ssize_t k
    for k in range(n_junctions):
        tail_pos = (tail_r[k], tail_c[k])
        if tail_pos not in tail_to_dest:
            tail_to_dest[tail_pos] = []
        tail_to_dest[tail_pos].append((dst_r[k], dst_c[k]))

    # 1. Loop over each reach
    for reach_ids_py in reaches:
        reach_ids = np.asarray(reach_ids_py, dtype=np.int32)
        reach_len = len(reach_ids)

        if reach_len > max_reach_length:
            raise ValueError(f"Reach length {reach_len} exceeds workspace {max_reach_length}")

        max_C = 0.0

        # 2. Pre‑compute hydraulics for the current reach EXACTLY ONCE
        for i in range(reach_len):
            r0 = id_to_row[reach_ids[i]]
            c0 = id_to_col[reach_ids[i]]
            rows[i] = r0
            cols[i] = c0

            Q_i = fmax(Q[r0, c0], 1e-30)
            S_i = fmax(S[r0, c0], 0.0)

            W_i = a * (Q_i ** b)
            H_i = c * (Q_i ** f)
            RH = (H_i * W_i) / (2.0 * H_i + W_i + 1e-30)
            v_i = mannings**(-1.0) * RH**(2.0/3.0) * sqrt(fmax(S_i, 0.0))

            A_i = fmax(Q_i / v_i, 1e-6)
            V_i = A_i * dx * 1000.0  # Volume in Liters

            Re = (997.1 * v_i * 4.0 * RH) / 0.894e-3
            f_fric = 64.0 / (Re + 1e-30)
            tau = (f_fric / 8.0) * 997.1 * v_i * v_i
            u_star = sqrt(fmax(tau / 997.1, 1e-30))
            D_i = 5.4 * ((W_i / (H_i + 1e-30))**0.7) * ((v_i / (u_star + 1e-30))**0.13) * H_i * v_i

            v_arr[i] = <F32>v_i
            A_arr[i] = A_i
            V_i_arr[i] = V_i
            D_arr[i] = D_i
            H_arr[i] = H_i

            if (v_i * dt / dx) > max_C:
                max_C = v_i * dt / dx

        n_sub = <I64>ceil(max_C)
        if n_sub < 1:
            n_sub = 1
        elif n_sub > max_substeps:
            n_sub = max_substeps
        dt_sub = dt / n_sub

        # Determine if the last cell of this reach is a junction tail
        last_is_tail = (rows[reach_len-1], cols[reach_len-1]) in tail_to_dest

        # 3. Process each chemical species
        for species_idx in range(5):
            if species_idx == 0:
                conc = conc_fe2
                conc_s = conc_s_fe2
                Q_lat = Q_lat_fe2
                C_lat = C_lat_fe2
                has_settling = False
                has_storage = True
            elif species_idx == 1:
                conc = conc_fe3
                conc_s = conc_s_fe3
                Q_lat = Q_lat_fe3
                C_lat = C_lat_fe3
                has_settling = False
                has_storage = True
            elif species_idx == 2:
                conc = conc_h
                conc_s = conc_s_h
                Q_lat = Q_lat_h
                C_lat = C_lat_h
                has_settling = False
                has_storage = True
            elif species_idx == 3:
                conc = conc_so4
                conc_s = conc_s_so4
                Q_lat = Q_lat_so4
                C_lat = C_lat_so4
                has_settling = False
                has_storage = True
            else:  # species_idx == 4 ('precip')
                conc = conc_precip
                conc_s = None
                Q_lat = Q_lat_precip
                C_lat = C_lat_precip
                has_settling = True
                has_storage = False

            # Reset outflow accumulator for this species
            outflow_sum[species_idx] = 0.0

            for sub in range(n_sub):
                bc_top = conc[rows[0], cols[0]] / fmax(V_i_arr[0], 1e-30)

                # Save old concentrations for this substep
                for i in range(reach_len):
                    r0 = rows[i]
                    c0 = cols[i]
                    C_old_arr[i] = conc[r0, c0] / fmax(V_i_arr[i], 1e-30)

                # Build Crank‑Nicolson Matrix
                for i in range(reach_len):
                    r0 = rows[i]
                    c0 = cols[i]

                    V_i = V_i_arr[i]
                    v_i = v_arr[i]
                    D_i = D_arr[i]
                    H_i = H_arr[i]

                    r_coef = D_i * dt_sub / (dx * dx)
                    s_adv = v_i * dt_sub / dx

                    q_lin = fmax(Q_lat[r0, c0], 0.0)
                    p = (q_lin * 1000.0) * dt_sub / (V_i + 1e-30)

                    lateral_source = (C_lat[r0, c0] * dt_sub) / (V_i + 1e-30)

                    C_i = C_old_arr[i]

                    if i == 0:
                        C_im1 = bc_top
                    else:
                        C_im1 = C_old_arr[i-1]

                    if i < reach_len - 1:
                        C_ip1 = C_old_arr[i+1]
                    else:
                        C_ip1 = C_i

                    # Standard CN Coefficients
                    ai = -(0.5 * r_coef + 0.5 * s_adv)
                    bi = 1.0 + 0.5 * (2.0 * r_coef + p) + 0.5 * s_adv
                    ci = -0.5 * r_coef

                    # Base RHS vector assignment including direct lateral mass entry
                    di = ((0.5 * r_coef + 0.5 * s_adv) * C_im1
                        + (1.0 - 0.5 * (2.0 * r_coef + p) - 0.5 * s_adv) * C_i
                        + 0.5 * r_coef * C_ip1
                        + lateral_source)

                    # Boundary adjustments
                    if i == 0:
                        bi += ai
                        ai = 0.0
                    if i == reach_len - 1:
                        bi += ci
                        ci = 0.0

                    # Transient storage (Runkel Eq. 17)
                    if has_storage:
                        alpha_m = alpha_s
                        alpha_st = alpha_s / A_s_ratio
                        gamma = 0.5 * alpha_st * dt_sub

                        C_s_old = conc_s[r0, c0] / fmax(A_s_ratio * V_i, 1e-30)

                        bi += (0.5 * alpha_m * dt_sub) / (1.0 + gamma)
                        di += (alpha_m * dt_sub * C_s_old) / (1.0 + gamma) - (0.5 * alpha_m * dt_sub * C_i) / (1.0 + gamma)

                    # Settling loss
                    if has_settling:
                        settling_term = wf / fmax(H_i, 0.1)
                        bi += 0.5 * settling_term * dt_sub
                        di -= 0.5 * settling_term * C_i * dt_sub

                    a_arr[i] = ai
                    b_arr[i] = bi
                    c_arr[i] = ci
                    d_arr[i] = di

                # Solve Thomas Algorithm
                c_prime[0] = c_arr[0] / b_arr[0]
                d_prime[0] = d_arr[0] / b_arr[0]

                for i in range(1, reach_len):
                    denom = b_arr[i] - a_arr[i] * c_prime[i-1]
                    if fabs(denom) < 1e-30: denom = 1e-30
                    c_prime[i] = c_arr[i] / denom
                    d_prime[i] = (d_arr[i] - a_arr[i] * d_prime[i-1]) / denom

                x_arr[reach_len-1] = d_prime[reach_len-1]
                for i in range(reach_len-2, -1, -1):
                    x_arr[i] = d_prime[i] - c_prime[i] * x_arr[i+1]

                # --- Compute outflow mass for the tail cell of this reach ---
                if last_is_tail:
                    i = reach_len - 1
                    v_tail = <double>v_arr[i]
                    A_tail = A_arr[i]
                    # time‑averaged concentration during this substep
                    mass_flux_sub = v_tail * A_tail * 1000.0 * 0.5 * (C_old_arr[i] + x_arr[i])
                    outflow_sum[species_idx] += mass_flux_sub * dt_sub

                # Update Main Channel, Bedload, and Transient Storage
                for i in range(reach_len):
                    r0 = rows[i]
                    c0 = cols[i]
                    V_i = V_i_arr[i]

                    C_new = fmax(x_arr[i], 0.0)
                    C_old = C_old_arr[i]

                    # Main channel update
                    conc[r0, c0] = C_new * V_i

                    # Settling / Bedload explicit update
                    if has_settling:
                        settling_term = wf / fmax(H_arr[i], 0.1)
                        settled_moles = settling_term * ((C_old + C_new) / 2.0) * dt_sub * V_i
                        bedload_storage[r0, c0] += fmax(settled_moles, 0.0)

                    # Storage zone update (Runkel Eq 25)
                    if has_storage:
                        alpha_st = alpha_s / A_s_ratio
                        gamma = (dt_sub * alpha_st) / 2.0

                        C_s_old = conc_s[r0, c0] / fmax(A_s_ratio * V_i, 1e-30)
                        C_s_new = (C_s_old * (1.0 - gamma) + gamma * (C_new + C_old)) / (1.0 + gamma)

                        conc_s[r0, c0] = fmax(C_s_new * (A_s_ratio * V_i), 0.0)

        # 4. After all species for this reach are solved, transfer accumulated outflow mass
        #    to the downstream cells' C_lat arrays.
        if last_is_tail:
            tail_key = (rows[reach_len-1], cols[reach_len-1])
            for (dr, dc) in tail_to_dest[tail_key]:
                # Accumulate moles per second for each species
                C_lat_fe2[dr, dc] += outflow_sum[0] / dt
                C_lat_fe3[dr, dc] += outflow_sum[1] / dt
                C_lat_h[dr, dc]   += outflow_sum[2] / dt
                C_lat_so4[dr, dc] += outflow_sum[3] / dt
                C_lat_precip[dr, dc] += outflow_sum[4] / dt

def _build_junction_inflows(
    F32[:, ::1] Q,               
    F32[:, ::1] Q_lat_out,       
    I32[::1] tail_r,
    I32[::1] tail_c,
    I32[::1] dst_r,
    I32[::1] dst_c,
    Py_ssize_t n_junctions,
):
    cdef Py_ssize_t k, tr, tc, dr, dc
    cdef double Q_t

    with nogil:
        for k in range(n_junctions):
            tr = tail_r[k]
            tc = tail_c[k]
            dr = dst_r[k]
            dc = dst_c[k]
            Q_t = <double>Q[tr, tc]
            if Q_t > 0.0:
                Q_lat_out[dr, dc] += <F32>Q_t