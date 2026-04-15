# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: language=c++

import numpy as np
cimport numpy as np
from libc.math cimport exp, fmax, fmin, ceil
from libc.stdlib cimport malloc, free

ctypedef np.float32_t FLOAT_t
ctypedef np.int32_t INT_t
ctypedef np.int64_t LONG_t

cdef extern from "math.h":
    double INFINITY

def _transport_cython(
    FLOAT_t[:, :, ::1] fe2_buf,      # buffer[2, nlat, nlon]
    FLOAT_t[:, :, ::1] fe3_buf,
    FLOAT_t[:, :, ::1] so4_buf,
    FLOAT_t[:, :, ::1] h_buf,
    FLOAT_t[:, ::1] Q,               # flow at current time step [nlat, nlon]
    INT_t[:, ::1] ID_grid,           # [nlat, nlon]
    INT_t[:, ::1] outID_grid,        # [nlat, nlon]
    INT_t[:] id_to_row,              # mapping from ID to row index (-1 if invalid)
    INT_t[:] id_to_col,              # mapping from ID to col index
    LONG_t[:] id_to_outid,           # mapping from ID to outID
    long time_idx,                   # current time index (unused but kept for interface)
    double time_step_seconds,
    double v,
    double D,
    double dx,
    double alpha,
    int max_substeps,
    long nlat,
    long nlon,
    LONG_t[:] src_rows,              # in/out: source rows, will be modified in-place
    LONG_t[:] src_cols,              # in/out: source cols, will be modified in-place
):
    """
    Cython kernel for advective‑dispersive‑decay transport along the flow network.
    Buffers are modified in‑place.
    """
    cdef:
        LONG_t n_cells = src_rows.shape[0]
        LONG_t i, sub, var_idx
        LONG_t src_r, src_c, dst_r, dst_c
        double Q_val, A, V_cell, C_courant, dt_sub, adv, disp, decay, trans_frac
        double src_val, moved
        LONG_t n_sub, sub_step
        LONG_t[:] dst_rows = np.empty(n_cells, dtype=np.int64)
        LONG_t[:] dst_cols = np.empty(n_cells, dtype=np.int64)
        int[:] valid_cell = np.ones(n_cells, dtype=np.int32)
        LONG_t valid_count
        LONG_t current_id, next_id
        double epsilon = 1e-12

        # For dynamic arrays inside loops
        double[:] transport_frac
        int[:] vol_valid
        int[:] has_next
        LONG_t max_cells = n_cells

    # 1. Determine destination cells for each source
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

    # Filter to valid source‑destination pairs
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

    # 2. Compute number of substeps (Courant‑like condition)
    cdef double max_C = 0.0
    for i in range(n_cells):
        src_r = src_rows[i]
        src_c = src_cols[i]
        Q_val = Q[src_r, src_c]
        if Q_val > 0:
            A = Q_val / v
            if A < 1e-6:
                A = 1e-6
            V_cell = A * dx
            C_courant = Q_val * time_step_seconds / V_cell
            if C_courant > max_C:
                max_C = C_courant

    n_sub = <LONG_t>ceil(max_C)
    if n_sub < 1:
        n_sub = 1
    elif n_sub > max_substeps:
        n_sub = max_substeps
    dt_sub = time_step_seconds / n_sub

    # 3. Prepare for substep loop
    cdef LONG_t[:] current_src_rows = src_rows
    cdef LONG_t[:] current_src_cols = src_cols
    cdef LONG_t[:] current_dst_rows = dst_rows
    cdef LONG_t[:] current_dst_cols = dst_cols
    cdef LONG_t current_n = n_cells

    cdef double disp_coeff = D * dt_sub / (dx * dx)
    if disp_coeff > 0.5:
        disp_coeff = 0.5

    # Pre‑allocate working arrays of maximum possible size
    transport_frac = np.empty(max_cells, dtype=np.float64)
    vol_valid = np.empty(max_cells, dtype=np.int32)
    has_next = np.empty(max_cells, dtype=np.int32)

    for sub_step in range(n_sub):
        # Merge incoming buffer into resident before each substep except first
        # Merge buffers at start of substep (chemicals from previous substep become available)
        if sub_step > 0:
            for i in range(current_n):
                src_r = current_src_rows[i]
                src_c = current_src_cols[i]
                fe2_buf[0, src_r, src_c] += fe2_buf[1, src_r, src_c]
                fe2_buf[1, src_r, src_c] = 0.0
                fe3_buf[0, src_r, src_c] += fe3_buf[1, src_r, src_c]
                fe3_buf[1, src_r, src_c] = 0.0
                so4_buf[0, src_r, src_c] += so4_buf[1, src_r, src_c]
                so4_buf[1, src_r, src_c] = 0.0
                h_buf[0, src_r, src_c] += h_buf[1, src_r, src_c]
                h_buf[1, src_r, src_c] = 0.0

        # Compute transport fractions and volume validity for this substep
        for i in range(current_n):
            src_r = current_src_rows[i]
            src_c = current_src_cols[i]
            Q_val = Q[src_r, src_c]
            if Q_val <= 0:
                vol_valid[i] = False
                continue
            A = Q_val / v
            if A < 1e-6:
                A = 1e-6
            V_cell = A * dx
            adv = Q_val * dt_sub / V_cell
            if adv > 1.0:
                adv = 1.0
            disp = disp_coeff
            decay = exp(-alpha * dt_sub)
            trans_frac = (adv + disp) * decay
            if trans_frac > 1.0:
                trans_frac = 1.0
            elif trans_frac < 0.0:
                trans_frac = 0.0
            transport_frac[i] = trans_frac

            # Optional volume check (consistent with original)
            src_vol = Q_val * time_step_seconds * 1000.0
            if src_vol <= 0:
                vol_valid[i] = False
            else:
                vol_valid[i] = True

        # Filter out invalid cells (zero volume, zero flow, etc.)
        valid_count = 0
        for i in range(current_n):
            if vol_valid[i]:
                current_src_rows[valid_count] = current_src_rows[i]
                current_src_cols[valid_count] = current_src_cols[i]
                current_dst_rows[valid_count] = current_dst_rows[i]
                current_dst_cols[valid_count] = current_dst_cols[i]
                transport_frac[valid_count] = transport_frac[i]
                valid_count += 1
        if valid_count == 0:
            break
        current_n = valid_count

        # Perform transport for all four chemical variables
        for i in range(current_n):
            src_r = current_src_rows[i]
            src_c = current_src_cols[i]
            dst_r = current_dst_rows[i]
            dst_c = current_dst_cols[i]
            # Fe2+ (ferrous iron)
            src_val = fe2_buf[0, src_r, src_c]
            moved = src_val * transport_frac[i]
            fe2_buf[0, src_r, src_c] = fmax(src_val - moved, 0.0)
            fe2_buf[1, dst_r, dst_c] += moved
            # Fe3+ (ferric iron)
            src_val = fe3_buf[0, src_r, src_c]
            moved = src_val * transport_frac[i]
            fe3_buf[0, src_r, src_c] = fmax(src_val - moved, 0.0)
            fe3_buf[1, dst_r, dst_c] += moved
            # SO4 (sulphate)
            src_val = so4_buf[0, src_r, src_c]
            moved = src_val * transport_frac[i]
            so4_buf[0, src_r, src_c] = fmax(src_val - moved, 0.0)
            so4_buf[1, dst_r, dst_c] += moved
            # H+ (hydrogen ion)
            src_val = h_buf[0, src_r, src_c]
            moved = src_val * transport_frac[i]
            h_buf[0, src_r, src_c] = fmax(src_val - moved, 0.0)
            h_buf[1, dst_r, dst_c] += moved

        # Cascade to next downstream cells
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
                        # Overwrite the source arrays in‑place for next iteration
                        current_src_rows[i] = dst_r
                        current_src_cols[i] = dst_c
                        current_dst_rows[i] = next_r
                        current_dst_cols[i] = next_c
                        has_next[i] = True

        # Compress to only cells that have a valid downstream neighbour
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

    # Final merge of any remaining incoming material into resident buffer
    for i in range(current_n):
        src_r = current_src_rows[i]
        src_c = current_src_cols[i]
        fe2_buf[0, src_r, src_c] += fe2_buf[1, src_r, src_c]
        fe2_buf[1, src_r, src_c] = 0.0
        fe3_buf[0, src_r, src_c] += fe3_buf[1, src_r, src_c]
        fe3_buf[1, src_r, src_c] = 0.0
        so4_buf[0, src_r, src_c] += so4_buf[1, src_r, src_c]
        so4_buf[1, src_r, src_c] = 0.0
        h_buf[0, src_r, src_c] += h_buf[1, src_r, src_c]
        h_buf[1, src_r, src_c] = 0.0