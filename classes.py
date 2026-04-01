# class py file for AMDFLOW
# contains main AMDModel class used for AMD modelling

from amd_chemistry import process_chemistry
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os


class AMDModel:

    def __init__(self, dataset, t_unit, do = 10 / 31998):
        self.dataset = dataset.copy(deep=True)
        self.t_unit = t_unit
        self.time_steps = self.dataset["time"]
        self.do = do
        self.tree = None

        for var in ["ferrous_iron", "ferric_iron", "sulphate", "hydrogen_ion", "iron_III_hydroxide"]:
            self.dataset[var] = xr.full_like(self.dataset["Q"], 0, dtype=float)

        attrs_dict = {
        "ferrous_iron": {"units": "mol/timestep", "description": "Fe²⁺"},
        "ferric_iron": {"units": "mol/timestep", "description": "Fe³⁺"},
        "sulphate": {"units": "mol/timestep", "description": "SO₄²⁻"},
        "hydrogen_ion": {"units": "mol/timestep", "description": "H⁺"},
        "iron_III_hydroxide": {"units": "mol/timestep", "description": "Fe(OH)₃"}}

        for var_name, attrs in attrs_dict.items():
            self.dataset[var_name].attrs = attrs

        self.dataset = self.dataset.set_coords("ID")
        self.time_step_seconds = {"month": 2628000, "week" : 604800, "day": 86400, "hour": 3600, "minute": 60}[self.t_unit]

        # init the hydrogen ion at a pH of 7: 10**-7 hydrogen ions per litre at step 0
        self.dataset["volume"] = self.dataset["Q"] * self.time_step_seconds * 1000  # L per timestep
        first_time = self.time_steps.values[0]
        self.dataset["hydrogen_ion"].loc[dict(time=first_time)] = 1e-7 * self.dataset["volume"].sel(time=first_time)
        
        low, high = 0.01, 1.0
        Q_ref = self.dataset["Q"].isel(time=0)
        norm = low + (Q_ref - Q_ref.min()) / (Q_ref.max() - Q_ref.min() + 1e-12) * (high - low)
        self._norm_transport = norm.values
        
        self._build_tree()
        self._build_cache()

        

    def run(self, chunk_size=1000, n_jobs = -1, backend = "threading"):

        ore_vals = self.dataset["ore"].values
        source_vals = self.dataset["source"].values
        id_vals = self.dataset["ID"].values

        upstream_mask = (ore_vals > 0) & (source_vals == 1)
        upstream_ids = id_vals[upstream_mask].astype(np.int64)
        upstream_ids = upstream_ids[upstream_ids >= 0]

        for ti, t in tqdm(enumerate(self.dataset.time.values)):
            if ti > 0:
                prev_t = self.time_steps.values[ti - 1]
                for var in ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                            "sulphate", "iron_III_hydroxide"]:
                    prev_vals = self.dataset[var].sel(time=prev_t).fillna(0)
                    self.dataset[var].loc[dict(time=t)] = \
                        self.dataset[var].loc[dict(time=t)].fillna(0) + prev_vals

            # track visited cells to prevent revisiting
            visited = set(upstream_ids.tolist())

            groups  = self._get_parallel_groups(upstream_ids, chunk_size)
            #results = [self._compute_slice(group, t, ti) for group in groups]
            results = Parallel(n_jobs=n_jobs, backend = backend)(
                delayed(self._compute_slice)(group, t, ti) for group in groups
            )
            results = [r for r in results if r is not None]

            for result in results:
                self._update_dataset(t, result)
            if ti < len(self.time_steps) - 1:
                for result in results:
                    self._transport(t, result)

            
            current_ids = upstream_ids

            while len(current_ids) > 0:
                # get downstream IDs from current frontier
                valid_mask = (current_ids <= len(self._id_to_outid) -1)
                out_ids = self._id_to_outid[current_ids[valid_mask]]
                out_ids = out_ids[
                    np.isfinite(out_ids) & (out_ids >= 0)
                ]

                # only process cells not yet visited this timestep
                out_ids = np.unique(out_ids)
                out_ids = out_ids[[i not in visited for i in out_ids]]

                if len(out_ids) == 0:
                    break

                visited.update(out_ids.tolist())

                groups = self._get_parallel_groups(out_ids, chunk_size)
                
                # parallel only when there are multiple groups
                if len(groups) > 4:
                    results = Parallel(n_jobs=n_jobs, backend = backend)(
                        delayed(self._compute_slice)(group, t, ti) for group in groups
                    )
                else:
                    results = [self._compute_slice(group, t, ti) for group in groups]
                
                results = [r for r in results if r is not None]

                for result in results:
                    self._update_dataset(t, result)

                
                if ti < len(self.time_steps) - 1:
                    for result in results:
                        self._transport(t, result)

                # advance frontier to the newly processed IDs
                current_ids = out_ids

    
    def _process_slice(self, current_slice, ti):
        volume = np.ascontiguousarray(current_slice["volume"].values.ravel(),             dtype=np.float64)
        ore    = np.ascontiguousarray(current_slice["ore"].values.ravel(),                dtype=np.float64)
        fe2    = np.ascontiguousarray(current_slice["ferrous_iron"].values.ravel().copy(), dtype=np.float64)
        fe3    = np.ascontiguousarray(current_slice["ferric_iron"].values.ravel().copy(),  dtype=np.float64)
        so4    = np.ascontiguousarray(current_slice["sulphate"].values.ravel().copy(),     dtype=np.float64)
        h      = np.ascontiguousarray(current_slice["hydrogen_ion"].values.ravel().copy(), dtype=np.float64)
        fe_oh3 = np.ascontiguousarray(current_slice["iron_III_hydroxide"].values.ravel().copy(), dtype=np.float64)

        shape = current_slice["ferrous_iron"].values.shape  # save 2D shape for reassignment

        # # debug!!!
        # nan_cells = ~np.isfinite(volume) | ~np.isfinite(ore)
        # if nan_cells.any():
        #     print(f"  {nan_cells.sum()} NaN cells in slice — fmax would zero these")
        # # debug !!!!!

        process_chemistry(fe2, fe3, so4, h, fe_oh3, ore, volume,
                        self.do, self.time_step_seconds)

        return current_slice.assign({
            "ferrous_iron":       (current_slice["ferrous_iron"].dims,       fe2.reshape(shape)),
            "ferric_iron":        (current_slice["ferric_iron"].dims,        fe3.reshape(shape)),
            "sulphate":           (current_slice["sulphate"].dims,           so4.reshape(shape)),
            "hydrogen_ion":       (current_slice["hydrogen_ion"].dims,       h.reshape(shape)),
            "iron_III_hydroxide": (current_slice["iron_III_hydroxide"].dims, fe_oh3.reshape(shape)),
        })

    def _compute_slice(self, cell_ids, t, ti):

        # Convert IDs → indices (vectorized, no Python loops)
        cell_ids = np.asarray(cell_ids, dtype=np.int64)

        rows = self._id_to_row[cell_ids]
        cols = self._id_to_col[cell_ids]

        # Filter invalid IDs
        valid = (rows >= 0) & (cols >= 0)
        if not np.any(valid):
            return None

        rows = rows[valid]
        cols = cols[valid]

        time_idx = self._time_index[t]

        # Direct NumPy access (NO xarray)
        volume = self.dataset["volume"].values[time_idx, rows, cols]
        ore    = self.dataset["ore"].values[rows, cols]

        fe2    = self.dataset["ferrous_iron"].values[time_idx, rows, cols]
        fe3    = self.dataset["ferric_iron"].values[time_idx, rows, cols]
        so4    = self.dataset["sulphate"].values[time_idx, rows, cols]
        h      = self.dataset["hydrogen_ion"].values[time_idx, rows, cols]
        fe_oh3 = self.dataset["iron_III_hydroxide"].values[time_idx, rows, cols]

        # Ensure contiguous arrays (NO unnecessary copies)
        volume = np.ascontiguousarray(volume, dtype=np.float64)
        ore    = np.ascontiguousarray(ore, dtype=np.float64)

        fe2    = np.ascontiguousarray(fe2, dtype=np.float64)
        fe3    = np.ascontiguousarray(fe3, dtype=np.float64)
        so4    = np.ascontiguousarray(so4, dtype=np.float64)
        h      = np.ascontiguousarray(h, dtype=np.float64)
        fe_oh3 = np.ascontiguousarray(fe_oh3, dtype=np.float64)

        # 🔥 Your Cython call (now actually matters)
        process_chemistry(
            fe2, fe3, so4, h, fe_oh3,
            ore, volume,
            self.do, self.time_step_seconds
        )

        # Return raw NumPy results instead of xarray
        return rows, cols, fe2, fe3, so4, h, fe_oh3
    
    def _get_parallel_groups(self, cell_ids, chunk_size=None):
        cell_ids = list(cell_ids)
        if chunk_size is None:
            n_workers = os.cpu_count()
            chunk_size = max(1, len(cell_ids) // n_workers)
            self.chunk_size = chunk_size
        return [cell_ids[i:i+chunk_size] for i in range(0, len(cell_ids), chunk_size)]

    def _update_dataset(self, t, result):
        """Update main dataset using vectorised scatter operation."""
        key_vars = ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                    "sulphate", "iron_III_hydroxide"]

        # Quick return if slice is empty
        if result is None:
            return
        
        rows, cols, fe2, fe3, so4, h, fe_oh3 = result
        time_idx = self._time_index[t]

        self.dataset["ferrous_iron"].values[time_idx, rows, cols] = fe2
        self.dataset["ferric_iron"].values[time_idx, rows, cols] = fe3
        self.dataset["sulphate"].values[time_idx, rows, cols] = so4
        self.dataset["hydrogen_ion"].values[time_idx, rows, cols] = h
        self.dataset["iron_III_hydroxide"].values[time_idx, rows, cols] = fe_oh3
        

    def _transport(self, t, result):
        key_vars = ["ferrous_iron", "ferric_iron", "hydrogen_ion", "sulphate"]
        next_time = self._next_time(t)
        if next_time is None:
            return

        # ── unpack the tuple returned by _compute_slice ──────────────────────
        rows, cols, fe2, fe3, so4, h, fe_oh3 = result

        # Reconstruct IDs and outIDs from the dataset using (rows, cols)
        ids     = self.dataset["ID"].values[rows, cols]
        out_ids = self.dataset["outID"].values[rows, cols]

        valid = np.isfinite(out_ids) & (out_ids != -1)

        if not valid.any():
            return

        src_ids = ids[valid]
        dst_ids = out_ids[valid]

        # filter to dst IDs that exist in the grid
        dst_exists = np.array([int(i) in self._id_to_rc for i in dst_ids])
        valid_both = valid.copy()
        valid_both[valid] = dst_exists

        src_ids = ids[valid_both]
        dst_ids = out_ids[valid_both]

        src_rc = np.array([self._id_to_rc[int(i)] for i in src_ids])
        dst_rc = np.array([self._id_to_rc[int(i)] for i in dst_ids])

        src_row, src_col = src_rc[:, 0], src_rc[:, 1]
        dst_row, dst_col = dst_rc[:, 0], dst_rc[:, 1]

        time_idx_t    = self._time_index[t]
        time_idx_next = self._time_index[next_time]

        src_trs = self._norm_transport[src_row, src_col]

        for var in key_vars:
            arr  = self._arrays[var]
            dims = self._var_dims[var]

            if dims == ('time', 'lat', 'lon'):
                src_vals = arr[time_idx_t, src_row, src_col]
                moved    = src_vals * src_trs

                arr[time_idx_t, src_row, src_col] -= moved
                arr[time_idx_t, src_row, src_col]  = np.maximum(
                    arr[time_idx_t, src_row, src_col], 0)
                np.add.at(arr[time_idx_next], (dst_row, dst_col), moved)

            elif dims == ('time', 'lon', 'lat'):
                src_vals = arr[time_idx_t, src_col, src_row]
                moved    = src_vals * src_trs

                arr[time_idx_t, src_col, src_row] -= moved
                arr[time_idx_t, src_col, src_row]  = np.maximum(
                    arr[time_idx_t, src_col, src_row], 0)
                np.add.at(arr[time_idx_next], (dst_col, dst_row), moved)


    def _build_cache(self):
        id_vals = self.dataset["ID"].values
        out_vals = self.dataset["outID"].values

        rows, cols = np.indices(id_vals.shape)

        flat_ids = id_vals.ravel().astype(np.int64)
        flat_rows = rows.ravel()
        flat_cols = cols.ravel()

        self._id_to_rc = dict(zip(
            flat_ids.tolist(),
            zip(
                flat_rows.tolist(),
                flat_cols.tolist()
            )
        ))

        max_id = int(np.nanmax(self.dataset["ID"].values))
        self._id_to_row = np.full(max_id + 1, -1, dtype=np.int32)
        self._id_to_col = np.full(max_id + 1, -1, dtype=np.int32)
        self._id_to_outid = np.full(max_id + 1, -1, dtype=np.int64)

        flat_out = out_vals.ravel()

        for id_val, r, c, out in zip(flat_ids, flat_rows, flat_cols, flat_out):
            if id_val >= 0:
                self._id_to_row[id_val] = r
                self._id_to_col[id_val] = c
                self._id_to_outid[id_val] = int(out) if np.isfinite(out) else -1

        self._arrays = {
            var: self.dataset[var].values
            for var in ["ferrous_iron", "ferric_iron", "sulphate", "hydrogen_ion", "iron_III_hydroxide"]
        }

        self._var_dims = {
            var: self.dataset[var].dims for var in self._arrays
        }

        ts = self.dataset.time.values
        self._next_time_map = {
            ts[i]: ts[i + 1] for i in range(len(ts) - 1)
        }
        self._next_time_map[ts[-1]] = None
                
    def _next_time(self, t):
        return self._next_time_map[t]
    
    def output_calc(self):

        # safety check to make sure calculations are not run twice
        if self.dataset["ferric_iron"].attrs["units"] == "g/L":
            print("Output calculations already run, skipped to ensure calculations are not run twice")
        
        # average molar mass per mole dict
        molar_masses = {
            "ferrous_iron": 55.845,
            "ferric_iron": 55.845,
            "sulphate": 96.056,
            "hydrogen_ion": 1.008,
            "iron_III_hydroxide": 106.866,
        }

        # convert moles to grams total per cell
        for var, mass in molar_masses.items():
            if var in self.dataset.data_vars:
                self.dataset[var] = self.dataset[var] * mass
                self.dataset[var].attrs["units"] = "g"
        steps_total = np.arange(1, len(self.time_steps) + 1)

        # calculate volume in liters through cell cumulative 
        cumulative_vol = (
            self.dataset["Q"] * self.time_step_seconds * 1000
        ) * xr.DataArray(steps_total, dims = ["time"])

        
        # convert to concentration (g/L) except hydrogen_ion
        for var in molar_masses.keys():
            if var == "hydrogen_ion":
                continue
            if var in self.dataset.data_vars:
                # avoid division by zero: replace zero volume with NaN or 0
                conc = xr.where(cumulative_vol > 0, 
                    self.dataset[var] / cumulative_vol, 
                    0
                    )
                self.dataset[var] = conc
                self.dataset[var].attrs["units"] = "g/L"
        
        # compute pH from H⁺ concentration (mol/L)
        if "hydrogen_ion" in self.dataset.data_vars:
            # H⁺ in mol/L = (H⁺ moles) / volume
            h_conc = xr.where(cumulative_vol > 0, 
                              self.dataset["hydrogen_ion"] / cumulative_vol, 
                              np.nan)
            # pH = -log10([H⁺]), clip to avoid log of zero/negative
            pH = -np.log10(h_conc.where(h_conc > 0, np.nan))
            self.dataset["pH"] = pH
            self.dataset["pH"].attrs = {"units": "pH", "description": "pH value"}
            self.dataset["hydrogen_ion"] = h_conc
            self.dataset["hydrogen_ion"].attrs["units"] = "mol/L"
    
    def _build_tree(self):

        target_lon = self.dataset.lon.values
        target_lat = self.dataset.lat.values
        lon_grid, lat_grid = np.meshgrid(target_lon, target_lat, indexing = "xy")

        target_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

        self.tree = cKDTree(target_points)
        self.n_lon = len(target_lon)

        self._time_index = {t: i for i, t in enumerate(self.dataset.time.values)}


        