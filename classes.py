# class py file for AMDFLOW
# contains main AMDModel class used for AMD modelling

from amd_chemistry import process_chemistry
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import netCDF4


class AMDModel:

    def __init__(self, dataset, t_unit, do = 10 / 31998, output_path = "amdflow_output.nc", calculation_output_path = "amdflow_calculated_output.nc"):
        self.dataset = dataset.copy(deep=True)
        self.dataset["Q"] = self.dataset["Q"].fillna(0.0)
        self.dataset["ore"] = self.dataset["ore"].fillna(0.0)
        self.dataset["ID"] = self.dataset["ID"].where(self.dataset["ID"] >= 0, -1)
        self.dataset["outID"] = self.dataset["outID"].where(self.dataset["outID"] >= 0, -1)
        self.dataset["source"] = self.dataset["source"].where(self.dataset["source"] == 1, 0)
        
        mask_source = (self.dataset["source"] == 1)
        cond1 = ~mask_source.values
        cond2 = (self.dataset["Q"].values > 0)
        condition = np.logical_or(cond1, cond2)
        self.dataset["Q"] = self.dataset["Q"].where(condition, 1e-12)
        self._Q = self.dataset["Q"].copy(deep=True)
        
        self.t_unit = t_unit
        self.time_steps = self.dataset["time"]
        self.do = do
        self.output_path = output_path
        self.calculation_output_path = calculation_output_path
        spatial_shape = (len(self.dataset.lat), len(self.dataset.lon))
        n_steps = len(self.dataset.time)

        self._chem_vars = ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                    "sulphate", "iron_III_hydroxide"]
        
        self._buffer = {
            var: np.zeros((2, *spatial_shape), dtype = np.float32)
            for var in self._chem_vars
        }

        self.time_step_seconds = {"month": 2628000, "week" : 604800, "day": 86400, "hour": 3600, "minute": 60}[self.t_unit]
        
        # init the hydrogen ion at a pH of 7: 10**-7 hydrogen ions per litre at step 0
        volume_0 = self.dataset["Q"].isel(time=0).values * self.time_step_seconds * 1000
        self._buffer["hydrogen_ion"][0] = (1e-7 * volume_0).astype(np.float32)

        self._volume = self.dataset["Q"].values * self.time_step_seconds * 1000
        self._cumulative_vol = np.zeros(spatial_shape, dtype = np.float64)
        self._prev_buffer = {
            var: np.zeros(spatial_shape, dtype = np.float32)
            for var in self._chem_vars
        }

        self._create_output_file(n_steps, spatial_shape)
        
        low, high = 0.01, 1.0
        Q_ref = self.dataset["Q"].isel(time=0)
        norm = low + (Q_ref - Q_ref.min()) / (Q_ref.max() - Q_ref.min() + 1e-12) * (high - low)
        if np.any(np.isnan(norm.values)):
            raise ValueError("_norm_transport contains NaN – check Q_ref for NaN/inf")
        self._norm_transport = norm.values
        
        self._build_cache()

        

    def run(self, chunk_size=1000, n_jobs = -1, backend = "threading"):
        # debug!!!!!!!
        np.seterr(all='raise')
        ore_vals = self.dataset["ore"].values
        source_vals = self.dataset["source"].values
        id_vals = self.dataset["ID"].values

        upstream_mask = (ore_vals > 0) & (source_vals == 1)
        upstream_ids = id_vals[upstream_mask].astype(np.int64)
        upstream_ids = upstream_ids[upstream_ids >= 0]
        with netCDF4.Dataset(self.output_path, "r+") as nc:
            for ti, t in tqdm(enumerate(self.dataset.time.values)):

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

                self._cumulative_vol += self._volume[ti]
                self._write_timestep(ti, nc)

                # write back to buffers
                for var in self._chem_vars:
                    self._buffer[var][0] = self._buffer[var][0] + self._buffer[var][1]
                    self._buffer[var][1] = 0.0
                
                # needs to be outside of previous loop to prevent weird behaviour at first timestep
                for var in self._chem_vars:
                    self._prev_buffer[var] = self._buffer[var][0].copy()

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
        volume = self._volume[time_idx, rows, cols]
        ore    = self.dataset["ore"].values[rows, cols]

        fe2    = self._buffer["ferrous_iron"][0, rows, cols]
        fe3    = self._buffer["ferric_iron"][0, rows, cols]
        so4    = self._buffer["sulphate"][0, rows, cols]
        h      = self._buffer["hydrogen_ion"][0, rows, cols]
        fe_oh3 = self._buffer["iron_III_hydroxide"][0, rows, cols]

        # Ensure contiguous arrays (NO unnecessary copies)
        volume = np.ascontiguousarray(volume, dtype=np.float64)
        ore    = np.ascontiguousarray(ore, dtype=np.float64)

        fe2    = np.ascontiguousarray(fe2, dtype=np.float64)
        fe3    = np.ascontiguousarray(fe3, dtype=np.float64)
        so4    = np.ascontiguousarray(so4, dtype=np.float64)
        h      = np.ascontiguousarray(h, dtype=np.float64)
        fe_oh3 = np.ascontiguousarray(fe_oh3, dtype=np.float64)

        # debug!!!!!!!
        if np.isnan(fe2).any():
            print(f"NaN in fe2 at timestep {ti}, cell IDs: {cell_ids[np.isnan(fe2)]}")
        # debug!!!!!!!
        if np.isnan(fe3).any():
            print(f"NaN in fe3 at timestep {ti}, cell IDs: {cell_ids[np.isnan(fe3)]}")

        # CPython chemistry call
        process_chemistry(
            fe2, fe3, so4, h, fe_oh3,
            ore, volume,
            self.do, self.time_step_seconds
        )

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

        self._buffer["ferrous_iron"][0, rows, cols] = fe2
        self._buffer["ferric_iron"][0, rows, cols] = fe3
        self._buffer["sulphate"][0, rows, cols] = so4
        self._buffer["hydrogen_ion"][0, rows, cols] = h
        self._buffer["iron_III_hydroxide"][0, rows, cols] = fe_oh3
        
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
        
        valid = np.isfinite(out_ids) & (out_ids >= 0)

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

        src_trs = self._norm_transport[src_row, src_col]

        # make and use mask for volume > 0
        time_idx = self._time_index[t]
        src_vol = self._volume[time_idx, src_row, src_col]
        valid_vol = src_vol > 0

        if not valid_vol.any():
            return
        
        src_row = src_row[valid_vol]
        src_col = src_col[valid_vol]
        dst_row = dst_row[valid_vol]
        dst_col = dst_col[valid_vol]
        src_trs = src_trs[valid_vol]
        
        for var in key_vars:
            arr  = self._arrays[var]

            src_vals = arr[0, src_row, src_col]

            # debug!!!!!!!!!!!!!!
            if np.isnan(src_vals).any():
                print(f"NaN in source values for variable '{var}' at timestep {t}, cell IDs: {src_ids[np.isnan(src_vals)]}")
            if np.isnan(src_trs).any():
                print(f"NaN in transport factors for variable '{var}' at timestep {t}, cell IDs: {src_ids[np.isnan(src_trs)]}")
            # debug!!!!!!!!!!!!!!

            moved    = src_vals * src_trs
            arr[0, src_row, src_col] -= moved
            arr[0, src_row, src_col] = np.maximum(arr[0, src_row, src_col], 0)
            np.add.at(arr[1], (dst_row, dst_col), moved) 

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

        max_id = int(np.nanmax(self.dataset["ID"].values[self.dataset["ID"].values >= 0]))
        self._id_to_row = np.full(max_id + 1, -1, dtype=np.int32)
        self._id_to_col = np.full(max_id + 1, -1, dtype=np.int32)
        self._id_to_outid = np.full(max_id + 1, -1, dtype=np.int64)

        flat_out = out_vals.ravel()

        for id_val, r, c, out in zip(flat_ids, flat_rows, flat_cols, flat_out):
            if id_val >= 0:
                self._id_to_row[id_val] = r
                self._id_to_col[id_val] = c
                self._id_to_outid[id_val] = int(out) if out >= 0 else -1

        self._arrays = {
            var: self._buffer[var]
            for var in ["ferrous_iron", "ferric_iron", "sulphate", "hydrogen_ion", "iron_III_hydroxide"]
        }

        self._var_dims = {
            var: ("lat", "lon") for var in self._arrays
        }

        ts = self.dataset.time.values
        self._next_time_map = {
            ts[i]: ts[i + 1] for i in range(len(ts) - 1)
        }
        self._next_time_map[ts[-1]] = None

        self._time_index = {t: i for i, t in enumerate(self.dataset["time"].values)}
                
    def _next_time(self, t):
        return self._next_time_map[t]
    
    def _create_output_file(self, n_steps, spatial_shape):
        with netCDF4.Dataset(self.output_path, "w", format = "NETCDF4") as nc:

            # dims
            nc.createDimension("time", n_steps)
            nc.createDimension("lat", spatial_shape[0])
            nc.createDimension("lon", spatial_shape[1])

            # coordinate vars
            t_var = nc.createVariable("time", "f8", ("time",))
            t_var[:] = netCDF4.date2num(
                [pd.Timestamp(t).to_pydatetime() for t in self.dataset.time.values],
                units = "hours since 1970-01-01",
                calendar = "standard"
            )
            t_var.units = "hours since 1970-01-01"
            t_var.calendar = "standard"

            lat_var = nc.createVariable("lat", "f4", ("lat",))
            lat_var[:] = self.dataset.lat.values

            lon_var = nc.createVariable("lon", "f4", ("lon",))
            lon_var[:] = self.dataset.lon.values

            # chem vars
            attrs = {
                "ferrous_iron": ("g/L", "Fe²⁺"),
                "ferric_iron": ("g/L", "Fe³⁺"),
                "sulphate": ("g/L", "SO₄²⁻"),
                "hydrogen_ion": ("g/L", "H⁺"),
                "iron_III_hydroxide": ("g/L", "Fe(OH)₃")
                }

            for var in self._chem_vars:
                v = nc.createVariable(
                    var, "f4",
                    ("time", "lat", "lon"),
                    chunksizes = (1, spatial_shape[0], spatial_shape[1]),
                    zlib = True,
                    complevel = 4,
                    fill_value = np.nan
                )

                v.units = attrs[var][0]
                v.description = f"{attrs[var][1]} - flow weighted mean (cumulative volume) concentration"
            
                v_instant = nc.createVariable(
                    f"{var}_instant", "f4",
                    ("time", "lat", "lon"),
                    chunksizes=(1, spatial_shape[0], spatial_shape[1]),
                    zlib=True, complevel=4, fill_value=np.nan
                )

                v_instant.units = attrs[var][0]
                v_instant.description = f"{attrs[var][1]} - instant concentration at timestep"

            ph_var = nc.createVariable(
                "pH", "f4",
                ("time", "lat", "lon"),
                chunksizes=(1, spatial_shape[0], spatial_shape[1]),
                zlib=True, complevel=4, fill_value=np.nan   
            )

            ph_var.units = "pH"
            ph_var.description = "pH value calculated from hydron concentration"
            
            ph_instant = nc.createVariable(
                "pH_instant", "f4",
                ("time", "lat", "lon"),
                chunksizes=(1, spatial_shape[0], spatial_shape[1]),
                zlib=True, complevel=4, fill_value=np.nan
            )

            ph_instant.units = "pH"
            ph_instant.description = "pH value calculated from instant hydron concentration at timestep"

    def _write_timestep(self, ti, nc):
        cum_vol = self._cumulative_vol
        step_vol = self._volume[ti]

        molar_masses = {
            "ferrous_iron":       55.845,
            "ferric_iron":        55.845,
            "sulphate":           96.056,
            "hydrogen_ion":       1.008,
            "iron_III_hydroxide": 106.866,
        }

        with np.errstate(under='ignore', divide='ignore', invalid='ignore'):
            for var in self._chem_vars:
                total_moles = self._buffer[var][0]
                delta_moles = total_moles - self._prev_buffer[var]
                delta_moles = np.maximum(delta_moles, 0)  # prevent negative deltas
                mass = molar_masses[var]

                if var == "hydrogen_ion":
                    conc_cum = np.where(cum_vol > 0, total_moles / cum_vol, np.nan)
                    nc[var][ti, :, :] = conc_cum.astype(np.float32)

                    ph = np.where(conc_cum > 0, -np.log10(conc_cum), np.nan)
                    nc["pH"][ti, :, :] = ph.astype(np.float32)

                    if ti == len(self.time_steps) - 1:
                        conc_instant = np.full_like(conc_cum, np.nan)
                    else:
                        conc_instant = np.where(step_vol > 0, delta_moles / step_vol, np.nan)
                    nc[f"{var}_instant"][ti, :, :] = conc_instant.astype(np.float32)

                    ph_instant = np.where(conc_instant > 0, -np.log10(conc_instant), np.nan)
                    nc["pH_instant"][ti, :, :] = ph_instant.astype(np.float32)

                else:
                    total_grams = total_moles * mass
                    conc_cum = np.where(cum_vol > 0, total_grams / cum_vol, np.nan)
                    nc[var][ti, :, :] = conc_cum.astype(np.float32)

                    delta_grams = delta_moles * mass
                    if ti == len(self.time_steps) - 1:
                        conc_instant = np.full_like(conc_cum, np.nan)
                    else:
                        conc_instant = np.where(step_vol > 0, delta_grams / step_vol, np.nan)
                    nc[f"{var}_instant"][ti, :, :] = conc_instant.astype(np.float32)

