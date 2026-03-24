# class py file for AMDFLOW
# contains main AMDModel class used for AMD modelling

import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os

class AMDModel:

    def __init__(self, dataset, t_unit, do = 0.2500094):
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
        # first_time = self.time_steps.values[0]
        # self.dataset["hydrogen_ion"].loc[dict(time=first_time)] = 1e-7 * self.dataset["volume"].sel(time=first_time)
        
        low, high = 0.01, 1.0
        Q_ref = self.dataset["Q"].isel(time=0)
        norm = low + (Q_ref - Q_ref.min()) / (Q_ref.max() - Q_ref.min() + 1e-12) * (high - low)
        self._norm_transport = norm.values
        
        self._build_tree()
        self._build_cache()

    def run(self, chunk_size=50, n_jobs = -1, backend = "loky"):
        mask_ores = self.dataset["ore"] > 0
        reactive_ores = self.dataset.where(mask_ores, drop=True)
        mask = reactive_ores["source"].where(reactive_ores["source"] == 1)
        most_upstream_reactive_ores = self.dataset.where(mask, drop=True)

        for ti, t in tqdm(enumerate(self.dataset.time.values)):
            if ti > 0:
                prev_t = self.time_steps.values[ti - 1]
                for var in ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                            "sulphate", "iron_III_hydroxide"]:
                    prev_vals = self.dataset[var].sel(time=prev_t).fillna(0)
                    self.dataset[var].loc[dict(time=t)] = \
                        self.dataset[var].loc[dict(time=t)].fillna(0) + prev_vals

            # fix 1: ravel to guarantee 1D flat array of IDs
            upstream_ids = most_upstream_reactive_ores["ID"].values.ravel()
            upstream_ids = upstream_ids[np.isfinite(upstream_ids)]
            upstream_ids = upstream_ids.astype(np.int64)

            # fix 2: track visited cells to prevent revisiting
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

            dataset_t = self.dataset.sel(time=t)
            current_ids = upstream_ids

            while len(current_ids) > 0:
                # get downstream IDs from current frontier
                current_ds = dataset_t.where(dataset_t["ID"].isin(current_ids), drop=True)
                out_ids = current_ds["outID"].values.ravel()
                out_ids = out_ids[np.isfinite(out_ids) & (out_ids != -1)]
                out_ids = out_ids.astype(np.int64)

                # fix 3: only process cells not yet visited this timestep
                out_ids = np.array([i for i in out_ids if i not in visited])

                if len(out_ids) == 0:
                    break

                visited.update(out_ids.tolist())

                groups  = self._get_parallel_groups(out_ids, chunk_size)
                #results = [self._compute_slice(group, t, ti) for group in groups]
                results = Parallel(n_jobs=n_jobs, backend = backend)(
                    delayed(self._compute_slice)(group, t, ti) for group in groups
                )
                results = [r for r in results if r is not None]

                for result in results:
                    self._update_dataset(t, result)

                dataset_t = self.dataset.sel(time=t)
                if ti < len(self.time_steps) - 1:
                    for result in results:
                        self._transport(t, result)

                # advance frontier to the newly processed IDs
                current_ids = out_ids

    def _process_slice(self, current_slice, ti):
        # extract all arrays to numpy once — no more xarray overhead inside the function
        volume   = current_slice["volume"].values
        ore      = current_slice["ore"].values
        fe2      = current_slice["ferrous_iron"].values.copy()
        fe3      = current_slice["ferric_iron"].values.copy()
        so4      = current_slice["sulphate"].values.copy()
        h        = current_slice["hydrogen_ion"].values.copy()
        fe_oh3   = current_slice["iron_III_hydroxide"].values.copy()

        if ti == 0:
            h = 1e-7 * volume

        h2o = (0.99704702 * (volume * 1000)) / 18.01528

        # 1) pyrite oxidation by ferric iron
        mask_ferric = (fe3 > 0) & (ore > 0)
        ferric_consumed = np.where(mask_ferric, fe3, 0)
        max_ferric = 1.75 * h2o
        ferric_consumed = np.minimum(ferric_consumed, max_ferric)

        fe2 += ferric_consumed * 1.07
        fe3 -= ferric_consumed
        h   += np.where(mask_ferric, ferric_consumed * 1.14, 0)

        # 2) rate-limited pyrite oxidation
        k = 10**-8.19
        mask_rate = (ore > 0) & ~mask_ferric
        h_conc = h / np.where(volume > 0, volume, 1)
        h_safe = np.where((h_conc <= 0) | ~np.isfinite(h_conc), 1e-7, h_conc)
        rate = k * (self.do ** 0.5) / (h_safe ** 0.11)

        ferrous_amount = np.where(mask_rate, rate * ore * self.time_step_seconds, 0.0)
        ferrous_amount = np.minimum(ferrous_amount, 1 * h2o)

        fe2 += ferrous_amount
        so4 += 2 * ferrous_amount
        h   += 2 * ferrous_amount

        # 3) ferrous to ferric oxidation
        fe3 += fe2
        h   -= fe2
        fe2  = np.zeros_like(fe2)
        h    = np.maximum(h, 0)

        # 4) ferric <> iron III hydroxide equilibrium
        diff       = fe3 - fe_oh3
        adjustment = 0.5 * diff
        fe3    -= adjustment
        fe_oh3 += adjustment
        h      += adjustment * 3

        # 5) clip negatives
        fe2    = np.maximum(fe2,    0)
        fe3    = np.maximum(fe3,    0)
        h      = np.maximum(h,      0)
        so4    = np.maximum(so4,    0)
        fe_oh3 = np.maximum(fe_oh3, 0)

        # write back to slice — one assignment per variable, no align/reindex
        current_slice = current_slice.assign({
            "ferrous_iron":       (current_slice["ferrous_iron"].dims,       fe2),
            "ferric_iron":        (current_slice["ferric_iron"].dims,        fe3),
            "sulphate":           (current_slice["sulphate"].dims,           so4),
            "hydrogen_ion":       (current_slice["hydrogen_ion"].dims,       h),
            "iron_III_hydroxide": (current_slice["iron_III_hydroxide"].dims, fe_oh3),
        })

        return current_slice
    
    def _compute_slice(self, cell_ids, t, ti):

        dataset_t = self.dataset.sel(time = t)
        cell_slice = dataset_t.where(dataset_t["ID"].isin(cell_ids), drop = True)
        cell_slice = cell_slice.where(cell_slice["Q"] > 0, drop = True)

        if cell_slice.sizes.get("lat", 0) == 0:
            return None
        return self._process_slice(cell_slice, ti)
    
    def _get_parallel_groups(self, cell_ids, chunk_size=None):
        cell_ids = list(cell_ids)
        if chunk_size is None:
            n_workers = os.cpu_count()
            chunk_size = max(1, len(cell_ids) // n_workers)
            self.chunk_size = chunk_size
        return [cell_ids[i:i+chunk_size] for i in range(0, len(cell_ids), chunk_size)]

    def _update_dataset(self, t, current_slice):
        """Update main dataset using vectorised scatter operation."""
        key_vars = ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                    "sulphate", "iron_III_hydroxide"]

        # Quick return if slice is empty
        if current_slice.sizes.get("lon", 0) == 0 or current_slice.sizes.get("lat", 0) == 0:
            return

        # Stack cells and drop those with any NaN in key variables
        stacked = current_slice.stack(cell=("lon", "lat"))
        stacked = stacked.dropna(dim="cell", subset=key_vars, how="any")
        n_cells = stacked.sizes.get("cell", 0)
        if n_cells == 0:
            return

        # Source coordinates (all valid cells)
        src_lon = stacked["lon"].values          # shape (n_cells,)
        src_lat = stacked["lat"].values
        src_points = np.column_stack([src_lon, src_lat])


        # For each source point, find the nearest target grid cell (index)
        distances, target_indices_flat = self.tree.query(src_points, k=1)  # shape (n_cells,)

        # Convert flat indices to 2D (lat, lon) indices
        # Since ravel order is (lon varies fastest), the conversion is:
        target_lon_idx = target_indices_flat % self.n_lon
        target_lat_idx = target_indices_flat // self.n_lon

        # Process each variable
        for var in key_vars:
            if var not in stacked.data_vars:
                continue

            src_vals = stacked[var].values               # shape (n_cells,)

            # Only consider source values > 0
            valid_mask = src_vals > 0
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                continue

            # Subset to valid sources
            valid_target_lat_idx = target_lat_idx[valid_indices]
            valid_target_lon_idx = target_lon_idx[valid_indices]
            valid_src_vals = src_vals[valid_indices]

            # Create unique cell IDs and keep last occurrence
            cell_ids = valid_target_lat_idx * self.n_lon + valid_target_lon_idx
            _, unique_idx = np.unique(cell_ids[::-1], return_index=True)
            unique_idx = len(cell_ids) - 1 - unique_idx

            # Extract final assignments
            final_lat_idx = valid_target_lat_idx[unique_idx]
            final_lon_idx = valid_target_lon_idx[unique_idx]
            final_vals = valid_src_vals[unique_idx]

            # Get time index
            time_idx = self._time_index[t]#[0][0]

            # Determine dimension order and assign
            dims = self.dataset[var].dims
            if dims == ('time', 'lat', 'lon'):
                self.dataset[var].values[time_idx, final_lat_idx, final_lon_idx] = final_vals
            elif dims == ('time', 'lon', 'lat'):
                self.dataset[var].values[time_idx, final_lon_idx, final_lat_idx] = final_vals
            else:
                # fallback loop
                for lat_i, lon_i, val in zip(final_lat_idx, final_lon_idx, final_vals):
                    idx_dict = {'time': time_idx, 'lat': lat_i, 'lon': lon_i}
                    idx_tuple = tuple(idx_dict.get(dim, slice(None)) for dim in dims)
                    self.dataset[var].values[idx_tuple] = val

    def _transport(self, t, current_slice):
        key_vars = ["ferrous_iron", "ferric_iron", "hydrogen_ion", "sulphate"]
        next_time = self._next_time(t)
        if next_time is None:
            return

        ids      = current_slice["ID"].values.ravel()
        out_ids  = current_slice["outID"].values.ravel()
        valid   = np.isfinite(out_ids) & (out_ids != -1)

        if not valid.any():
            return

        src_ids     = ids[valid]
        dst_ids     = out_ids[valid]

        # look up (row, col) directly from cache — no unravel needed
        src_rc = np.array([self._id_to_rc[int(i)] for i in src_ids])
        dst_rc = np.array([self._id_to_rc[int(i)] for i in dst_ids if int(i) in self._id_to_rc])


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
            arr = self.dataset[var].values  # (time, lat, lon) or (time, lon, lat)
            dims = self.dataset[var].dims

            if dims == ('time', 'lat', 'lon'):
                src_vals  = arr[time_idx_t,    src_row, src_col]
                moved     = src_vals * src_trs
                np.subtract.at(arr[time_idx_t],
                            (src_row, src_col), moved)
                arr[time_idx_t, src_row, src_col] = np.maximum(
                    arr[time_idx_t, src_row, src_col], 0)
                np.add.at(arr[time_idx_next], (dst_row, dst_col), moved)
            elif dims == ('time', 'lon', 'lat'):
                src_vals  = arr[time_idx_t,    src_col, src_row]
                moved     = src_vals * src_trs
                np.subtract.at(arr[time_idx_t],
                            (src_col, src_row), moved)
                arr[time_idx_t, src_col, src_row] = np.maximum(
                    arr[time_idx_t, src_col, src_row], 0)
                np.add.at(arr[time_idx_next], (dst_col, dst_row), moved)

            self.dataset[var].values[:] = arr

    def _build_cache(self):
        id_2d = self.dataset["ID"].isel(time=0) if "time" in self.dataset["ID"].dims else self.dataset["ID"]
        self._nrows, self._ncols = id_2d.shape
        self._time_index = {t: i for i, t in enumerate(self.dataset.time.values)}
        
        # map ID value → (row, col) directly — safe regardless of ID range
        id_vals = id_2d.values
        rows, cols = np.indices(id_vals.shape)
        self._id_to_rc = {
            int(id_vals[r, c]): (r, c)
            for r in range(self._nrows)
            for c in range(self._ncols)
        }
                
    def _next_time(self, t):
        idx = np.where(self.time_steps.values == t)[0][0]
        if idx + 1 >= len(self.time_steps):
            return None
        return self.time_steps.values[idx + 1]
    
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
        
        # calculate volume in liters for each cell and timestep
        volume = self.dataset["Q"] * self.time_step_seconds * 1000  # L
        volume.attrs = {"units": "L", "description": "Water volume per timestep"}
        
        # convert to concentration (g/L) except hydrogen_ion
        for var in molar_masses.keys():
            if var == "hydrogen_ion":
                continue
            if var in self.dataset.data_vars:
                # avoid division by zero: replace zero volume with NaN or 0
                conc = xr.where(volume > 0, self.dataset[var] / volume, 0)
                self.dataset[var] = conc
                self.dataset[var].attrs["units"] = "g/L"
        
        # compute pH from H⁺ concentration (mol/L)
        if "hydrogen_ion" in self.dataset.data_vars:
            # H⁺ in mol/L = (H⁺ moles) / volume
            h_conc = xr.where(volume > 0, self.dataset["hydrogen_ion"] / volume, np.nan)
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


        