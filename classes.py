# class py file for AMDFLOW
# contains main AMDModel class used for AMD modelling

from amd_chemistry import process_chemistry
from transport import _transport_cython
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import netCDF4


class AMDModel:
    """AMDModel class for Acid Mine Drainage modelling
    This class takes in a dataset containing variables (Q, ore, ID, outID, source) and runs the AMD flow model over time (.run()),
    results are written to output_path as a netCDF file
    """
    def __init__(self, dataset, t_unit, do = 10 / 31998, output_path = "amdflow_output.nc", v = 1,
                 a = 2.71, b = 0.557, c = 0.349, f = 0.341, wf = 0.00142):
        """class initialisation

        Parameters
        ----------
        dataset : xr.Dataset
            dataset containing variables: Q (time, lat, lon), ore (lat, lon), ID (lat, lon), outID (lat, lon), source (lat, lon)
        t_unit : str
            time unit for the model: "month", "week", "day", "hour", or "minute", should align with timesteps of the dataset
        do : float, optional
            dissolved oxygen concentration, by default 10/31998
        output_path : str, optional
            path to the output netCDF file, by default "amdflow_output.nc"
        v : float, optional
            average velocity used for area calculation in transport, by default 0.5 m/s
        D : float, optional
            dispersion coefficient used in transport, by default 10 m**2/s

        """
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
        self.v = v
        self.dx = 1000 
        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.wf = wf
        
        self.t_unit = t_unit
        self.time_steps = self.dataset["time"]
        self.do = do
        self.output_path = output_path
        spatial_shape = (len(self.dataset.lat), len(self.dataset.lon))
        n_steps = len(self.dataset.time)

        self._chem_vars = ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                    "sulphate", "iron_III_hydroxide", "bedload_storage"]
        
        self._buffer = {
            var: np.zeros((2, *spatial_shape), dtype = np.float32)
            for var in self._chem_vars
        }

        self.time_step_seconds = {"month": 2628000, "week" : 604800, "day": 86400, "hour": 3600, "minute": 60}[self.t_unit]
        
        # init the hydrogen ion at a pH of 7: 10**-7 hydrogen ions per litre at step 0
        volume_0 = (self.dataset["Q"].isel(time=0).values / self.v) * self.dx * 1000 # V = (Q / v) * dx = m**3, *1000 = L
        self._buffer["hydrogen_ion"][0] = (1e-7 * volume_0).astype(np.float32)

        self._Q_dataset = self.dataset["Q"]

        self._prev_buffer = {
            var: np.zeros(spatial_shape, dtype = np.float32)
            for var in self._chem_vars
        }

        self._create_output_file(n_steps, spatial_shape)
        
        self._build_cache()

        self.molar_masses = {
            "ferrous_iron":       55.845 * 1000,
            "ferric_iron":        55.845 * 1000,
            "sulphate":           96.056 * 1000,
            "hydrogen_ion":       1.008 * 1000,
            "iron_III_hydroxide": 55.845 * 1000,
            "bedload_storage": 106.87 * 1000
        }

    def run(self, chunk_size=1000, n_jobs = -1, backend = "threading"):
        """Runs model over all time steps and spatial extent, writes results to output_path netCDF file

        Parameters
        ----------
        chunk_size : int, optional
            size of chunks to process in parallel, by default 1000
                should be tuned based on dataset size, system memory, etc.
        n_jobs : int, optional
            number of parallel jobs to run, by default -1
                should be tuned based on dataset size, system memory, etc.
        backend : str, optional
            parallel processing backend, by default "threading"
                should be tuned based on dataset size, system memory, etc.
        """
        ore_vals = self.dataset["ore"].values
        source_vals = self.dataset["source"].values
        id_vals = self.dataset["ID"].values

        upstream_mask = (ore_vals > 0) & (source_vals == 1)
        upstream_ids = id_vals[upstream_mask].astype(np.int64)
        #upstream_ids = upstream_ids[upstream_ids >= 0]
        upstream_ids = np.array(
            [self._id_remap[i] for i in upstream_ids.tolist() if i in self._id_remap],
            dtype=np.int64
        )

        with netCDF4.Dataset(self.output_path, "r+") as nc:
            for ti, t in tqdm(enumerate(self.dataset.time.values)):
                # upper loop of all source cells with water and ore
                visited = set(upstream_ids.tolist())

                groups  = self._get_parallel_groups(upstream_ids, chunk_size)
                
                results = Parallel(n_jobs=n_jobs, backend = backend)(
                    delayed(self._compute_slice)(group, t, ti) for group in groups
                )
                results = [r for r in results if r is not None]

                for result in results:
                    self._update_buffer(t, result)
                
                Q_2d = self._Q_np[ti].astype(np.float32)
                if ti < len(self.time_steps) - 1:
                    for result in results:
                        self._transport(t, result, Q_2d)

                
                current_ids = upstream_ids

                while len(current_ids) > 0:
                    # lower loop to propagate through flow network from upper loop, until no more downstream cells to process
                    
                    # get downstream IDs from current frontier
                    out_ids = np.full_like(current_ids, -1, dtype=np.int64)
                    valid_idx = (current_ids >= 0) & (current_ids < len(self._id_to_outid))
                    out_ids[valid_idx] = self._id_to_outid[current_ids[valid_idx]]
                    out_ids = out_ids[out_ids >= 0]

                    # only process cells not yet visited this timestep
                    out_ids = np.unique(out_ids)
                    out_ids = out_ids[~np.isin(out_ids, np.fromiter(visited, dtype=np.int64))]

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
                        self._update_buffer(t, result)

                    if ti < len(self.time_steps) - 1:
                        for result in results:
                            self._transport(t, result, Q_2d)

                    # advance frontier to the newly processed IDs
                    current_ids = out_ids

                # write back to buffers
                for var in self._chem_vars:
                    self._buffer[var][0] = self._buffer[var][0] + self._buffer[var][1]
                    self._buffer[var][1] = 0.0
                
                self._write_timestep(ti, nc)
                
                # needs to be outside of previous loop to prevent weird behaviour at first timestep
                for var in self._chem_vars:
                    self._prev_buffer[var] = self._buffer[var][0].copy()

    def _compute_slice(self, cell_ids, t, ti):
        """Calculate chemistry for slice of cells at timestep t/ti, passes arrays to CPython file (see: amd_chemistry.pyx) for processing,
        returns arrays of row/col indices and chemistry outputs for input cells to be written back to main dataset

        Parameters
        ----------
        cell_ids : np.ndarray
            array of cell IDs to process in slice
        t : np.datetime64
            timestep to process
        ti : int
            index of timestep to process

        Returns
        -------
        rows, cols, fe2, fe3, so4, h, fe_oh3: tuple of np.ndarrays
            arrays containing the row, column indices, and chemistry outputs for input cells at timestep t/ti
        """
        # convert IDs to indices 
        cell_ids = np.asarray(cell_ids, dtype=np.int64)

        valid_ids = (cell_ids >= 0) & (cell_ids < len(self._id_to_row))
        rows = np.full_like(cell_ids, -1, dtype=np.int32)
        cols = np.full_like(cell_ids, -1, dtype=np.int32)
        rows[valid_ids] = self._id_to_row[cell_ids[valid_ids]]
        cols[valid_ids] = self._id_to_col[cell_ids[valid_ids]]

        # filter invalid IDs
        valid = (rows >= 0) & (cols >= 0)
        if not np.any(valid):
            return None

        rows = rows[valid]
        cols = cols[valid]

        time_idx = self._time_index[t]

        # xarray to numpy arrays for CPython
        volume = self._get_volume(time_idx)[rows, cols]
        ore    = self._ore_np[rows, cols]

        fe2    = self._buffer["ferrous_iron"][0, rows, cols]
        fe3    = self._buffer["ferric_iron"][0, rows, cols]
        so4    = self._buffer["sulphate"][0, rows, cols]
        h      = self._buffer["hydrogen_ion"][0, rows, cols]
        fe_oh3 = self._buffer["iron_III_hydroxide"][0, rows, cols]
        bedload_storage = self._buffer["bedload_storage"][0, rows, cols]


        volume = np.ascontiguousarray(volume, dtype=np.float64)
        ore    = np.ascontiguousarray(ore, dtype=np.float64)

        fe2    = np.ascontiguousarray(fe2, dtype=np.float64)
        fe3    = np.ascontiguousarray(fe3, dtype=np.float64)
        so4    = np.ascontiguousarray(so4, dtype=np.float64)
        h      = np.ascontiguousarray(h, dtype=np.float64)
        fe_oh3 = np.ascontiguousarray(fe_oh3, dtype=np.float64)
        bedload_storage = np.ascontiguousarray(bedload_storage, dtype=np.float64)

        # CPython chemistry call
        process_chemistry(
            fe2, fe3, so4, h, fe_oh3, bedload_storage,
            ore, volume,
            self.do, self.time_step_seconds
        )

        return rows, cols, fe2, fe3, so4, h, fe_oh3, bedload_storage
    
    def _get_parallel_groups(self, cell_ids, chunk_size=None):
        """Groups cell IDs into chunks for parallel processing, if chunk_size is None, it will be automatically determined based on number of CPU cores and number of cell IDs

        Parameters
        ----------
        cell_ids : np.ndarray
            array of cell IDs to group into chunks
        chunk_size : int, optional
            size of chunk, by default None,
                if None, automatic chunk size calculation: not that great, should be set by user

        Returns
        -------
        list of np.ndarray
            list of chunks, each containing an array of cell IDs
        """
        if chunk_size is None:
            n_workers = os.cpu_count()
            chunk_size = max(1, len(cell_ids) // n_workers)
            self.chunk_size = chunk_size
        return [cell_ids[i:i+chunk_size] for i in range(0, len(cell_ids), chunk_size)]

    def _update_buffer(self, t, result):
        """Updates self._buffer with chemistry results from the computation

        Parameters
        ----------
        t : np.datetime64
            timestep to update buffer for
        result : tuple of np.ndarrays
            arrays containing the chemistry outputs for input cells at timestep t
        """

        # quick return if slice is empty
        if result is None:
            return
        
        rows, cols, fe2, fe3, so4, h, fe_oh3, bedload_storage = result
        with np.errstate(under='ignore'):
            self._buffer["ferrous_iron"][0, rows, cols] = fe2
            self._buffer["ferric_iron"][0, rows, cols] = fe3
            self._buffer["sulphate"][0, rows, cols] = so4
            self._buffer["hydrogen_ion"][0, rows, cols] = h
            self._buffer["iron_III_hydroxide"][0, rows, cols] = fe_oh3
            self._buffer["bedload_storage"][0, rows, cols] = bedload_storage
        
    def _transport(self, t, result, Q_2d):
        """Transport chemistry downstream based on flow network, updating self._buffer with transported chemistry for next timestep
            uses transport_cython from transport.pyx 

        Parameters
        ----------
        t : np.datetime64
            timestep of model
        result : tuple of np.ndarrays
            arrays containing the row, column indices and chemistry outputs for cells at timestep t
        Q_2d : np.ndarray
            2d array of flow values at timestep t, used for transport calculations
        """
        next_time = self._next_time(t)
        if next_time is None:
            return

        rows, cols, fe2, fe3, so4, h, fe_oh3, bedload_storage = result
        time_idx = self._time_index[t]

        # Get source rows/cols (already filtered by _compute_slice)
        # Convert to numpy arrays of int64 for Cython
        src_rows = rows.astype(np.int64)
        src_cols = cols.astype(np.int64)

        # Extract 2D arrays at current time step
        ID_grid = self._ID_grid
        outID_grid = self.outID_grid

        # Call Cython kernel
        _transport_cython(
            self._buffer["ferrous_iron"],
            self._buffer["ferric_iron"],
            self._buffer["sulphate"],
            self._buffer["hydrogen_ion"],
            self._buffer["iron_III_hydroxide"],
            self._buffer["bedload_storage"],
            Q_2d,
            ID_grid,
            outID_grid,
            self._id_to_row,      
            self._id_to_col,      
            self._id_to_outid,    
            time_idx,
            self.time_step_seconds,
            self.v,
            self.dx,
            self.a,
            self.b,
            self.c,
            self.f,
            self.wf,          
            max_substeps=1000,
            nlat=len(self.dataset.lat),
            nlon=len(self.dataset.lon),
            src_rows=src_rows,
            src_cols=src_cols,
        )

    def _build_cache(self):
        """Build cache at initialisation to pre-process certain static variables and mappings for faster access during model run, 
        such as ID to row/col mapping, chemistry variable arrays, and next time mapping for timesteps
        """
        id_vals = self.dataset["ID"].values
        out_vals = self.dataset["outID"].values

        rows, cols = np.indices(id_vals.shape)

        flat_ids = id_vals.ravel().astype(np.int64)
        flat_rows = rows.ravel()
        flat_cols = cols.ravel()
        flat_out = out_vals.ravel().astype(np.int64)

        valid = flat_ids >= 0
        valid_ids  = flat_ids[valid]
        
        unique_ids = np.unique(np.concatenate([
            valid_ids,
            flat_out[flat_out >= 0]
        ]))
        id_remap = {orig: new for new, orig in enumerate(unique_ids.tolist())}
        array_size = len(unique_ids)  # ~36,300 instead of 67,586,153

        self._id_to_row   = np.full(array_size, -1, dtype=np.int32)
        self._id_to_col   = np.full(array_size, -1, dtype=np.int32)
        self._id_to_outid = np.full(array_size, -1, dtype=np.int64)

        remapped_ids = np.array([id_remap[i] for i in valid_ids.tolist()], dtype=np.int64)
        self._id_to_row[remapped_ids] = flat_rows[valid]
        self._id_to_col[remapped_ids] = flat_cols[valid]

        remapped_out = np.array(
            [id_remap.get(i, -1) for i in flat_out[valid].tolist()], dtype=np.int64
        )
        self._id_to_outid[remapped_ids] = remapped_out

        
        self._ID_grid = np.vectorize(lambda x: id_remap.get(x, -1))(
            id_vals).astype(np.int32)
        self.outID_grid = np.vectorize(lambda x: id_remap.get(x, -1))(
            out_vals).astype(np.int32)

        
        self._id_remap = id_remap

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

        self._Q_np = self.dataset["Q"].values
        self._ore_np = self.dataset["ore"].values
        self._sink_mask = self.outID_grid < 0
                
    def _next_time(self, t):
        """Get next timestep from _next_time_map cache

        Parameters
        ----------
        t : np.datetime64
            current timestep

        Returns
        -------
        np.datetime64 or None
            timestep after t, or None if t is last timestep
        """
        return self._next_time_map[t]
    
    def _create_output_file(self, n_steps, spatial_shape):
        """Create the initial netCDF output file (output_path) with the right dimensions, variables, etc.
        to be written back to during model run

        Parameters
        ----------
        n_steps : int
            amount of timesteps in dataset
        spatial_shape : tuple of ints
            shape of spatial dimensions (lat, lon) in dataset
        """
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
                "ferrous_iron": ("µg/L", "Fe²⁺"),
                "ferric_iron": ("µg/L", "Fe³⁺"),
                "sulphate": ("µg/L", "SO₄²⁻"),
                "hydrogen_ion": ("µg/L", "H⁺"),
                "iron_III_hydroxide": ("µg/L", "Fe in Fe(OH)₃ (suspended)"),
                "bedload_storage": ("mol total", "Fe(OH)₃ deposited on riverbed")
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
                if var == "bedload_storage": 
                    v.description = attrs[var][1]
                else:
                    v.description = f"{attrs[var][1]} - instant concentration at timestep"

            ph_var = nc.createVariable(
                "pH", "f4",
                ("time", "lat", "lon"),
                chunksizes=(1, spatial_shape[0], spatial_shape[1]),
                zlib=True, complevel=4, fill_value=np.nan   
            )

            ph_var.units = "pH"
            ph_var.description = "pH value calculated from hydron concentration"

    def _write_timestep(self, ti, nc):
        """Writes chemistry results for timestep index ti from self._buffer to netCDF dataset nc at output_path
        Parameters
        ----------
        ti : int
            index of timestep
        nc : netCDF4.Dataset()
            the open netCDF dataset to write to, created in _create_output_file
        """
        A_vals = np.maximum(self.dataset["Q"].values[ti] / self.v, 1e-6)  # m²
        V_cell = A_vals * self.dx          # m³
        step_vol = V_cell * 1000            # litres, storage volume

        with np.errstate(under='ignore', divide='ignore', invalid='ignore'):
            for var in self._chem_vars:
                # set sinks to 0 concentration, as the system is closed all chemistry piles here making it unreliable 
                mask = self._sink_mask
                self._buffer[var][0][mask] = 0
                
                if var == "bedload_storage":
                    nc.variables[var][ti, :, :] = (self._buffer[var][0] + self._buffer[var][1])
                
                else:
                    # get both buffer indices and compute concentration
                    mol_amount = (self._buffer[var][0] + self._buffer[var][1])
                    concentration_molar = mol_amount / step_vol  # moles per litre
                    concentration_mg_per_L = concentration_molar * self.molar_masses[var]  # mg/L
                    concentration_ug_per_L = concentration_mg_per_L * 1000  # µg/L
                    nc.variables[var][ti, :, :] = concentration_ug_per_L.astype(np.float32)

            h_mol = self._buffer["hydrogen_ion"][0] + self._buffer["hydrogen_ion"][1]
            h_conc = h_mol / step_vol  # mol/L
            ph = np.where(h_conc > 0, -np.log10(np.maximum(h_conc, 1e-14)), np.nan)
            nc.variables["pH"][ti, :, :] = ph.astype(np.float32)

    def _get_volume(self, ti):
        return (self._Q_np[ti] / self.v) * self.dx * 1000