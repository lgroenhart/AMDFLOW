# class py file for AMDFLOW
# contains main AMDModel class used for AMD modelling

from amd_chemistry_fast import process_chemistry
from transport import _transport_cn, _transport_ad_dep, _build_junction_inflows
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import netCDF4
from collections import deque



class AMDModel:
    """AMDModel class for Acid Mine Drainage modelling
    This class takes in a dataset containing variables (Q, ore, ID, outID, source) and runs the AMD flow model over time (.run()),
    results are written to output_path as a netCDF file
    """
    def __init__(self, dataset, t_unit, do = 10 / 31998, output_path = "amdflow_output.nc",
                 a = 2.71, b = 0.557, c = 0.349, f = 0.341, wf = 0.00142, alpha_s = 1e-5, A_s_ratio = 0.5,
                    buffer_capacity = 0.0, v = 1.0):
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
        a : float, optional
            geometry relation parameter for equation: width (W) = a * Q**b, by default 2.71
        b : float, optional
            geometry relation parameter for equation: width (W) = a * Q**b, by default 0.557
        c : float, optional
            geometry relation parameter for equation: depth (H) = c * Q**f, by default 0.349
        f : float, optional
            geometry relation parameter for equation: depth (H) = c * Q**f, by default 0.341
        wf : float, optional
            settling velocity of iron hydroxides, by default 0.00142 m/s
        alpha_s : float, optional
            storage exchange coefficient, by default 1e-5
        A_s_ratio : float, optional
            storage zone cross sectional area ratio relative to main channel cross sectional area, by default 0.5
        ssa : float, optional
            specific surface area of pyrite, by default 1
        buffer_capacity : float, optional
            buffer capacity for pH, non-physical instrument to buffer pH, by default 0.1
        v : float, optional
            global set velocity
        """
        self.dataset = dataset.copy(deep=True)
        self.dataset["Q"] = self.dataset["Q"].fillna(0.0)
        self.dataset["ore"] = self.dataset["ore"].fillna(0.0)
        self.dataset["ID"] = self.dataset["ID"].where(self.dataset["ID"] >= 0, -1)
        self.dataset["outID"] = self.dataset["outID"].where(self.dataset["outID"] >= 0, -1)
        self.dataset["source"] = self.dataset["source"].where(self.dataset["source"] == 1, 0)

        # mask_source = (self.dataset["source"] == 1)
        # cond1 = ~mask_source.values
        # cond2 = (self.dataset["Q"].values > 0)
        # condition = np.logical_or(cond1, cond2)
        # self.dataset["Q"] = self.dataset["Q"].where(condition, 1e-3) # 
        self._Q = self.dataset["Q"].copy(deep=True)
        self.dx = 1000 
        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.wf = wf
        self.alpha_s = alpha_s
        self.A_s_ratio = A_s_ratio
        self.v = v
        self.t_unit = t_unit
        self.time_steps = self.dataset["time"]
        self.do = do
        self.buffer_capacity = buffer_capacity
        self.output_path = output_path
        spatial_shape = (len(self.dataset.lat), len(self.dataset.lon))
        n_steps = len(self.dataset.time)

        self._chem_vars = ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                    "sulphate", "iron_III_hydroxide", "bedload_storage"]
        
        self._transport_vars = [v for v in self._chem_vars
                                if v not in ["iron_III_hydroxide", "bedload_storage"]]
        
        self._buffer = {
            var: np.zeros((2, *spatial_shape), dtype = np.float64)
            for var in self._chem_vars
        }

        self._sbuffer = {
            var: np.zeros((2, *spatial_shape), dtype = np.float64)
            for var in self._transport_vars
        }

        self._Q_lat_buff_dict = {var: np.zeros(spatial_shape, dtype = np.float32) for var in self._transport_vars}
        self._C_lat_buff_dict = {var: np.zeros(spatial_shape, dtype = np.float64) for var in self._transport_vars}
        self._C_lat_num_buff_dict = {var: np.zeros(spatial_shape, dtype = np.float64) for var in self._transport_vars}

        self.time_step_seconds = {"month": 2628000, "week" : 604800, "day": 86400, "hour": 3600, "minute": 60}[self.t_unit]
        
        # init the hydrogen ion at a pH of 7: 10**-7 hydrogen ions per litre at step 0
        volume_0 = (self.dataset["Q"].isel(time=0).values / self.v) * self.dx * 1000 # V = (Q / v) * dx = m**3, *1000 = L
        self._buffer["hydrogen_ion"][0] = (1e-7 * volume_0).astype(np.float64)
        self._sbuffer["hydrogen_ion"][0] = self._buffer["hydrogen_ion"][0].copy()

        self._Q_dataset = self.dataset["Q"]
        self._median_vol = np.full(spatial_shape, 1000.0, dtype = np.float32)

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

    def run(self):
        """Runs model over all time steps and spatial extent, writes results to output_path netCDF file
        """


        with netCDF4.Dataset(self.output_path, "r+") as nc:
            for ti, t in tqdm(enumerate(self.dataset.time.values)):
                
            
                Q_2d = self._Q_np[ti].astype(np.float32)
                self._chemistry(Q_2d)
                
                self._transport(t, Q_2d)

                for var in self._chem_vars:
                    # 1. Update main buffer
                    self._buffer[var][0] = self._buffer[var][0] + self._buffer[var][1]
                    self._buffer[var][1] = 0.0
                    self._buffer[var][0][self._off_network] = 0.0
                    
                    # 2. Update storage buffer ONLY if the variable exists in it
                    if var in self._sbuffer:
                        self._sbuffer[var][0] = self._sbuffer[var][0] + self._sbuffer[var][1]
                        self._sbuffer[var][1] = 0.0
                        self._sbuffer[var][0][self._off_network] = 0.0
                
                self._write_timestep(ti, nc, Q_2d)
                
    def _chemistry(self, Q_2d):
        
        volume_2d = (Q_2d / self.v) * self.dx * 1000  # litres
        mask = (np.isfinite(volume_2d)) & (volume_2d > 0) & (Q_2d > 1e-6)
        rows, cols = np.where(mask)
        valid_rows = rows.astype(np.intp)
        valid_cols = cols.astype(np.intp)
        num_valid = len(valid_rows)

        total_h_mol = (self._buffer["hydrogen_ion"][0] + self._sbuffer["hydrogen_ion"][0])
        total_vol = (Q_2d / self.v) * self.dx * 1000 * (1 + self.A_s_ratio)
        safe_vol = np.where(total_vol > 0, total_vol, np.inf)
        total_h_conc = total_h_mol / safe_vol
        capped_conc = np.minimum(total_h_conc, 1e4)
        with np.errstate(divide='ignore', invalid='ignore'):
            scaling = np.divide(capped_conc, total_h_conc, out=np.ones_like(total_h_conc), where=total_h_conc>0)
        self._buffer["hydrogen_ion"][0] *= scaling
        self._sbuffer["hydrogen_ion"][0] *= scaling

        process_chemistry(
            self._buffer["ferrous_iron"][0],
            self._buffer["ferric_iron"][0],
            self._buffer["sulphate"][0],
            self._buffer["hydrogen_ion"][0],
            self._buffer["iron_III_hydroxide"][0],
            self._buffer["bedload_storage"][0],
            self._ore_np,
            volume_2d,
            self._median_vol,
            self.do,
            self.buffer_capacity,
            self.time_step_seconds,
            valid_rows,
            valid_cols,
            num_valid
        )
        
    def _transport(self, t, Q_2d):
        """Transport chemistry downstream based on flow network, updating self._buffer with transported chemistry for next timestep
            uses transport_cython from transport.pyx 

        Parameters
        ----------
        t : np.datetime64
            timestep of model
        Q_2d : np.ndarray
            2d array of flow values at timestep t, used for transport calculations
        """
        def _run_transport_for_var(var):     
            Q_lat, C_lat = self._build_junction_inflows(Q_2d, var)
            
            work_arrays = self._cn_working_arrays_dict[var]
            if self._cn_working_arrays_dict is not None:
                _transport_cn(
                    self._buffer[var][0],
                    self._sbuffer[var][0],
                    Q_2d,
                    Q_lat,
                    C_lat,
                    self._median_vol,
                    self._reaches,
                    self._id_to_row,      
                    self._id_to_col,  
                    self.dx,    
                    self.a,
                    self.b,
                    self.c,
                    self.f,
                    self.time_step_seconds,
                    0.5,
                    0.5,
                    self.alpha_s,
                    self.A_s_ratio,
                    self.v,
                    1000,
                    work_arrays["a"],
                    work_arrays["b"],
                    work_arrays["c"],
                    work_arrays["d"],
                    work_arrays["c_prime"],
                    work_arrays["d_prime"],
                    work_arrays["x"],
                    work_arrays["rows"],
                    work_arrays["cols"],
                    work_arrays["V"],
                    work_arrays["v"],
                    work_arrays["A"],
                    work_arrays["D"],
                    self._max_reach_length
                    )
            else:
                for reach in self._reaches: 
                    hr = int(self._id_to_row[reach[0]])
                    hc = int(self._id_to_col[reach[0]])
                    Q_l = float(Q_lat[hr, hc])
                    if Q_l > 0.0:
                        m_in = Q_l * float(C_lat[hr, hc]) * self.time_step_seconds
                        self._buffer[var][0, hr, hc] += m_in

        ti = self._time_index[t]

        Parallel(n_jobs=4, require = "sharedmem")(
            delayed(_run_transport_for_var)(var) for var in self._transport_vars
        )

        mask = self._buffer["iron_III_hydroxide"][0] > 0
        src_rows, src_cols = np.where(mask)
        src_rows = src_rows.astype(np.int64)
        src_cols = src_cols.astype(np.int64)

        _transport_ad_dep(
            self._buffer["iron_III_hydroxide"],
            self._buffer["bedload_storage"],
            Q_2d,
            self._ID_grid,
            self.outID_grid,
            self._id_to_row,
            self._id_to_col,
            self._id_to_outid,
            ti,
            self.time_step_seconds,
            self.dx,
            self.a,
            self.b,
            self.c, 
            self.f,
            self.wf,
            self.v,
            1000,
            len(self.dataset.lat),
            len(self.dataset.lon),
            src_rows,
            src_cols,
            self._addep_working_arrays["dst_rows"],
            self._addep_working_arrays["dst_cols"],
            self._addep_working_arrays["valid_cell"],
            self._addep_working_arrays["vol_valid"],
            self._addep_working_arrays["has_next"]
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
        self._id_to_outid = np.full(array_size, -1, dtype=np.int32)

        remapped_ids = np.array([id_remap[i] for i in valid_ids.tolist()], dtype=np.int64)
        self._id_to_row[remapped_ids] = flat_rows[valid]
        self._id_to_col[remapped_ids] = flat_cols[valid]

        remapped_out = np.array(
            [id_remap.get(i, -1) for i in flat_out[valid].tolist()], dtype=np.int64
        )
        self._id_to_outid[remapped_ids] = remapped_out

        
        self._ID_grid = np.vectorize(lambda x: id_remap.get(x, -1))(
            id_vals).astype(np.int64)
        self.outID_grid = np.vectorize(lambda x: id_remap.get(x, -1))(
            out_vals).astype(np.int64)

        
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

        self._Q_np = self.dataset["Q"].values.astype(np.float32)

        self._ore_np = self.dataset["ore"].values.astype(np.float32)
        self._sink_mask = self.outID_grid < 0
        
        self._build_reaches()
        self._build_network_mask()
        self._off_network = ~self._network_mask

        if self._max_reach_length >= 1:
            self._cn_working_arrays_dict = {
                var: {
                "a": np.empty((self._max_reach_length,), dtype=np.float64),
                "b": np.empty((self._max_reach_length,), dtype=np.float64),
                "c": np.empty((self._max_reach_length,), dtype=np.float64),
                "d": np.empty((self._max_reach_length,), dtype=np.float64),
                "c_prime": np.empty((self._max_reach_length,), dtype=np.float64),
                "d_prime": np.empty((self._max_reach_length,), dtype=np.float64),
                "x": np.empty((self._max_reach_length,), dtype=np.float64),
                "rows": np.empty((self._max_reach_length,), dtype=np.int64),
                "cols": np.empty((self._max_reach_length,), dtype=np.int64),
                "V": np.empty((self._max_reach_length,), dtype=np.float64),
                "v": np.empty((self._max_reach_length,), dtype=np.float32),
                "A": np.empty((self._max_reach_length,), dtype=np.float64),
                "D": np.empty((self._max_reach_length,), dtype=np.float64)
                } for var in self._transport_vars
            }

        else:
            self._cn_working_arrays = None
        
        nlat, nlon = self.dataset.lat.size, self.dataset.lon.size
        self._addep_working_arrays = {
            "dst_rows": np.empty(nlat * nlon, dtype = np.int64),
            "dst_cols": np.empty(nlat * nlon, dtype = np.int64),
            "valid_cell": np.empty(nlat * nlon, dtype = np.int32),
            "vol_valid": np.empty(nlat * nlon, dtype = np.int32),
            "has_next": np.empty(nlat * nlon, dtype = np.int32)
        }

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
                elif var == "iron_III_hydroxide":
                    v.description = f"{attrs[var][1]} - suspended concentration at timestep in main channel"
                else:
                    v.description = f"{attrs[var][1]} - instant concentration at timestep in both storage zone and main channel (sum)" 

            ph_var = nc.createVariable(
                "pH", "f4",
                ("time", "lat", "lon"),
                chunksizes=(1, spatial_shape[0], spatial_shape[1]),
                zlib=True, complevel=4, fill_value=np.nan   
            )

            ph_var.units = "pH"
            ph_var.description = "pH value calculated from hydron concentration"

    def _write_timestep(self, ti, nc, Q_2d):
        """Writes chemistry results for timestep index ti from self._buffer to netCDF dataset nc at output_path
        Parameters
        ----------
        ti : int
            index of timestep
        nc : netCDF4.Dataset()
            the open netCDF dataset to write to, created in _create_output_file
        """
        A_vals = (Q_2d / self.v)
        A_vals = np.maximum(A_vals, 1e-6)  # m²
        V_cell = A_vals * self.dx # m³
        step_vol = V_cell * 1000 # litres, main storage volume

        storage_V = self.A_s_ratio * V_cell * 1000 # litres, storage zone storage volume

        with np.errstate(under='ignore', divide='ignore', invalid='ignore'):
            for var in self._chem_vars:

                # set sinks to 0 concentration, as the system is closed all chemistry piles here making it unreliable 
                mask = self._sink_mask
                self._buffer[var][0][mask] = 0
                if var in self._sbuffer:
                    self._sbuffer[var][0][mask] = 0
                
                if var == "bedload_storage":
                    nc.variables[var][ti, :, :] = (self._buffer[var][0] + self._buffer[var][1])
                elif var == "iron_III_hydroxide":
                    # output concentration in main channel, as precip is split into bedload storage and suspended
                    mol_amount = (self._buffer[var][0] + self._buffer[var][1])
                    concentration_molar = mol_amount / step_vol #+ storage_V)  # moles per litre
                    concentration_mg_per_L = concentration_molar * self.molar_masses[var]  # mg/L
                    concentration_ug_per_L = concentration_mg_per_L * 1000  # µg/L

                    nc.variables[var][ti, :, :] = concentration_ug_per_L.astype(np.float32)
                else:
                    # output concentration of total chem (storage + main channel) of dissolved chems
                    mol_amount = (self._buffer[var][0] + self._buffer[var][1] + self._sbuffer[var][0] + self._sbuffer[var][1])
                    concentration_molar = mol_amount / (step_vol + storage_V)  # moles per litre
                    concentration_mg_per_L = concentration_molar * self.molar_masses[var]  # mg/L
                    concentration_ug_per_L = concentration_mg_per_L * 1000  # µg/L
                    nc.variables[var][ti, :, :] = concentration_ug_per_L.astype(np.float32)

            h_mol = self._buffer["hydrogen_ion"][0] + self._buffer["hydrogen_ion"][1] + self._sbuffer["hydrogen_ion"][0] + self._sbuffer["hydrogen_ion"][1]
            h_conc = h_mol / (step_vol + storage_V)  # mol/L
            ph = np.where(h_conc > 0, -np.log10(np.maximum(h_conc, 1e-14)), np.nan)
            nc.variables["pH"][ti, :, :] = ph.astype(np.float32)

    def _get_volume(self, ti):
        return (self._Q_np[ti] / self.v) * self.dx * 1000
    
    def _build_reaches(self):
        """Builds reaches and junction network for transport,
        reach = headwater --> confluence, or confluence --> confluence, or headwater/confluence --> sink,
        junction = tail of reach --> head of downstream reach, or tail of reach --> sink
        confluence = cell with inflow cells > 1, headwater = cell with inflow cells = 0, sink = cell with outflow ID < 0
        """
        up_count = np.zeros(len(self._id_to_row), dtype=np.int32)
        for remap_id, out_id in enumerate(self._id_to_outid):
            if out_id >= 0:
                up_count[out_id] += 1

        headwaters  = np.where((up_count == 0) & (self._id_to_row >= 0))[0]
        confluences = np.where((up_count > 1)  & (self._id_to_row >= 0))[0]
        start_cells = np.concatenate([headwaters, confluences])

        reaches = []
        visited = set()
        for hw in start_cells:
            reach = []
            current = int(hw)
            while current >= 0 and current not in visited:
                if self._id_to_row[current] < 0:
                    break
                visited.add(current)
                reach.append(current)
                out = int(self._id_to_outid[current])
                if out < 0 or up_count[out] > 1:
                    break
                current = out
            if len(reach) >= 1:
                reaches.append(reach)

        # --- topological sort (Kahn's algorithm) ---
        # map each reach's head cell → reach index
        head_to_reach = {r[0]: i for i, r in enumerate(reaches)}

        # for each reach, which reach index is immediately downstream (-1 = none)
        downstream_of = []
        for reach in reaches:
            out_id = int(self._id_to_outid[reach[-1]])
            downstream_of.append(head_to_reach.get(out_id, -1))

        n = len(reaches)
        in_degree = [0] * n
        for d in downstream_of:
            if d >= 0:
                in_degree[d] += 1

        from collections import deque
        queue = deque(i for i in range(n) if in_degree[i] == 0)
        sorted_order = []
        while queue:
            i = queue.popleft()
            sorted_order.append(i)
            d = downstream_of[i]
            if d >= 0:
                in_degree[d] -= 1
                if in_degree[d] == 0:
                    queue.append(d)

        # guard against cycles (shouldn't occur in a valid river network)
        if len(sorted_order) < n:
            visited_set = set(sorted_order)
            sorted_order.extend(i for i in range(n) if i not in visited_set)

        self._reaches = [np.array(reaches[i], dtype = np.int64) for i in sorted_order]

        # --- junction cache: one entry per reach ---
        # each entry is (tail_r, tail_c, dst_r, dst_c) or None for sinks
        self._reach_junctions = []
        for reach in self._reaches:
            tail_id = reach[-1]
            tail_r  = int(self._id_to_row[tail_id])
            tail_c  = int(self._id_to_col[tail_id])
            out_id  = int(self._id_to_outid[tail_id])
            if out_id >= 0 and out_id < len(self._id_to_row):
                dst_r = int(self._id_to_row[out_id])
                dst_c = int(self._id_to_col[out_id])
                if dst_r >= 0 and dst_c >= 0:
                    self._reach_junctions.append((tail_r, tail_c, dst_r, dst_c))
                    continue
            self._reach_junctions.append(None)

        self._max_reach_length = max(len(r) for r in self._reaches)

        _valid = [j for j in self._reach_junctions if j is not None]

        if _valid:
            _arr = np.array(_valid, dtype=np.int32)  # shape (n, 4)
            self._junc_tail_r = np.ascontiguousarray(_arr[:, 0])
            self._junc_tail_c = np.ascontiguousarray(_arr[:, 1])
            self._junc_dst_r  = np.ascontiguousarray(_arr[:, 2])
            self._junc_dst_c  = np.ascontiguousarray(_arr[:, 3])
        else:
            self._junc_tail_r = np.empty(0, dtype=np.int32)
            self._junc_tail_c = np.empty(0, dtype=np.int32)
            self._junc_dst_r  = np.empty(0, dtype=np.int32)
            self._junc_dst_c  = np.empty(0, dtype=np.int32)
        self._n_junctions = len(_valid)

    def diagnose_reach_lengths(self):
        """Count and show how the reaches of the network are distributed, diagnosing tool"""
        up_count = np.zeros(len(self._id_to_row), dtype=np.int32)
        for remap_id, out_id in enumerate(self._id_to_outid):
            if out_id >= 0:
                up_count[out_id] += 1

        headwaters  = np.where((up_count == 0) & (self._id_to_row >= 0))[0]
        confluences = np.where((up_count > 1)  & (self._id_to_row >= 0))[0]
        start_cells = np.concatenate([headwaters, confluences])

        visited = set()
        length_counts = {}

        for hw in start_cells:
            reach = []
            current = int(hw)
            while current >= 0 and current not in visited:
                visited.add(current)
                reach.append(current)
                out = int(self._id_to_outid[current])
                if out < 0 or up_count[out] > 1:
                    break
                current = out
            n = len(reach)
            length_counts[n] = length_counts.get(n, 0) + 1

        for length, count in sorted(length_counts.items()):
            print(f"  length {length:>3}: {count:>6} reaches")

    def _build_junction_inflows(self, Q_2d, var):
        """
        Calls build_junction_inflows Cython function in transport.pyx,
        calculates the in/outflow from junctions as lateral in/outflow
        """
        _build_junction_inflows(
            self._buffer[var][0],
            Q_2d,
            self._Q_lat_buff_dict[var],
            self._C_lat_buff_dict[var],
            self._C_lat_num_buff_dict[var],
            self._junc_tail_r,
            self._junc_tail_c,
            self._junc_dst_r,
            self._junc_dst_c,
            self._n_junctions,
            self.v,
            self.dx,
            self.time_step_seconds,
        )
        return self._Q_lat_buff_dict[var], self._C_lat_buff_dict[var]
    
    def _build_network_mask(self):
        """Builds a 2d bolean mask of stream network downstream from mine cells, 
            used to set all buffers of cells outside the mask to 0
            stored as self._network_mask,
        """
        shape = (len(self.dataset.lat), len(self.dataset.lon))
        network_mask = np.zeros(shape, dtype = bool)

        source_rows, source_cols = np.where(self.dataset["ore"] > 0)
        queue = deque()
        visited = set()

        for r, c in zip(source_rows, source_cols):
            cell_id = int(self._ID_grid[r, c])
            if cell_id >= 0 and cell_id not in visited:
                visited.add(cell_id)
                network_mask[r, c] = True
                queue.append(cell_id)
        
        while queue:
            current_id = queue.popleft()
            out_id = int(self._id_to_outid[current_id])
            if out_id < 0 or out_id in visited:
                continue
            r = int(self._id_to_row[out_id])
            c = int(self._id_to_col[out_id])
            if r < 0 or c < 0:
                continue
            visited.add(out_id)
            network_mask[r, c] = True
            queue.append(out_id)
        
        self._network_mask = network_mask
        n_network = network_mask.sum()
        n_total = network_mask.size
        print(f"Network mask built: {n_network:,} / {n_total:,} cells on AMD network")