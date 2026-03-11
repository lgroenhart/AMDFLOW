# class py file for AMDFLOW
# contains main AMDModel class used for AMD modelling

import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


class AMDModel:

    def __init__(self, dataset, t_unit, do = 0.2500094):
        self.dataset = dataset.copy(deep=True)
        self.t_unit = t_unit
        self.time_steps = self.dataset["time"]
        self.do = do

        self.dataset = self.dataset.assign(ferrous_iron=xr.full_like(self.dataset.Q, 0))
        self.dataset = self.dataset.assign(ferric_iron=xr.full_like(self.dataset.Q, 0))
        self.dataset = self.dataset.assign(sulphate=xr.full_like(self.dataset.Q, 0))
        self.dataset = self.dataset.assign(hydrogen_ion=xr.full_like(self.dataset.Q, 0))
        self.dataset = self.dataset.assign(iron_III_hydroxide=xr.full_like(self.dataset.Q, 0))

        attrs_dict = {
        "ferrous_iron": {"units": "mol/timestep", "description": "Fe²⁺"},
        "ferric_iron": {"units": "mol/timestep", "description": "Fe³⁺"},
        "sulphate": {"units": "mol/timestep", "description": "SO₄²⁻"},
        "hydrogen_ion": {"units": "mol/timestep", "description": "H⁺"},
        "iron_III_hydroxide": {"units": "mol/timestep", "description": "Fe(OH)₃"}}

        for var_name, attrs in attrs_dict.items():
            self.dataset[var_name].attrs = attrs

        self.dataset = self.dataset.set_coords("ID")
        self.time_step_seconds = {"month": 2628000, "week" : 604800, "day": 86400}[self.t_unit]

        # init the hydrogen ion at a pH of 7: 10**-7 hydrogen ions per litre
        volume = self.dataset["Q"] * self.time_step_seconds * 1000  # L per timestep as m3/s * seconds per timestep * 1000
        self.dataset["hydrogen_ion"] = 1e-7 * volume

    def run(self):

        # get only cells where reactive ores are present
        mask_ores = self.dataset["ore"] > 0
        reactive_ores = self.dataset.where(mask_ores, drop=True)

        # get the most upstream cells (cells with no inflow) with source == 1
        # and ores
        mask = reactive_ores["source"].where(reactive_ores["source"] == 1)
        most_upstream_reactive_ores = self.dataset.where(mask, drop = True)

        # start timestep t
        for ti, t in tqdm(enumerate(self.dataset.time.values)):
            # add mass from previous timestep to current timestep
            if ti > 0:
                prev_t = self.time_steps.values[ti - 1]

                for var in ["ferrous_iron", "ferric_iron",
                            "hydrogen_ion", "sulphate",
                            "iron_III_hydroxide"]:
                    prev_vals = self.dataset[var].sel(time=prev_t).fillna(0)
                    self.dataset[var].loc[dict(time=t)] = \
                        self.dataset[var].loc[dict(time=t)].fillna(0) + prev_vals
                    
            dataset_t = self.dataset.sel(time = t)
            
            if ti > 0:
                current_slice = dataset_t.where(dataset_t["ID"].isin(most_upstream_reactive_ores["ID"].values), drop = True)
            else:
                current_slice = most_upstream_reactive_ores.sel(time = t)

            # processing step of most upstream cells with water and reactive ores at t 
            # --------------------------------------------------------------------------------------------
                   
            # check for water > 0
            mask = current_slice["Q"] > 0
            current_slice = current_slice.where(mask, drop=True)

            current_slice = self.process_slice(current_slice)

            # safety checks
            if current_slice.sizes == {}:
                print(f"Warning: Empty slice at time {t}, skipping update")
                continue  
            if "lon" not in current_slice.coords or "lat" not in current_slice.coords:
                current_slice = current_slice.set_coords(["lon", "lat"])

            self.update_dataset(t, current_slice)
           # transport only the processed cells
            if ti < len(self.time_steps) - 1:
                self.transport(t, current_slice)

            # ----------------------------------------------------------------------------------------------

            # loop to process downstream cells until no more downstream cells exist
            #-----------------------------------------------------------------------------------------------
            while current_slice["ID"].size > 0:
                
                # get next current slice
                out_ids = current_slice["outID"].values
                out_ids = out_ids[out_ids != -1]

                current_slice = dataset_t.where(dataset_t["ID"].isin(out_ids), drop = True)
                
                # process current slice cells 
                mask = current_slice["Q"] > 0
                current_slice = current_slice.where(mask, drop=True)

                current_slice = self.process_slice(current_slice)
                
                # safety_checks
                if current_slice.sizes == {}:
                    print(f"Warning: Empty slice at time {t}, skipping update")
                    continue  
                if "lon" not in current_slice.coords or "lat" not in current_slice.coords:
                    current_slice = current_slice.set_coords(["lon", "lat"])
                    
                self.update_dataset(t, current_slice)
                dataset_t = self.dataset.sel(time=t)
                if ti < len(self.time_steps) - 1:
                    self.transport(t, current_slice)
            # -----------------------------------------------------------------------------------------------

    def process_slice(self, current_slice):

        k = 10**-8.19
        do_term = self.do ** 0.5
        

       
        # 1) pyrite oxidation by ferric iron 
        mask_ferric = (current_slice["ferric_iron"] > 0) & (current_slice["ore"] > 0)
        mask_rate = (current_slice["ore"] > 0) & (~mask_ferric)

        ferric_consumed = xr.where(
            mask_ferric,
            current_slice["ferric_iron"],
            0
        )

        ferrous_produced = ferric_consumed * 1.07
        hydrogen_produced = xr.where(
            mask_ferric,
            ferric_consumed * 1.14,
            0
        )

        current_slice = current_slice.assign(
            ferric_iron=current_slice["ferric_iron"] - ferric_consumed,
            ferrous_iron=current_slice["ferrous_iron"] + ferrous_produced,
            hydrogen_ion=current_slice["hydrogen_ion"] + hydrogen_produced,
        )

        # 2) rate-limited pyrite oxidation 
        h_conc = xr.where(
            current_slice["Q"] * self.time_step_seconds * 1000 > 0,
            current_slice["hydrogen_ion"] / (current_slice["Q"] * self.time_step_seconds * 1000) ,
            1e-7                     
        )

        h_safe = xr.where(
            (h_conc <= 0) | h_conc.isnull(),
            1e-7,                
            h_conc
        )

        rate = k * (do_term) / (h_safe ** 0.11)

        reaction_amount = xr.where(
            mask_rate,
            rate * current_slice["ore"] * self.time_step_seconds,
            0
        )

        current_slice = current_slice.assign(
            ferrous_iron=current_slice["ferrous_iron"] + reaction_amount,
            sulphate=current_slice["sulphate"] + 2 * reaction_amount,
            hydrogen_ion=current_slice["hydrogen_ion"] + 2 * reaction_amount,
        )



        # 3) ferrous to ferric oxidation
        ferrous_available = current_slice["ferrous_iron"]
        current_slice = current_slice.assign(
            ferric_iron=current_slice["ferric_iron"] + ferrous_available,
            ferrous_iron=xr.zeros_like(current_slice["Q"]),   
            hydrogen_ion=current_slice["hydrogen_ion"] - 1 * ferrous_available,
        )

        # prevent negative hydrogen
        current_slice["hydrogen_ion"] = current_slice["hydrogen_ion"].clip(min=0)

        
        # 4) ferric <> iron III hydroxide equilibrium
        ferric = current_slice["ferric_iron"]
        hydroxide = current_slice["iron_III_hydroxide"]
        hydrogen_ion = current_slice["hydrogen_ion"]

        diff = ferric - hydroxide
        adjustment = 0.5 * diff

        current_slice = current_slice.assign(
            ferric_iron=ferric - adjustment,
            iron_III_hydroxide=hydroxide + adjustment,
            hydrogen_ion = hydrogen_ion + (adjustment * 3)
        )


        # 5) numerical cleanup

        for var in ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                    "sulphate", "iron_III_hydroxide"]:
            current_slice[var] = current_slice[var].fillna(0)
            current_slice[var] = current_slice[var].clip(min=0)

        return current_slice




    def update_dataset(self, t, current_slice):
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

        # Target grid (all lon/lat points of the main dataset)
        target_lon = self.dataset.lon.values
        target_lat = self.dataset.lat.values
        
        n_lon = len(target_lon)
        n_lat = len(target_lat)
        
        # Create a grid of target points
        # Use sparse=False to get full meshgrid, and ravel in 'C' order
        lon_grid, lat_grid = np.meshgrid(target_lon, target_lat, indexing='xy', sparse=False)
        target_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

        # Build KD‑Tree of target points
        tree = cKDTree(target_points)

        # For each source point, find the nearest target grid cell (index)
        distances, target_indices_flat = tree.query(src_points, k=1)  # shape (n_cells,)

        # Convert flat indices to 2D (lat, lon) indices
        # Since ravel order is (lon varies fastest), the conversion is:
        target_lon_idx = target_indices_flat % n_lon
        target_lat_idx = target_indices_flat // n_lon

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
            cell_ids = valid_target_lat_idx * n_lon + valid_target_lon_idx
            _, unique_idx = np.unique(cell_ids[::-1], return_index=True)
            unique_idx = len(cell_ids) - 1 - unique_idx

            # Extract final assignments
            final_lat_idx = valid_target_lat_idx[unique_idx]
            final_lon_idx = valid_target_lon_idx[unique_idx]
            final_vals = valid_src_vals[unique_idx]

            # Get time index
            time_idx = np.where(self.dataset.time.values == t)[0][0]

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

    def transport(self, t, current_slice):
        """Accumulate mass to downstream cells using vectorised operations."""
        key_vars = ["ferrous_iron", "ferric_iron", "hydrogen_ion", "sulphate"]
        next_time = self._next_time(t)
        if next_time is None:
            return

        # Filter source cells with valid outflow
        source = current_slice.where(current_slice["outID"] != -1, drop=True)
        if source.sizes["lat"] == 0:
            return

        # Convert source to DataFrame for fast groupby
        source_df = source[key_vars + ["outID"]].to_dataframe().reset_index()
        source_df = source_df.dropna(subset=["outID"])

        # Group by outID and sum each variable
        grouped = source_df.groupby("outID")[key_vars].sum()

        # Get target grid at next_time
        ds_next = self.dataset.sel(time=next_time)

        # Build mapping from ID to (lon, lat)
        target_df = ds_next[["ID", "lon", "lat"]].to_dataframe().reset_index()[["ID", "lon", "lat"]]
        target_df = target_df.set_index("ID")

        # Merge grouped sums with target coordinates
        merged = grouped.join(target_df, how="inner")
        if merged.empty:
            return

        # Prepare coordinate to index mappings
        lon_to_idx = {lon: i for i, lon in enumerate(self.dataset.lon.values)}
        lat_to_idx = {lat: i for i, lat in enumerate(self.dataset.lat.values)}
        time_idx = np.where(self.dataset.time.values == next_time)[0][0]

        # For each variable, add contributions
        for var in key_vars:
            sum_vals = merged[var].values
            lon_vals = merged["lon"].values
            lat_vals = merged["lat"].values

            # Convert coordinates to indices
            lon_idx = np.array([lon_to_idx[lon] for lon in lon_vals])
            lat_idx = np.array([lat_to_idx[lat] for lat in lat_vals])

            # Check dimension order before assignment
            dims = self.dataset[var].dims
            if dims == ('time', 'lat', 'lon'):
                # Current values
                current_vals = self.dataset[var].values[time_idx, lat_idx, lon_idx]
                current_vals = np.nan_to_num(current_vals, nan=0.0)
                
                # New values
                new_vals = current_vals + sum_vals
                
                # Assign back
                self.dataset[var].values[time_idx, lat_idx, lon_idx] = new_vals
                
            elif dims == ('time', 'lon', 'lat'):
                # Current values
                current_vals = self.dataset[var].values[time_idx, lon_idx, lat_idx]
                current_vals = np.nan_to_num(current_vals, nan=0.0)
                
                # New values
                new_vals = current_vals + sum_vals
                
                # Assign back
                self.dataset[var].values[time_idx, lon_idx, lat_idx] = new_vals
                
    def _next_time(self, t):
        idx = np.where(self.time_steps.values == t)[0][0]
        if idx + 1 >= len(self.time_steps):
            return None
        return self.time_steps.values[idx + 1]
    
    def output_calc(self):

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
        