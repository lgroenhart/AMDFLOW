# class py file for AMDFLOW
# contains main AMDModel class used for AMD modelling

import numpy as np
import xarray as xr
import pandas as pd

class AMDModel:

    def __init__(self, dataset, t_unit, do = 10, h_plus = 1e-7):
        self.dataset = dataset.copy(deep=True)
        self.t_unit = t_unit
        self.time_steps = self.dataset["time"]
        self.do = do
        self.h_plus = h_plus

        self.dataset = self.dataset.assign(ferrous_iron=xr.full_like(self.dataset.Q, 0))
        self.dataset = self.dataset.assign(ferric_iron=xr.full_like(self.dataset.Q, 0))
        self.dataset = self.dataset.assign(sulphate=xr.full_like(self.dataset.Q, 0))
        self.dataset = self.dataset.assign(hydrogen_ion=xr.full_like(self.dataset.Q, 0))
        self.dataset = self.dataset.assign(iron_III_hydroxide=xr.full_like(self.dataset.Q, 0))

        attrs_dict = {
        'ferrous_iron': {'units': 'mol/timestep', 'description': 'Fe²⁺'},
        'ferric_iron': {'units': 'mol/timestep', 'description': 'Fe³⁺'},
        'sulphate': {'units': 'mol/timestep', 'description': 'SO₄²⁻'},
        'hydrogen_ion': {'units': 'mol/timestep', 'description': 'H⁺'},
        'iron_III_hydroxide': {'units': 'mol/timestep', 'description': 'Fe(OH)₃'}}

        for var_name, attrs in attrs_dict.items():
            self.dataset[var_name].attrs = attrs

        self.dataset = self.dataset.set_coords("ID")
        self.time_step_seconds = {"month": 2628000, "week" : 604800}[self.t_unit]
    def run(self):

        # get the most upstream cells (cells with no inflow)
        mask = xr.apply_ufunc(pd.isna, self.dataset["inID"], vectorize = True, dask = "parallelized", output_dtypes=[bool]) # redo with "source" bolean band
        mask_np = pd.isna(self.dataset["inID"].values)
        mask = xr.DataArray(mask_np, dims = self.dataset["inID"].dims, coords = self.dataset["inID"].coords)
        most_upstream = self.dataset.where(mask, drop = True)
        
        # get only most upstream cells where reactive ores are present
        mask_ores = most_upstream["ore"] > 0
        most_upstream_reactive_ores = most_upstream.where(mask_ores, drop=True)

        # start timestep t
        for ti, t in enumerate(self.time_steps.values):

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
            if 'i' not in current_slice.coords or 'j' not in current_slice.coords:
                current_slice = current_slice.set_coords(['i', 'j'])

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
                if 'i' not in current_slice.coords or 'j' not in current_slice.coords:
                    current_slice = current_slice.set_coords(['i', 'j'])
                    
                self.update_dataset(t, current_slice)
                dataset_t = self.dataset.sel(time=t)
                if ti < len(self.time_steps) - 1:
                    self.transport(t, current_slice)
            # -----------------------------------------------------------------------------------------------

    def process_slice(self, current_slice):

        k = 10**-8.19
        do_term = self.do * 0.5
        h_background = self.h_plus

       
        # 1) pyrite oxidation by ferric iron 
        mask_ferric = (current_slice["ferric_iron"] > 0) & (current_slice["ore"] > 0)
        mask_rate = (current_slice["ore"] > 0) & (~mask_ferric)

        ferric_consumed = xr.where(
            mask_ferric,
            current_slice["ferric_iron"] * 1.07,
            0
        )

        ferrous_produced = ferric_consumed
        hydrogen_produced = xr.where(
            mask_ferric,
            current_slice["ferric_iron"] * 1.14,
            0
        )

        current_slice = current_slice.assign(
            ferric_iron=current_slice["ferric_iron"] - ferric_consumed,
            ferrous_iron=current_slice["ferrous_iron"] + ferrous_produced,
            hydrogen_ion=current_slice["hydrogen_ion"] + hydrogen_produced,
        )

        # 2) rate-limited pyrite oxidation 
        effective_h = xr.where(
            (current_slice["hydrogen_ion"] <= 0) | current_slice["hydrogen_ion"].isnull(),
            h_background,
            current_slice["hydrogen_ion"]
        )

        rate = k * do_term / (effective_h ** 0.01)

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

        diff = ferric - hydroxide
        adjustment = 0.5 * diff

        current_slice = current_slice.assign(
            ferric_iron=ferric - adjustment,
            iron_III_hydroxide=hydroxide + adjustment,
        )


        # 5) numerical cleanup

        for var in ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                    "sulphate", "iron_III_hydroxide"]:
            current_slice[var] = current_slice[var].fillna(0)
            current_slice[var] = current_slice[var].clip(min=0)

        return current_slice


    def update_dataset(self, t, current_slice):
        """Update main dataset using stacked cell iteration (handles any grid shape)."""
        key_vars = ["ferrous_iron", "ferric_iron", "hydrogen_ion",
                    "sulphate", "iron_III_hydroxide"]

        if current_slice.sizes.get('i', 0) == 0 or current_slice.sizes.get('j', 0) == 0:
            return

        stacked = current_slice.stack(cell=('i', 'j'))
        stacked = stacked.dropna(dim='cell', subset=key_vars, how='any')
        n_cells = stacked.sizes.get('cell', 0)
        if n_cells == 0:
            return

        i_vals = stacked['i'].values   # shape (n_cells,)
        j_vals = stacked['j'].values

        for idx in range(n_cells):
            i_val = int(i_vals[idx])
            j_val = int(j_vals[idx])

            for var in key_vars:
                if var in stacked.data_vars:
                    val_array = stacked[var].values
                    # Handle both 0‑d (scalar) and 1‑d cases
                    if val_array.ndim == 0:
                        val = val_array.item()          # scalar value applies to all cells
                    else:
                        val = val_array[idx]            # per‑cell value

                    if not np.isnan(val) and val > 0:
                        try:
                            self.dataset[var].loc[dict(i=i_val, j=j_val, time=t)] = val
                        except Exception as e:
                            print(f"Warning: Could not assign {var} at (i={i_val}, j={j_val}, t={t}): {e}")

    def transport(self, t, current_slice):
        next_time = self._next_time(t)
        if next_time is None:
            return

        # remove cells with no downstream
        valid = current_slice["outID"] != -1
        source = current_slice.where(valid, drop=True)
        if source["ID"].size == 0:
            return

        # group contributions by downstream ID
        grouped = source.groupby("outID")
        for downstream_id, group in grouped:
            if downstream_id == -1:
                continue
            downstream_id = int(downstream_id)

            # find downstream cell at next timestep
            dataset_next = self.dataset.sel(time=next_time)
            dataset_next["ID"] = dataset_next["ID"].astype(int)
            target = dataset_next.where(dataset_next["ID"] == downstream_id, drop=True)
            if target["ID"].size == 0:
                continue

            i = int(target["i"].values[0])
            j = int(target["j"].values[0])

            for var in ["ferrous_iron", "ferric_iron", "hydrogen_ion", "sulphate"]:
                # added_mass = group[var].sum().values
                # if added_mass > 0:
                #     self.dataset[var].loc[dict(i=i, j=j, time=next_time)] += float(added_mass)
                added_mass = group[var].sum(skipna=True).values.item()
                if not np.isnan(added_mass) and added_mass > 0:
                    current_val = self.dataset[var].loc[dict(i=i, j=j, time=next_time)]
                    if np.isnan(current_val):
                        current_val = 0.0
                    self.dataset[var].loc[dict(i=i, j=j, time=next_time)] = current_val + added_mass

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
                self.dataset[var].attrs['units'] = 'g'
        
        # calculate volume in liters for each cell and timestep
        volume = self.dataset['Q'] * self.time_step_seconds * 1000  # L
        volume.attrs = {'units': 'L', 'description': 'Water volume per timestep'}
        
        # convert to concentration (g/L) except hydrogen_ion
        for var in molar_masses.keys():
            if var == "hydrogen_ion":
                continue
            if var in self.dataset.data_vars:
                # avoid division by zero: replace zero volume with NaN or 0
                conc = xr.where(volume > 0, self.dataset[var] / volume, 0)
                self.dataset[var] = conc
                self.dataset[var].attrs['units'] = 'g/L'
        
        # compute pH from H⁺ concentration (mol/L)
        if "hydrogen_ion" in self.dataset.data_vars:
            # H⁺ in mol/L = (H⁺ moles) / volume
            h_conc = xr.where(volume > 0, self.dataset["hydrogen_ion"] / volume, np.nan)
            # pH = -log10([H⁺]), clip to avoid log of zero/negative
            pH = -np.log10(h_conc.where(h_conc > 0, np.nan))
            self.dataset["pH"] = pH
            self.dataset["pH"].attrs = {'units': 'pH', 'description': 'pH value'}
            self.dataset["hydrogen_ion"] = h_conc
            self.dataset["hydrogen_ion"].attrs['units'] = 'mol/L'
        