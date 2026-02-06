from openmindat import LocalitiesRetriever, GeomaterialRetriever
import os
import json
import pandas as pd
import re

def main(region, material_id = 3314, mineral_strings = "(Fe|S)", material_name = "pyrite"):

    # set API key
    with open("mindat_API_key.txt") as f:
        key = f.read()
        os.environ["MINDAT_API_KEY"] = key

    # try opening material file containing worldwide material localities
    try:
        with open(f"../data/mindat_data/v1_geomaterials_{material_id}/v1_geomaterials.json") as file:
            material = json.load(file)

    # if no file available get data from mindat and save and load to json in data
    except:
        gr = GeomaterialRetriever()
        gr.expand("locality").id_in(material_id)
        gr.saveto(f"../data/mindat_data/v1_geomaterials_{material_id}")
        with open(f"../data/mindat_data/v1_geomaterials_{material_id}/v1_geomaterials.json") as file:
            material = json.load(file)

    # try to open region file containing region localities
    try:
        with open(f"../data/mindat_data/localities_{region}/v1_localities.json", "r") as f:
            region_json = json.load(f)
            # return if file/data already exists
            print("File already exists, delete if you want new data")
            return
    
    # if no file avaible get data from mindat and save/load as json in data
    except:
        # get region specific localities and save and load as json in data
        lr = LocalitiesRetriever()
        lr.country(region).page_size(100) # explicit page_size to circumvent brocli errors
        lr.saveto(f"../data/mindat_data/localities_{region}")
        with open(f"../data/mindat_data/localities_{region}/v1_localities.json") as file:
            region_json = json.load(file)

    # start filtering data for strings in mineral_strings and mine|mining|quarry
    df_region = pd.json_normalize(region_json['results'])
    df_region = df_region[
    # check for mine/mining/quarry in description (case insensitive)
    df_region["description_short"].str.contains(r"\b(mine|mining|quarry)\b", 
                                        case=False, na=False, regex=True) &
    # check for Fe or S in elements string
    df_region["elements"].str.contains(f'{mineral_strings}', na=False, regex=True)
                            ]
    # filter region dataframe localities by only keeping the ids that are also in the material localities
    df_material_ids = pd.DataFrame(material["results"][0]["locality"], columns = ["id"])
    ids_to_keep = df_material_ids["id"].unique() 
    df_region_material = df_region[df_region["id"].isin(ids_to_keep)]
    df_region_material = df_region_material[["id", "latitude", "longitude"]]
    df_region_material.to_csv(f"../data/mindat_data/{region}_{material_name}.csv")
    print(f"\nSuccesfully saved data to: ../data/mindat_data/{region}_{material_name}.csv")
    return

if __name__ == "__main__":
    main(region = "Portugal")
        

    
