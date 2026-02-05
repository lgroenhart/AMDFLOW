import openmindat
import os
import csv

# set API key as user input
API_KEY = input()
os.environ["MINDAT_API_KEY"] = API_KEY

from openmindat import GeomaterialRetriever

material_id = str(3314)
# get and save pyrite (3314) data
gr = GeomaterialRetriever()
gr.expand("locality").id_in(material_id)
gr.saveto("../data/mindat_data")


import json 

# open pyrite json as dict
with open("../data/mindat_data/v1_geomaterials.json", "r") as file:
    pyrite_dict = json.load(file)

from openmindat import LocalitiesRetriever

# get pyrite localities from dict 
pyrite_locs = [str(r) for r in pyrite_dict["results"][0]["locality"]]

country = "Portugal"

# loop over batch size batches of pyrite locations and get their data
# save to total data list
batch_size = 100
total_data = []
counter = 0
for i in range(0, len(pyrite_locs), batch_size):
    batch_ids = pyrite_locs[i:i + batch_size]
    id_str = ",".join(batch_ids)
    lr = LocalitiesRetriever()
    lr.id_in(id_str).country(country).fields("longitude,latitude,guid,txt")
    batch_data = lr.get_dict()
    total_data.extend(batch_data["results"])
    print(f"batch {counter} of {len(pyrite_locs) / batch_size} done")
    counter += 1


# output data
output_data = {
    "count": len(total_data),
    "next": None,
    "previous": None,
    "results": total_data
}

# Save to file
with open(f"../data/mindat_data/v1_localities_{country}_.json", "w") as f:
    json.dump(output_data, f, indent=2)




