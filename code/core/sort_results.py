import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import re
import pandas as pd
import json

def get_data(root_dir: str):
    # pattern = r"(\[.+?\]|[0-9]+[.]?[0-9]*|\w+)"
    pattern = r"(\[.+?\]|\d+\.?\d*|\w+)"
    data = []
    with os.scandir(root_dir) as iter:
        for entry in iter:
            if entry.name.endswith(".txt") and entry.is_file():
                with open(entry.path, "r") as infile:
                    keys = infile.readline()
                    keys = keys.split()
                    vals = re.findall(pattern, infile.readline())
                    tmp_data = {key: val for key, val in zip(keys, vals)}
                    data.append(tmp_data)
    return data


# root_dir = "./results"
# data = get_data(root_dir=root_dir)
# with open("./results/all_results.json", "w") as outfile:
#     json.dump(data, outfile)

# fname = "./results/all_results.json"
fname = "./results/all_results_formatted.json"
with open(fname, "r") as infile:
    data = json.load(infile)
print(data[-1])

merged_dict = {key: [] for key in data[-1].keys()}
for d in data:
    for key in merged_dict:
        merged_dict[key].append(d.get(key))
    
for key in merged_dict:
    print(len(merged_dict.get(key)))

fname = "./results/results_merged.json"
with open(fname, "w") as outfile:
    json.dump(merged_dict, outfile)

df = pd.DataFrame(merged_dict)
print(df)
df.to_pickle(fname.replace(".json", ".pkl"))





# fname = "./results/all_results_formatted.json"
# with open(fname, "w") as outfile:
#     json.dump(data, outfile)


# for d in data:
#     d["layers"] = eval(d.get("layers"))


# for d in data:
#     if d.get("accept_ratio") is not None:
#         accept_ratio = eval(d.get("accept_ratio"))
#         d["accept_ratio"] = sum(accept_ratio)

# for d in data:
#     # layers = list(d.get("layers"))
#     layers = eval(d.get("layers"))
#     # print(d.get("layers"))
#     num_params = 0
#     for a, b, in zip(layers[:-1], layers[1:]):
#         num_params += a * b + b
#     d["num_params"] = num_params



