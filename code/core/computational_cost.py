import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import re
import pandas as pd

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


root_dir = "./results"
data = get_data(root_dir=root_dir)


# max_len = -1
# all_keys = None
# for d in data:
#     if len(d.keys()) > max_len:
#         max_len = len(d.keys())
#         all_keys = list(d.keys())
# merged_data = {key: [] for key in all_keys}
# for d in data:
#     for key in d:
#         merged_data[key].append(d.get(key))
# print(merged_data)



