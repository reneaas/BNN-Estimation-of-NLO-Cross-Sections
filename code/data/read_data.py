import sys
import os
import pyslha
import tqdm
import numpy as np
import pandas as pd
import collections


ids = ["1000022", "1000023", "1000025", "1000035", "1000037"]
ids = [int(id) for id in ids]


blocks = ["MASS", "NMIX", "VMIX", "UMIX"]

def get_features(data, blocks, ids):
    features = {}
    for block in blocks:
        if block == "MASS":
            mass = {}
            for id in ids:
                mass[id] = data.blocks[block][id]
            features[block] = mass
        else:
            features[block] = []
            for key in data.blocks[block].keys():
                features[block].append(data.blocks[block][key])
    return features

def get_targets(data, col_idx = 8, block = "PROSPINO_OUTPUT", excl_id = 0, with_keys = True):
    targets = {}
    if with_keys == False:
        targets = []
        for key in data.blocks[block].keys():
            if not excl_id in key and not -excl_id in key:
                if len(key) == 3:
                    targets.append(data.blocks[block][key][col_idx])
                else:
                    targets.append(-1)
        return targets

    else:
        targets = {}
        for key in data.blocks[block].keys():
            if not excl_id in key and not -excl_id in key:
                if len(key) == 3:
                    targets[key[:2]] = data.blocks[block][key][col_idx]
                else:
                    targets[key[:2]] = -1
        return targets


def sort_targets(targets):
    keys = targets[0].keys()
    sorted_targets = {key : [] for key in keys}
    for d in targets:
        for key in keys:
            sorted_targets[key].append(d.get(key))
    sorted_targets = {str(key) : np.asarray(sorted_targets[key]) for key in sorted_targets.keys()}
    return sorted_targets



def get_data(root_dir, blocks, ids, col_idxs = [8, 9]):

    # Extract number of files in the root ri
    if not os.path.exists(".num_files.txt"):
        os.system(f"ls {root_dir}/*.slha | wc -l >> num_files.txt")
    with open("num_files.txt", "r") as infile:
        lines = infile.readlines()
        num_files = int(lines[0].split()[0])

    features = {block : [] for block in blocks}
    targets = {col_idx : [] for col_idx in col_idxs}

    with os.scandir(root_dir) as it:
        for entry in tqdm.tqdm(it, total=num_files):
            if entry.name.endswith("slha") and entry.is_file():
                data = pyslha.read(entry.path)
                new_feat = get_features(data, blocks, ids)
                features = {block : features[block] + [new_feat[block]] for block in blocks}
                for col_idx in targets.keys():
                    targets[col_idx].append(get_targets(data, col_idx))
        features = {key : np.asarray(features[key]) for key in features.keys()}
        for col_idx in targets.keys():
            targets[col_idx] = sort_targets(targets[col_idx])
    return features, targets

def load_targets(col_idxs):
    col_idxs = [8, 9]
    targets = {}
    processes = {}
    for col_idx in col_idxs:
        targets[col_idx] = np.load(f"targets_col_{col_idx}.npz")
    return targets

blocks = ["MASS", "NMIX", "VMIX", "UMIX"]
ids = [1000022, 1000024, 1000023, 1000025, 1000035, 1000037] #particle ids
path = "./EWonly"
filename = "./EWonly/1_3_1.slha"
data = pyslha.read(filename)
# # print(data.blocks["PROSPINO_OUTPUT"].keys())
# targets1 = get_targets(data)
# filename2 = "./EWonly/1_4_1.slha"
# targets2 = get_targets(data)
# targets = [targets1, targets2]
#
# # print(targets)
# targets = sort_targets(targets)
# print(targets)
# df_targets = pd.DataFrame(targets)
# print(df_targets)

# print(data.blocks["PROSPINO_OUTPUT"])
# print(targets)
# print(targets)
# features = get_features(data, blocks, ids)
# print(features["MASS"])
# print("*"*20)
# print(features["NMIX"])
# print("*"*20)
# print(features["UMIX"])
# print("*"*20)
# print(features["VMIX"])



##########################################################################
# Reads data from file and writes it to numpy compressed zip file.
##########################################################################


features, targets = get_data(path, blocks, ids)
#Sort mass dictionary.
print(features["MASS"])
for i in range(len(features["MASS"])):
    features["MASS"][i] = {key : [features["MASS"][i][key]] for key in features["MASS"][i].keys()}
features["MASS"] = [pd.DataFrame(features["MASS"][i]) for i in range(len(features["MASS"]))]
features["MASS"] = pd.concat(features["MASS"], ignore_index = True)
print("--"*30)
print(features["MASS"])

# targets = {}
# for col_idx in targets.keys():
#     targets[col_idx] = pd.DataFrame(targets[col_idx])
#     cols = targets[col_idx].columns
#     arrs = {col : targets[col_idx][col] for col in cols}
#     np.savez_compressed(f"targets_col_{col_idx}.npz", **arrs)




##########################################################################
# Loads data from numpy zip file.
##########################################################################


# targets = load_targets(8)
# print(targets[8])
