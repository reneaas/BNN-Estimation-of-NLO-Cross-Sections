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

def get_features(data, blocks, ids = None):
    features = {}
    for block in blocks:
        if block == "MASS":
            mass = {}
            if ids == None:
                for id in data.blocks[block].keys():
                    mass[id] = data.blocks[block][id]
                features[block] = mass
            else:
                for id in ids:
                    mass[id] = data.blocks[block][id]
                features[block] = mass
        else:
            features[block] = []
            for key in data.blocks[block].keys():
                features[block].append(data.blocks[block][key])
    return features

def get_features2(data, blocks, ids):
    features = {}
    for block in blocks:
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


def sort_list_of_dicts(dicts):
    keys = dicts[0].keys()
    sorted_dict = {key : [] for key in keys}
    for d in dicts:
        for key in keys:
            sorted_dict[key].append(d.get(key))
    sorted_dict = {str(key) : np.asarray(sorted_dict.get(key)) for key in sorted_dict.keys()}
    return sorted_dict



def sort_targets(targets):
    return sort_list_of_dicts(targets)



def get_data(root_dir, blocks, ids, col_idxs = [8, 9]):

    # Extract number of files in the root ri
    os.system(f"ls {root_dir}/*.slha | wc -l > num_files.txt")
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
        features = sort_features(features, blocks)
    return features, targets

def load_targets(col_idxs):
    col_idxs = [8, 9]
    targets = {}
    processes = {}
    for col_idx in col_idxs:
        targets[col_idx] = np.load(f"targets_col_{col_idx}.npz")
    return targets



##########################################################################
# Reads data from file and writes it to numpy compressed zip file.
##########################################################################


def sort_features(features, blocks):
    for block in blocks:
        if block == "MASS":
            features[block] = sort_list_of_dicts(features[block])
        else:
            shape = features[block].shape
            features[block] = features[block].reshape(shape[0], int(np.sqrt(shape[1])), int(np.sqrt(shape[1])))
    return features


def save_targets(targets):
    #targets = {}
    for col_idx in targets.keys():
        targets[col_idx] = pd.DataFrame(targets[col_idx])
        cols = targets[col_idx].columns
        arrs = {col : targets[col_idx][col] for col in cols}
        np.savez_compressed(f"targets_col_{col_idx}.npz", **arrs)
    del targets

def save_features(features):
    np.savez_compressed("mass.npz", **features["MASS"])
    del features["MASS"]
    np.savez_compressed("feat_no_mass.npz", **features)
    del features


def load_features():
    mass = np.load("mass.npz")
    feat_no_mass = np.load("feat_no_mass.npz")

    features = {}
    features["MASS"] = { key : mass[key] for key in mass.files}
    for key in feat_no_mass.files:
        features[key] = feat_no_mass[key]
    return features



blocks = ["MASS", "NMIX", "VMIX", "UMIX", "MINPAR"]
ids = [1000022, 1000024, 1000023, 1000025, 1000035, 1000037] #particle ids
path = "./EWonly"
# path = "./dummy_data"




# features, targets = get_data(path, blocks, ids = None)
# save_targets(targets)
# save_features(features)



##########################################################################
# Loads data from numpy zip file.
##########################################################################


features = load_features()
targets = load_targets(8)
# print(targets[8].files)
# print(targets[8].get("(1000022, 1000022)"))

print(features)
