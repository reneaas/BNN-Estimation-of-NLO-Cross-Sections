import os
import pyslha
import tqdm
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict


ids = ["1000022", "1000023", "1000025", "1000035", "1000024", "1000037"]
ids = [int(id) for id in ids]
blocks = ["MASS", "NMIX", "VMIX", "UMIX"]


def get_features(data, blocks: list[str], ids: Optional[list[str]] = None) -> dict:
    """Extracts features from a pyslha data object.

    Args:
        data                    : pyslha data object
        blocks (list)           : which blocks to extract.
        ids (list, optional)    : particle ids to extract.

    Returns:
        features (dict) :   dictionary containing the extracted features.
                            using blocks as its keys.
    """
    features = {}
    for block in blocks:
        if block == "MASS":
            mass = {}
            if ids is None:
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


def get_targets(data, col_idx: int = 8, block: str = "PROSPINO_OUTPUT") -> dict:
    """Extracts targets from a pyslha object.

    Args:
        data                :   pyslha data object
        col_idx (int)       :   which column in the slha file to extract.
        block (str)         :   defaults to PROSPINO_OUTPUT. Not subject to change.

    Returns:
        targets (dict)  : Dictionary of targets.
    """

    targets = {}
    for key in data.blocks[block].keys():
        if len(key) == 3:
            targets[key[:2]] = data.blocks[block][key][col_idx]
        else:
            targets[key[:2]] = -1
    return targets


def merge_list_of_dicts(dicts: list[dict]) -> dict:
    """Merges a list of dictionaries to a single dictionary.

    Args:
        dicts (list[dict])    :   list of dictionaries to merge.

    Returns:
        merged_dict (dict)  :   Merged dictionary.

    """
    keys = dicts[0].keys()
    merged_dict = {key: [] for key in keys}
    for d in dicts:
        for key in keys:
            merged_dict[key].append(d.get(key))
    merged_dict = {
        str(key): np.asarray(merged_dict.get(key)) for key in merged_dict.keys()
    }
    return merged_dict


def sort_targets(targets):
    return merge_list_of_dicts(targets)


def get_data(
    root_dir: str, blocks: list[str], ids: list[str], col_idxs: list[int] = [8, 7]
) -> tuple[dict]:
    """Extracts data from a set of .slha files.

    Args:
        root_dir (str):
            root directory of .slha files.
        blocks (list):
            blocks in .slha files to extract
        ids (list):
            particle ids to extract data for
        col_idx (list):
            column index of targets to extract in PROSPINO_OUTPUT block.

    Returns:
        features (dict)     :   dictionary with extracted features
        targets (dict)      :   dictionary with extracted targets

    """

    # Extract number of files in the root ri
    os.system(f"ls {root_dir}/*.slha | wc -l > num_files.txt")
    with open("num_files.txt", "r") as infile:
        lines = infile.readlines()
        num_files = int(lines[0].split()[0])

    features = {block: [] for block in blocks}
    targets = {col_idx: [] for col_idx in col_idxs}

    with os.scandir(root_dir) as it:
        for entry in tqdm.tqdm(it, total=num_files):
            if entry.name.endswith("slha") and entry.is_file():
                data = pyslha.read(entry.path)
                new_feat = get_features(data, blocks, ids)
                features = {
                    block: features[block] + [new_feat[block]] for block in blocks
                }
                for col_idx in targets.keys():
                    targets[col_idx].append(get_targets(data, col_idx))
        features = {key: np.asarray(features[key]) for key in features.keys()}
        for col_idx in targets.keys():
            targets[col_idx] = sort_targets(targets[col_idx])
        features = merge_features(features, blocks)
    return features, targets


def merge_features(features: dict, blocks: list[str]) -> dict:
    """Merge features with respect to the blocks in the .slha files.

    Args:
        features (dict):   
            dictionary containing features to merge.
        blocks (list[str]):   
            List of blocks to merge.

    Returns:
        Merged dictionary
    """

    for block in blocks:
        if block == "MASS":
            features[block] = merge_list_of_dicts(features[block])
        else:
            shape = features[block].shape
            if int(shape[1]) != 1:
                features[block] = features[block].reshape(
                    shape[0], int(np.sqrt(shape[1])), int(np.sqrt(shape[1]))
                )
            else:
                features[block] = features[block].reshape(shape[0], 1)
    return features


def save_targets(targets: dict) -> None:
    """Saves targets to a set of .npz file using the keys as filenames.

    Args:
        targets (dict)  :   Dictionary of targets

    """

    for col_idx in targets.keys():
        targets[col_idx] = pd.DataFrame(targets[col_idx])
        cols = targets[col_idx].columns
        arrs = {col: targets[col_idx][col] for col in cols}
        np.savez_compressed(f"targets_col_{col_idx}.npz", **arrs)
    del targets


def sort_nmix(features: dict) -> dict:
    """Sort NMIX according to particle ids

    Args:
        features (dict) :   dictionary of features

    Returns:
        features (dict) :   dictionary of features
                            with NMIX sorted.
    """
    neutralinos = [
        "1000022",
        "1000023",
        "1000025",
        "1000035",
    ]
    nmix_dict = {}
    for i, id in enumerate(neutralinos):
        nmix_dict[id] = features["NMIX"][:, i, :]
    features["NMIX"] = nmix_dict
    return features


def sort_vmix_umix(features: dict) -> dict:
    """Sort VMIX and UMIX according to particle ids

    Args:
        features (dict) :   dictionary of features

    Returns:
        features (dict) :   dictionary of features
                            with UMIX and VMIX sorted.
    """

    charginos = ["1000024", "1000037"]

    vmix_dict = {}
    umix_dict = {}
    for i, id in enumerate(charginos):
        vmix_dict[id] = features["VMIX"][:, i, :]
        umix_dict[id] = features["UMIX"][:, i, :]
    features["VMIX"] = vmix_dict
    features["UMIX"] = umix_dict
    return features


def save_features(features: dict) -> None:
    # Store masses
    features["MASS"] = pd.DataFrame(features["MASS"])
    features["MASS"].to_pickle("../features/mass.pkl")

    # Store NMIX matrices
    for key in features["NMIX"].keys():
        features["NMIX"][key] = pd.DataFrame(
            features["NMIX"][key], columns=[f"{key},N{i}" for i in range(1, 5)]
        )
    for key in features["NMIX"].keys():
        path = f"../features/nmix_{key}.pkl"
        features["NMIX"][key].to_pickle(path)

    # Store UMIX matrices
    for key in features["UMIX"].keys():
        features["UMIX"][key] = pd.DataFrame(
            features["UMIX"][key], columns=[f"{key},U{i}" for i in range(1, 3)]
        )
        features["VMIX"][key] = pd.DataFrame(
            features["VMIX"][key], columns=[f"{key},V{i}" for i in range(1, 3)]
        )

    for key in features["UMIX"].keys():
        path = f"../features/umix_{key}.pkl"
        features["UMIX"][key].to_pickle(path)

        path = f"../features/vmix_{key}.pkl"
        features["VMIX"][key].to_pickle(path)


def save_targets(targets: dict) -> None:
    for key in targets.keys():
        targets[key] = pd.DataFrame(targets[key])

    targets[7].to_pickle("../targets/nlo.pkl")
    targets[8].to_pickle("../targets/nlo_rel_err.pkl")


blocks = ["MASS", "NMIX", "VMIX", "UMIX", "MINPAR"]
ids = [1000022, 1000024, 1000023, 1000025, 1000035, 1000037]  # particle ids
path = "../dataset/EWonly"
# path = "./dummy_data"
# path = "./dummy_data/1_317_1.slha"
# data = pyslha.read(path)
# print(data.blocks["PROSPINO_OUTPUT"])

##########################################################################
# Parse data and write it to file
##########################################################################

features, targets = get_data(path, blocks, ids=None)
features = sort_nmix(features)
features = sort_vmix_umix(features)
save_features(features)
save_targets(targets)
print(targets)


# del features["VMIX"]
# del features["UMIX"]


# Store targets


##########################################################################
# Loads data from numpy zip file.
##########################################################################


# features = load_features()
# targets = load_targets(8)
# print(targets[8].files)
# print(targets[8].get("(1000022, 1000022)"))
#
# print(features)


##########################################################################
# Loads data from pickle
##########################################################################

# df_nmix = pd.read_pickle("./features/nmix_1000022.pkl")
# print(type(df_nmix))
# df_mass = pd.read_pickle("./features/mass.pkl")
# df_mass1 = pd.DataFrame(df_mass["1000022"])
# print(pd.concat([df_mass1, df_nmix], axis=1))
#
#
# df_nmix1 = pd.read_pickle("./features/nmix_1000022.pkl")
# df_nmix2 = pd.read_pickle("./features/nmix_1000025.pkl")
# df_mass1 = pd.DataFrame(df_mass["1000022"])
# df_mass2 = pd.DataFrame(df_mass["1000025"])
#
# df = pd.concat([df_mass1, df_mass2, df_nmix1, df_nmix2], axis=1)
# print(df)
# print(df.to_numpy())
# print(df.to_numpy().shape)
