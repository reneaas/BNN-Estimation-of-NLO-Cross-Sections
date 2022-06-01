from typing import Optional, Tuple, List, Dict
import os


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import tqdm
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from slha_loader import SLHALoader



def get_data(root_dir: str) -> Tuple[Dict]:
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

    l = 0
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.name.endswith("slha") and entry.is_file():
                print(entry.name)
                l += 1
                if l == 5:
                    break
    return None


def main():
    # get_data(root_dir="../dataset/EWonly")

    target_dir = "./targets"
    feat_dir = "./features"
    dl = SLHALoader(
        particle_ids=["1000022"] * 2,
        feat_dir=feat_dir,
        target_dir=target_dir,
        target_keys=["nlo"],
    )

    print(dl.features)

    target_dir = "../targets"
    feat_dir = "../features"
    dl = SLHALoader(
        particle_ids=["1000022"] * 2,
        feat_dir=feat_dir,
        target_dir=target_dir,
        target_keys=["nlo"],
    )

    print(dl.features)




if __name__ == "__main__":
    main()




