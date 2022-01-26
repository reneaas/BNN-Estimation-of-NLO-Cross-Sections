import pyslha
import os
import numpy as np
import pandas as pd

class SLHAReader(object):
    """docstring for SLHAReader."""

    def __init__(self, root_dir=None, blocks=None):
        super(SLHAReader, self).__init__()
        self.root_dir = root_dir
        
        if blocks is None:
            self.blocks = [
                "MASS", 
                "NMIX",
                "VMIX",
                "UMIX",
            ]
        else:
            self.blocks = blocks

        

    def read_slha_file(self, fname, particle_ids):
        if not fname.endswith(".slha"):
            raise ValueError(f"fname = {fname} does not end with .slha.")

        particle_ids = list(set(particle_ids))
        if self.root_dir is not None:
            fname = "/".join([self.root_dir, fname])
        
        features = {block: [] for block in self.blocks}
        data = pyslha.read(fname)


if __name__ == "__main__":
    particle_ids = [1000022, 1000022]
    root_dir = "../dataset/EWonly"

    data_reader = SLHAReader(root_dir=root_dir)
    features = data_reader.read_slha_file(fname="10_119_1.slha", particle_ids=particle_ids)
    print(features)