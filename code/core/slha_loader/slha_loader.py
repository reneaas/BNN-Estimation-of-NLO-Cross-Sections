import numpy as np
import os
import pandas as pd


class SLHAloader(object):
    """ SLHAloader provides a flexible and easy-to-use dataloader
        for data from slha files.

        Args:
            particle_ids (list)     :       List of particle ids for a process
                                            (particle pair)
            
            feat_dir (str)          :       root directory with features

            target_dir (str)        :       root directory with targets


    """

    def __init__(self, particle_ids, feat_dir, target_dir, target_keys = ["nlo"]):

        possible_ids = ["1000022", "1000023", "1000025",
                        "1000035", "1000024", "1000037"
        ]


        #Raise ValueError if a particle id is not a valid id.
        for id in particle_ids:
            if id not in possible_ids:
                err_message = f"particle id = {id} is not a supported particle id. \n"
                err_message += "Supported particle ids: \n"
                err_message += "\n".join(possible_ids)
                raise ValueError(err_message)

        features = {"MASS" : {}, "NMIX" : {}, "VMIX" : {}, "UMIX" : {}}

        #Extract mass of particle_ids.
        df_mass = pd.read_pickle(feat_dir + "/" + "mass.pkl")
        dfs = []

        #Extract rest of features
        for id in particle_ids:
            features["MASS"][id] = df_mass[id]
            fname = feat_dir + "/nmix_"  + id + ".pkl"
            if os.path.isfile(fname):
                features["NMIX"][id] = pd.read_pickle(fname)

            fname = feat_dir + "/vmix_" + id + ".pkl"
            if os.path.isfile(fname):
                features["VMIX"][id] = pd.read_pickle(fname)

            fname = feat_dir + "/umix_" + id + ".pkl"
            if os.path.isfile(fname):
                features["UMIX"][id] = pd.read_pickle(fname)

        #Extract features and delete temporary dict
        self.features = {}
        for key in features.keys():
            if len(features[key]) != 0:
                self.features[key] = features.get(key)
        del features  #Clear memory.

        # Merge dataframes
        dfs = []
        for block in self.features.keys():
            for id in particle_ids:
                dfs.append(self.features[block][id])
        self.features = pd.concat(dfs, axis=1)

        #Extract targets
        targets = {}
        for key in target_keys:
            fname = target_dir  + f"/{key}.pkl"
            targets[key] = pd.read_pickle(fname)

        #Extract process
        self.targets = {}
        process = str(tuple( [int(i) for i in particle_ids] ))
        for key in targets.keys():
            self.targets[key] = targets[key][process]
            self.targets[key] = self.targets[key]


    def to_numpy(self):
        """Converts the data to numpy arrays."""
        self.features = self.features.to_numpy()
        for key in self.targets.keys():
            self.targets[key] = self.targets[key].to_numpy()


    def __getitem__(self, idx):
        """ Returns a datapoint (feature, target)

            Args:
                idx (int)   :   index of datapoint
        """

        return None


if __name__ == "__main__":

    #ids = ["1000022", "1000023"]
    ids = ["2", "4"]
    target_dir = "../targets"
    feat_dir = "../features"
    dl = SLHAloader(ids, feat_dir, target_dir)
    #dl.to_numpy()
    print(dl.features)
    print(dl.targets)
    # print(targets)
