import numpy as np
import os
import pandas as pd


class SLHAloader(object):
    """ SLHAloader provides a flexible and easy-to-use dataloader
        for data from slha files.

        Args:
            particle_ids (list)  :      List of particle ids for a process
                                        (particle pair)
    """

    def __init__(self, particle_ids, feat_dir, target_dir):

        possible_ids = ["1000022", "1000023", "1000025", "1000035",
                        "1000024", "1000037"
        ]


        #Raise ValueError if a particle id is not a valid id.
        for id in particle_ids:
            if id not in possible_ids:
                raise ValueError(f"{id} is not a supported particle id. Supported particle ids: \n" + ", ".join(possible_ids))

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
        target_keys = ["nlo", "nlo_rel_err"]
        targets = {}
        for key in target_keys:
            fname = target_dir  + f"/{key}.pkl"
            targets[key] = pd.read_pickle(fname)

        #Extract process
        self.targets = {}
        process = str(tuple( [int(i) for i in particle_ids] ))
        for key in targets.keys():
            self.targets[key] = targets[key][process]




    def load_targets(self, fnames):
        """Loads targets from .npz files.

            Args:
                fnames: filenames of .npz files containing targets

            Returns:
                targets:    dictionary with numpy ndarrays contaning targets
                            obtained from the input fnames.
        """
        targets = {}
        processes = {}
        for fname in fnames:
            if os.path.isfile(fname):
                targets[fname.strip(".npz")] = np.load(fname, allow_pickle=True)
        return targets

    def load_features(self, mass_fname, feat_exc_mass_fname):
        """ Loads features from .npz files.

            Args:
                mass_fname              :   .npz filename containing mass of particles
                feat_exc_mass_fname     :   .npz filename containing all other features
                                            in the .slha files.

            Returns:
                features:   dictionary containing all the features stored
                            as numpy ndarrays.
        """
        mass = np.load(mass_fname, allow_pickle=True)
        feat_exc_mass = np.load(feat_exc_mass_fname, allow_pickle=True)

        features = {}
        features["MASS"] = { key : mass[key] for key in mass.files}
        for key in feat_exc_mass.files:
            features[key] = feat_exc_mass[key]
        return features

    def extract_blocks(self, blocks):
        """ Extracts the blocks the user wants from the dataset.

            Args:
                blocks: specified blocks in the .slha files the user wants.
        """
        features = {}
        for block in blocks:
            features[block] = self.features[block]
        self.features = features

    def extract_targets(self, process):
        """ Returns targets relevant for a given particle pair

            Args:
                process (tuple) :   particle pair.

            Returns:
                Relevant targets for a the particle pair (process).
        """

        process = str(process)
        targets = {}
        for key in self.targets.keys():
            targets[key] = self.targets[key][process]
        return targets

    def extract_features(self, process):
        """ Returns features for a given particle pair.

            !!! Lacks functionality for UMIX and VMIX features. !!!


            Args:
                process (tuple) :   particle pair.


        """
        ids = process
        features = []
        if abs(ids[0]) == abs(ids[1]):
            features.append(self.features["MASS"][ids[0]])
            features.append(self.features["NMIX"][ids[0]])
        else:
            features.append(self.features["MASS"][ids[0]])
            features.append(self.features["MASS"][ids[1]])
            features.append(self.features["NMIX"][ids[0]])
            features.append(self.features["NMIX"][ids[1]])
        return features








if __name__ == "__main__":

    ids = ["1000022", "1000023"]
    target_dir = "./targets"
    feat_dir = "./features"
    dl = SLHAloader(ids, feat_dir, target_dir)
    print(dl.features)
    print(dl.targets)

    # mass_fname = "mass.npz"
    # feat_exc_mass_fname = "feat_no_mass.npz"
    # target_fnames = ["targets_col_8.npz", "targets_col_9.npz"]
    # dl = SLHAloader(mass_fname, feat_exc_mass_fname, target_fnames)
    # # print(dl.features["NMIX"])
    # print("--"*20)
    # # print(dl.features["NMIX"])
    # # print(dl.targets["targets_col_8"].files)
    # process = (10000_22, 10000_22)
    # targets = dl.extract_targets(process)
    # print(dl.features["MASS"])
    # print("--"*20)
    # print(dl.features["NMIX"])
    # # print(type(dl.features["NMIX"]["1000022"]))
    # print(dl.features)
    # features = dl.extract_features(process)
    # print(features)
    # print(targets)
