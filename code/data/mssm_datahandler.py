import numpy as np
import os


class MSSMDataHandler(object):
    """ MSSMDataHandler provides a flexible and easy-to-use datahandler
        for the MSSM dataset.

        Args:
            mass_fname          : .npz file contaning masses
            feat_exc_mass_fname : .npz filename contaning the remaining features
            target_fnames       : .npz filenames containing the targets
    """

    def __init__(self, mass_fname, feat_exc_mass_fname, target_fnames):

        self.features = self.load_features(mass_fname, feat_exc_mass_fname)
        self.targets = self.load_targets(target_fnames)



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
                targets[fname.strip(".npz")] = np.load(fname)
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
        mass = np.load(mass_fname)
        feat_exc_mass = np.load(feat_exc_mass_fname)

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



if __name__ == "__main__":

    mass_fname = "mass.npz"
    feat_exc_mass_fname = "feat_no_mass.npz"
    target_fnames = ["targets_col_8.npz", "targets_col_9.npz"]
    datahandler = MSSMDataHandler(mass_fname, feat_exc_mass_fname, target_fnames)
    print(datahandler.features["NMIX"])
    print("--"*20)
