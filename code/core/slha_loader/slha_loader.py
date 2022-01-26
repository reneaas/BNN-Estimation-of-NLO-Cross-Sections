import numpy as np
import os
import pandas as pd


class SLHAloader(object):
    """SLHAloader provides a flexible and easy-to-use dataloader
    for data from slha files.

    Args:
        particle_ids (list)     :       List of particle ids for a process
                                        (particle pair)

        feat_dir (str)          :       root directory with features

        target_dir (str)        :       root directory with targets
    """

    def __init__(
        self,
        particle_ids: list[str],
        feat_dir: str,
        target_dir: str,
        target_keys: list[str] = ["nlo"],
    ):

        self.supported_ids = [
            "1000022",
            "1000023",
            "1000025",
            "1000035",
            "1000024",
            "1000037",
        ]

        self.supported_processes = [
            str((1000022, 1000022)),
            str((1000022, 1000023)),
            str((1000022, 1000025)),
            str((1000022, 1000035)),
            str((1000022, 1000024)),
            str((1000022, 1000037)),
            str((1000022, -1000024)),
            str((1000022, -1000037)),
            str((1000023, 1000023)),
            str((1000023, 1000025)),
            str((1000023, 1000035)),
            str((1000023, 1000024)),
            str((1000023, 1000037)),
            str((1000023, -1000024)),
            str((1000023, -1000037)),
            str((1000025, 1000025)),
            str((1000025, 1000035)),
            str((1000025, 1000024)),
            str((1000025, 1000037)),
            str((1000025, -1000024)),
            str((1000025, -1000037)),
            str((1000035, 1000035)),
            str((1000035, 1000024)),
            str((1000035, 1000037)),
            str((1000035, -1000024)),
            str((1000035, -1000037)),
            str((1000024, -1000024)),
            str((1000024, -1000037)),
            str((1000037, -1000024)),
            str((1000037, -1000037)),
        ]

        # particle_ids = sorted(particle_ids) #Sort in ascending order.
        self.particle_ids = list(set(particle_ids))
        # Raise ValueError if a particle id is invalid.
        for id in self.particle_ids:
            if str(abs(int(id))) not in self.supported_ids:
                err_message = f"particle {id=} is not a supported particle id. \n"
                err_message += "Supported particle ids: \n"
                err_message += "\n".join(self.supported_ids)
                raise ValueError(err_message)

        # Raise Valueerror if a particle process is invalid.
        processes = [
            str(tuple([int(i) for i in particle_ids])),
            str(tuple([int(i) for i in particle_ids[::-1]])) #Permuted tuple.
        ]
        state = False
        for process in processes:
            if process in self.supported_processes:
                state = True
                self.process = process
        if state is False:
            err_message = "\n"
            for process in processes:
                err_message += f"{process=} is not a valid process.\n"
            err_message += "Supported processes:\n"
            err_message += "\n".join(self.supported_processes)
            raise ValueError(err_message)

        features = {"MASS": {}, "NMIX": {}, "VMIX": {}, "UMIX": {}}

        # Extract mass of particle_ids.
        df_mass = pd.read_pickle(feat_dir + "/" + "mass.pkl")[self.particle_ids]

        dfs = []
        # Extract rest of features
        for id in self.particle_ids:
            features["MASS"][id] = df_mass[id]
            fname = feat_dir + "/nmix_" + id + ".pkl"
            if os.path.isfile(fname):
                features["NMIX"][id] = pd.read_pickle(fname)

            fname = feat_dir + "/vmix_" + id + ".pkl"
            if os.path.isfile(fname):
                features["VMIX"][id] = pd.read_pickle(fname)

            fname = feat_dir + "/umix_" + id + ".pkl"
            if os.path.isfile(fname):
                features["UMIX"][id] = pd.read_pickle(fname)

        # Extract features and delete temporary dict
        self.features = {}
        for key in features:
            if features.get(key) is not None:
                self.features[key] = features.get(key)
        del features  # Clear memory.

        # Merge dataframes
        dfs = []
        for block in self.features:
            for id in self.particle_ids:
                if self.features[block].get(id) is not None:
                    dfs.append(self.features[block][id])
        self.features = pd.concat(dfs, axis=1)

        # Extract targets
        targets = {}
        for key in target_keys:
            fname = target_dir + f"/{key}.pkl"
            targets[key] = pd.read_pickle(fname)

        # Extract process
        self.targets = {}
        for key in targets.keys():
            self.targets[key] = targets[key].get(self.process)

    def to_numpy(self) -> tuple[np.ndarray]:
        """Converts the data to numpy arrays."""
        self.features = self.features.to_numpy()
        features = self.features.to_numpy()
        targets = {key : val.numpy() for key, val in self.targets.items()}
        return features, targets



def main():
    ids = ["1000022", "1000024"]
    # ids = ["2", "4"]
    target_dir = "../targets"
    feat_dir = "../features"
    dl = SLHAloader(ids, feat_dir, target_dir)
    # dl.to_numpy()
    print(dl.features)
    print(dl.targets)
    targets = dl.targets.get("nlo")
    print(targets.to_numpy())

    # print(targets)



if __name__ == "__main__":
    main()

