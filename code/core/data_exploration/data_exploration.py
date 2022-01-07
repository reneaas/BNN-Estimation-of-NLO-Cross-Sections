from slha_loader.slha_loader import SLHAloader
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import altair_viewer
import pandas as pd

target_dir = "../targets"
feat_dir = "../features"
neutralinos = [
        "1000022",
        "1000023",
        "1000025",
        "1000035",
    ]  # neutralinos[i] = neutralino i.

processes = []
for id_m in neutralinos:
    for id_n in neutralinos:
        if int(id_n) >= int(id_m):
            processes.append([id_m, id_n])


# for process in processes:
#     dl = SLHAloader(process, feat_dir, target_dir)
#     id1, id2 = process
#     #print(dl.features)
#     #print(f"{dl.features[particle_ids[0]]=}")
#     #print(f"{dl.features[particle_ids[1]]=}")
#     sns.scatterplot(dl.features[id1], dl.features[id2])
#     plt.figure()
# plt.show()

def plot_neutralinos():
    neutralinos = [
        "1000022",
        "1000023",
        "1000025",
        "1000035",
    ]  # neutralinos[i] = neutralino i.
    for neutralino in neutralinos:
        process = [neutralino] * 2
        dl = SLHAloader(process, feat_dir, target_dir)
        features = dl.features[neutralino]
        targets = dl.targets["nlo"]
        plt.figure()
        sns.scatterplot(features, targets)
        plt.yscale("log")
        plt.show()


def neutralino_chart():
    neutralinos = [
        "1000022",
        "1000023",
        "1000025",
        "1000035",
    ]  # neutralinos[i] = neutralino i.
    dfs = []
    for neutralino in neutralinos:
        process = [neutralino] * 2
        dl = SLHAloader(process, feat_dir, target_dir)

        features = dl.features[neutralino]
        targets = dl.targets["nlo"]
        df = pd.concat([features, targets], axis=1)
        dfs.append(df)
    
    df = pd.concat([dfs], axis=1)
    


    

neutralino_chart()