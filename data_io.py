import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
import matplotlib.pyplot as plt


def load_json_file(data_num):
    cols = ['id', 'loc.lat', 'loc.lng', 'weather.date.hour', 'weather.tempm', 'weather.hum', 'weather.wgustm']
    file_path = '/mnt/fs2/2018/matsuzaki/dataset_fromnitta/metadata.json'
    with open(file_path) as f:
        df = json.load(f, object_hook=True)
    df_norm = json_normalize(df)
    df_norm = df_norm.sample(frac=1).reset_index(drop=True)
    df_norm = df_norm[:data_num]
    df_ = pd.DataFrame()
    df_[cols[0]] = df_norm[cols[0]]
    for c in cols[1:]:
        df_[c] = df_norm[c].apply(float_force)
        df_.loc[df_[c] <= -999, c] = np.nan
    df_.dropna(how='any', axis=0, thresh=3)
    for c in cols[1:]:
        df_[c] = df_[c].fillna(df_[c].mean())
    return df_.dropna(how='any', axis=0)

def float_force(x):
    try: return np.float(x)
    except ValueError:
        return np.nan

def main():
    df = load_json_file(100)
    for c in cols[1:]:
        fig = plt.hist(df[c], bins=24)
        plt.savefig("/mnt/fs2/2018/matsuzaki/dataset_fromnitta/graph/"+c+".png")
        plt.clf()

if __name__ == "__main__":
    main()
