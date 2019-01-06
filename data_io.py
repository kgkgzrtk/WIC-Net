import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def load_json_file(data_num, load_csv=False, save_csv=False):
    cols = ['id', 'loc.lat', 'loc.lng', 'weather.date.hour', 'weather.tempm', 'weather.hum', 'weather.wgustm']
    file_path = '/mnt/fs2/2018/matsuzaki/dataset_fromnitta/metadata.json'
    save_path = '/mnt/data2/matsuzaki/data/i2w_dropna_'+ str(data_num) +'.csv'
    if load_csv:
        return pd.read_csv(save_path, dtype = {'id': str})
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = json_normalize(data)

        #random sample
        df = df.sample(frac=1).reset_index(drop=True)
        df = df[:data_num]

        df_ = pd.DataFrame()
        df_[cols[0]] = df[cols[0]]
        for c in cols[1:]:
            df_[c] = df[c].apply(float_force)
            df_.loc[df_[c] <= -999, c] = np.nan
        df_.dropna(how='any', axis=0, thresh=3)
        for c in cols[1:]:
            df_[c] = df_[c].fillna(df_[c].mean())
        df_ =  df_.dropna(how='any', axis=0)
        if save_csv: df_.to_csv(save_path)
        return df_

def write_tfrecord(images, labels, filename):
    filename = 'data.tfrecord'
    writer = tf.python_io.TFRecordWriter(filename)
    for image, label in zip(images, labels)
        img_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        label_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()]))
        ex = tf.train.Example({'image' : img_feat, 'label' : label_faet})
        writer.write(ex.SerializeToString())
    writer.close()


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
