from Utility_LSTM import *
import pickle
import pandas as pd
import numpy as np

def main():
    print("init")
    path = '../results/'
    title = 'rischio4_voronoi'

    datas = open(path+"prato_pistoia_traj_voronoi.p","rb")
    dataset = pickle.load(datas)
    print(dataset)
    labels = pd.read_csv(path + title + '.csv')

    label = dict()
    label_o = list()
    nuovo = dict()
    # select the most long trajectory for padding later
    maxlen = 2800
    for user in dataset.keys():
        if len(dataset[user]) > maxlen:
            nuovo[user] = dataset[user][0:maxlen]
        else:
            nuovo[user] = dataset[user]

    for i, j in labels.iterrows():
        label[j['uid']] = j['label']
        label_o.append(j['label'])
    for user in label.keys():
        if len(nuovo[user]) > maxlen:
            maxlen = len(nuovo[user])
    new_data = list()
    for u in label:
        new_data.append(nuovo[u])
    new_data = np.array(new_data)
    label_o = np.array(label_o)
    print(new_data)
    print(label_o.shape)
    classifier = Utility_LSTM(path, 'rischio5_voronoi')

    #call to the method for k fold
    classifier.k_fold_validation(5, 'sigmoid', 'binary_crossentropy', 32, 20, new_data, label_o, 'Adamax')



if __name__ == "__main__":
    main()
