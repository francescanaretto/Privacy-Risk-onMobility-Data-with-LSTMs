from Utility_LSTM import *
import pickle
import pandas as pd
import numpy as np

def main():
    print("init")
    title = 'rischio2_istat'
    datas = open("../../datasets/Prato_Pistoia_trajectories_ids.p","rb")
    dataset = pickle.load(datas)
    print(dataset)
    labels = pd.read_csv('../../datasets/' + title + '.csv')
    path = '../results/'
    maxlen = 0  # select the most long trajectory for padding later
    label = dict()
    label_o = list()
    nuovo = dict()
    for user in dataset.keys():
        if len(dataset[user]) > 340:
            nuovo[user] = dataset[user][0:340]
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
    print(new_data.shape)
    print(label_o.shape)
    classifier = Utility_LSTM(path, 'rischio2_istat')
    print(new_data)
    classifier.train_test_split(new_data, label_o, 'strat', 0.20, 0.10, maxlen)
    model = classifier.compile_classifier('sigmoid', 'binary_crossentropy', 2, maxlen, 'Adadelta' )
    classifier.fit_model(model, 64, 20, 'strat', False)



if __name__ == "__main__":
    main()


