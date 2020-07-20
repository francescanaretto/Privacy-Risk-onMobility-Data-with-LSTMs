import matplotlib
from Utility_LSTM import *
matplotlib.use("TkAgg")  # Do this before importing pyplot!
import matplotlib.pyplot as plt
plt.interactive(True)
import pickle
import shap
from keras.utils import plot_model
import time
def main():
    print("init")
    trained_model = open('../results/trained_net_rischio4_voronoi.p',"rb")
    model = pickle.load(trained_model)
    model_history = open('../results/history_rischio4_voronoi.p',"rb")
    history = pickle.load(model_history)
    title = "../datasets/test_set_rischio4_voronoi_strat.p"
    test = open(title,"rb")
    test_set = pickle.load(test)
    title = "../datasets/test_label_rischio4_voronoi_strat.p"
    test_l = open(title,"rb")
    test_label = pickle.load(test_l)
    path = '../results/'
    classifier = Utility_LSTM(path, 'rischio4_voronoi')
    start_time = time.time()
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    classifier.prediction_classifier(model, test_set, test_label)
    print("--- %s seconds ---" % (time.time() - start_time))

    #classifier.plot_accuracy(history)
    classifier.plot_loss_vs_epoch(history)



if __name__ == "__main__":
    main()
