from sklearn.model_selection import train_test_split
import pickle
from imblearn.under_sampling import RandomUnderSampler
import keras
import tensorflow
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import StratifiedKFold


class Utility_LSTM:

    def __init__(self, path, title):
        print('Utility LSTM up and running')
        self.path = path
        self.title = title

    #it splits the dataset into training and testing, ratio specified is the one for test.
    #datasets saved in path
    def train_test_split(self, dataset, labels, method, ratio_test, ratio_val, maxlen):
        #padding for every record up to the maxlen
        dataset = keras.preprocessing.sequence.pad_sequences(dataset, maxlen=maxlen)

        #stratified case
        if method == 'strat':
            train_set, test_set, train_label, test_label = train_test_split(dataset, labels,stratify=labels,test_size=ratio_test)
            train_set, validation_set, train_label, validation_label = train_test_split(train_set,train_label,stratify=train_label,test_size=ratio_val)

        #undersampling case
        else:
            train_set, test_set, train_label, test_label = train_test_split(dataset, labels, stratify=labels,test_size=ratio_test)
            sampler = RandomUnderSampler(sampling_strategy=method, random_state=42)
            train_set, train_label = sampler.fit_sample(train_set, train_label)
            train_set, validation_set, train_label, validation_label = train_test_split(train_set, train_label, stratify=train_label,test_size=ratio_val)
            sampler = RandomUnderSampler(sampling_strategy=method, random_state=42)
            validation_set, validation_label = sampler.fit_sample(validation_set, validation_label)

        #save training and testing
        title = self.path + "train_set_" + self.title + "_" + method + ".p"
        with open(title, 'wb') as fp:
            pickle.dump(train_set, fp)
        title = self.path+"test_set_"+self.title+"_"+method+".p"
        with open(title, 'wb') as fp:
            pickle.dump(test_set, fp)
        title = self.path + "validation_set_" + self.title + "_" + method + ".p"
        with open(title, 'wb') as fp:
            pickle.dump(validation_set, fp)
        title = self.path+"train_label_"+self.title+"_"+method+".p"
        with open(title, 'wb') as fp:
            pickle.dump(train_label, fp)
        title = self.path+"test_label_"+self.title+"_"+method+".p"
        with open(title, 'wb') as fp:
            pickle.dump(test_label, fp)
        print(validation_label)
        title = self.path + "validation_label_" + self.title + "_" + method + ".p"
        with open(title, 'wb') as fp:
            pickle.dump(validation_label, fp)


    def fit_model(self, model, batch, epoch, method, flag, train_set=None, train_label=None, validation_set=None, validation_label=None):
        if flag is False:
            title_train_set = open(self.path + 'train_set_' + self.title + "_"+method+".p", 'rb')
            train_set = pickle.load(title_train_set)
            title_train_label = open(self.path + 'train_label_' + self.title + "_"+method+".p", 'rb')
            train_label = pickle.load(title_train_label)
            title_validation_set = open(self.path + 'validation_set_' + self.title + "_"+method+".p", 'rb')
            validation_set = pickle.load(title_validation_set)
            title_validation_label = open(self.path + 'validation_label_' + self.title + "_"+method+".p", 'rb')
            validation_label = pickle.load(title_validation_label)
        model = self.fit_classifier(model, batch, epoch, train_set, train_label, validation_set, validation_label)
        return model


    #fit the classifier
    def fit_classifier(self, model, batch, epoch, train_set, train_label, validation_set, validation_label):

        # early stopping
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=4, verbose=1)

        print(train_label)
        train_label = keras.utils.to_categorical(train_label)
        validation_label = keras.utils.to_categorical(validation_label)

        history = model.fit(train_set, train_label, epochs=epoch, batch_size=batch, validation_data=(validation_set, validation_label), callbacks=[es])

        #saving the model and its history to file
        title_net = '../results/trained_net_'+str(self.title)+'.p'
        history_net = '../results/history_'+str(self.title)+'.p'
        pickle.dump(model, open(title_net, 'wb'))
        pickle.dump(history, open(history_net, 'wb'))
        return model


    #compile the classifier
    def compile_classifier(self, activation, loss, number_classes, maxlen, optimizer):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(maxlen, 40))
        model.add(keras.layers.LSTM(35, return_sequences=True, recurrent_dropout=0.3))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.LSTM(20, recurrent_dropout=0.3))
        model.add(keras.layers.Dense(number_classes, activation=activation))

        # compile the model
        model.compile(loss=loss, optimizer=optimizer,  metrics=['accuracy'])
        return model


    #for the evaluation phase
    #predict the value of the test set, then it creates the confusion matrix
    def prediction_classifier(self, model, test_set, test_label):
        test_label = keras.utils.to_categorical(test_label)
        #retrieve the predicted lables
        predicted_labels = model.predict(test_set, verbose=1)
        count_correct = [0]*2
        count_incorrect = [0]*2
        #create two lists: one for the predictions and one for the correct classes
        predictions = list()
        reals = list()
        for pred in range(0, len(predicted_labels)):
            predicted = predicted_labels[pred].argmax(axis=0)
            predictions.append(predicted)
            real = test_label[pred].argmax(axis=0)
            reals.append(real)
            if real == predicted:
                count_correct[real] += 1
            else:
                count_incorrect[real] += 1
        report = classification_report(reals, predictions)
        write_report = open(self.path+'measures_'+self.title+'.txt',"w")
        write_report.write(report)
        print('correct ', count_correct)
        print('incorrect ', count_incorrect)
        cm = metrics.confusion_matrix(test_label.argmax(axis=1), predicted_labels.argmax(axis=1))
        self.plot_confusion_matrix_multi(cm, ['Low risk', 'High risk'])


    #method for plotting the confusion matrix
    def plot_confusion_matrix_multi(self, cm,target_names, cmap=None, normalize=True):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        if cmap is None:
            cmap = plt.get_cmap('Blues')
        plt.figure(figsize=(15, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('Confusion Matrix')
        plt.colorbar()
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names)
            plt.yticks(tick_marks, target_names)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig(self.path+'cm_'+str(self.title)+'.png', dpi=400)

    #method for plotting a plot with the accuracy in the test and training set
    def plot_accuracy(self, history):
        plt.figure()
        print(history.history)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='bottom right')
        title = '../results/accuracy_history'+str(self.title)+'.png'
        plt.savefig(title)

    #method for plotting the behaviour of the loss during the epochs
    def plot_loss_vs_epoch(self, history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        title = '../results/loss_epochs'+str(self.title)+'.png'
        plt.savefig(title)

    def k_fold_validation(self, k, activation, loss, batch, epoch, data, labels, optimizer):
        data =keras.preprocessing.sequence.pad_sequences(data, maxlen=1800)
        kfold = StratifiedKFold(k, True, 1)
        scores = dict()
        count = 0
        for train_index, test_index in kfold.split(data, labels):
            # select samples
            train_set, test_set = data[train_index], data[test_index]
            print(train_set)
            print(train_set.astype)
            train_label, test_label = labels[train_index], labels[test_index]
            # compile the model
            model = self.compile_classifier(activation, loss, 2, 1800, optimizer)
            scores[count] = dict()
            # evaluate the model
            model = self.fit_model(model, batch, epoch, 'strat', True, train_set, train_label, test_set, test_label)
            scores[count] = self.evaluate_model_kfold(model, test_set, test_label)
            print(scores)
            # saving the model and its history to file
            title_net = '../results/trained_net_' + str(self.title) + str(count) + '.p'
            pickle.dump(model, open(title_net, 'wb'))
            count += 1
            write_index = open('index_' + self.title + str(count) + '.txt', "w")
            write_index.write(str(train_index))
            write_index.write(str(test_index))

        history_net = '../results/scores' + str(self.title) + '.p'
        pickle.dump(scores, open(history_net, 'wb'))


    def evaluate_model_kfold(self, model, test_set, test_label):
        #retrieve the predicted lables
        test_label = keras.utils.to_categorical(test_label)
        predicted_labels = model.predict(test_set, verbose=1)
        count_correct = [0]*2
        count_incorrect = [0]*2
        #create two lists: one for the predictions and one for the correct classes
        predictions = list()
        reals = list()
        for pred in range(0, len(predicted_labels)):
            predicted = predicted_labels[pred].argmax(axis=0)
            predictions.append(predicted)
            real = test_label[pred].argmax(axis=0)
            reals.append(real)
            if real == predicted:
                count_correct[real] += 1
            else:
                count_incorrect[real] += 1
        report = classification_report(reals, predictions, output_dict=True)
        return report

