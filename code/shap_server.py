import shap
import pickle
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Embedding
from keras.wrappers.scikit_learn import KerasClassifier

#load of the traning and test set on which the model was trained on + load of the model
path = '../results/'
name = 'rischio2_istat_strat'
trained_model = open(path+'trained_net_'+name+'.p',"rb")
model = pickle.load(trained_model)
title = path+"test_set_"+name+".p"
test = open(title,"rb")
test_set = pickle.load(test)
title = path+"train_set_"+name+".p"
test = open(title,"rb")
train_set = pickle.load(test)
title = path+"test_label_"+name+".p"
test_l = open(title,"rb")
test_label = pickle.load(test_l)

#train of the explainer
explainer = shap.DeepExplainer(model, train_set)

#shap values
shap_values = explainer.shap_values(test_set)

#dump shap values
title = path+"values_shap_"+name+".p"
with open(title, 'wb') as fp:
    pickle.dump(shap_values, fp)

#dump shap values
title = path+"explainer_0"+name+".p"
with open(title, 'wb') as fp:
    pickle.dump(explainer[0], fp)

#dump shap values
title = path+"explainer_1"+name+".p"
with open(title, 'wb') as fp:
    pickle.dump(explainer[1], fp)
