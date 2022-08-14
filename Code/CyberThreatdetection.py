
import pandas as pd

import numpy as np
from numpy.random import rand, seed

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import ExtraTreesClassifier,  RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, MinMaxScaler, Binarizer
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from keras import backend as K
#from keras import layers

import tensorflow as tf
from tensorflow.keras.models import Model
#from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Flatten, Embedding
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import time

#To be able to replicate the results
tf.random.set_seed(1)
np.random.seed(1)
tf.keras.initializers.GlorotUniform(seed=1)

#Load data
data = pd.read_csv(r'C:\Users\erika\OneDrive\Desktop\AML\CW\train_imperson_without4n7_balanced_data.csv')
data = data.reindex(np.random.permutation(data.index))
test_data = pd.read_csv(r'C:\Users\erika\OneDrive\Desktop\AML\CW\test_imperson_without4n7_balanced_data.csv')

#Get rid of data having 0 variance given that are going to be suitable variables
print(data.head(5))
columns = list(data)

list_col_to_drop = []
for i in columns:
    if data[i].std() == 0:
        list_col_to_drop.append(i)

print(len(list_col_to_drop))

#74 variables have 0 variance and are dropped from the dataset
new_data = data.drop(list_col_to_drop, axis=1)
col = list(new_data.columns)
test_data = test_data.drop(list_col_to_drop, axis=1)

shape = new_data.shape
array = new_data.values
test_array = test_data.values
x = array[:,0:shape[1]-1]
y = array[:,shape[1]-1]
shape_y = y.shape
length = len(y)
x_test = test_array[:,0:test_data.shape[1]-1]
y_test = test_array[:,test_data.shape[1]-1]

#---- correlation and density plot ---

#correlation
correlation = new_data.corr(method='pearson')
pd.set_option('precision',3)
print(correlation)

f = plt.figure(figsize=(7,7))
plt.matshow(new_data.corr(), fignum=f.number)
plt.xticks(range(new_data.select_dtypes(['number']).shape[1]), new_data.select_dtypes(['number']).columns, fontsize=5, rotation=90 )
plt.yticks(range(new_data.select_dtypes(['number']).shape[1]), new_data.select_dtypes(['number']).columns, fontsize=5)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=7)
plt.title('Correlation Matrix', fontsize=10)
plt.show()

new_data2 = new_data[['wlan.fc.subtype','wlan.fc.pwrmgt',
'radiotap.channel.type.cck','radiotap.channel.type.ofdm','radiotap.datarate','wlan.fc.ds','radiotap.mactime','wlan.seq','wlan.fc.protected',
'wlan.qos.tid','wlan.qos.priority','frame.len','frame.cap_len','data.len','wlan.fc.type','wlan.duration','wlan.fc.retry',
'wlan_mgt.fixed.reason_code','wlan.da','wlan_mgt.tim.dtim_period','wlan.wep.iv','wlan_mgt.fixed.capabilities.preamble','wlan_mgt.fixed.timestamp',
'wlan_mgt.rsn.version','class']]
#univariate density plots
new_data2.plot(kind='density',subplots=True,layout=(5,5),sharex=False, sharey=False, figsize=(5,5), use_index=False)
plt.show()

#Scale features matrix so that they are bounded between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
x_scaled = scaler.fit_transform(x)
x_test_scaled = scaler.fit_transform(x_test)

#--------------------------------------------
#------------Feature Generation--------------
#--------------------------------------------
#1 - PCA
pca = PCA(n_components=10)
x_pca = pca.fit_transform(x_scaled)

plt.bar(range(0, len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=1)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title('Explain Variance Ratio')
plt.tight_layout()
plt.show()

#from the PVE chart is easy to note that after the 3 principal component the explained variance decrease significantly
#given this we decided to consider just the forst 3 principal components
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_scaled)
x_pca_scaled = scaler.fit_transform(x_pca)
x_pca_scaled_y = np.insert(x_pca_scaled, 3, y, axis=1)

fig = plt.figure()
ax = plt.axes(projection='3d')
g = ax.scatter3D(x_pca_scaled_y[:,0],x_pca_scaled_y[:,1],x_pca_scaled_y[:,0], c=x_pca_scaled_y[:,3], marker='x')
ax.set_xlabel('PCA_1')
ax.set_ylabel('PCA_2')
ax.set_zlabel('PCA_3')
legend = ax.legend(*g.legend_elements(), loc="lower right")
plt.title("PCA")
plt.show()

#2 - Autoencoder (standard and sparse)
#Data Normalization
scaler = Normalizer().fit(x)
x_norm = scaler.transform(x)
x_test_norm = scaler.transform(x_test)


class AutoEncoderModel:

    def one_layer(x=x,p_hidden1_size=50, p_code_size=3, sparse=False, regval=10e-6):
        input_size = len(x[0])
        hidden1_size = p_hidden1_size
        code_size = p_code_size
        input_data = Input(shape=(input_size,))
        hidden_1 = Dense(hidden1_size, activation='relu')(input_data)
        if sparse:
            code = Dense(code_size, activation='relu', activity_regularizer=l1(regval))(hidden_1)
        else:
            code = Dense(code_size, activation='relu')(hidden_1)
        hidden_2 = Dense(hidden1_size, activation='relu')(code)
        output_data = Dense(input_size, activation='sigmoid')(hidden_2)
        autoencoder = Model(input_data, output_data)
        encoded = Model(input_data, code)
        return autoencoder, encoded

    def two_layers(x=x,p_hidden1_size=50,p_hidden2_size=35, p_code_size=3, sparse=False, regval=10e-6):
        input_size = len(x[0])
        hidden1_size = p_hidden1_size
        hidden2_size = p_hidden2_size
        code_size = p_code_size
        input_data = Input(shape=(input_size,))
        if sparse:
            hidden_1 = Dense(hidden1_size, activation='relu',activity_regularizer=l1(regval))(input_data)
            hidden_2 = Dense(hidden2_size, activation='relu',activity_regularizer=l1(regval))(hidden_1)
            code = Dense(code_size, activation='relu',activity_regularizer=l1(regval))(hidden_2)
        else:
            hidden_1 = Dense(hidden1_size, activation='relu')(input_data)
            hidden_2 = Dense(hidden2_size, activation='relu')(hidden_1)
            code = Dense(code_size, activation='relu')(hidden_2)
        hidden_3 = Dense(hidden2_size, activation = 'relu')(code)
        hidden_4 = Dense(hidden1_size, activation='relu')(hidden_3)
        output_data = Dense(input_size, activation = 'sigmoid')(hidden_4)
        autoencoder = Model(input_data, output_data)
        encoded = Model(input_data, code)
        return autoencoder, encoded

    def sampling(args):
        code_size = 3
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], code_size),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def vae(x=x,input_size=78,hidden_size = 35, code_size = 3):

        input_data = Input(shape=(input_size,))
        hidden_1 = Dense(hidden_size, activation='relu')(input_data)
        code = Dense(code_size, activation='relu')(hidden_1)
        x_mean = Dense(code_size, name="x_mean")(code)
        x_log_var = Dense(code_size, name="x_log_var")(code)
        #z = AutoEncoderModel.sampling([x_mean, x_log_var])
        var_encoder = Model(input_data, [x_mean, x_log_var])
        vae = var_encoder.predict(x)
        z1 = AutoEncoderModel.sampling([vae[0], vae[1]])
        vae_array = np.array(z1)
        return var_encoder, vae_array

#one hidden Layer
#Stacked (sparse) AE - just the selected AE has been kept
ae_sparse_1, enc_sparse_1 = AutoEncoderModel.one_layer(x,p_hidden1_size=50,p_code_size=10,sparse=True, regval=10e-6)
ae_sparse_1.compile(optimizer='adam',loss='binary_crossentropy')
ae_sparse_fit_1 = ae_sparse_1.fit(x_norm,x_norm,epochs=3)
en_sparse_1 = enc_sparse_1.predict(x_norm)
ae_sparse_loss_1 = ae_sparse_fit_1.history.items()

#VAE
var_encoder, vae_array = AutoEncoderModel.vae(x_norm, input_size=78,hidden_size = 35, code_size = 3)

#plot
encoded_sparse_y = np.insert(en_sparse_1, 3, y, axis=1)
fig = plt.figure()
ax = plt.axes(projection='3d')
g = ax.scatter3D(encoded_sparse_y[:,2],encoded_sparse_y[:,1],encoded_sparse_y[:,0], c=encoded_sparse_y[:,3], marker='^')
legend = ax.legend(*g.legend_elements(), loc="lower right")
plt.title("Sparse Autoencoder 1")
plt.show()


#3 - Variational Autoencoder
z_array_y = np.insert(vae_array, 3, y, axis=1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(z_array_y[:,0],z_array_y[:,1],z_array_y[:,2], c=z_array_y[:,3], marker='o')
plt.title("Variational Autoencoder")
plt.show()

#4 - Decision tree classifier - features
def create_decision_tree_features(new_data):
    new_data.loc[data['wlan.ra'] <= 0.001, 'wlan.ra_binary' ] = 1
    new_data.loc[data['wlan.ra'] > 0.001, 'wlan.ra_binary' ] = 0

    new_data['bin_tree_f'] = 0
    new_data.loc[(data['wlan.ra'] <= 0.001) & (new_data['frame.cap_len'] <= 0.039) & (new_data['radiotap.mactime'] <= 0.855),'bin_tree_f'] = 1
    new_data.loc[(data['wlan.ra'] > 0.001) & (new_data['wlan_mgt.ssid'] > 0.001), 'bin_tree_f'] = 1

    bin_tree_df = new_data[['wlan.ra_binary','bin_tree_f']]
    binary_tree_f = bin_tree_df.values
    return binary_tree_f

binary_tree_f = create_decision_tree_features(new_data)

#6 - KMeans
col_x = col[0:len(col)-1]
df = pd.DataFrame(x,columns=col_x)
kmeans = KMeans(n_clusters=2)
cluster = kmeans.fit_predict(df[col_x])
df['Cluster'] = cluster

#Combine original Data with Generated features
columns_new = col[0:len(col)-1]
new_features_list = {'PCA':x_pca,'VAE':vae_array, 'SPARSE_AE':en_sparse_1, 'DECISION_TREE':binary_tree_f,'CLUSTER':cluster}

for key in new_features_list:
    j = new_features_list.get(key)
    tupl = j.shape
    myrange = 1
    if len(tupl) > 1:
        myrange = tupl[1]
    for i in range(myrange):
       columns_new.append(key+'_'+str(i))

#Create the enriched datasets
new_x = np.column_stack([x, x_pca, vae_array, en_sparse_1, binary_tree_f, cluster])
scaler = MinMaxScaler(feature_range=(0,1))
new_x_scaled = scaler.fit_transform(new_x)
enr_full_df = pd.DataFrame(new_x_scaled, columns=columns_new)

#apply to the test set
#PCA
x_pca_test = pca.transform(x_test_scaled)
#Binary tree
binary_tree_test = create_decision_tree_features(test_data)
#Stacked (sparse) AE
x_ae_sparse_test = enc_sparse_1.predict(x_test_norm)
#VAE
vae_test = var_encoder.predict(x_test_norm)
z_test = AutoEncoderModel.sampling([vae_test[0], vae_test[1]])
vae_test_array = np.array(z_test)
#Cluster
cluster_test = kmeans.predict(x_test)
x_test = np.column_stack([x_test, x_pca_test, vae_test_array, x_ae_sparse_test, binary_tree_test, cluster_test ])
x_test_scaled = scaler.transform(x_test)
x_test_df = pd.DataFrame(x_test_scaled, columns=columns_new)

#x_test_df.to_csv('x_test_df.csv')
#enr_full_df.to_csv('x_train_df.csv')
#y_test_df = test_data[['class']]
#y_train_df = data[['class']]
#y_test_df.to_csv('y_test_df.csv')
#y_train_df.to_csv('y_train_df.csv')

#------------------------------------------------#
#------ Feature Selction & Model Selction -------#
#------------------------------------------------#

#Modify the function to get the feature names
def get_features_names_feature_union(feature_union):
    features_names = feature_union.get_feature_names_out()
    final_ft_st = []
    for i in features_names:
        x_position = i.rfind('x')
        index_i = i[x_position+1:]
        final_ft_st.append(index_i)
    final_feature_names = []
    for i in final_ft_st:
        ft_index = int(i)
        ft = columns_new[ft_index]
        final_feature_names.append(ft)
    return final_feature_names


#Feature Selection
features = []
features.append(('select_best', SelectKBest(mutual_info_classif, k=10)))
#--- all the other attempt for features selection have been commented ----
#LR = LogisticRegression(solver='liblinear')
#estimator = ExtraTreesClassifier(n_estimators=10)
#rfe_estimator = SVC(kernel="linear")
#features.append(('rfe', RFE(estimator=rfe_estimator,n_features_to_select=10)))
#estimator = ExtraTreesClassifier(n_estimators=10)
#features.append(('LR',SelectFromModel(LR)))
#estimator = LinearSVC(LinearSVC(C=0.01, penalty="l1", dual=False))
#features.append(('LR', estimator))
feature_union = FeatureUnion(features)

models = []
models.append(('LogisticRegression', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC(probability=True)))
models.append(('MLP',MLPClassifier(hidden_layer_sizes=(30,10),solver='sgd',learning_rate='adaptive',random_state=1, max_iter=100)))

dict_time = dict()
dict_auc = dict()
dict_score = dict()
dict_test = dict()
dict_tp = dict()
dict_fp = dict()
dict_fn = dict()
dict_tn = dict()


for name, model in models:
    estimators = []
    estimators.append(('feature_union', feature_union))
    estimators.append((name, model))
    pipe = Pipeline(estimators)
    start = time.time()
    model_fit = pipe.fit(new_x_scaled,y)
    final_feature_names = get_features_names_feature_union(feature_union)
    end = time.time()

    start_test = time.time()
    score = model_fit.score(x_test_scaled,y_test)
    pred = pipe.predict(x_test_scaled)
    end_test = time.time()
    c_matrix = confusion_matrix(y_test, pred)
    prob = pipe.predict_proba(x_test_scaled)
    auc = roc_auc_score(y_test, prob[:, 1])
    fpr, tpr, threshold = roc_curve(y_test, prob[:, 1])

    print(model, 'score is: ', score)
    print(c_matrix)
    print('AUC is: ', auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='o')
    plt.title(name)
    plt.show()
    print(model, 'execution time is: ', end - start)

    model_time = end - start
    model_test = end_test - start_test
    dict_time[name] = model_time
    dict_test[name] = model_test
    dict_auc[name] = auc
    dict_score[name] = score
    dict_tp[name] = c_matrix[0][0]
    dict_fp[name] = c_matrix[0][1]
    dict_fn[name] = c_matrix[1][0]
    dict_tn[name] = c_matrix[1][1]

print(final_feature_names)
#get a dictionary with all the results
df_dict = dict()
for name, model in models:
    lst = []
    lst.append(dict_auc[name])
    lst.append(dict_score[name])
    lst.append(dict_tp[name])
    lst.append(dict_fp[name])
    lst.append(dict_fn[name])
    lst.append(dict_tn[name])
    lst.append(dict_time[name])
    lst.append(dict_test[name])
    df_dict[name] = lst

#---- Get the best Decision Tree in the Forest ---
select_best_ft = SelectKBest(mutual_info_classif, k=10)
x_train = select_best_ft.fit_transform(new_x_scaled,y)
model = RandomForestClassifier()
model.fit(x_train,y)
x_test = select_best_ft.transform(x_test_scaled)
score = model.score(x_test, y_test)
pred = model.predict(x_test)

c_matrix = confusion_matrix(y_test, pred)
prob = model.predict_proba(x_test)
auc = roc_auc_score(y_test, prob[:, 1])
fpr, tpr, threshold = roc_curve(y_test, prob[:, 1])
ft_importance = model.feature_importances_

dict = {}
count = 0
myobject = model.estimators_[0]
actualscore = myobject.score(x_test, y_test)

for i in model.estimators_:
    i.fit(x_train, y)
    score_dt = i.score(x_test, y_test)
    if score_dt > actualscore:
        myobject = i
        actualscore = myobject.score(x_test, y_test)

ft_names = ['frame.len', ' frame.cap_len', ' radiotap.mactime', ' PCA_1', ' PCA_2',' PCA_3',' SPARSE_AE_1', ' SPARSE_AE_3',' SPARSE_AE_5',' DECISION_TREE_1']

feat_importance = pd.Series(myobject.feature_importances_, index=ft_names)
feat_importance.nlargest(10).plot(kind='barh')
plt.show()

plt.figure(figsize=(9,6))
plot_tree(decision_tree=myobject, max_depth=4, feature_names=ft_names, label=None, precision=2, fontsize=7, node_ids=False, impurity=True, filled=True)
plt.title('Best Decision Tree - score 94%')

plt.show()

#Grid search has been commented has is too comuntationally expensive

#param_grid = {'n_estimators': [100,200],
# 'criterion':   ['gini'],
# 'max_depth': [ None],
# 'max_features': [10],
# 'min_samples_leaf': [1,2],
# 'min_samples_split': [2,4]}

#grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid)
#grid.fit(x_test,y)
#print(grid.best_score_)
#print(grid.best_index_)