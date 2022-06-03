import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.GraphDescriptors import BalabanJ, BertzCT, Chi0, Chi1
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn import model_selection,metrics
from sklearn.metrics import r2_score, mean_squared_error

import xgboost as xgb

import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint

def get_rxn_fingerprint(smi_reactant, smi_product):
    vector_dim = 4096
    R = np.zeros((vector_dim))
    R_graph_features = np.zeros((4))
    for smi in smi_reactant:
        mol = Chem.MolFromSmiles(smi)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, \
                                                            nBits=vector_dim, useFeatures=True))

        R_graph_features = np.add(R_graph_features,  np.array([BalabanJ(mol), BertzCT(mol), Chi0(mol), Chi1(mol)]))
        R = np.add(R, fp)

    P = np.zeros((vector_dim))
    P_graph_features = np.zeros((4))
    for smi in smi_product:
        mol = Chem.MolFromSmiles(smi)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, \
                                                            nBits=vector_dim, useFeatures=True))
        P_graph_features = np.add(P_graph_features,  np.array([BalabanJ(mol), BertzCT(mol), Chi0(mol), Chi1(mol)]))
        P = np.add(P, fp)

    return np.concatenate((P-R, P_graph_features - R_graph_features), axis=None)


data = pd.read_csv('Ea_rmg_2017_subset.csv')

rxn_fingerprints = []
for _, row in data.iterrows():
    reactant_smiles = row['Reactant'][1:-1].replace("'","").replace(' ','').split(',')
    product_smiles = row['Product'][1:-1].replace("'","").replace(' ','').split(',')

    rxn_fingerprints.append(get_rxn_fingerprint(reactant_smiles, product_smiles))

print(np.array(rxn_fingerprints).shape)
rxn_fp_transformed = PCA(n_components=256, svd_solver = 'arpack').fit(np.array(rxn_fingerprints)).transform(np.array(rxn_fingerprints))
print(rxn_fp_transformed.shape)

data['rxn_fingerprint_before_PCA'] = [list(x) for x in rxn_fingerprints]
#data['rxn_fingerprint'] = [list(x) for x in list(rxn_fp_transformed)]
data['rxn_fingerprint'] = list(rxn_fp_transformed)

train_data = data.sample(frac=.8, random_state=1)
valid_data = data[~data.index.isin(train_data.index)]

#train_data['Train/Valid'] = ['Train'] * len(train_data)
#valid_data['Train/Valid'] = ['Valid'] * len(valid_data)
#pd.concat([train_data,valid_data])[['Reactant','Product','Ea', 'Train/Valid', 'rxn_fingerprint','rxn_fingerprint_before_PCA']].to_csv('data_with_split_and_fingerprints.csv')

X_train, X_valid = tf.constant(list(train_data.rxn_fingerprint), dtype=tf.float32), \
                   tf.constant(list(valid_data.rxn_fingerprint), dtype=tf.float32)
Y_train, Y_valid = tf.constant([[x] for x in train_data.Ea], dtype=tf.float32), \
                   tf.constant([[x] for x in valid_data.Ea], dtype=tf.float32)

#print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)


#ANN
model = tf.keras.Sequential()
#model.add(layers.InputLayer(input_shape=(256,)))
model.add(tf.keras.Input(shape=(256,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer = optimizer, loss='mae', metrics = [tf.keras.metrics.RootMeanSquaredError()])

model_path = './best_model.h5'
checkpoint = ModelCheckpoint(model_path, monitor = 'val_loss', \
                             verbose=2, save_best_only = True, mode = 'auto', period=1)
model.fit(X_train, Y_train, batch_size=128, epochs=500, verbose=2, validation_data = (X_valid, Y_valid), callbacks = [checkpoint])

model.load_weights(model_path)
_Y_train = model.predict(X_train)
_Y_valid = model.predict(X_valid)

print('R^2 train: ', r2_score(Y_train, _Y_train),
      'R^2 valid: ', r2_score(Y_valid, _Y_valid))
print('MAE train: ', np.mean(np.abs(Y_train - _Y_train)),
      'MAE valid: ', np.mean(np.abs(Y_valid - _Y_valid)))
print('RMSE train: ', np.sqrt(mean_squared_error(Y_train, _Y_train)) ,
      'RMSE valid: ', np.sqrt(mean_squared_error(Y_valid, _Y_valid)) )


Y_train, Y_valid = tf.constant(train_data.Ea, dtype=tf.float32), \
                   tf.constant(valid_data.Ea, dtype=tf.float32)

#SVR
#https://medium.com/it-paragon/support-vector-machine-regression-cf65348b6345
print("starting CV w/ grid searching")
#{'n_estimators': 10000, 'learning_rate': 0.004, 'min_child_weight': 3, 'max_depth': 10, 'subsample': 0.5}
#param_grid = {'C': [1400.0], 'epsilon': [0.03] }
param_grid = {'C': [800.0,1000.0,1200.0,1400.0,1600.0,1800.0], 'epsilon': [0.02,0.03,0.04,0.05] }
print(param_grid)
grid_search = model_selection.GridSearchCV(\
                                estimator=SVR(cache_size=1200),\
                                #param_grid=param_grid, cv=5,n_jobs=2,scoring="r2",)
                                param_grid=param_grid, cv=5,n_jobs=2,scoring="neg_mean_absolute_error",)

grid_search.fit(np.array(X_train),np.array(Y_train))

print(grid_search.best_params_)
print(grid_search.best_score_)

_Y_train = grid_search.predict(np.array(X_train))
_Y_valid = grid_search.predict(np.array(X_valid))

print('R^2 train: ', r2_score(Y_train, _Y_train),
      'R^2 valid: ', r2_score(Y_valid, _Y_valid))
print('MAE train: ', np.mean(np.abs(Y_train - _Y_train)),
      'MAE valid: ', np.mean(np.abs(Y_valid - _Y_valid)))
print('RMSE train: ', np.sqrt(mean_squared_error(Y_train, _Y_train)) ,
      'RMSE valid: ', np.sqrt(mean_squared_error(Y_valid, _Y_valid)) )


#XGB
dtrain = xgb.DMatrix(X_train, Y_train)
dvalid = xgb.DMatrix(X_valid, Y_valid)

param = { 'booster':'dart','silent':0,'learning_rate':0.05,'subsample':0.6,'max_depth':60, 'metrics':['mae']  }
bst = xgb.train(param,dtrain,10000,[(dvalid,'eval'),(dtrain,'train')], early_stopping_rounds=20)

_Y_train = bst.predict(xgb.DMatrix(X_train))
_Y_valid = bst.predict(xgb.DMatrix(X_valid))

print('R^2 train: ', r2_score(Y_train, _Y_train),
      'R^2 valid: ', r2_score(Y_valid, _Y_valid))
print('MAE train: ', np.mean(np.abs(Y_train - _Y_train)),
      'MAE valid: ', np.mean(np.abs(Y_valid - _Y_valid)))
print('RMSE train: ', np.sqrt(mean_squared_error(Y_train, _Y_train)) ,
      'RMSE valid: ', np.sqrt(mean_squared_error(Y_valid, _Y_valid)) )

