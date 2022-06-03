import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from molgraph import dgl_molgraph_one_molecule
from dgl import batch
from dgl.nn.tensorflow import GraphConv

class GCN(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 activation):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = []
        self.activation = activation

        # input projection (no residual)
        self.gcn_layers.append(GraphConv(
            in_dim, num_hidden, activation=self.activation, allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, num_layers):
            self.gcn_layers.append(GraphConv(
                num_hidden, num_hidden, activation=self.activation, allow_zero_in_degree=True))
        # output projection
        self.gcn_layers.append(GraphConv(num_hidden, num_hidden, activation=None, allow_zero_in_degree=True))
        # Readout layers after averaging atom feature vectors
        self.readout_layers = [layers.Dense(32, activation='relu'), layers.Dense(16, activation='relu'), layers.Dense(1)  ]


    def call(self, features, g, segment, Max_atoms):        
        h = features

        for l in range(self.num_layers):
            h = self.gcn_layers[l](g, h)

        num_molecules = int(h.shape[0] / Max_atoms)

        atom_features_mean = tf.math.unsorted_segment_mean( h, segment_ids = segment, num_segments = num_molecules )

        # readout
        h = self.readout_layers[0](atom_features_mean)

        for l in range(1,len(self.readout_layers)):
            h = self.readout_layers[l](h)

        # (batch_size, 1) -> (batch_size)
        _Y = tf.reshape(h, [-1])
        
        return _Y

    def save_model(self, name):
        super(GCN, self).save_weights(name)
        #self.save(name)
    
    def load_model(self, name):
        super(GCN, self).load_weights(name)

def create_segment(Max_atoms, features):
    seg = tf.math.count_nonzero(features, axis=1).numpy()
    seg[seg==0] = -1

    for i, elem in enumerate(seg):
        mol_index = int( i / Max_atoms )
        if elem != -1: 
            seg[i] = mol_index
    return tf.constant(seg)

def evaluate(model, features, g, segment, Max_atoms, Y): 
    _Y = model(features, g, segment, Max_atoms)
    loss_value = tf.reduce_mean(tf.math.abs(Y-_Y))

    return loss_value, _Y



##### AqsolDB
data = pd.read_csv('AqsolDB_CHO.csv')
data = data[data.HeavyAtomCount <= 100]

train_data = data.sample(frac=.8, random_state=1)
valid_data = data[~data.index.isin(train_data.index)].sample(frac=.5, random_state=1)
test_data = data[~data.index.isin(train_data.index) & ~data.index.isin(valid_data.index)]

Y_train, Y_valid, Y_test = tf.constant(train_data.Solubility, dtype=tf.float32), \
                           tf.constant(valid_data.Solubility, dtype=tf.float32), \
                           tf.constant(test_data.Solubility, dtype=tf.float32)

#ANN
features_of_interest = ['MolWt', 'MolLogP', 'MolMR', 'HeavyAtomCount',\
'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', \
'NumValenceElectrons', 'NumAromaticRings', 'NumSaturatedRings',\
'NumAliphaticRings', 'RingCount', 'TPSA', 'LabuteASA', 'BalabanJ',\
'BertzCT']

X_train, X_valid, X_test = tf.constant(np.array(train_data[features_of_interest]), dtype=tf.float32), \
                           tf.constant(np.array(valid_data[features_of_interest]), dtype=tf.float32), \
                           tf.constant(np.array(test_data[features_of_interest]), dtype=tf.float32)



model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(16,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer = optimizer, loss='mae')
model.fit(X_train, Y_train, batch_size=128, epochs=100, verbose=2, validation_data = (X_valid, Y_valid))
print(model.evaluate(X_test, Y_test))

#GCN
Max_atoms = 100
data['molgraph'] = data.SMILES.apply(lambda x: dgl_molgraph_one_molecule(x, Max_atoms))
train_data = data.sample(frac=.8, random_state=1)
valid_data = data[~data.index.isin(train_data.index)].sample(frac=.5, random_state=1)
test_data = data[~data.index.isin(train_data.index) & ~data.index.isin(valid_data.index)]

Graphs_train = batch( list(train_data.molgraph) )
Graphs_valid = batch( list(valid_data.molgraph) )
Graphs_test  = batch( list(test_data.molgraph) )
seg_valid = create_segment(Max_atoms, Graphs_valid.ndata['feat'])
seg_test  = create_segment(Max_atoms, Graphs_test.ndata['feat'])
atom_feat_dim = Graphs_valid.ndata['feat'].shape[-1]

model = GCN(num_layers=3, 
            in_dim = atom_feat_dim, 
            num_hidden = 32, 
            activation = tf.nn.relu)

Epoch = 100
Batch_size = 128 
weight_decay = 5e-4

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, epsilon=1e-8)

train_valid_costs = []
batch_costs = []

for epoch in range(Epoch):
    #shuffle batches
    train_data_shuffled = train_data.sample(frac = 1.0, random_state = epoch)
    num_batches = int(np.ceil(len(Y_train) / Batch_size))

    for _iter in range(num_batches):
        data_batch = train_data_shuffled.iloc[_iter*Batch_size:(_iter+1)*Batch_size]
        Y_batch    = tf.constant(data_batch.Solubility, dtype=tf.float32)

        train_graphs_batch = batch( list(data_batch.molgraph) )
        seg_train_batch = create_segment(Max_atoms, train_graphs_batch.ndata['feat'])

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            logits = model(train_graphs_batch.ndata['feat'], train_graphs_batch, seg_train_batch, Max_atoms, training=True)

            #loss_value = tf.math.sqrt(tf.reduce_mean(tf.pow((Y_batch-logits),2)))
            loss_value = tf.reduce_mean(tf.math.abs(Y_batch-logits))
            for weight in model.trainable_weights:
                loss_value = loss_value + weight_decay * tf.nn.l2_loss(weight)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #print('batch '+str(_iter)+': MAE- '+str(loss_value.numpy()))
        batch_costs.append(loss_value.numpy())

    seg_train = create_segment(Max_atoms, Graphs_train.ndata['feat'])
    train_loss, _Y_train = evaluate(model, Graphs_train.ndata['feat'], Graphs_train, \
                                                        seg_train, Max_atoms,  Y_train)
    valid_loss, _Y_valid = evaluate(model, Graphs_valid.ndata['feat'], Graphs_valid, \
                                                        seg_valid, Max_atoms,  Y_valid)
    print('epoch '+str(epoch)+': train loss- '+str(train_loss.numpy()))
    print('epoch '+str(epoch)+': valid loss- '+str(valid_loss.numpy()))

    train_valid_costs.append([train_loss.numpy(), valid_loss.numpy()])
test_loss, _Y_test = evaluate(model, Graphs_test.ndata['feat'], Graphs_test, \
                                                 seg_test, Max_atoms,  Y_test)
print('test loss- '+str(test_loss.numpy()))




