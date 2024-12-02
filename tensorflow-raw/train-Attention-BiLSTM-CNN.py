import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout,Layer,Lambda
from keras.layers import Flatten, Dense, Concatenate, Reshape, LSTM,BatchNormalization,Dropout
from keras.models import Sequential, Model
from keras.utils import plot_model

import keras
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.config.list_physical_devices('GPU'))
from keras import backend as K
import datetime
from sklearn.model_selection import KFold
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping



#==================================Data Loading and reshaping=====================================
num_classes = 4
batch_size = 128
img_rows, img_cols, num_chan = 8, 9, 4

falx = np.load("D:/BigData/SEED_IV/SEED_IV/DE0.5s/session_1_2_3/t6x_89.npy")
y = np.load("D:/BigData/SEED_IV/SEED_IV/DE0.5s/session_1_2_3/t6y_89.npy")
print('{}-{}'.format('falx shape', falx.shape))
print('{}-{}'.format('y shape', y.shape))

one_y = to_categorical(y, num_classes)  
print('{}-{}'.format('one_y categorical shape', one_y.shape))

one_falx_1 = falx.reshape((-1, 6, img_rows, img_cols, 5))  # reshape 15xeach person's segments
one_falx = one_falx_1[:,:,:,:,1:5]  # only 4 bands since last band reflect sleep feature



# Define the AttentionLayer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zero', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)

        # Reshape e for correct dot product
        e = Reshape((6, 256))(e)  # Reshape to (None, 6, 256)
        e = K.dot(e, self.W)  # Dot product with W (512, 512)
        e = Reshape((1, 6, 256))(e)  # Reshape to (None, 1, 6, 256)

        e = Lambda(lambda x: K.sum(x, axis=-1))(e)  # Sum over the last dimension
        a = K.softmax(e, axis=2)  # Apply softmax over the time dimension (axis 2)
        output = x * K.expand_dims(a, axis=-1)  # Expand dims to match x's shape
        return K.sum(output, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
#def base-network
def create_base_network(input_dim):

            seq = Sequential()
            seq.add(Conv2D(32, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
            seq.add(BatchNormalization())
            seq.add(Dropout(0.2))
            seq.add(Conv2D(128, 4, activation='relu', padding='same', name='conv2'))
            seq.add(BatchNormalization())
            seq.add(Dropout(0.2))
            seq.add(Conv2D(256, 4, activation='relu', padding='same', name='conv3'))
            seq.add(BatchNormalization())
            seq.add(Dropout(0.2))
            seq.add(Conv2D(64, 1, activation='relu', padding='same', name='conv4'))
            seq.add(MaxPooling2D(2, 2, name='pool1'))
            seq.add(Flatten(name='fla1'))
            seq.add(Dense(256, activation='relu', name='dense1'))
            seq.add(BatchNormalization())
            seq.add(Dropout(0.3))
            seq.add(Reshape((1, 256), name='reshape'))

            return seq

#====================================start train=================================

acc_list = []
std_list = []
all_acc = []

# Initialize KFold with the desired number of splits and split ratio
kf = KFold(n_splits=5, shuffle=True, random_state=42)

K.clear_session()
start = datetime.time()
  
for train_val_indices, test_indices in kf.split(one_falx):
        X_train_val, X_test = one_falx[train_val_indices], one_falx[test_indices]
        y_train_val, y_test = one_y[train_val_indices], one_y[test_indices]

        # X_train_val,y_train_val are for training, 
        # X_test, y_test used for the final evaluation of the model after training

        # Split the training folds further with a 1:1 ratio, or 7:3 ratio
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.3, random_state=42, stratify=y_train_val.argmax(1)
        )

        #4 folds (80% of the data) used for training and validation combined.
        # Within those 4 folds, a 7:3 or 1:1 split is created for training and validation.
        # 1 fold (20% of the data) used as the test set.

        img_size = (img_rows, img_cols, num_chan)

        base_network = create_base_network(img_size)
        input_1 = Input(shape=img_size)
        input_2 = Input(shape=img_size)
        input_3 = Input(shape=img_size)
        input_4 = Input(shape=img_size)
        input_5 = Input(shape=img_size)
        input_6 = Input(shape=img_size)



        out_all = Concatenate(axis=1)([base_network(input_1), base_network(input_2), base_network(input_3), base_network(input_4), base_network(input_5), base_network(input_6)])
        attention_output = AttentionLayer()(out_all)
        lstm_layer = Bidirectional(LSTM(128, name='lstm'))(out_all)
        out_layer = Dense(4, activation='softmax', name='out')(lstm_layer)
        model = Model([input_1, input_2, input_3, input_4, input_5, input_6], out_layer)
        plot_model(model, to_file='model.png', show_shapes=True)
        #model.summary()

        # Compile model
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.adam_v2.Adam(learning_rate=0.0005),
                      metrics=['accuracy'])
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)




        #tensorboard
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
       
        # Fit the model
        model.fit([
           X_train[:, 0, :, :, :], 
           X_train[:, 1, :, :, :], 
           X_train[:, 2, :, :, :], 
           X_train[:, 3, :, :, :], 
           X_train[:, 4, :, :, :], 
           X_train[:, 5, :, :, :]], 
          y_train, epochs=150, batch_size=batch_size, verbose=1,
          validation_data=([X_val[:, 0, :, :, :],
                                   X_val[:, 1, :, :, :],
                                   X_val[:, 2, :, :, :],
                                   X_val[:, 3, :, :, :],
                                   X_val[:, 4, :, :, :],
                                   X_val[:, 5, :, :, :]],
                                  y_val),
                  callbacks=[early_stopping,tensorboard_callback])
        
        # evaluate the model() provides the performance of the model on the test set of that specific fold
        scores = model.evaluate([
           X_test[:, 0, :, :, :], 
           X_test[:, 1, :, :, :], 
           X_test[:, 2, :, :, :], 
           X_test[:, 3, :, :, :], 
           X_test[:, 4, :, :, :], 
           X_test[:, 5, :, :, :]], 
          y_test,  verbose=0)
        model.save('Attention-BiLSTM-CNN.h5')
        print("%.2f%%" % (scores[1] * 100)) # Accuracy
        all_acc.append(scores[1] * 100)
    

print("mean acc: {}".format(np.mean(all_acc)))
print("std acc: {}".format(np.std(all_acc)))
acc_list.append(np.mean(all_acc))
std_list.append(np.std(all_acc))
end = datetime.time()
print("%.2f" % (end - start))   # run time        


