import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, Reshape, BatchNormalization, Bidirectional, GRU, Add, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.models import  Model
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout,Layer,Lambda
from keras.layers import Flatten, Dense, Concatenate, Reshape, LSTM,BatchNormalization,Dropout
from keras.models import Model

import keras
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.config.list_physical_devices('GPU'))
from keras import backend as K
import time
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping
import datetime

# Set GPU configuration
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.config.list_physical_devices('GPU'))

# Data Loading and Reshaping
num_classes = 4
batch_size = 128
img_rows, img_cols, num_chan = 8, 9, 4

falx = np.load("D:/BigData/SEED_IV/SEED_IV/DE0.5s/session_1_2_3/t6x_89.npy")
y = np.load("D:/BigData/SEED_IV/SEED_IV/DE0.5s/session_1_2_3/t6y_89.npy")
print('{}-{}'.format('falx shape', falx.shape))
print('{}-{}'.format('y shape', y.shape))

one_y = to_categorical(y, num_classes)
print('{}-{}'.format('one_y categorical shape', one_y.shape))

one_falx_1 = falx.reshape((-1, 6, img_rows, img_cols, 5))  # reshape each person's segments
one_falx = one_falx_1[:, :, :, :, 1:5]  # only 4 bands since last band reflects sleep feature

# Base CNN Network with Residual Blocks
def create_base_network(input_dim):
    input_layer = Input(shape=input_dim)
    x = Conv2D(32, 5, activation='relu', padding='same', name='conv1')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # First Residual Block
    residual = Conv2D(128, 4, activation='relu', padding='same', name='conv2')(x)
    residual = BatchNormalization()(residual)
    residual = Dropout(0.2)(residual)
    x = Conv2D(128, 4, activation='relu', padding='same', name='conv2_residual')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Add()([x, residual])
    
    # Second Residual Block
    residual = Conv2D(256, 4, activation='relu', padding='same', name='conv3')(x)
    residual = BatchNormalization()(residual)
    residual = Dropout(0.2)(residual)
    x = Conv2D(256, 4, activation='relu', padding='same', name='conv3_residual')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Add()([x, residual])
    
    x = Conv2D(64, 1, activation='relu', padding='same', name='conv4')(x)
    x = MaxPooling2D(2, 2, name='pool1')(x)
    x = Flatten(name='fla1')(x)
    x = Dense(256, activation='relu', name='dense1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Reshape((1, 256), name='reshape')(x)
    
    return Model(inputs=input_layer, outputs=x)

# Vision Transformer Encoder Block
def transformer_encoder(inputs, embed_dim, num_heads, ff_dim, rate=0.1):
    # Multi-Head Self Attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = Dropout(rate)(attn_output)
    # Add & Norm
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    # Feed Forward Network
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    # Add & Norm
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# Training and Evaluation
acc_list = []
std_list = []
all_acc = []

# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

K.clear_session()
start = time.time()

for train_val_indices, test_indices in kf.split(one_falx):
    X_train_val, X_test = one_falx[train_val_indices], one_falx[test_indices]
    y_train_val, y_test = one_y[train_val_indices], one_y[test_indices]

    # Split training into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.3, random_state=42, stratify=y_train_val.argmax(1)
    )

    img_size = (img_rows, img_cols, num_chan)

    # Create base network
    base_network = create_base_network(img_size)
                        
    inputs = [Input(shape=img_size) for _ in range(6)]
    out_all = Concatenate(axis=1)([base_network(inp) for inp in inputs])

    # Transformer Encoder Block
    embed_dim = 256
    num_heads = 4
    ff_dim = 512
    transformer_output = transformer_encoder(out_all, embed_dim, num_heads, ff_dim)

    # Bidirectional GRU Layer
    bidir_gru = Bidirectional(GRU(128, return_sequences=False))(transformer_output)

    # Output Layer
    out_layer = Dense(4, activation='softmax', name='out')(bidir_gru)
    model = Model(inputs, out_layer)

    #tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.adam_v2.Adam(learning_rate=0.0005),
                      metrics=['accuracy'])
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

    #tensorboard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit the model
    model.fit(
    [X_train[:, i, :, :, :] for i in range(6)],
    y_train,
    epochs=150,
    batch_size=batch_size,
    verbose=1,
    validation_data=([X_val[:, i, :, :, :] for i in range(6)], y_val),
    callbacks=[early_stopping, reduce_lr,tensorboard_callback])


    # Evaluate the model
    scores = model.evaluate(
    [X_val[:, i, :, :, :] for i in range(6)],
    y_val,
    verbose=0)
    model.save('ResCNN-TransGRU.h5')
    
    print("%.2f%%" % (scores[1] * 100))
    all_acc.append(scores[1] * 100)

print("mean acc: {}".format(np.mean(all_acc)))
print("std acc: {}".format(np.std(all_acc)))
acc_list.append(np.mean(all_acc))
std_list.append(np.std(all_acc))
end = time.time()
print("%.2f" % (end - start))  # Run time
