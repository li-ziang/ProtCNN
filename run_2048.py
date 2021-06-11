

'''
train data: train_ohe_b in shape(1086741, 2048, 21), label: y_train_b in shape(1086741,17929)
valid data: val_ohe_b in shape(126171, 2048, 21), label: val_y_b in shape(126171,17929)

The tensors in the last dimension of train_ohe_b are all one-hot tensors.
The length of 21 represents 21 types of amino acids in the sequence. 
The position of the amino acid in this position is 1 and the others are 0. 

The labels are one-hot tensors in length 17929, science there are 17929 classes in PFAM seed dataset.

'''

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,4,5,6,7"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization,GlobalMaxPool1D
from tensorflow.keras.layers import Embedding, Bidirectional,LSTM, GlobalMaxPooling1D
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import math

def residual_block(data, filters, d_rate,layer_index, bottleneck_factor=0.5, first_dilated_layer=2,training=False):
    shifted_layer_index = layer_index - first_dilated_layer + 1
    dilation_rate = max(1, d_rate**shifted_layer_index)
    num_bottleneck_units = math.floor(
        bottleneck_factor * filters)
    kernel_size = 21

    shortcut = data

    x = BatchNormalization()(data,training=training)
    x = Activation('relu')(x)
    x = Conv1D(num_bottleneck_units, kernel_size, dilation_rate=dilation_rate, padding='same', kernel_regularizer=l2(0.001))(x)

    #bottleneck convolution
    x = BatchNormalization()(x,training=training)
    x = Activation('relu')(x)

    x = Conv1D(filters, 1, dilation_rate=1, padding='same', kernel_regularizer=l2(0.001))(x)

    #skip connection
    x = Add()([x, shortcut])

    return x

strategy = tf.distribute.MirroredStrategy()

# # Open a strategy scope.
with strategy.scope():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    x_input = Input(shape=(2048, 21))

    #initial conv
    filters=1024
    kernel_size = 21
    x = Conv1D(filters, kernel_size, padding='same')(x_input) 

    # per-residue representation
    # res = residual_block(conv, filters, 1)
    for i in range(5):
        x = residual_block(x, filters, 3,i,training=True)

    # res = Conv1D(128, 1, padding='same', kernel_regularizer=l2(0.001))(res)
    # print(res5.shape)
    x = GlobalMaxPooling1D()(x)
    # x = Dropout(0.5)(x)
    # x = Conv1D(128, 1, padding='same')(x) 


    # softmax classifier
    # x = Flatten()(x)
    x_output = Dense(17929, kernel_regularizer=l2(0.0001))(x)
    x_output = Activation('softmax', dtype='float32', name='predictions')(x_output)

    model2 = Model(inputs=x_input, outputs=x_output)

# lr_schedule according to paper

    initial_learning_rate = 0.0001
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=1000,
    #     decay_rate=0.997,
    #     staircase=True)

    # model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
    #                 loss='categorical_crossentropy',
    #                 metrics=['accuracy'])
    model2.compile(optimizer=tf.keras.optimizers.Adam(initial_learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model2.summary()

# import tensorboard
# log_dir="./logs/"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 

import numpy as np
from scipy import sparse

train_ohe_sp= sparse.load_npz('train_ohe_2048.npz')
train_ohe_s=train_ohe_sp.toarray()
train_ohe_b=np.reshape(train_ohe_s, (1086741, 2048, 21))

y_train_sp= sparse.load_npz('y_train_2048.npz') 
y_train_b=y_train_sp.toarray()

val_ohe_sp= sparse.load_npz('val_ohe_2048.npz') 
val_ohe_s=val_ohe_sp.toarray()
val_ohe_b=np.reshape(val_ohe_s, (126171, 2048, 21))

y_val_sp= sparse.load_npz('y_val_2048.npz')
val_y_b=y_val_sp.toarray()


batch_size=32
checkpoint_filepath="model_brand_new"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
train_dataset=tf.data.Dataset.from_tensor_slices((train_ohe_b, y_train_b)).batch(batch_size)
val_dataset=tf.data.Dataset.from_tensor_slices((val_ohe_b, val_y_b)).batch(batch_size)


es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
history2=model2.fit(train_dataset, epochs=30, validation_data=val_dataset,callbacks=[model_checkpoint_callback])

model2.save_weights('model_2048_all.h5')




# def residual_block(data, filters, d_rate):
#   """
#   _data: input
#   _filters: convolution filters
#   _d_rate: dilation rate
#   """

#     shortcut = data

#     bn1 = BatchNormalization()(data)
#     act1 = Activation('relu')(bn1)
#     conv1 = Conv1D(filters, 21, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)

#     #bottleneck convolution
#     bn2 = BatchNormalization()(conv1)
#     act2 = Activation('relu')(bn2)
#     conv2 = Conv1D(filters*2, 1, padding='same', kernel_regularizer=l2(0.001))(act2)
#     bn3 = BatchNormalization()(conv2)
#     act3 = Activation('relu')(bn3)
#     conv3 = Conv1D(filters*2, 21, padding='same', kernel_regularizer=l2(0.001))(act3)
#     bn4 = BatchNormalization()(conv3)
#     act4 = Activation('relu')(bn4)
#     conv4 = Conv1D(filters, 1, padding='same', kernel_regularizer=l2(0.001))(act4)

#     #skip connection
#     x = Add()([conv4, shortcut])

#     return x