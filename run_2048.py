

'''
train data: train_ohe_b in shape(1086741, 2048, 21), label: y_train_b in shape(1086741,17929)
valid data: val_ohe_b in shape(126171, 2048, 21), label: val_y_b in shape(126171,17929)

The tensors in the last dimension of train_ohe_b are all one-hot tensors.
The length of 21 represents 21 types of amino acids in the sequence. 
The position of the amino acid in this position is 1 and the others are 0. 

The labels are one-hot tensors in length 17929, science there are 17929 classes in PFAM seed dataset.

'''

import numpy as np
from scipy import sparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
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
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import tensorflow as tf


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn,
        warmup_steps: int,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.

            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done
            learning_rate = tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )
            return learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }



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
    x = BatchNormalization()(x,training=training)
    x = Activation('relu')(x)
    x = Conv1D(filters, 1, dilation_rate=1, padding='same', kernel_regularizer=l2(0.001))(x)
    x = Add()([x, shortcut])
    return x

strategy = tf.distribute.MirroredStrategy()
import tensorboard
import datetime

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

    for i in range(5):
        x = residual_block(x, filters, 3,i,training=True)

    x = GlobalMaxPooling1D()(x)

    x_output = Dense(17929, kernel_regularizer=l2(0.0001))(x)
    x_output = Activation('softmax', dtype='float32', name='predictions')(x_output)

    model2 = Model(inputs=x_input, outputs=x_output)

    log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(log_dir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    initial_learning_rate = 0.0001
    lr_func = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.997,
        staircase=True)
    lr_schedule = WarmUp(initial_learning_rate,lr_func,1000)
    # model2.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
    #                 loss='categorical_crossentropy',
    #                 metrics=['accuracy'])
    model2.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model2.summary()

 
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
history2=model2.fit(train_dataset, epochs=30, validation_data=val_dataset,callbacks=[model_checkpoint_callback,tensorboard_callback])

model2.save_weights('model_2048_all.h5')
