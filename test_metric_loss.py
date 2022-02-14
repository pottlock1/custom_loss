# testing metric based loss functions
import loss
import util
from loss import precision_at_recall_loss
import tensorflow as tf
import os
import numpy as np 
import random as rn
import pandas as pd
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def set_seed():
    os.environ['PYTHONHASHSEED'] = '0'                      
    np.random.seed(123)
    rn.seed(123)
    tf.random.set_seed(123) # for TF == 2.x 

set_seed()

class precision_at_recall(tf.keras.losses.Loss):
    def __init__(self, target_recall=0.1, name="precision_at_recall"):
        super().__init__(name=name)
        self.target_recall = target_recall

    def call(self, y_true, y_pred):
        y_true = K.reshape(y_true, (-1, 5))
        y_pred = K.reshape(y_pred, (-1, 5))  
        return precision_at_recall_loss(y_true, y_pred, self.target_recall)[0]

# creating training and test data
x = np.random.normal(loc = 0, scale = 1, size = (10000, 10))
print(x.shape)

y = np.float64((np.random.normal(loc = 0.5, scale = 1, size = (10000, 5))>0.5)*1)
print(y.shape)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = 123)



# creating model
loss_fn=precision_at_recall()
inputs = tf.keras.Input(shape=(train_x.shape[1],))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)

outputs = tf.keras.layers.Dense(train_y.shape[1], activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 4, mode = 'min')

adam=tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam,loss=loss_fn,metrics=['accuracy'],run_eagerly=True)
out=model.fit(train_x,train_y, epochs=10,batch_size=10, verbose=1, validation_split = 0.2, callbacks = [es])

# prediction and testing performance
y_pred = (model.predict(test_x)>0.5)*1

print(classification_report(test_y, y_pred))