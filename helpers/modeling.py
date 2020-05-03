import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)    

class KerasValInfo(tf.keras.callbacks.Callback):
    
    def __init__(self, n_verbose_step=25):
        super(KerasValInfo, self).__init__()
        self.n_verbose = n_verbose_step
        self.accs = []
        self.val_accs = []

    def on_train_begin(self, logs={}):
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        self.val_accs.append(logs['val_accuracy'])
        self.accs.append(logs['accuracy'])
        if epoch % self.n_verbose == 0:
            print("Epoch: {} - accuracy: {:.4f} - val_accuracy: {:.4f}".format(epoch, logs['accuracy'], logs['val_accuracy']))
            
    def on_train_end(self, epoch, logs={}):
        print("Train Finished.")
        print("Total epochs: {}".format(len(self.accs)))
        print("Average Accuracy for training: {:.4f}".format(np.mean(self.accs)))
        print("Std Accuracy for training: {:.4f}".format(np.std(self.accs)))
        print("Average Val Accuracy for training: {:.4f}".format(np.mean(self.val_accs)))        
        print("Std Val Accuracy for training: {:.4f}".format(np.std(self.val_accs)))                
    
    
    

def create_cnn_model(input_shape, n_classes=8):
    import tensorflow as tf
    
    
    
    METRICS = [
          tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
#           tf.keras.metrics.Precision(name='precision'),
#           tf.keras.metrics.Recall(name='recall'),
#           f1_macro
    ]     
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),padding='same', input_shape=input_shape , activation='relu'
                               , kernel_initializer='he_normal')
        , tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')
        , tf.keras.layers.Dropout(0.25)
        , tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),padding='same', activation='relu')
        , tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        , tf.keras.layers.Dropout(0.25)
        , tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')
        , tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')
        , tf.keras.layers.Dropout(0.25)
        , tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')
        , tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same')
        , tf.keras.layers.Dropout(0.25)
        
        
        , tf.keras.layers.Flatten()
        , tf.keras.layers.Dense(512, activation='relu')
#         , tf.keras.layers.Dropout(0.5)
        , tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(loss=tf.keras.losses.categorical_crossentropy
                 , optimizer=tf.keras.optimizers.Adam()
                 , metrics=METRICS)
    
    return model


def create_mlp_model(input_shape, n_classes=8, learning_rate=0.001):
    import tensorflow as tf
    
    
    
    METRICS = [
          tf.keras.metrics.Accuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          f1_macro
    ]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape)
        , tf.keras.layers.Dense(256, activation='relu')
        , tf.keras.layers.Dense(256, activation='relu')
        , tf.keras.layers.Dense(256, activation='relu')
        , tf.keras.layers.Dense(256, activation='relu')
        , tf.keras.layers.Dense(256, activation='relu')
        , tf.keras.layers.Dropout(0.25)       
        , tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(loss=tf.keras.losses.categorical_crossentropy
                 , optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
                 #, optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
                 #, optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                 #, optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                 , metrics=METRICS)
    
    return model    


def test_train_split(features, target, val=True, test_size= 0.10, val_size=0.15):
    test_size = test_size
    val_size = val_size

    test_weight = 0.10 / 0.25 # after first split, weights of data changes. 
    la = LabelEncoder()

    la.fit(target) 
    
    if val:
        x_train, x_other, y_train, y_other = train_test_split(features, target, stratify=target, test_size=test_size+val_size)
        x_val, x_test, y_val, y_test = train_test_split(x_other, y_other, stratify=y_other, test_size=test_weight)
        x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape

        #normalization
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        ranges_max = np.max(x_train, axis=0)
        ranges_min = np.min(x_train, axis=0)

        #z_scoore
    #     x_train = (x_train - mean) / std
    #     x_va = (x_val - mean) / std
    #     x_test = (x_test - mean) / std


        # normalization
        x_train = (x_train - mean) / (ranges_max - ranges_min)
        x_val = (x_val - mean) / (ranges_max - ranges_min)
        x_test = (x_test - mean) / (ranges_max - ranges_min)   

        y_train = la.transform(y_train)
        y_test = la.transform(y_test)
        y_val = la.transform(y_val)    

        #x_train.shape, y_train.shape, x_test.shape, y_test.shape    
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        x_train, x_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=test_size)
        #normalization
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        ranges_max = np.max(x_train, axis=0)
        ranges_min = np.min(x_train, axis=0)
        
        # normalization
        x_train = (x_train - mean) / (ranges_max - ranges_min)
        x_test = (x_test - mean) / (ranges_max - ranges_min)   
        
        y_train = la.transform(y_train)
        y_test = la.transform(y_test)        
        
        return x_train, y_train, x_test, y_test