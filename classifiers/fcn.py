# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os
import time
from copy import deepcopy

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import model_compile_and_callback, freeze_and_make_layer_trainable, fit_model, predict_model

class Classifier_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model, self.callbacks = model_compile_and_callback(model, self.output_directory)
        
        return model

    def fit(self, x_train, y_train, x_val, y_val, x_test, y_test, y_true, do_pred_only=False, nb_epochs=2000, batch_size=16, train_method='normal', trainable_layers=None, nb_classes=None):
        df_metrics, self.model, self.output_directory, self.callbacks, self.verbose = fit_model(self.model, 
                                                                                        self.output_directory, 
                                                                                        self.callbacks, 
                                                                                        self.verbose, 
                                                                                        x_train, 
                                                                                        y_train, 
                                                                                        x_val, 
                                                                                        y_val, 
                                                                                        x_test, 
                                                                                        y_test, 
                                                                                        y_true, 
                                                                                        do_pred_only=do_pred_only, 
                                                                                        nb_epochs=nb_epochs, 
                                                                                        batch_size=batch_size, 
                                                                                        train_method=train_method, 
                                                                                        trainable_layers=trainable_layers, 
                                                                                        nb_classes=nb_classes)
        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics = True):
        return predict_model(self.output_directory, x_test, y_true, x_train, y_train, y_test, return_df_metrics = return_df_metrics)
