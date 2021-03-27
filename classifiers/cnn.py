# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import model_compile_and_callback, freeze_and_make_layer_trainable, fit_model, predict_model


class Classifier_CNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, min_lr=0.0001):
        self.output_directory = output_directory

        if build == True:
            self.min_lr = min_lr
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        # padding = 'valid'
        padding = 'same'
        # activation = 'sigmoid'
        activation = 'relu'
        input_layer = keras.layers.Input(input_shape)

        # if input_shape[0] < 60:  # for italypowerondemand dataset
        #     padding = 'same'

        conv1 = keras.layers.Conv1D(
            filters=6, kernel_size=7, padding=padding, activation=activation)(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(
            filters=12, kernel_size=7, padding=padding, activation=activation)(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        # output_layer = keras.layers.Dense(
        #     units=nb_classes, activation='sigmoid')(flatten_layer)

        output_layer = keras.layers.Dense(
            units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model, self.callbacks = model_compile_and_callback(
            model, self.output_directory, min_lr=self.min_lr)

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
                                                                                                nb_classes=nb_classes,
                                                                                                min_lr=self.min_lr)
        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        return predict_model(self.output_directory, x_test, y_true, x_train, y_train, y_test, return_df_metrics=return_df_metrics)
