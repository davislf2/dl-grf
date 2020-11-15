# MLP model
import matplotlib.pyplot as plt
from utils.utils import model_compile_and_callback, freeze_and_make_layer_trainable, fit_model, predict_model
from utils.utils import calculate_metrics
from utils.utils import save_logs
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
matplotlib.use('agg')


class Classifier_MLP:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if(verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        # flatten/reshape because when multivariate all should be on the same axis
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(
            nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model, self.callbacks = model_compile_and_callback(
            model, self.output_directory, optimizer=keras.optimizers.Adadelta())
        # model, self.callbacks = model_compile_and_callback(model, self.output_directory, optimizer='adam')

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

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        return predict_model(self.output_directory, x_test, y_true, x_train, y_train, y_test, return_df_metrics=return_df_metrics)
