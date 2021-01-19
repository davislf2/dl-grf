import time
import ast
import sklearn
import tensorflow.keras as keras
import tensorflow as tf
# import tensorflow.keras.backend as keras_backend
import tensorflow.compat.v1.keras.backend as keras_backend
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from utils.constants import MTS_DATASET_NAMES
from utils.constants import ITERATIONS
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
import utils
import operator
import os
import matplotlib.pyplot as plt
from builtins import print
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
# import shap
# tf.compat.v1.disable_eager_execution()

matplotlib.use('agg')

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'


def readucr(filename, remove_docstr=False):
    if remove_docstr:
        pass
    data = np.loadtxt(filename, delimiter=',')
    # Y = data[:, 0]
    # X = data[:, 1:]
    Y = data[:, -1]
    X = data[:, 0:-1]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + \
        classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def read_dataset(root_dir, archive_name, dataset_name, file_ext='', remove_docstr=False):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')

    if archive_name == 'mts_archive':
        file_name = cur_root_dir + '/archives/' + \
            archive_name + '/' + dataset_name + '/'
        x_train = np.load(file_name + 'x_train.npy')
        y_train = np.load(file_name + 'y_train.npy')
        x_test = np.load(file_name + 'x_test.npy')
        y_test = np.load(file_name + 'y_test.npy')
        if os.path.exists(file_name + 'x_val.npy') and os.path.exists(file_name + 'y_val.npy'):
            x_val = np.load(file_name + 'x_val.npy')
            y_val = np.load(file_name + 'y_val.npy')
            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy(), x_val.copy(), y_val.copy())
        else:
            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    elif archive_name == 'pretrain':
        file_name = cur_root_dir + '/archives/' + \
            archive_name + '/' + dataset_name + '/'
        p_x_train = np.load(file_name + 'PreTrain_x_train.npy')
        p_y_train = np.load(file_name + 'PreTrain_y_train.npy')
        p_x_test = np.load(file_name + 'PreTrain_x_test.npy')
        p_y_test = np.load(file_name + 'PreTrain_y_test.npy')
        o_x_train = np.load(file_name + 'OneShot_x_train.npy')
        o_y_train = np.load(file_name + 'OneShot_y_train.npy')
        o_x_test = np.load(file_name + 'OneShot_x_test.npy')
        o_y_test = np.load(file_name + 'OneShot_y_test.npy')

        datasets_dict[dataset_name] = (p_x_train.copy(), p_y_train.copy(), p_x_test.copy(), p_y_test.copy(),
                                       o_x_train.copy(), o_y_train.copy(), o_x_test.copy(), o_y_test.copy())

    elif archive_name == 'UCRArchive_2018':
        root_dir_dataset = cur_root_dir + '/archives/' + \
            archive_name + '/' + dataset_name + '/'
        df_train = pd.read_csv(
            root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

        df_test = pd.read_csv(root_dir_dataset + '/' +
                              dataset_name + '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]

        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])

        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    else:
        file_name = cur_root_dir + '/archives/' + \
            archive_name + '/' + dataset_name + '/' + dataset_name
        x_train, y_train = readucr(
            file_name + '_TRAIN' + file_ext, remove_docstr)
        x_test, y_test = readucr(file_name + '_TEST' + file_ext, remove_docstr)
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    return datasets_dict


def read_all_datasets(root_dir, archive_name, split_val=False):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    dataset_names_to_sort = []

    if archive_name == 'mts_archive':

        for dataset_name in MTS_DATASET_NAMES:
            root_dir_dataset = cur_root_dir + '/archives/' + \
                archive_name + '/' + dataset_name + '/'

            x_train = np.load(root_dir_dataset + 'x_train.npy')
            y_train = np.load(root_dir_dataset + 'y_train.npy')
            x_test = np.load(root_dir_dataset + 'x_test.npy')
            y_test = np.load(root_dir_dataset + 'y_test.npy')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
    elif archive_name == 'UCRArchive_2018':
        for dataset_name in DATASET_NAMES_2018:
            root_dir_dataset = cur_root_dir + '/archives/' + \
                archive_name + '/' + dataset_name + '/'

            df_train = pd.read_csv(
                root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

            df_test = pd.read_csv(
                root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]

            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])

            x_train.columns = range(x_train.shape[1])
            x_test.columns = range(x_test.shape[1])

            x_train = x_train.values
            x_test = x_test.values

            # znorm
            std_ = x_train.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

            std_ = x_test.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    else:
        for dataset_name in DATASET_NAMES:
            root_dir_dataset = cur_root_dir + '/archives/' + \
                archive_name + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_TRAIN')
            x_test, y_test = readucr(file_name + '_TEST')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

            dataset_names_to_sort.append((dataset_name, len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATASET_NAMES)):
            DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict

# Note: Original version, which can work
# def model_compile_and_callback(model, output_directory):
#     # model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(),
#     #               metrics=['accuracy'])

#     print("loss:", 'categorical_crossentropy')
#     print("optimizer:", keras.optimizers.Adam())
#     model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics=['accuracy'])

#     reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
#         min_lr=0.0001)

#     file_path = output_directory+'best_model.hdf5'

#     model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
#         save_best_only=True)

#     callbacks = [reduce_lr, model_checkpoint]
#     return model, callbacks


def model_compile_and_callback(model, output_directory, loss='categorical_crossentropy', optimizer='adam', min_lr=0.0001):

    # model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(),
    #               metrics=['accuracy'])

    # Note: it can't use optimizer = keras.optimizers.Adam() in args of model_compile_and_callback, which will cause an error
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
    #     min_lr=0.0001)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                  min_lr=min_lr)

    file_path = output_directory + 'best_model.hdf5'

    # Note: it can't use optimizer = keras.optimizers.Adam() in args of model_compile_and_callback, you need save_weights_only
    # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
    #     save_best_only=True, save_weights_only=True)

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                       save_best_only=True)

    callbacks = [reduce_lr, model_checkpoint]
    return model, callbacks


# def freeze_and_make_layer_trainable(model, nb_classes, trainable_layers=None):
#     num_layers = len(model.layers)
#     # print("*"*30, "old_model summary:", model.summary(), "*"*30)
#     print("*"*30, "old_model summary:", "*"*30)
#     model.summary()

#     new_model = keras.Sequential()
#     for layer in model.layers[:-1]: # go through until last layer
#         layer._name = layer.name + str("_2")
#         new_model.add(layer)
#     new_model.add(keras.layers.Dense(activation='softmax', units=nb_classes))
#     new_model._name = 'finetuned_model'

#     print("*"*30, "new_model summary:", new_model.summary(), "*"*30)
#     trainable_layers = ast.literal_eval(trainable_layers)
#     for i, layer in enumerate(new_model.layers):
#         if trainable_layers and i in trainable_layers:
#             layer.trainable = True
#         else:
#             layer.trainable = False
#         print(f'i:{i}, layer.name:{layer.name}, layer.trainable:{layer.trainable}')
#     return new_model


def freeze_and_make_layer_trainable(model, nb_classes, trainable_layers=None):
    num_layers = len(model.layers)
    # print("*"*30, "old_model summary:", model.summary(), "*"*30)
    print("*"*30, "old_model summary:", "*"*30)
    model.summary()

    # layer number obtained from model summary above
    base_output = model.layers[num_layers-2].output
    new_output = keras.layers.Dense(
        activation='softmax', units=nb_classes, name='softmax_layer')(base_output)
    new_model = keras.models.Model(
        inputs=model.inputs, outputs=new_output, name='finetuned_model')

    # model = deepcopy(new_model)
    print("*"*30, "new_model summary:", new_model.summary(), "*"*30)
    trainable_layers = ast.literal_eval(trainable_layers)
    for i, layer in enumerate(new_model.layers):
        if trainable_layers and i in trainable_layers:
            layer.trainable = True
        else:
            layer.trainable = False
        print(f'i:{i}, layer.name:{layer.name}, layer.trainable:{layer.trainable}')
    return new_model


def fit_model(model, output_directory, callbacks, verbose, x_train, y_train, x_val, y_val, x_test, y_test, y_true, do_pred_only=False, nb_epochs=2000, batch_size=16, train_method='normal', trainable_layers=None, nb_classes=None, min_lr=0.0001):
    print('-'*20, 'fit_model', '-'*20)
    print("train_method:", train_method)
    nb_epochs = int(nb_epochs)
    batch_size = int(batch_size)
    if not tf.test.is_gpu_available:
        print('error')
        exit()
    if do_pred_only:
        results = model.evaluate(x_test, y_test, batch_size=128)
        print("results:", results)
        y_pred = model.predict(x_test)
        print("y_pred:", y_pred)
        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)
    else:
        # x_val and y_val are only used to monitor the test loss and NOT for training

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()
        if 'finetune' in train_method:
            model = freeze_and_make_layer_trainable(
                model, nb_classes, trainable_layers=trainable_layers)
            model, callbacks = model_compile_and_callback(
                model, output_directory, min_lr=min_lr)

        if x_val is not None and y_val is not None:
            hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                             verbose=verbose, validation_data=(x_val, y_val), callbacks=callbacks)
        else:
            hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                             verbose=verbose, callbacks=callbacks)

        duration = time.time() - start_time

        if 'normal' not in train_method:
            last_model_file_path = output_directory + f'{train_method}_last_model.hdf5'
            best_model_file_path = output_directory + f'{train_method}_best_model.hdf5'
            os.rename(output_directory + 'best_model.hdf5',
                      best_model_file_path)
        else:
            last_model_file_path = output_directory + 'last_model.hdf5'
            best_model_file_path = output_directory + 'best_model.hdf5'

        model.save(last_model_file_path)
        model_loaded = keras.models.load_model(best_model_file_path)

        # y_pred = model.predict(x_val)
        y_pred = model_loaded.predict(x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(
            output_directory, hist, y_pred, y_true, duration, train_method=train_method)

        keras.backend.clear_session()

        return df_metrics, model, output_directory, callbacks, verbose


##### shap
# # explain how the input to the 37th layer of the model explains the top two classes
# def map2layer(x, model, layer):
#     feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
#     return keras_backend.get_session().run(model.layers[layer].input, feed_dict)


# def fit_model(model, output_directory, callbacks, verbose, x_train, y_train, x_val, y_val, x_test, y_test, y_true, do_pred_only=False, nb_epochs=2000, batch_size=16, train_method='normal', trainable_layers=None, nb_classes=None, min_lr=0.0001):
#     print('-'*20, 'fit_model', '-'*20)
#     print("train_method:", train_method)
#     nb_epochs = int(nb_epochs)
#     batch_size = int(batch_size)

#     layer_to_check = 36 # 37
#     to_explain = x_train[[39, 41]]
#     e = shap.GradientExplainer(
#         (model.layers[layer_to_check].input, model.layers[-1].output),
#         map2layer(x_train, model, layer_to_check),
#         local_smoothing=0 # std dev of smoothing noise
#     )
#     shap_values, indexes = e.shap_values(map2layer(to_explain, model, layer_to_check), ranked_outputs=2)

#     # get the names for the classes
#     # index_names = np.vectorize(y_train)
#     index_names = y_train

#     # plot the explanations
#     # print("index_names.shape[0]:", index_names.shape[0])
#     print("len(index_names):", len(index_names))
#     print("shap_values[0].shape:", shap_values[0].shape)
#     print("shap_values[0].shape[0]:", shap_values[0].shape[0])

#     shap.image_plot(shap_values, to_explain, index_names)

#     df_metrics = None
#     model = None 
#     output_directory = None 
#     callbacks = None 
#     verbose = None

#     return df_metrics, model, output_directory, callbacks, verbose


def predict_model(output_directory, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
    start_time = time.time()
    model_path = output_directory + 'best_model.hdf5'
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_test)
    if return_df_metrics:
        y_pred = np.argmax(y_pred, axis=1)
        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        return df_metrics
    else:
        test_duration = time.time() - start_time
        save_test_duration(output_directory +
                           'test_duration.csv', test_duration)
        return y_pred


def get_func_length(x_train, x_test, func):
    if func == min:
        func_length = np.inf
    else:
        func_length = 0

    n = x_train.shape[0]
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[1])

    n = x_test.shape[0]
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[1])

    return func_length


def transform_to_same_length(x, n_var, max_length):
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)

    # loop through each time series
    for i in range(n):
        mts = x[i]
        curr_length = mts.shape[1]
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, idx.max(), max_length)
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            f = interp1d(idx, ts, kind='cubic')
            new_ts = f(idx_new)
            ucr_x[i, :, j] = new_ts

    return ucr_x


def transform_mts_to_ucr_format():
    mts_root_dir = '/mnt/Other/mtsdata/'
    mts_out_dir = '/mnt/nfs/casimir/archives/mts_archive/'
    for dataset_name in MTS_DATASET_NAMES:
        # print('dataset_name',dataset_name)

        out_dir = mts_out_dir + dataset_name + '/'

        # if create_directory(out_dir) is None:
        #     print('Already_done')
        #     continue

        a = loadmat(mts_root_dir + dataset_name + '/' + dataset_name + '.mat')
        a = a['mts']
        a = a[0, 0]

        dt = a.dtype.names
        dt = list(dt)

        for i in range(len(dt)):
            if dt[i] == 'train':
                x_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'test':
                x_test = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'trainlabels':
                y_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'testlabels':
                y_test = a[i].reshape(max(a[i].shape))

        # x_train = a[1][0]
        # y_train = a[0][:,0]
        # x_test = a[3][0]
        # y_test = a[2][:,0]

        n_var = x_train[0].shape[0]

        max_length = get_func_length(x_train, x_test, func=max)
        min_length = get_func_length(x_train, x_test, func=min)

        print(dataset_name, 'max', max_length, 'min', min_length)
        print()
        # continue

        x_train = transform_to_same_length(x_train, n_var, max_length)
        x_test = transform_to_same_length(x_test, n_var, max_length)

        # save them
        np.save(out_dir + 'x_train.npy', x_train)
        np.save(out_dir + 'y_train.npy', y_train)
        np.save(out_dir + 'x_test.npy', x_test)
        np.save(out_dir + 'y_test.npy', y_test)

        print('Done')


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def generate_results_csv(output_file_name, root_dir):
    res = pd.DataFrame(data=np.zeros((0, 7), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name',
                                'precision', 'accuracy', 'recall', 'duration'])
    for classifier_name in CLASSIFIERS:
        for archive_name in ARCHIVE_NAMES:
            datasets_dict = read_all_datasets(root_dir, archive_name)
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                for dataset_name in datasets_dict.keys():
                    output_dir = root_dir + '/results/' + classifier_name + '/' \
                        + curr_archive_name + '/' + dataset_name + '/' + 'df_metrics.csv'
                    if not os.path.exists(output_dir):
                        continue
                    df_metrics = pd.read_csv(output_dir)
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['archive_name'] = archive_name
                    df_metrics['dataset_name'] = dataset_name
                    res = pd.concat((res, df_metrics), axis=0, sort=False)

    res.to_csv(root_dir + output_file_name, index=False)
    # aggreagte the accuracy for iterations on same dataset
    res = pd.DataFrame({
        'accuracy': res.groupby(
            ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
    }).reset_index()

    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    if hist.history.get('val_' + metric):
        plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    if hist.history.get('val_' + metric):
        plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs_t_leNet(output_directory, hist, y_pred, y_true, duration):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None, train_method='normal'):
    hist_df = pd.DataFrame(hist.history)
    if 'normal' not in train_method:
        hist_df.to_csv(output_directory + f'{train_method}_history.csv', index=False)
    else:
        hist_df.to_csv(output_directory + 'history.csv', index=False)

    # print("len(y_true):", len(y_true))
    # print("len(y_pred):", len(y_pred))
    count = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            count += 1
    # print("count:", count)
    print("accuracy:", count/len(y_pred))
    df_metrics = calculate_metrics(
        y_true, y_pred, duration, y_true_val, y_pred_val)
    if 'normal' not in train_method:
        df_metrics.to_csv(output_directory + f'{train_method}_df_metrics.csv', index=False)
    else:
        df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model.get('val_loss')
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model.get('val_accuracy')
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    if 'normal' not in train_method:
        df_best_model.to_csv(output_directory + f'{train_method}_df_best_model.csv', index=False)
    else:
        df_best_model.to_csv(output_directory +
                             'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    if 'normal' not in train_method:
        plot_epochs_metric(hist, output_directory + f'{train_method}_epochs_loss.png')
    else:
        plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics


def visualize_filter(root_dir):
    import tensorflow.keras as keras
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    filters = model.layers[1].get_weights()[0]

    new_input_layer = model.inputs
    new_output_layer = [model.layers[1].output]

    new_feed_forward = keras.backend.function(
        new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    colors = [(255 / 255, 160 / 255, 14 / 255),
              (181 / 255, 87 / 255, 181 / 255)]
    colors_conv = [(210 / 255, 0 / 255, 0 / 255),
                   (27 / 255, 32 / 255, 101 / 255)]

    idx = 10
    idx_filter = 1

    filter = filters[:, 0, idx_filter]

    plt.figure(1)
    plt.plot(filter + 0.5, color='gray', label='filter')
    for c in classes:
        c_x_train = x_train[np.where(y_train == c)]
        convolved_filter_1 = new_feed_forward([c_x_train])[0]

        idx_c = int(c) - 1

        plt.plot(c_x_train[idx], color=colors[idx_c],
                 label='class' + str(idx_c) + '-raw')
        plt.plot(convolved_filter_1[idx, :, idx_filter],
                 color=colors_conv[idx_c], label='class' + str(idx_c) + '-conv')
        plt.legend()

    plt.savefig(root_dir + 'convolution-' + dataset_name + '.pdf')

    return 1


def viz_perf_themes(root_dir, df):
    df_themes = df.copy()
    themes_index = []
    # add the themes
    for dataset_name in df.index:
        themes_index.append(utils.constants.dataset_types[dataset_name])

    themes_index = np.array(themes_index)
    themes, themes_counts = np.unique(themes_index, return_counts=True)
    df_themes.index = themes_index
    df_themes = df_themes.rank(axis=1, method='min', ascending=False)
    df_themes = df_themes.where(df_themes.values == 1)
    df_themes = df_themes.groupby(level=0).sum(axis=1)
    df_themes['#'] = themes_counts

    for classifier in CLASSIFIERS:
        df_themes[classifier] = df_themes[classifier] / df_themes['#'] * 100
    df_themes = df_themes.round(decimals=1)
    df_themes.to_csv(root_dir + 'tab-perf-theme.csv')


def viz_perf_train_size(root_dir, df):
    df_size = df.copy()
    train_sizes = []
    datasets_dict_ucr = read_all_datasets(
        root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(datasets_dict[dataset_name][0])
        train_sizes.append(train_size)

    train_sizes = np.array(train_sizes)
    bins = np.array([0, 100, 400, 800, 99999])
    train_size_index = np.digitize(train_sizes, bins)
    train_size_index = bins[train_size_index]

    df_size.index = train_size_index
    df_size = df_size.rank(axis=1, method='min', ascending=False)
    df_size = df_size.groupby(level=0, axis=0).mean()
    df_size = df_size.round(decimals=2)

    print(df_size.to_string())
    df_size.to_csv(root_dir + 'tab-perf-train-size.csv')


def viz_perf_classes(root_dir, df):
    df_classes = df.copy()
    class_numbers = []
    datasets_dict_ucr = read_all_datasets(
        root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(np.unique(datasets_dict[dataset_name][1]))
        class_numbers.append(train_size)

    class_numbers = np.array(class_numbers)
    bins = np.array([0, 3, 4, 6, 8, 13, 9999])
    class_numbers_index = np.digitize(class_numbers, bins)
    class_numbers_index = bins[class_numbers_index]

    df_classes.index = class_numbers_index
    df_classes = df_classes.rank(axis=1, method='min', ascending=False)
    df_classes = df_classes.groupby(level=0, axis=0).mean()
    df_classes = df_classes.round(decimals=2)

    print(df_classes.to_string())
    df_classes.to_csv(root_dir + 'tab-perf-classes.csv')


def viz_perf_length(root_dir, df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(
        root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths = np.array(lengths)
    bins = np.array([0, 81, 251, 451, 700, 1001, 9999])
    lengths_index = np.digitize(lengths, bins)
    lengths_index = bins[lengths_index]

    df_lengths.index = lengths_index
    df_lengths = df_lengths.rank(axis=1, method='min', ascending=False)
    df_lengths = df_lengths.groupby(level=0, axis=0).mean()
    df_lengths = df_lengths.round(decimals=2)

    print(df_lengths.to_string())
    df_lengths.to_csv(root_dir + 'tab-perf-lengths.csv')


def viz_plot(root_dir, df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(
        root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths_index = np.array(lengths)

    df_lengths.index = lengths_index

    plt.scatter(x=df_lengths['fcn'], y=df_lengths['resnet'])
    plt.ylim(ymin=0, ymax=1.05)
    plt.xlim(xmin=0, xmax=1.05)
    # df_lengths['fcn']
    plt.savefig(root_dir + 'plot.pdf')


def viz_for_survey_paper(root_dir, filename='results-ucr-mts.csv'):
    df = pd.read_csv(root_dir + filename, index_col=0)
    df = df.T
    df = df.round(decimals=2)

    # get table performance per themes
    # viz_perf_themes(root_dir,df)

    # get table performance per train size
    # viz_perf_train_size(root_dir,df)

    # get table performance per classes
    # viz_perf_classes(root_dir,df)

    # get table performance per length
    # viz_perf_length(root_dir,df)

    # get plot
    viz_plot(root_dir, df)


def viz_cam(root_dir, classifier_name, archive_name, dataset_name, itr, file_ext='', remove_docstr=False):
    # # classifier = 'resnet'
    # classifier = 'cnn'
    # # archive_name = 'UCRArchive_2018'
    # archive_name = 'new'
    # dataset_name = 'GunPoint'
    classifier = classifier_name

    if dataset_name == 'Gun_Point':
        save_name = 'GunPoint'
    else:
        save_name = dataset_name
    # datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
    datasets_dict = read_dataset(
        root_dir, archive_name, dataset_name, file_ext, remove_docstr)

    if True:  # 'finetune'
        x_train = datasets_dict[dataset_name][4]
        y_train = datasets_dict[dataset_name][5]
        x_test = datasets_dict[dataset_name][6]
        y_test = datasets_dict[dataset_name][7]
    else:
        x_train = datasets_dict[dataset_name][0]
        y_train = datasets_dict[dataset_name][1]
        x_test = datasets_dict[dataset_name][2]
        y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    print("nb_classes:", nb_classes)

    # TODO: change to multivariate
    # # transform to binary labels
    # enc = sklearn.preprocessing.OneHotEncoder()
    # enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    # y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()

    print("x_train.shape:", x_train.shape)
    print("y_train before:", y_train)
    # enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    # enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    # y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    # y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    print("y_train after:", y_train)
    print("y_train[20:] after:", y_train[20:])
    print("y_train.shape after:", y_train.shape)
    # print("y_test after:", y_test)

    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    pre_model_name = 'finetune_'
    # pre_model_name = ''
    model = keras.models.load_model(
        root_dir + '/' + 'results/' + classifier + '/' + archive_name + itr + '/' + dataset_name + '/' + pre_model_name + 'best_model.hdf5')

    # filters
    # weights for each filter k for each class c
    w_k_c = model.layers[-1].get_weights()[0]
    print("w_k_c.shape:", w_k_c.shape)  # (128, 100) 128 filter 100 class
    # w_k_c = model.layers[-1].get_weights()
    # print("w_k_c:", w_k_c)
    # print("len(w_k_c):", len(w_k_c))
    # print("w_k_c[0].shape:", w_k_c[0].shape)

    # the same input
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(
        new_input_layer, new_output_layer)
    print("new_input_layer:", new_input_layer)

    classes = np.unique(y_train)
    print("classes:", classes)
    count_all = 0

    # all classes
    plt.figure()
    for c in classes:
        if c > 110:
            break
        print("c:", c)
        # per class
        # plt.figure()
        count = 0
        print("np.where(y_train == c):", np.where(y_train == c))
        c_x_train = x_train[np.where(y_train == c)]
        print("c_x_train.shape:", c_x_train.shape)
        for ts in c_x_train:
            print("before ts:", ts)
            print("before ts.shape:", ts.shape)
            # ts = ts.reshape(1, -1, 1)
            # ts = ts.reshape(1, 10, -1)
            # TODO: this is hotfix
            ts = ts.reshape(1, 10, -1)
            print("after ts:", ts)
            print("after ts.shape:", ts.shape)
            [conv_out, predicted] = new_feed_forward([ts])
            print("conv_out:", conv_out)
            print("predicted:", predicted)
            pred_label = np.argmax(predicted[0])
            print("predicted[0][pred_label]:", predicted[0][pred_label])
            # pred_label += 101  # TODO: hotfix, should be fixed with one-hot encoder
            print("pred_label:", pred_label)
            # orig_label = np.argmax(enc.transform([[c]]))
            orig_label = int(c)
            orig_label -= 101 # TODO: hotfix, should be fixed with one-hot encoder
            print("orig_label:", orig_label)
            # assert 0
            if pred_label == orig_label:
                print("conv_out.shape:", conv_out.shape)
                cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
                # cas = np.zeros(dtype=np.float, shape=(conv_out.shape[2]))
                for k, w in enumerate(w_k_c[:, orig_label]):
                    # print("w * conv_out[0, :, k]:", w * conv_out[0, :, k])
                    cas += w * conv_out[0, :, k]
                    # cas += w * conv_out[0, k, :]  # k filter
                print("before cas:", cas)
                minimum = np.min(cas)

                cas = cas - minimum

                cas = cas / max(cas)
                cas = cas * 100
                print("after cas:", cas)

                # shape_ind = 1
                max_length = 2000

                shape_ind = 2
                # # max_length = 2020
                # max_length = 2000

                print("ts.shape:", ts.shape)
                print("ts.shape[1] - 1:", ts.shape[shape_ind] - 1)  # 9 or 100
                x = np.linspace(0, ts.shape[shape_ind] - 1, max_length, endpoint=True)
                # linear interpolation to smooth
                print("x:", x)
                print("x.shape:", x.shape)
                print("ts.shape[shape_ind]:", ts.shape[shape_ind])

                # print("ts[0, :, 0]:", ts[0, :, 0])
                # print("len(ts[0, :, 0]):", len(ts[0, :, 0]))
                # print("range(ts.shape[shape_ind]):", range(ts.shape[shape_ind]))
                # print("ts[0, :, 0].shape:", ts[0, :, 0].shape)  # ts[0, :, 0].shape: (10,)
                # f = interp1d(range(ts.shape[shape_ind]), ts[0, :, 0])

                frame = 3
                print("ts[0, frame, :]:", ts[0, frame, :])
                print("len(ts[0, frame, :]):", len(ts[0, frame, :]))
                # print("len(ts[0, frame, :][:100]):", len(ts[0, frame, :][:100]))
                # print("len(ts[0, frame, :][1:]):", len(ts[0, frame, :][1:]))
                print("range(ts.shape[shape_ind]):", range(ts.shape[shape_ind]))
                print("ts[0, frame, :].shape:", ts[0, frame, :].shape)  # ts[0, frame, :].shape: (101,)
                # f = interp1d(range(ts.shape[shape_ind]-1), ts[0, frame, :][:100])
                # f = interp1d(range(ts.shape[shape_ind]), ts[0, frame, :][1:])
                f = interp1d(range(ts.shape[shape_ind]), ts[0, frame, :])

                y = f(x)
                print("y:", y)
                print("y.shape:", y.shape)
                print("range(ts.shape[1]):", range(ts.shape[1]))
                print("cas.shape:", cas.shape)
                # if (y < -2.2).any():
                #     continue
                # f = interp1d(range(ts.shape[shape_ind]), cas)
                f = interp1d(range(ts.shape[1]), cas, fill_value="extrapolate")  # 

                cas = f(x).astype(int)
                plt.scatter(x=x, y=y, c=cas, cmap='jet', marker='.',
                            s=2, vmin=0, vmax=100, linewidths=0.0)
                # if dataset_name == 'Gun_Point':
                # if c == 1:
                #     plt.yticks([-1.0, 0.0, 1.0, 2.0])
                # else:
                #     plt.yticks([-2, -1.0, 0.0, 1.0, 2.0])
                count += 1

        print("count:", count)
        count_all += count

        # # save pic per class
        # cbar = plt.colorbar()
        # # cbar.ax.set_yticklabels([100,75,50,25,0])
        # pic_path = root_dir + '/temp/' + classifier + '-cam-' + save_name + '-class-' + str(int(c)) + '.png'
        # Path(root_dir + '/temp/').mkdir(parents=True, exist_ok=True)
        # plt.savefig(pic_path,
        #             bbox_inches='tight', dpi=1080)

    # save pic all together
    cbar = plt.colorbar()
    # cbar.ax.set_yticklabels([100,75,50,25,0])
    pic_path = root_dir + '/temp/' + classifier + '-cam-' + save_name + '-class-' + str(int(c)) + '.png'
    Path(root_dir + '/temp/').mkdir(parents=True, exist_ok=True)
    plt.savefig(pic_path,
                bbox_inches='tight', dpi=1080)

    print("count_all:", count_all)