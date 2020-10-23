from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import shutil
import numpy as np
import sys
import sklearn
import utils
import argparse
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(0)

import random
import tensorflow as tf
import numpy as np
import tfdeterminism


def set_random_seed(seed: int = 1):
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def data_preprocessing(datasets_dict, dataset_name, val_proportion=0.0):
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    if val_proportion > 0.0:
        try:
            sskf = StratifiedShuffleSplit(n_splits=1, test_size=val_proportion)
            splits = sskf.split(x_train, y_train)

            for n, (train_index, val_index) in enumerate(splits):
                x_train_small = np.array([x_train[i] for i in train_index])
                x_val = np.array([x_train[i] for i in val_index])
                y_train_small = np.array([y_train[i] for i in train_index])
                y_val = np.array([y_train[i] for i in val_index])
        except:
            x_train_small = x_train
            x_val = None
            y_train_small = y_train
            y_val = None
    else:
        x_train_small = x_train
        x_val = None
        y_train_small = y_train
        y_val = None

    print("x_train_small.shape:", x_train_small.shape)
    print("y_train_small.shape:", y_train_small.shape)
    if x_val is not None:
        print("x_val.shape:", x_val.shape)
        print("y_val.shape:", y_val.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    # y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    # y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    y_train_small = enc.transform(y_train_small.reshape(-1, 1)).toarray()
    if y_val is not None:
        y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    print("y_train_small.shape:", y_train_small.shape)
    if y_val is not None:
        print("y_val.shape:", y_val.shape)
    print("y_test.shape:", y_test.shape)

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)
    print("y_true.shape:", y_true.shape)

    # if len(x_train.shape) == 2:  # if univariate
    if len(x_train_small.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        # x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        x_train_small = x_train_small.reshape((x_train_small.shape[0], x_train_small.shape[1], 1))
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    print("x_train_small.shape:", x_train_small.shape)
    if x_val is not None:
        print("x_val.shape:", x_val.shape)
    print("x_test.shape:", x_test.shape)
    # input_shape = x_train.shape[1:]
    input_shape = x_train_small.shape[1:]
    print("input_shape:", input_shape)
    return x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, input_shape, nb_classes

def fit_classifier(datasets_dict, dataset_name, verbose, val_proportion, do_pred_only, nb_epochs=None, batch_size=None, trainable_layers=None, nb_epochs_finetune=None):
    print("len(datasets_dict[dataset_name]):", len(datasets_dict[dataset_name]))
    if len(datasets_dict[dataset_name]) == 8:
        train_method = 'pretrain'
        p_datasets_dict = {dataset_name: None}
        p_datasets_dict[dataset_name] = datasets_dict[dataset_name][:4]
        x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, input_shape, nb_classes = data_preprocessing(p_datasets_dict, dataset_name, val_proportion=0.0)
        classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose)
        classifier_fit(classifier, x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, do_pred_only, nb_epochs, batch_size, train_method, nb_classes=nb_classes)
        
        train_method = 'finetune'
        o_datasets_dict = {dataset_name: None}
        o_datasets_dict[dataset_name] = datasets_dict[dataset_name][4:]
        x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, input_shape, nb_classes = data_preprocessing(o_datasets_dict, dataset_name, val_proportion=0.0)
        if nb_epochs_finetune:
            nb_epochs = nb_epochs_finetune
        classifier_fit(classifier, x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, do_pred_only, nb_epochs, batch_size, train_method, trainable_layers=trainable_layers, nb_classes=nb_classes)
        
    else:
        train_method = 'normal'
        x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, input_shape, nb_classes = data_preprocessing(datasets_dict, dataset_name, val_proportion=val_proportion)
        classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose)
        classifier_fit(classifier, x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, do_pred_only, nb_epochs, batch_size, train_method, nb_classes=nb_classes)


def classifier_fit(classifier, x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, do_pred_only, nb_epochs=None, batch_size=None, train_method='normal', trainable_layers=None, nb_classes=None):
    if nb_epochs:
        if batch_size:
            classifier.fit(x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, do_pred_only, nb_epochs=nb_epochs, batch_size=batch_size, train_method=train_method, trainable_layers=trainable_layers, nb_classes=nb_classes)
        else:
            classifier.fit(x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, do_pred_only, nb_epochs=nb_epochs, train_method=train_method, trainable_layers=trainable_layers, nb_classes=nb_classes)
    else:
        if batch_size:
            classifier.fit(x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, do_pred_only, batch_size=batch_size, train_method=train_method, trainable_layers=trainable_layers, nb_classes=nb_classes)
        else:
            classifier.fit(x_train_small, y_train_small, x_val, y_val, x_test, y_test, y_true, do_pred_only, train_method=train_method, trainable_layers=trainable_layers, nb_classes=nb_classes)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True, train_method='normal'):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### main

if __name__ == '__main__':
    '''
    Example:
        python3 main.py \
        --dir ./data \
        --action single \
        --archive TSC \
        --dataset Coffee \
        --classifier fcn \
        --itr _itr_8 \
        --file_ext .arff \
        --remove_header True
    
    '''
    parser = argparse.ArgumentParser(description="Data Augmentation Modules (including SWAT, EDA, Multi-Label, "
                                                 "Overlapping, ..." +
                                                 "https://confluence.oraclecorp.com/confluence/display/IBS/SWAT"
                                                 "+Sensless+Patterns")
    parser.add_argument('--dir',
                        help='root dir',
                        default='./data')
    parser.add_argument('--action',
                        help='action',
                        default='single')
    parser.add_argument('--archive',
                        help='archive name',
                        default='TSC')
    parser.add_argument('--dataset',
                        help='input dataset name',
                        default='Coffee')
    parser.add_argument('--classifier',
                        help='classifier name',
                        default='fcn')
    parser.add_argument('--itr',
                        help='iteration times',
                        default='_itr_8')
    parser.add_argument('--file_ext',
                        help='input file extension',
                        default='')
    parser.add_argument('--remove_docstr',
                        help='remove the doc string of the data file',
                        default=True)
    parser.add_argument('--verbose',
                        help='make training progress verbose',
                        default=True)
    parser.add_argument('--val_proportion',
                        help='make training progress verbose',
                        default=0.1)
    parser.add_argument('--do_pred_only',
                        help='skip training, do prediction only',
                        default=False)
    parser.add_argument('--nb_epochs',
                        help='number of epochs',
                        default=None)
    parser.add_argument('--nb_epochs_finetune',
                        help='number of epochs',
                        default=None)
    parser.add_argument('--batch_size',
                        help='batch size in train',
                        default=None)
    parser.add_argument('--trainable_layers',
                        help='trainable layers during finetuning',
                        default=None)
    parser.add_argument('--retrain',
                        help='remove existing models and retrain',
                        default=True)
    # parser.add_argument('--train_method',
    #                     help='3 approaches: pretrain, pretrain_finetune, finetune',
    #                     default='pretrain')
    args = parser.parse_args()
    root_dir = args.dir

    # change this directory for your machine
    # root_dir = '/b/home/uha/hfawaz-datas/dl-tsc-temp/'
    # root_dir = '/oradiskvdb/work/side/Gait/dl-4-tsc/data'
    
    archive_name = args.archive
    dataset_name = args.dataset
    classifier_name = args.classifier
    itr = args.itr
    if itr == '_itr_0':
        itr = ''

    if args.action == 'run_all':
        for classifier_name in CLASSIFIERS:
            print('classifier_name', classifier_name)

            for archive_name in ARCHIVE_NAMES:
                print('\tarchive_name', archive_name)

                datasets_dict = read_all_datasets(root_dir, archive_name)

                for iter in range(ITERATIONS):
                    print('\t\titer', iter)

                    trr = ''
                    if iter != 0:
                        trr = '_itr_' + str(iter)

                    tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + trr + '/'

                    for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                        print('\t\t\tdataset_name: ', dataset_name)

                        output_directory = tmp_output_directory + dataset_name + '/'

                        create_directory(output_directory)

                        fit_classifier()

                        print('\t\t\t\tDONE')

                        # the creation of this directory means
                        create_directory(output_directory + '/DONE')

    elif args.action == 'transform_mts_to_ucr_format':
        transform_mts_to_ucr_format()
    elif args.action == 'visualize_filter':
        visualize_filter(root_dir)
    elif args.action == 'viz_for_survey_paper':
        viz_for_survey_paper(root_dir)
    elif args.action == 'viz_cam':
        viz_cam(root_dir, classifier_name, archive_name, dataset_name, itr, args.file_ext, args.remove_docstr)
    elif args.action == 'generate_results_csv':
        res = generate_results_csv('results.csv', root_dir)
        print(res.to_string())
    else:
        # this is the code used to launch an experiment on a dataset
        # archive_name = sys.argv[1]
        # dataset_name = sys.argv[2]
        # classifier_name = sys.argv[3]
        # itr = sys.argv[4]

        output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + \
                        dataset_name + '/'

        test_dir_df_metrics = output_directory + 'df_metrics.csv'

        print('Method: ', archive_name, dataset_name, classifier_name, itr)

        if args.retrain:
            if os.path.exists(test_dir_df_metrics):
                os.remove(test_dir_df_metrics)

            create_directory(output_directory)
            datasets_dict = read_dataset(root_dir, archive_name, dataset_name, args.file_ext, args.remove_docstr)
            fit_classifier(datasets_dict, dataset_name, args.verbose, args.val_proportion, args.do_pred_only, args.nb_epochs, args.batch_size, args.trainable_layers, args.nb_epochs_finetune)
            
            print('DONE')

            # the creation of this directory means
            create_directory(output_directory + '/DONE')
        else:
            if os.path.exists(test_dir_df_metrics):
                print(f'Already done in {test_dir_df_metrics}')
            else:
                print(f'Not retrain and no {test_dir_df_metrics}. Please set --retrain True')
