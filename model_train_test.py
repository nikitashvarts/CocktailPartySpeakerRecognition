"""Performs training and testing of a neural network.
Takes at the input audio features extracted from voice cleared audio
For dataset preparation use gen_audio_features.py script"""

#region Imports
import datetime
import random
import argparse
import math
from os.path import join, isdir, isfile
from os import makedirs
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, CuDNNLSTM, Dropout, Reshape, TimeDistributed
from keras.utils import to_categorical, multi_gpu_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from feature_extractor import FeatureExtractor
from progress_bar import print_progress_bar
#endregion


def compose_samples_with_timesteps(raw_features, num_timesteps, overlap_coeff=0.75):
    num_coeffs = raw_features.shape[1]
    num_windows = raw_features.shape[0]
    num_overlapped_timesteps = math.floor(num_timesteps*overlap_coeff)
    num_samples = math.floor(num_windows/num_overlapped_timesteps)
    # For simplicity Skip the last sample for the case
    # if num_windows is not divisible by num_samples
    if num_samples:
        num_samples -= 1
    timestep_samples = np.zeros((num_samples, num_timesteps, num_coeffs))
    for idx in range(num_samples):
        sample_start = idx*num_overlapped_timesteps
        sample_end = sample_start + num_timesteps
        timestep_samples[idx] = raw_features[sample_start:sample_end]
    return timestep_samples


def load_feature_from_file(path, num_timesteps):
    features_all = np.load(path)
    timestep_samples = compose_samples_with_timesteps(features_all, num_timesteps)
    return timestep_samples


def load_voxceleb_features(dataset_dir, list_path, num_timesteps, debug_mode):
    debug_break_number = 1000

    print('Started loading features from files...')
    time_start = datetime.datetime.now()

    f = open(list_path)
    f_list = list(f)
    f.close()

    features = []
    labels = []
    names_list = []

    num_files = len(f_list)
    print('Files to process: ', num_files)
    file_idx = 0
    print_progress_bar(iteration=file_idx, total=num_files,
                       prefix='{}/{}'.format(file_idx, num_files), suffix='complete')

    for line in f_list:
        path, label = line.rstrip().split(' ')
        path = path[:-4] + '.npy'
        path = join(dataset_dir, path)
        label = int(label) - 1  # in Voxceleb classes starts from 1, shift to 0

        timestep_samples = load_feature_from_file(path, num_timesteps)
        stretched_label = np.full(timestep_samples.shape[0], label)
        stretched_name = np.full(timestep_samples.shape[0], line)

        if timestep_samples.shape[0] != 0:
            features.extend(timestep_samples)
            labels.extend(stretched_label)
            names_list.extend(stretched_name)

        file_idx += 1
        print_progress_bar(iteration=file_idx, total=num_files,
                           prefix='{}/{}'.format(file_idx, num_files), suffix='complete')
        if debug_mode and file_idx == debug_break_number:
            print('\nDebug mode. Interrupted on ', file_idx)
            break

    print('Dataset successfully processed and saved')
    time_end = datetime.datetime.now()
    print('Elapsed time: ', time_end-time_start)

    features = np.array(features)
    labels = np.array(labels)
    return features, labels, names_list


def load_cleared_audio(dataset_dir, files_list):
    print('Loading cleared audio files...')
    audio_all = []
    labels = []
    file_idx = 0
    num_files = len(files_list)
    print_progress_bar(iteration=file_idx, total=num_files,
                       prefix='{}/{}'.format(file_idx, num_files), suffix='complete')
    for line in files_list:
        path, label = line.rstrip().split(' ')
        path = path[:-4] + '.npy'  # eliminate .wav and add .npy
        path_to_file = join(dataset_dir, path)
        audio = np.load(path_to_file)
        audio_all.append(audio)
        labels.append(int(label)-1)  # in Voxceleb classes starts from 1, shift to 0
        file_idx += 1
        print_progress_bar(iteration=file_idx, total=num_files,
                           prefix='{}/{}'.format(file_idx, num_files), suffix='complete')
    return np.array(audio_all), np.array(labels)


def feature_mixup_augmentation(features, labels, alpha, num_to_mix):
    print('Started data augmentation using feature mixup...')
    time_start = datetime.datetime.now()

    mixup_features = []
    mixup_labels = []
    num_records = len(labels)

    indices = np.arange(num_records)
    np.random.shuffle(indices)
    features = features[indices]
    labels = labels[indices]

    num_same_class = 0
    for idx in range(num_records-num_to_mix+1):
        for mix_idx in range(num_to_mix-1):
            lam = np.random.beta(a=alpha, b=alpha)
            mixed_feat = lam * features[idx] + (1-lam) * features[idx+mix_idx+1]
            mixed_lab = lam * labels[idx] + (1-lam) * labels[idx+mix_idx+1]
            mixup_features.append(mixed_feat)
            mixup_labels.append(mixed_lab)

            if np.array_equal(labels[idx], labels[idx+mix_idx+1]):
                num_same_class += 1

        print_progress_bar(idx, num_records-num_to_mix,
                           prefix='{}/{}'.format(idx+1, num_records-num_to_mix+1),
                           suffix='complete')

    ratio = num_same_class/(num_to_mix*num_records)
    print('%.2f%% are mixed from the same class,\
          \n%.2f%% are mixed from different classes' % (ratio, 1-ratio))

    time_end = datetime.datetime.now()
    print('Elapsed time: ', time_end-time_start)
    return np.array(mixup_features), np.array(mixup_labels)


def audio_mixup_augmentation(paths_list, cleared_audio_dir, alpha, num_classes, sample_rate,
                             num_coeffs, num_timesteps, num_to_mix):
    print('Started data augmentation using audio mixup...')
    time_start = datetime.datetime.now()
    mixup_features = []
    mixup_labels = []

    paths_list = list(dict.fromkeys(paths_list))  # remove all duplicates
    num_records = len(paths_list)
    np.random.shuffle(paths_list)
    audio, labels = load_cleared_audio(cleared_audio_dir, paths_list)
    labels = to_categorical(labels, num_classes)

    print('Augmentation:')
    num_same_class = 0
    for idx in range(num_records-num_to_mix+1):
        for mix_idx in range(num_to_mix-1):
            lam = np.random.beta(a=alpha, b=alpha)
            audio_size = len(audio[idx]) if len(audio[idx])<len(audio[idx+mix_idx+1]) else len(audio[idx+mix_idx+1])
            mixed_audio = lam * audio[idx][:audio_size] + (1-lam) * audio[idx+mix_idx+1][:audio_size]
            mixed_lab = lam * labels[idx] + (1-lam) * labels[idx+mix_idx+1]

            extractor = FeatureExtractor(mixed_audio, sample_rate, num_coeffs)
            mixed_feat = extractor.extract_log_mel_filterbank_energies()
            timestep_samples = compose_samples_with_timesteps(mixed_feat, num_timesteps)
            mixup_features.extend(timestep_samples)
            stretched_label = np.tile(mixed_lab, (timestep_samples.shape[0], 1))
            mixup_labels.extend(stretched_label)

            if np.array_equal(labels[idx], labels[idx+mix_idx+1]):
                num_same_class += 1

        print_progress_bar(idx, num_records-num_to_mix,
                           prefix='{}/{}'.format(idx+1, num_records-num_to_mix+1),
                           suffix='complete')

    time_end = datetime.datetime.now()
    print('Elapsed time: ', time_end-time_start)
    ratio = num_same_class/(num_to_mix*num_records)
    print('%.2f%% are mixed from the same class,\
          \n%.2f%% are mixed from different classes' % (ratio, 1-ratio))
    return np.array(mixup_features), np.array(mixup_labels)


def list_train_test_val_split(list_path, num_classes, test_size=0.2, val_size=0.1):
    f = open(list_path)
    f_list = list(f)
    f.close()

    classes_list = [[]] * num_classes
    for line in f_list:
        label = line.rstrip().split(' ')[1]
        label = int(label) - 1  # classes starts from 1, shift to 0
        tmp_list = list(classes_list[label])
        tmp_list.append(line)
        classes_list[label] = list(tmp_list)

    train_list = []
    test_list = []
    val_list = []
    for line in classes_list:
        tmp_list = list(line)
        random.shuffle(tmp_list)
        idx_train_part = math.floor(len(tmp_list)*(1-test_size-val_size))
        idx_test_part = idx_train_part + math.floor(len(tmp_list)*test_size)
        train_list.extend(tmp_list[:idx_train_part])
        test_list.extend(tmp_list[idx_train_part:idx_test_part])
        val_list.extend(tmp_list[idx_test_part:])
    return train_list, test_list, val_list


#TODO: something wrong with generator, model in not training using it
# probably the problem is with different batch size
def arrays_generator(dataset_dir, subset_list, subset, num_timesteps, num_coeffs,
                     num_classes, batch_size):
    """Custom file generator for keras.model.fit_generator method"""
    while True:
        if subset == 'train':
            random.shuffle(subset_list)

        batch_index = 0
        for line in subset_list:
            if batch_index == 0:
                features = np.zeros((0, num_timesteps, num_coeffs))
                labels = np.array([])

            path, label = line.rstrip().split(' ')
            path = path[:-4] + '_features.npy'
            path = join(dataset_dir, path)
            label = int(label) - 1  # in Voxceleb classes starts from 1, shift to 0

            timestep_samples = load_feature_from_file(path, num_timesteps)
            stretched_label = np.full(timestep_samples.shape[0], label)

            features = np.concatenate((features, timestep_samples), axis=0)
            labels = np.concatenate((labels, stretched_label))

            batch_index += timestep_samples.shape[0]

            if batch_index >= batch_size:
                labels = to_categorical(labels, num_classes)
                batch_index = 0
                yield (features[:batch_size], labels[:batch_size])
            #TODO: what if count of images is not divisible by batch_size?


def initialize_base_model(num_timesteps, num_coeffs, output_size):
    model = Sequential()

    model.add(CuDNNLSTM(256, return_sequences=True,
                        input_shape=(num_timesteps, num_coeffs)))
    model.add(Dropout(0.2))

    model.add(CuDNNLSTM(128, return_sequences=True))
    model.add(Dropout(0.4))

    model.add(CuDNNLSTM(output_size, return_sequences=True))
    model.add(Dropout(0.5))

    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Performing training and testing of \
                                     a neural network")
    parser.add_argument('--dataset_dir', required=True, default='',
                        help="Path of directory containing features of voice \
                            cleared audio dataset in .npy binary format")
    parser.add_argument('--dataset_list_path', required=True, default='',
                        help="Path to txt file containing relative path to all \
                             audio in dataset")
    parser.add_argument('--property_file_loc', required=True, default='',
                        help="Path to property file")
    parser.add_argument('--num_timesteps', type=int, default=20,
                        help="(Optional) Number of timesteps per audio sample \
                            (default is 5 for better training of CuDNNLSTM layers)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_folds', type=int, default=5,
                        help="Number of folds to do K-Fold cross validation")
    parser.add_argument('--num_runs', type=int, default=10,
                        help="Number of times for cross validation run")
    parser.add_argument('--gpus', type=int, default=1,
                        help="Number of GPUs to use in training")
    parser.add_argument('--use_audio_mixup', action='store_true',
                        help="Whether to use mixup of input audio or not")
    parser.add_argument('--use_feature_mixup', action='store_true',
                        help="Whether to use mixup of audio features or not")
    parser.add_argument('--cleared_audio_dir', default='',
                        help="Path of directory containing silence cleared \
                            audion dataset in .npy binary format")
    parser.add_argument('--debug_mode', action='store_true',
                        help="Whether to interrupt loading of all files to \
                            increase debugging speed")
    args = parser.parse_args()
    return args


def main():
    #region Arguments reading
    args = parse_arguments()
    num_timesteps = args.num_timesteps
    batch_size = args.batch_size * args.gpus
    num_epochs = args.num_epochs
    num_gpus = args.gpus
    num_folds = args.num_folds
    num_runs = args.num_runs
    #endregion

    output_size_lstm = 64

    #region Property reading
    with open(args.property_file_loc, 'r') as prop_file:
        num_classes, num_coeffs = prop_file.readline().split(',')
        num_classes = int(num_classes)
        num_coeffs = int(num_coeffs)
    #endregion

    #region Data loading and preparation
    features, labels, names_list = load_voxceleb_features(dataset_dir=args.dataset_dir,
                                                          list_path=args.dataset_list_path,
                                                          num_timesteps=num_timesteps,
                                                          debug_mode=args.debug_mode)
    names_list = np.array(names_list)
    #endregion

    #region Generators initializing
    # train_list, test_list, val_list = list_train_test_val_split(list_path=args.dataset_list_path,
    #                                                             num_classes=num_classes,
    #                                                             test_size=0.2, 
    #                                                             val_size=0.1)
    # train_generator = arrays_generator(dataset_dir=args.dataset_dir,
    #                                    subset_list=train_list,
    #                                    subset='train',
    #                                    num_timesteps=num_timesteps,
    #                                    num_coeffs=num_coeffs,
    #                                    num_classes=num_classes,
    #                                    batch_size=batch_size)
    # test_generator = arrays_generator(dataset_dir=args.dataset_dir,
    #                                   subset_list=test_list,
    #                                   subset='test',
    #                                   num_timesteps=num_timesteps,
    #                                   num_coeffs=num_coeffs,
    #                                   num_classes=num_classes,
    #                                   batch_size=batch_size)
    # val_generator = arrays_generator(dataset_dir=args.dataset_dir,
    #                                  subset_list=val_list,
    #                                  subset='val',
    #                                  num_timesteps=num_timesteps,
    #                                  num_coeffs=num_coeffs,
    #                                  num_classes=num_classes,
    #                                  batch_size=batch_size)
    #endregion

    time_start = datetime.datetime.now()  # time for training

    # define k-fold cross validation test harness
    kfold = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_runs)
    acc_scores = []
    run_idx = 1
    fold_idx = 1

    for train, test in kfold.split(features, labels):
        fold_time_start = datetime.datetime.now()
        print('--------------- Run {}: fold {} ---------------'.format(run_idx, fold_idx))
        if fold_idx == num_folds:
            fold_idx = 0
            run_idx += 1
        fold_idx += 1

        #region Model initializing
        if num_gpus <= 1:
            base_model = initialize_base_model(num_timesteps, num_coeffs, output_size_lstm)
            x = base_model.output
            # reshape 3D output of LSTM to 2D form
            x = Reshape((num_timesteps*output_size_lstm,))(x)
            predictions = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        else:
            with tf.device("/cpu:0"):
                base_model = initialize_base_model(num_timesteps, num_coeffs, output_size_lstm)
                x = base_model.output
                # reshape 3D output of LSTM to 2D form
                x = Reshape((num_timesteps*output_size_lstm,))(x)
                predictions = Dense(num_classes, activation='softmax')(x)
                model = Model(inputs=base_model.input, outputs=predictions)
            model = multi_gpu_model(model, gpus=num_gpus)
        #endregion

        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.0, amsgrad=False)
        loss = 'categorical_crossentropy'

        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=['accuracy'])

        es = EarlyStopping(monitor='acc', patience=10, verbose=1, restore_best_weights=True)

        onehot_labels = to_categorical(labels, num_classes)

        x_train = features[train]
        y_train = onehot_labels[train]

        if args.use_feature_mixup:
            mixup_features, mixup_labels = feature_mixup_augmentation(x_train, y_train,
                                                                      alpha=5,
                                                                      num_to_mix=2)
            x_train = np.concatenate((x_train, mixup_features), axis=0)
            y_train = np.concatenate((y_train, mixup_labels), axis=0)

        if args.use_audio_mixup:
            mixup_features, mixup_labels = audio_mixup_augmentation(paths_list=names_list[train],
                                                                    cleared_audio_dir=args.cleared_audio_dir,
                                                                    alpha=1000,
                                                                    num_classes=num_classes,
                                                                    sample_rate=16000,
                                                                    num_coeffs=40,
                                                                    num_timesteps=25,
                                                                    num_to_mix=2)  #TODO: eliminate consts
            x_train = np.concatenate((x_train, mixup_features), axis=0)
            y_train = np.concatenate((y_train, mixup_labels), axis=0)

        #region Model.fit
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  verbose=1,
                  shuffle=True,
                  callbacks=[es])
        #endregion

        #region Model.fit_generator
        # num_train_samples = len(train_list)
        # train_gen_len = num_train_samples // batch_size
        # num_val_samples = len(val_list)
        # val_gen_len = num_val_samples // batch_size
        # model.fit_generator(generator=train_generator,
        #                     steps_per_epoch=train_gen_len,
        #                     epochs=num_epochs,
        #                     verbose=1,
        #                     validation_data=val_generator,
        #                     validation_steps=val_gen_len,
        #                     shuffle=True)
        #endregion

        print('\nModel evaluation...')

        print('Start preparing test data...')
        x_test, y_test = audio_mixup_augmentation(paths_list=names_list[test],
                                                  cleared_audio_dir=args.cleared_audio_dir,
                                                  alpha=1000,
                                                  num_classes=num_classes,
                                                  sample_rate=16000,
                                                  num_coeffs=40,
                                                  num_timesteps=25,
                                                  num_to_mix=2)  #TODO: eliminate consts

        #region Model.evaluate
        loss_and_metrics = model.evaluate(x_test, y_test,
                                          batch_size=batch_size,
                                          verbose=1)
        #endregion

        #region Model.evaluate_generator
        # num_test_samples = len(test_list)
        # test_gen_len = num_test_samples // batch_size
        # loss_and_metrics = model.evaluate_generator(generator=test_generator,
        #                                             steps=test_gen_len,
        #                                             verbose=1)
        #endregion

        print("%s: %.5f\n%s: %.2f%%" % (model.metrics_names[0], loss_and_metrics[0],
                                        model.metrics_names[1], (loss_and_metrics[1]*100)))
        acc_scores.append(loss_and_metrics[1] * 100)

        fold_time_end = datetime.datetime.now()
        print('Elapsed time: ', fold_time_end-fold_time_start)

        #region Model saving
        model_name = 'model_000{}_{}_folds'.format(fold_idx-1, num_folds)
        save_loc = './models/' + model_name  #TODO: will not work on Windows
        if not isdir(save_loc):
            makedirs(save_loc)
        model.save(join(save_loc, (model_name+'.h5')))
        print('Model saved in root directory of project in ', save_loc)

        with open(join(save_loc, 'train_list.txt'), 'w') as lst:
            for line in names_list[train]:
                lst.write(line)
        with open(join(save_loc, 'test_list.txt'), 'w') as lst:
            for line in names_list[test]:
                lst.write(line)
        #endregion

    print('Average accuracy: %.2f%% (+/- %.2f%%)' % (np.mean(acc_scores), np.std(acc_scores)))

    time_end = datetime.datetime.now()
    print('Elapsed time for the whole process: ', time_end-time_start)


if __name__ == "__main__":
    main()
