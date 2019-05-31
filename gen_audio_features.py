"""Takes at the input audio files from dataset in .wav format
then applies voice activity detection (VAD) and finally extracts
features from cleared data and saves it in binary format .npy"""

#region Imports
import datetime
import argparse
from os import walk, makedirs
from os.path import isdir, join
import numpy as np
import scipy.io.wavfile as wf
from voice_activity_detector import VoiceActivityDetector
from feature_extractor import FeatureExtractor
from progress_bar import print_progress_bar
#endregion

slash_sign = '/'  # for Linux
# slash_sign = '\\'  # for Windows

def load_wave(path_to_file):
    sample_rate, wave_data = wf.read(path_to_file)
    return sample_rate, wave_data


def count_files_in_dataset_dir(dataset_dir, file_format):
    count = 0
    for root, dirs, files in walk(dataset_dir):
        for i in range(len(files)):
            if files[i].endswith(file_format):
                count += 1
    return count


def gen_property_file(num_classes, num_coeffs, path_to_dataset):
    file_name = path_to_dataset + "_property.txt"
    with open(file_name, "w") as text_file:
        text_file.write("{},{}".format(num_classes, num_coeffs))
    print('Created property file containing info about number of classes and features')


def parse_label_voxceleb(path_to_file):
    path_list = path_to_file.split(slash_sign)
    label = path_list[-3][3:]
    return int(label)


def clear_audio_from_voice(wave_data, sample_rate, path_to_file):
    detector = VoiceActivityDetector(wave_data=wave_data,
                                     sample_rate=sample_rate,
                                     save_visual_results=False,
                                     file_name=path_to_file)
    cleared_audio = detector.detect_voice_activity()
    return cleared_audio


def extract_mel_filterbank_energies(data, sample_rate, num_coeffs):
    extractor = FeatureExtractor(data_array=data,
                                 sample_rate=sample_rate,
                                 num_of_coeffs=num_coeffs)
    features = extractor.extract_log_mel_filterbank_energies()
    return features


def save_numpy_array(np_array, path_to_file, root_dir, postfix):
    path_to_file = path_to_file[:-4]  # eliminate '.wav'
    path_to_file = path_to_file.split(slash_sign)
    dataset_name = root_dir.split(slash_sign)[-1]
    name_idx = path_to_file.index(dataset_name)
    path_to_file[name_idx] = dataset_name + postfix
    new_dir_path = slash_sign.join(path_to_file[:-1])
    if not isdir(new_dir_path):
        makedirs(new_dir_path)
    path_to_file = slash_sign.join(path_to_file)
    np.save(path_to_file, np_array)


def gen_audio_features(dataset_dir, num_coeffs, vad_only):
    """Takes at input path of root directory of dataset (or subset).
    For example, it can be the path to train/test folder
    in main dataset directory"""

    print('Start loading and processing of dataset...')

    count_of_records = count_files_in_dataset_dir(dataset_dir, file_format='.wav')
    print('Number of files to process: ', count_of_records)

    list_name = dataset_dir + '_list.txt'
    dataset_list = open(list_name, "w")
    labels = np.array([])

    time_start = datetime.datetime.now()
    record_idx = 0
    print_progress_bar(iteration=record_idx, total=count_of_records,
                       prefix='{}/{}'.format(record_idx, count_of_records),
                       suffix='complete')
    for root, dirs, files in sorted(walk(dataset_dir)):
        for wave_file in files:
            if wave_file.endswith('.wav'):
                path_to_file = join(root, wave_file)
                sample_rate, wave_data = load_wave(path_to_file)

                cleared_audio = clear_audio_from_voice(wave_data, sample_rate, path_to_file)
                if vad_only:
                    save_numpy_array(cleared_audio, path_to_file, root_dir=dataset_dir, postfix='_silencecleared')
                else:
                    features = extract_mel_filterbank_energies(cleared_audio, sample_rate, num_coeffs)
                    save_numpy_array(features, path_to_file, root_dir=dataset_dir, postfix='_features')

                label = parse_label_voxceleb(path_to_file)
                if label not in labels:
                    labels = np.append(labels, label)
                realtive_path_to_file = slash_sign.join(
                                                    [item for item in path_to_file.split(slash_sign) \
                                                         if item not in dataset_dir.split(slash_sign)])
                dataset_list.write('{} {}\n'.format(realtive_path_to_file, label))

                record_idx += 1
                print_progress_bar(iteration=record_idx, total=count_of_records,
                                   prefix='{}/{}'.format(record_idx, count_of_records),
                                   suffix='complete')
    dataset_list.close()
    num_classes = len(labels)
    print('Dataset successfully processed and saved')

    gen_property_file(num_classes, num_coeffs, path_to_dataset=(dataset_dir))

    time_end = datetime.datetime.now()
    print('Elapsed time: ', time_end-time_start)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Performing speaker diarization")
    parser.add_argument('--dataset_dir', required=True, default='',
                        help="Path of directory containig audio dataset")
    parser.add_argument('--num_coeffs', type=int, default=40,
                        help="(Optional) Number of coefficients to extract from \
                            each audio frame (default is 40 in accordance with article)")
    parser.add_argument('--vad_only', action='store_true',
                        help="Wheter to use voice activity detection only without \
                            feature extraction")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    gen_audio_features(args.dataset_dir, int(args.num_coeffs), args.vad_only)
