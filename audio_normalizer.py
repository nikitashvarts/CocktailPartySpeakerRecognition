"""Audio Normalizer
Normalize input audio to one volume level and save it to new location"""

#region Imports
import argparse
import datetime
from os import walk, makedirs
from os.path import join, isdir
from pydub import AudioSegment
from progress_bar import print_progress_bar
#endregion

slash_sign = '/'  # for Linux
# slash_sign = '\\'  # for Windows

class AudioNormalizer:
    def __init__(self, dataset_dir, audio_format, volume_level):
        self.dataset_dir = dataset_dir
        self.format = audio_format
        self.level = volume_level

    def __match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def __save_audio(self, sound, path_to_file, audio_format, root_dir):
        path_to_file = path_to_file.split(slash_sign)
        dataset_name = root_dir.split(slash_sign)[-1]
        name_idx = path_to_file.index(dataset_name)
        path_to_file[name_idx] = dataset_name + '_normalized'
        new_dir_path = slash_sign.join(path_to_file[:-1])
        if not isdir(new_dir_path):
            makedirs(new_dir_path)
        path_to_file = slash_sign.join(path_to_file)
        sound.export(path_to_file, format=audio_format)

    def normalize_audio(self):
        print('Start normalizing audio files...')

        count_of_records = count_files_in_dataset_dir(self.dataset_dir, file_format=self.format)
        print('Number of files to process: ', count_of_records)

        time_start = datetime.datetime.now()
        record_idx = 0
        print_progress_bar(iteration=record_idx, total=count_of_records,
                           prefix='{}/{}'.format(record_idx, count_of_records),
                           suffix='complete')

        for root, dirs, files in sorted(walk(self.dataset_dir)):
            for audio_file in files:
                if audio_file.endswith(self.format):
                    path_to_file = join(root, audio_file)
                    sound = AudioSegment.from_file(path_to_file, self.format)
                    normalized_sound = self.__match_target_amplitude(sound, self.level)
                    self.__save_audio(normalized_sound, path_to_file, self.format, root_dir=self.dataset_dir)

                    record_idx += 1
                    print_progress_bar(iteration=record_idx, total=count_of_records,
                                       prefix='{}/{}'.format(record_idx, count_of_records),
                                       suffix='complete')
        print('All audio files successfully normalized and saved')

        time_end = datetime.datetime.now()
        print('Elapsed time: ', time_end-time_start)


def count_files_in_dataset_dir(dataset_dir, file_format):
    count = 0
    for root, dirs, files in walk(dataset_dir):
        for i in range(len(files)):
            if files[i].endswith(file_format):
                count += 1
    return count


def parse_arguments():
    parser = argparse.ArgumentParser(description="Normalize input audio to one volume level")
    parser.add_argument('--dataset_dir', required=True,
                        help="Path of directory containing audio files")
    parser.add_argument('--audio_format', required=True,
                        help='Format of input audio')
    parser.add_argument('--volume_level', type=float, default='-20.0',
                        help='Target level of sound')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    normalizer = AudioNormalizer(dataset_dir=args.dataset_dir,
                                 audio_format=args.audio_format,
                                 volume_level=args.volume_level)
    normalizer.normalize_audio()
