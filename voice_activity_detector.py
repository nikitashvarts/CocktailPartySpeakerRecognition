"""Voice activity detector (VAD)
Detects non-speech areas in the record and deletes it"""

#region Imports
import math
import numpy as np
from scipy.stats import mstats
from scipy import signal
import matplotlib.pyplot as plt
#endregion

class VoiceActivityDetector:
    def __init__(self, wave_data, sample_rate, save_visual_results, file_name):
        self.data = np.float64(wave_data)
        # self.data = self.__peak_normalization(self.data)
        self.data = self.__rms_normalization(self.data)
        self.rate = sample_rate
        self.channels = len(self.data.shape)
        if self.channels == 2:
            self.__convert_to_mono()
        self.save_results = save_visual_results
        self.file_name = file_name

        # Overlapping of frames is not required
        self.sizeof_frame = int(0.01 * self.rate)  # 10 ms
        self.num_of_frames = len(self.data)//self.sizeof_frame

        # Short-term features thresholds
        self.speech_energy_threshold = 0.05  # Short-term Energy
        self.frequency_threshold = 150  # Most Dominant Frequency Component (Hz)
        self.sfm_threshold = 4  # SFM - Spectral Flatness Measure

    def __peak_normalization(self, data):
        max_val = np.amax(data)
        return data / max_val

    def __rms_normalization(self, data):
        # root mean square normalization
        rms = np.sqrt(np.mean(np.square(data)))
        return data / rms

    def __convert_to_mono(self):
        self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
        self.channels = 1
        return self

    def __calculate_energy(self, frame):
        frame_square = np.square(frame)
        data_energy = np.sum(frame_square)
        return data_energy

    def __calculate_spectrum(self, frame):
        frame_fft = np.fft.fft(frame)
        return np.abs(frame_fft)

    def __calculate_frequencies(self, data):
        data_freq = np.fft.fftfreq(len(data), 1.0/self.rate)
        return data_freq

    def __calculate_dominant_frequency(self, frame, spectrum):
        frequencies = self.__calculate_frequencies(frame)
        idx_of_max = np.argmax(spectrum)
        return frequencies[idx_of_max]

    def __calculate_spectral_flatness(self, spectrum):
        arithmetic_mean = np.mean(spectrum)
        geometric_mean = mstats.gmean(spectrum)
        ratio = geometric_mean / arithmetic_mean
        if ratio == 0:
            sfm = 0
        else:
            sfm = -10*math.log10(ratio)
        return sfm

    def __calculate_measurements(self):
        """Returns three crucial measurements for voice detection:
        energy, most dominant frequency component, spectral flatness"""
        data = self.data
        sizeof_frame = self.sizeof_frame
        number_of_frames = self.num_of_frames
        total_energy = np.array([])
        total_dominant_freqs = np.array([])
        total_flatness = np.array([])

        for frame_idx in range(number_of_frames):
            frame = data[frame_idx*sizeof_frame:(frame_idx+1)*sizeof_frame]
            frame_spectrum = self.__calculate_spectrum(frame)

            energy = self.__calculate_energy(frame)
            total_energy = np.append(total_energy, energy)

            dominant_freq = self.__calculate_dominant_frequency(frame, frame_spectrum)
            total_dominant_freqs = np.append(total_dominant_freqs, dominant_freq)
            total_dominant_freqs = np.abs(total_dominant_freqs)

            spectral_flatness = self.__calculate_spectral_flatness(frame_spectrum)
            total_flatness = np.append(total_flatness, spectral_flatness)
        # total_energy = self.__peak_normalization(total_energy)
        total_energy = self.__rms_normalization(total_energy)

        if self.save_results:
            self.__save_measurements(original_data=data,
                                     energy=total_energy,
                                     dominant_freqs=total_dominant_freqs,
                                     flatness=total_flatness,
                                     file_name=self.file_name)
        return total_energy, total_dominant_freqs, total_flatness

    def __save_measurements(self, original_data, energy, dominant_freqs, flatness, file_name):
        figure, plots = plt.subplots(2, 2, figsize=(14, 7))
        title = 'Characteriatics of a signal (' + file_name[:-4] + ')'
        figure.suptitle(title, fontsize=14)

        plots[0, 0].plot(original_data)
        plots[0, 0].set_title('Original signal')

        plots[1, 0].plot(energy)
        plots[1, 0].set_title('Short-term energy')

        plots[0, 1].plot(dominant_freqs)
        plots[0, 1].set_title('Most dominant frequency components')

        plots[1, 1].plot(flatness)
        plots[1, 1].set_title('Spectral flatness measurement')

        image_name = file_name[:-4] + '_measurements' + '.png'
        plt.savefig(image_name)
        plt.close()

    def __stretch_detected_frames(self, detected_frames):
        sizeof_frame = self.sizeof_frame
        num_of_frames = len(detected_frames)
        detected_samples = np.zeros(len(self.data))
        for frame_idx in range(num_of_frames):
            for sample_idx in range(sizeof_frame):
                frame_start = frame_idx * sizeof_frame
                detected_samples[frame_start+sample_idx] = detected_frames[frame_idx]
        return detected_samples

    def __save_detection_resutls(self, original_data, detected_frames, file_name):
        figure, plots = plt.subplots(2, 1, figsize=(14, 7))
        title = 'Detected voice in file ' + file_name[:-4]
        figure.suptitle(title, fontsize=14)

        plots[0].plot(original_data)
        plots[0].set_title('Original signal')

        detected_samples = self.__stretch_detected_frames(detected_frames)
        plots[1].plot(detected_samples)
        plots[1].set_title('Detected voice')

        image_name = file_name[:-4] + '_detected' + '.png'
        plt.savefig(image_name)
        plt.close()

    def __get_cleared_data(self, detected_frames):
        number_of_frames = len(detected_frames[1])
        cleared_data = np.array([])
        for idx in range(number_of_frames):
            if detected_frames[1][idx] == 1:
                cleared_data = np.append(cleared_data, detected_frames[0][idx])
        # plt.figure()
        # plt.plot(cleared_data)
        # plt.show()
        return cleared_data

    def detect_voice_activity(self):
        data = self.data
        sizeof_frame = self.sizeof_frame
        number_of_frames = self.num_of_frames
        detected_frames = [[], []]

        energy, dominant_freqs, spectral_flatness = self.__calculate_measurements()

        for frame_idx in range(number_of_frames):
            isvoice = 0
            frame = data[frame_idx*sizeof_frame:(frame_idx+1)*sizeof_frame]

            if energy[frame_idx] > self.speech_energy_threshold:
                isvoice += 1
            if dominant_freqs[frame_idx] > self.frequency_threshold:
                isvoice += 1
            if spectral_flatness[frame_idx] > self.sfm_threshold:
                isvoice += 1

            detected_frames[0].append(frame)
            if isvoice > 1:
                detected_frames[1].append(1)
            else:
                detected_frames[1].append(0)

        detected_frames[1] = signal.medfilt(np.array(detected_frames[1]), 25)
        if self.save_results:
            self.__save_detection_resutls(original_data=data,
                                          detected_frames=detected_frames[1],
                                          file_name=self.file_name)

        voice_cleared_audio = self.__get_cleared_data(detected_frames)
        return voice_cleared_audio
