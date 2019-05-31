"""Extractor of features from audio data
Extracts the following features: log mel filterbank energies"""

#region Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
#endregion

class FeatureExtractor:
    def __init__(self, data_array, sample_rate, num_of_coeffs):
        self.data = data_array
        self.channels = len(self.data.shape)
        if self.channels == 2:
            self.__convert_to_mono()
        self.rate = sample_rate
        self.num_of_coeffs = num_of_coeffs

        self.sizeof_frame = int(0.240 * self.rate)  # 240 ms
        self.sliding_step = int(0.120 * self.rate)  # 120 ms
        self.num_of_frames = len(self.data)//self.sliding_step
        tail_len = (self.num_of_frames * self.sliding_step + self.sizeof_frame - self.sliding_step)\
                    - len(self.data)
        padded_data = np.zeros(len(self.data) + tail_len)
        padded_data[:len(self.data)] = self.data
        self.data = padded_data

    def __convert_to_mono(self):
        self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
        self.channels = 1
        return self

    def __convert_hertz_to_mel(self, frequency):
        return 1125*math.log1p(frequency/700)

    def __convert_mel_to_hertz(self, mel):
        return 700 * (math.exp(mel / 1125) - 1)

    def __convert_mel_array_to_hertz(self, mel_array):
        return 700 * (np.exp(mel_array / 1125) - 1)

    def __calculate_mel_filterbanks(self, number_of_filters, dft_length):
        lower_freq = 300
        upper_freq = self.rate / 2
        lower_mel = self.__convert_hertz_to_mel(lower_freq)
        upper_mel = self.__convert_hertz_to_mel(upper_freq)

        # here we will write mel-spaced frequencies
        mel_freq = np.linspace(lower_mel, upper_mel, number_of_filters + 2)
        mel_freq = self.__convert_mel_array_to_hertz(mel_freq)
        mel_freq = np.floor((dft_length + 1) * mel_freq / self.rate)

        mel_filterbanks = np.zeros((number_of_filters, dft_length//2))
        for i in range(1, number_of_filters + 1):
            left = int(mel_freq[i-1])
            center = int(mel_freq[i])
            right = int(mel_freq[i+1])
            for k in range(left, center):
                mel_filterbanks[i-1, k] = (k - mel_freq[i-1]) / (mel_freq[i] - mel_freq[i-1])
            for k in range(center, right):
                mel_filterbanks[i-1, k] = (mel_freq[i+1] - k) / (mel_freq[i+1] - mel_freq[i])
        # Show mel filterbanks
        # plt.figure()
        # for idx in range(number_of_filters):
        #     plt.plot(mel_filterbanks[idx])
        # plt.show()
        return mel_filterbanks

    def __calculate_power_spectrum(self, spectrum):
        data_length = len(spectrum)
        power_spectrum = (1.0 / data_length) * (spectrum ** 2)
        return power_spectrum

    def extract_log_mel_filterbank_energies(self):
        data = self.data
        sizeof_frame = self.sizeof_frame
        num_of_frames = self.num_of_frames
        num_of_coeffs = self.num_of_coeffs
        dft_len = 512  # we would perform a 512 point Discrete Fourier Transform
        spectrogram = np.zeros((self.num_of_frames, num_of_coeffs))
        mel_filterbanks = self.__calculate_mel_filterbanks(num_of_coeffs, dft_length=dft_len)

        frame_start = 0
        sliding_step = self.sliding_step
        for frame_idx in range(num_of_frames):
            frame_end = frame_start + sizeof_frame
            frame = data[frame_start:frame_end]

            hamming_window = np.hamming(sizeof_frame)
            frame_spectrum = np.abs(np.fft.rfft(frame * hamming_window, n=dft_len))
            power_spectrum = self.__calculate_power_spectrum(frame_spectrum[1:])

            power_spectrum = np.tile(power_spectrum, (num_of_coeffs, 1))
            filterbank_energies = np.sum(power_spectrum*mel_filterbanks, axis=1)
            log_filterbank_energies = np.log(filterbank_energies)
            # spectrogram[frame_idx] = log_filterbank_energies

            mfcc_coeffs = dct(log_filterbank_energies)
            spectrogram[frame_idx] = mfcc_coeffs

            frame_start += sliding_step
        return spectrogram
