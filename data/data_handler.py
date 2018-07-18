#! /opt/conda/bin/python3
""" DataHandler class with methods to work with the dataset """

# Copyright 2018 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import gzip
import json
from typing import List, Any, Dict, Union, Tuple
import numpy as np
# noinspection PyProtectedMember
from scipy.signal.windows import _len_guards, _extend, _truncate, exponential  # type: ignore


class DataHandler:
    """
    Class to handle the dataset and methods to work with it

    The class contains methods for validating the analytically created dataset,
    loading samples from the dataset and methods to convert the data samples
    from the more general dataset format into the required format for training
    the neural network.
    """

    def __init__(self, path: str = './') -> None:
        """
        Initializer of the class

        :param path: Default dataset path
        """
        super(DataHandler, self).__init__()
        self.path = path
        self.dataset = None  # type: str
        self._file_format = None  # type: str
        self._max_file_count = 0

    @staticmethod
    def makedir(path: str) -> None:
        """
        Create a directory if not existing

        :param path: Path to the directory
        """
        if not os.path.exists(path):
            # noinspection PyBroadException
            try:
                os.makedirs(path)
            except OSError:
                pass

    @staticmethod
    def _meyer_wavelet(length: int, symmetric: bool = True, width: int = 5) -> np.ndarray:
        """
        Calculate the meyer wavelet for a given length

        :param length: Length to calculate wavelet for
        :param symmetric: Symmetric wavelet
        :param width: Meyer wavelet width parameter
        :return: Time domain meyer wavelet
        """
        if _len_guards(length):
            return np.ones(length)
        length, needs_trunc = _extend(length, symmetric)
        t_lin = np.linspace(-width, width, length, endpoint=True)  # type: ignore
        t11 = (t_lin - 0.5)
        t23 = 2 * np.pi / 3 * (t_lin - 0.5)
        t43 = 2 * t23
        t83 = 2 * t43
        psi1 = (4 / (3 * np.pi) * t11 * np.cos(t23) - np.sin(t43) / np.pi) / (t11 - 16 / 9 * (t_lin - 0.5) ** 3)
        psi2 = (8 / (3 * np.pi) * t11 * np.cos(t83) + np.sin(t43) / np.pi) / (t11 - 64 / 9 * (t_lin - 0.5) ** 3)
        result = psi1 + psi2
        return _truncate(result, needs_trunc)

    def validate_dataset(self, path: str = None) -> bool:
        """
        Validate that the directory contains one of the dataset

        :param path: Path to check for the dataset
        :return: Whether a valid dataset was found
        :raises ValueError: Dataset seems not valid
        """
        checked = {}  # type: Dict[str, Any]
        # Check path exists
        path = path if path is not None else self.path
        if not os.path.exists(path):
            raise ValueError('The path to the dataset does not exists! ({:s})'.format(path))
        # Check readme
        checked['readme'] = os.path.exists(os.path.join(path, 'Readme.md'))
        # Check the different file formats
        file_formats = {
            'simple-zip': 'single_oscillation_{:04d}.zip',
            'advanced-gz': 'single_oscillation_0-2_{:04d}.gz',
            'big2-gz': 'single_oscillation_01-2_{:04d}.gz',
        }
        for file_key, file_format in file_formats.items():
            file_count = 0
            while os.path.exists(os.path.join(path, file_format.format(file_count))):
                file_count += 1
            checked[file_key] = file_count
        # Test for valid dataset
        if checked['readme'] and checked['simple-zip'] == 64:
            self.dataset = 'simple'
            self._file_format = 'single_oscillation_{:04d}.zip'
            self._max_file_count = 64
            return True
        elif checked['advanced-gz'] == 1024:
            self.dataset = 'big'
            self._file_format = 'single_oscillation_0-2_{:04d}.gz'
            self._max_file_count = 1024
            return True
        elif checked['big2-gz'] == 1024:
            self.dataset = 'big2'
            self._file_format = 'single_oscillation_01-2_{:04d}.gz'
            self._max_file_count = 1024
            return True
        return False

    @property
    def file_count(self) -> int:
        """
        Get the number of files for the dataset

        :return: Number of files
        :raises ValueError: No dataset has been validated before
        """
        if self.dataset is None:
            raise ValueError('No known dataset found!')
        return self._max_file_count

    def load_source_file(self, idx: int, path: str = None) -> List[Any]:
        """
        Load the data from the file with id idx

        :param idx: Index of the file to load
        :param path: Optional path where the files are
        :return: List of dictionaries representing the data
        :raises ValueError: No dataset has been validated before
        """
        if self.dataset is None:
            raise ValueError('No known dataset found!')
        path = path if path is not None else self.path
        idx = int(max(0, min(idx, self._max_file_count-1)))
        file_name = self._file_format.format(idx)
        with gzip.open(os.path.join(path, file_name), 'r') as file:
            json_data = file.read().decode()
        data = json.loads(json_data)
        return data

    # noinspection SpellCheckingInspection,PyUnresolvedReferences,PyTypeChecker
    def signal_to_training(  # pylint: disable=too-many-locals
            self,
            signal: Union[Dict, List[Dict]]
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], np.ndarray, Dict[str, Any]]:
        """
        Extract training data from the dataset

        :param signal: List of or single dataset entry
        :return: Time signals, Tuple of FFTs (with different windows), peak counting vector and config data
        """
        dict_list = list(signal) if isinstance(signal, list) else list((signal, ))

        # Initialize the return values
        time_length = len(dict_list[0]['signal']['time']['data'])  # type: ignore
        length = int(time_length / 2)
        signals = np.zeros((0, time_length))
        result_r = np.zeros((0, length))
        result_b = np.zeros((0, length))
        result_h = np.zeros((0, length))
        result_m = np.zeros((0, length))
        result_p = np.zeros((0, length))
        answer = np.zeros((0, length))
        config = {
            'SNR': [],
            'count': [],
            'frequencies': [],
            'amplitudes': [],
            'minamplitude': [],
            'mindist': []
        }  # type: Dict[str, Any]

        # Calculate window functions
        window_bartlett = np.bartlett(time_length)
        window_hanning = np.hanning(time_length)
        window_meyer = self._meyer_wavelet(time_length)
        window_poisson = exponential(time_length, sym=True, tau=(time_length/2)*(8.69/60.0))

        # Loop all data entries
        for data in dict_list:
            time = np.asarray(data['signal']['time']['data'])
            signals = np.concatenate((signals, np.reshape(time, (1,) + time.shape)))
            config['SNR'].append(data['signal']['SNR'])

            # Assemble the FFTs
            fft = np.fft.fft(time)[:length] / time_length
            result_r = np.concatenate((result_r, np.reshape(fft, (1,) + fft.shape)))
            fft = np.fft.fft(time * window_bartlett)[:length] / time_length
            result_b = np.concatenate((result_b, np.reshape(fft, (1,) + fft.shape)))
            fft = np.fft.fft(time * window_hanning)[:length] / time_length
            result_h = np.concatenate((result_h, np.reshape(fft, (1,) + fft.shape)))
            fft = np.fft.fft(time * window_meyer)[:length] / time_length
            result_m = np.concatenate((result_m, np.reshape(fft, (1,) + fft.shape)))
            fft = np.fft.fft(time * window_poisson)[:length] / time_length
            result_p = np.concatenate((result_p, np.reshape(fft, (1,) + fft.shape)))

            # Assemble all the frequencies and amplitudes
            count = 0
            freqs = []
            ampls = []
            counting = np.zeros((1, length))
            for subsig in data['signal']['parts']:
                if subsig['signal']['type'] == 'SingleOscillation':
                    count += 1
                    freq = subsig['signal']['frequency']
                    counting[0, int(max(0, min(length - 1, round(freq))))] += 1
                    freqs.append(freq)
                    ampls.append(subsig['signal']['amplitude'])
            config['count'].append(count)

            # Sort frequencies and amplitudes by frequency
            np_freqs = np.asarray(freqs)
            sorting = np.unravel_index(np.argsort(np_freqs), np_freqs.shape)
            np_freqs = np_freqs[sorting]
            np_ampls = np.asarray(ampls)[sorting]

            # Assemble some statistics
            config['mindist'].append(999999. if len(np_freqs) < 2 else np.min(np.diff(np_freqs)))
            config['minamplitude'].append(np.min(np_ampls) if len(np_ampls) > 0 else 999999.)
            config['frequencies'].append(np_freqs)
            config['amplitudes'].append(np_ampls)
            answer = np.concatenate((answer, counting))

        # Assemble results
        ffts = (result_r, result_b, result_h, result_m, result_p)
        return signals, ffts, answer, config
