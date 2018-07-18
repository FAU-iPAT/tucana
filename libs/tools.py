#! /opt/conda/bin/python3
""" Tools class with methods to help working with the data """

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


import math
import numpy as np
import matplotlib.pyplot as plt


class Tools:
    """
    Class to provide helper method for working with the data
    """

    @staticmethod
    def fft(data, normalize=False):
        """
        Method to calculate the FFT (removed average value and symmetric spectral part)

        :param data: Time series data
        :return: FFT frequency spectrum
        """
        mean = np.mean(data)
        # noinspection PyTypeChecker
        fft = np.fft.fftshift(np.fft.fft(data - mean) / len(data))
        fft = np.split(fft, 2)[1]
        if normalize is True:
            fft = fft / np.max(np.abs(fft))
        return fft

    @staticmethod
    def cleanup(predict):
        """
        Method to cleanup the prediction vector when prediction between two values

        This method takes a prediction certainty vector and adds the smaller
        certainty of two neighbouring predictions to the larger one to better
        account for frequencies detected between two bins.

        :param predict: Initial prediction to be cleaned
        :return: Cleaned predict vector
        """
        is_peak = (predict > 0.)
        result = predict.copy()
        for i in range(len(predict) - 3):
            if list(is_peak[i:i + 4]) == list([False, True, True, False]):
                if result[i + 1] >= result[i + 2]:
                    result[i + 1] += result[i + 2]
                    result[i + 2] = 0.
                else:
                    result[i + 2] += result[i + 1]
                    result[i + 1] = 0.
        return result

    @staticmethod
    def features2blocks(features):
        """
        Method to split arbitrary long features vectors into 256-length blocks

        The arbitrary long feature vector is split into 256-length blocks since
        the neural network Tucana was only trained to process feature vectors of
        this length. The last block is zero-padded if the length is no exact
        multiple of 256.

        :param features: Features vector to be split
        :return: List of 256-length feature vector blocks
        """
        block_count = int(math.ceil(features.shape[0] / 256))
        req_length = block_count * 256
        fill_block = np.ones((req_length - features.shape[0], features.shape[1]))
        features = np.concatenate([features, fill_block])
        blocks = []
        for i in range(block_count):
            blocks.append(features[i * 256:(i + 1) * 256])
        return blocks

    @staticmethod
    def predict(model, features):
        """
        Method to apply model.predict to feature vectors of arbitrary size

        :param model: Model to use for predictions
        :param features: Feature vector to predict for
        :return: Resulting prediction vector
        """
        blocks = Tools.features2blocks(features)
        predictions = []
        for feature in blocks:
            predictions.append(model.predict(np.reshape(feature, (1,) + feature.shape))[0])
        return np.concatenate(predictions)[0:features.shape[0]]

    # noinspection SpellCheckingInspection
    @staticmethod
    def plot_prediction(
            signal,
            prediction=None,
            answer=None,
            xlabel=True,
            ylabel=True,
            legend=True
    ):
        """
        Method to plot a signal with predictions and ground truth

        :param signal: Signal to be plotted
        :param prediction: Predictions (either binary vector or list)
        :param answer: Ground truth (either binary vector or list)
        :param xlabel: Include labels for x-axis
        :param ylabel: Include labels for y-axis
        :param legend: Include a legend
        """
        # Plot the answers
        if answer is not None and len(answer) == len(signal):
            ans = np.arange(len(answer))[answer > 0.]
            plt.stem(ans, -0.1 * np.ones(len(ans)), 'go-', label='Ground truth')
        elif answer is not None and len(answer) > 0:
            plt.stem(answer, -0.1 * np.ones(len(answer)), 'go-', label='Ground truth')
        # Plot the prediction
        if prediction is not None and len(prediction) == len(signal):
            pred = np.arange(len(prediction))[prediction > 0.]
            plt.stem(pred, np.ones(len(pred)), 'ro--', label='Prediction')
        elif prediction is not None and len(prediction) > 0:
            plt.stem(prediction, np.ones(len(prediction)), 'ro--', label='Prediction')
        # Plot the signal
        plt.plot(np.abs(signal), 'b', label='FFT signal')
        # Plot the zero y-axis
        plt.plot(np.arange(len(signal)), np.zeros(len(signal)), 'k')
        # Plot legend and axis
        if xlabel is not False:
            plt.xlabel('Frequency' if xlabel is True else xlabel)
        if ylabel is not False:
            plt.ylabel('Amplitude' if ylabel is True else ylabel)
        if legend is True:
            plt.legend()
