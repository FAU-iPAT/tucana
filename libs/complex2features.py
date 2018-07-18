#! /opt/conda/bin/python3
""" Method to convert complex number series into feature vectors (Re,Im, Abs) """

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


from typing import Tuple, Union
import numpy as np


_TData = np.ndarray
_TDataSet = Tuple[_TData, ...]


def c2f(
        data: Union[_TData, _TDataSet],
        normalize: Union[bool, int] = False,
        real: bool = True,
        imaginary: bool = True,
        absolute: bool = True,
        angle: bool = False,
        group_by_signal: bool = False
) -> _TData:
    """
    Wrapper function for Complex2Features.apply method

    :param data: Complex valued data or tuple of data
    :param normalize: Apply normalization to the data
    :param real: Include real part in result
    :param imaginary: Include imaginary part in result
    :param absolute: Include absolute value in result
    :param angle: Include angle in result
    :param group_by_signal: Group results by signal,
                else result is group by complex component
                (real, imaginary, absolute, angle)
    :return: Feature vectors for each data point
    """
    return Complex2Features.apply(
        data,
        normalize,
        real,
        imaginary,
        absolute,
        angle,
        group_by_signal
    )


class Complex2Features:  # pylint: disable=too-few-public-methods
    """
    Class to convert complex data to feature vectors

    Input data are a single data series or a tuple of data series of complex valued
    data. Results of the conversion is a feature vector for each entry consisting of
    real, imaginary, absolute and angle value for each data point.
    """

    @staticmethod
    def _data_to_dataset(data: Union[_TData, _TDataSet]) -> _TDataSet:
        """
        Method to ensure data being a tuple of data series

        :param data: Data or tuple of data
        :return: Tuple of data
        :raises ValueError: Empty dataset or mismatching dataset shapes
        """
        if isinstance(data, (tuple, list)):
            dataset = tuple(data)  # type: _TDataSet
        else:
            dataset = (data, )
        if not dataset:
            raise ValueError('Dataset needs to contain at least one entry!')
        shape = dataset[0].shape
        for entry in dataset:
            if entry.shape != shape:
                raise ValueError('All data sets need to have the same shape!')
        return dataset

    @staticmethod
    def _normalize_dataset(dataset: _TDataSet, normalize: Union[bool, int]) -> _TDataSet:
        """
        Apply dataset normalization

        Normalization is applied along the last dimension of the data. If the
        parameter is set to True the maximum absolute value among all dataset
        in the input tuple is used for scaling. If the parameter is set to an
        integer, the according entry in the tuple's maximum absolute value is
        used for all other entries of the tuple.

        :param dataset: Dataset to be normalized
        :param normalize: Normalization parameter
        :return: Normalized dataset
        """
        if normalize is not False and normalize is not None:
            shape = dataset[0].shape
            scale_values = np.zeros((*shape[0:-1], 0))
            for data in dataset:
                max_values = np.amax(np.abs(data), axis=-1, keepdims=True)
                scale_values = np.concatenate((scale_values, max_values), axis=-1)
            if normalize is True:
                scale_values = np.amax(scale_values, axis=-1, keepdims=True)
            else:
                scale_values = scale_values[..., normalize:(normalize+1)]
            result = ()  # type: _TDataSet
            for data in dataset:
                result += (data * np.reciprocal(scale_values.astype(float)), )
            return result
        return dataset

    @staticmethod
    def _convert_data_entry(
            data: _TData,
            real: bool = True,
            imaginary: bool = True,
            absolute: bool = True,
            angle: bool = False
    ) -> _TData:
        """
        Convert a single complex data series to a feature vector

        :param data: Single complex data series to be converted
        :param real: Include real part in result
        :param imaginary: Include imaginary part in result
        :param absolute: Include absolute value in result
        :param angle: Include angle in result
        :return: Resulting feature vector for input data
        """
        reshaped = np.reshape(data, data.shape + (1,))
        result = np.zeros(data.shape + (0,))
        if real:
            result = np.concatenate((result, np.real(reshaped)), axis=-1)
        if imaginary:
            result = np.concatenate((result, np.imag(reshaped)), axis=-1)
        if absolute:
            result = np.concatenate((result, np.abs(reshaped)), axis=-1)
        if angle:
            result = np.concatenate((result, np.angle(reshaped)), axis=-1)
        return result

    @classmethod
    def _convert_dataset(
            cls,
            dataset: _TDataSet,
            real: bool = True,
            imaginary: bool = True,
            absolute: bool = True,
            angle: bool = False
    ) -> _TData:
        """
        Convert a tuple of complex data series to a single feature vector

        :param dataset: Tuple of complex data series to be converted
        :param real: Include real part in result
        :param imaginary: Include imaginary part in result
        :param absolute: Include absolute value in result
        :param angle: Include angle in result
        :return: Resulting feature vector for input data (input signal grouped)
        """
        shape = dataset[0].shape
        result = np.zeros(shape + (0,))
        for data in dataset:
            part = cls._convert_data_entry(data, real, imaginary, absolute, angle)
            result = np.concatenate((result, part), axis=-1)
        return result

    @staticmethod
    def _assemble_dataset(dataset: _TDataSet) -> _TData:
        """
        Assemble a dataset into a single numpy array

        :param dataset: Dataset to be assembled (tuple of numpy arrays)
        :return: Single numpy array with whole dataset
        """
        result = np.zeros(dataset[0].shape + (0,))
        for data in dataset:
            result = np.concatenate((result, np.reshape(data, data.shape + (1,))), axis=-1)
        return result

    @classmethod
    def _block_convert_dataset(
            cls,
            dataset: _TDataSet,
            real: bool = True,
            imaginary: bool = True,
            absolute: bool = True,
            angle: bool = False
    ) -> _TData:
        """
        Convert a tuple of complex data series to a single feature vector

        :param dataset: Tuple of complex data series to be converted
        :param real: Include real part in result
        :param imaginary: Include imaginary part in result
        :param absolute: Include absolute value in result
        :param angle: Include angle in result
        :return: Resulting feature vector for input data (complex component grouped)
        """
        data = cls._assemble_dataset(dataset)
        result = np.zeros(tuple(list(data.shape[:-1])) + (0,))
        if real:
            result = np.concatenate((result, np.real(data)), axis=-1)
        if imaginary:
            result = np.concatenate((result, np.imag(data)), axis=-1)
        if absolute:
            result = np.concatenate((result, np.abs(data)), axis=-1)
        if angle:
            result = np.concatenate((result, np.angle(data)), axis=-1)
        return result

    @classmethod
    def apply(  # pylint: disable=too-many-arguments
            cls,
            data: Union[_TData, _TDataSet],
            normalize: Union[bool, int] = False,
            real: bool = True,
            imaginary: bool = True,
            absolute: bool = True,
            angle: bool = False,
            group_by_signal: bool = False
    ) -> _TData:
        """
        Method to convert complex data to feature vectors

        Whether the input data are a single data series or a tuple of multiple
        data series, the result will always be only a single data set containing
        the according feature vectors for all input data series. For returning
        the feature vectors an additional dimension is added to the data.

        x = np.ones((10,256))
        result = Complex2Features.apply((x,-x), normalize=1)
        print(result.shape)
        >>> (10,256,6)

        Normalization is applied along the last dimension of the input data. Thus
        batch-processing of multiple data is possible. For the normalization the
        maximum absolute value of the complex input data is always used. If the
        normalization parameter is set to True the maximum value among all data
        within the input tuple of data is used for scaling. If normalization
        parameter is set to a number, the maximum absolute value of the according
        entry in the tuple is used for all data within the tuples.

        :param data: Complex valued data or tuple of data
        :param normalize: Apply normalization to the data
        :param real: Include real part in result
        :param imaginary: Include imaginary part in result
        :param absolute: Include absolute value in result
        :param angle: Include angle in result
        :param group_by_signal: Group results by signal,
                else result is group by complex component
                (real, imaginary, absolute, angle)
        :return: Feature vectors for each data point
        """
        dataset = cls._data_to_dataset(data)
        dataset = cls._normalize_dataset(dataset, normalize)
        result = cls._block_convert_dataset(dataset, real, imaginary, absolute, angle) \
            if group_by_signal is not True \
            else cls._convert_dataset(dataset, real, imaginary, absolute, angle)
        return result
