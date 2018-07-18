#! /opt/conda/bin/python3
""" Training script for tucana model """

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

import argparse

parser = argparse.ArgumentParser(description='Tucana v5 training script')
# Data source parameters
parser.add_argument('--databasepath', type=str)
parser.add_argument('--datapath', action='append', type=str)
parser.add_argument('--answerpath', action='append', type=str)
parser.add_argument('--configpath', action='append', type=str)
parser.add_argument('--modelfile', type=str)
parser.add_argument('--fileformat', type=str)
parser.add_argument('--maxfilecount', type=int)
parser.add_argument('--nocache', type=int)
# Resume parameters
parser.add_argument('--resumefile', type=str)
parser.add_argument('--initialepoch', type=int)
parser.add_argument('--checkpoint', type=int)
# Solver parameters
parser.add_argument('--learningrate', type=float)
parser.add_argument('--decay', type=float)
# Batch parameters
parser.add_argument('--batchsize', type=int)
# Results parameters
parser.add_argument('--resultpath', type=str)
parser.add_argument('--bestfile', type=str)
parser.add_argument('--runstatsfile', type=str)
parser.add_argument('--checkpointfile', type=str)
parser.add_argument('--cprunstatsfile', type=str)
parser.add_argument('--weightfile', type=str)
parser.add_argument('--tensorboard', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--verbose', type=int)
parser.add_argument('--histogramfreq', type=int)
# Data filter parameters
parser.add_argument('--mindist', type=float)
# Data enhancement parameters
# Specific parameters
parser.add_argument('--userectangle', type=int)
parser.add_argument('--usebartlett', type=int)
parser.add_argument('--usehanning', type=int)
parser.add_argument('--usemeyer', type=int)
parser.add_argument('--usepoisson', type=int)
parser_args = parser.parse_args()
cmdargs = dict(vars(parser_args))

def getarg(name, default):
    global cmdargs
    return cmdargs[name] if name in cmdargs and cmdargs[name] is not None else default



print('')
print('#################################')
print('###   Start training script   ###')
print('#################################')
print('')

for key in cmdargs.keys():
    print('{:16s} = {}'.format(key, cmdargs[key]))



# Import the libraries

print('')
print('###############################')
print('###   Importing libraries   ###')
print('###############################')
print('')

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.callbacks import Callback
from dieFFT.toolbox.models import Model, model_from_json
from dieFFT.toolbox.callbacks import StopFileInterrupt, RunStatsCallback
from dieFFT.toolbox import c2f, DataGenerator, DataGeneratorSelection, DataGeneratorEnhancement
import numpy as np
import os



# Check optional arguments

print('')
print('#########################################')
print('###   Checking setup and parameters   ###')
print('#########################################')
print('')

if getarg('nocache', 0) > 0:
    DataGenerator.clear_cache()

data_path = getarg('databasepath', './')
result_path = getarg('resultpath', './')
model_file = getarg('modelfile', './lyra.json')
resume_file = getarg('resumefile', None)
file_format = getarg('fileformat', 'batch_{0:05d}.npy')

paths = []
if getarg('userectangle', 1) > 0:
    paths.append(data_path+'rectangle/')
if getarg('usebartlett', 1) > 0:
    paths.append(data_path+'bartlett/')
if getarg('usehanning', 0) > 0:
    paths.append(data_path+'hanning/')
if getarg('usemeyer', 1) > 0:
    paths.append(data_path+'meyer/')
if getarg('usepoisson', 0) > 0:
    paths.append(data_path+'poisson/')
path_data = []
for path in getarg('datapath', paths):
    if os.path.isdir(path):
        path_data.append(path)
        
path_answer = []
for path in getarg('answerpath', [data_path+'answer/']):
    if os.path.isdir(path):
        path_answer.append(path)

path_config = []
for path in getarg('configpath', [data_path+'config/']):
    if os.path.isdir(path):
        path_config.append(path)
        
file_best = getarg('bestfile', result_path + 'best.hdf5')
file_final = getarg('weightfile', result_path+'final.hdf5')
file_checkpoint = getarg('checkpointfile', result_path + 'checkpoint.hdf5')
path_tensorboard = getarg('tensorboard', result_path)
file_runstats = getarg('runstatsfile', result_path+'runstats.npy')
file_cp_runstats = getarg('cprunstatsfile', result_path+'checkpoint_runstats.npy')

config_batchsize = getarg('batchsize', 128)
config_epochs = getarg('epochs', 150)
config_verbose = getarg('verbose', 1)
config_checkpoint = bool(getarg('checkpoint', 0) > 0)
config_histogram_freq = getarg('histogramfreq', 0)
config_initial_epoch = getarg('initialepoch', 0)

config_min_dist = getarg('mindist', None)


    
# Print the resulting setup
    
print('Model File = '+model_file)
if isinstance(path_data, (tuple,list)):
    for path in path_data:
        print('Data Path = '+path)
else:
    print('Data Path = '+path_data)
if isinstance(path_answer, (tuple,list)):
    for path in path_answer:
        print('Answer Path = '+path)
else:
    print('Answer Path = '+path_answer)
if isinstance(path_config, (tuple,list)):
    for path in path_config:
        print('Config Path = '+path)
else:
    print('Config Path = '+path_config)
print('File Format = '+file_format)
if resume_file is not None:
    print('Resuming from file = '+resume_file)
print('')
print('Best File = ' + file_best)
print('Final Weights File = '+file_final)
print('Tensorboard Path = '+path_tensorboard)
print('Runtime Statistic File = '+file_runstats)
print('')
print('Batch Size = {0}'.format(config_batchsize))
print('')
print('Minimal Distance = {0:5.3f}'.format(config_min_dist if config_min_dist is not None else 0.0))



# Build the result directory

def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            pass
        except:
            pass
mkdir(result_path)



# Load the model and compile it

print('')
print('####################################')
print('###   Load and compiling model   ###')
print('####################################')
print('')

optimizer = Adam(
    lr = getarg('learningrate', 0.001),
    # beta_1=0.9,
    # beta_2=0.999,
    # epsilon=1e-8,
    decay = getarg('decay', 0.0)
)

with open(model_file) as file:
    json = file.read()
model = model_from_json(json)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_accuracy"])
model.summary()
if resume_file is not None:
    model.load_weights(resume_file)



# Define the data preparation class

class DataPrepare(DataGeneratorEnhancement):

    def __init__(self):
        super(DataPrepare, self).__init__()

    def enhance(self, data_in, answer_in, config):
        data = c2f(data_in, normalize=0, real=True, imaginary=True, absolute=True)
        config.append(answer_in[0])
        answer = np.minimum(answer_in[0], np.ones(answer_in[0].shape))
        return [data], [answer], config

prepare = DataPrepare()



# Define the data enhancement class

class DataEnhance(DataGeneratorEnhancement):

    def __init__(self):
        super(DataEnhance, self).__init__()

    def enhance(self, data, answer, config):
        return data, answer, config

enhance = DataEnhance()



# Define the data selection class

class DataSelect(DataGeneratorSelection):

    def __init__(self, mindist=None):
        super(DataSelect, self).__init__()
        self.mindist = mindist

    def select(self, idx, data, answer, config):
        valid = np.equal(idx, idx)
        if self.mindist is not None:
            valid = np.logical_and(valid, np.asarray(config[0]['mindist']) >= self.mindist)
        return valid

select = DataSelect(
    mindist=config_min_dist,
)



# Define the generators

print('')
print('###############################')
print('###   Building generators   ###')
print('###############################')
print('')

dg = DataGenerator(
    path_data=path_data,
    path_answer=path_answer,
    path_config=path_config,
    file_format=file_format,
    file_limit=getarg('maxfilecount', None),  # Automatic counting
    validation=0.2,
    testing=0.1,
    data_loader='numpy_dict',
    file_size=None,  # Get automatically from first data file
    batch_size=config_batchsize,
    verbose=1,
    nocache=(getarg('nocache', 0) > 0),
)

gen = dg.generator(
    dataset='training',
    batch_size=None,
    shuffle=True,
    prepend_idx=False,
    append_config=False,
    preparation=(prepare),
    selection=(select),
    enhancement=(enhance),
)

gen_count = dg.batches(dataset='training', batch_size=None)

val_gen = dg.generator(
    dataset='validation',
    batch_size=None,
    shuffle=True,
    prepend_idx=False,
    append_config=False,
    preparation=(prepare),
    selection=(select),
    enhancement=(enhance),
)

val_gen_count = dg.batches(dataset='validation', batch_size=None)

test_gen = dg.generator(
    dataset='testing',
    batch_size=None,
    shuffle=True,
    prepend_idx=False,
    append_config=False,
    preparation=(prepare),
    selection=(select),
    enhancement=(enhance),
)

test_gen_count = dg.batches(dataset='testing', batch_size=None)



# Counting data sets remaining after filtering

print('')
print('#################################')
print('###   Counting data samples   ###')
print('#################################')
print('')

if gen_count > 0 :
    count_samples = dg.count(dataset='training', preparation=(prepare), selection=(select))
    print('Valid Training Samples = {0}   ({1:5.2f}%)'.format(
        count_samples,
        100 * count_samples / max(1, dg.batches(dataset='training', batch_size=1))
    ))

if val_gen_count > 0:
    val_count_samples = dg.count(dataset='validation', preparation=(prepare), selection=(select))
    print('Valid Validation Samples = {0}   ({1:5.2f}%)'.format(
        val_count_samples,
        100 * val_count_samples / max(1, dg.batches(dataset='validation', batch_size=1))
    ))

if test_gen_count > 0:
    test_count_samples = dg.count(dataset='testing', preparation=(prepare), selection=(select))
    print('Valid Testing Samples = {0}   ({1:5.2f}%)'.format(
        test_count_samples,
        100 * test_count_samples / max(1, dg.batches(dataset='testing', batch_size=1))
    ))



# Setup the fitting, start training and save final model

print('')
print('#################################')
print('###   Run training of model   ###')
print('#################################')
print('')

class CPRunStats(Callback):
    def __init__(self, runstats, filename):
        super(CPRunStats, self).__init__()
        self._runstats = runstats
        self._filename = filename
    def on_epoch_end(self, epoch, logs):
        runstats = self._runstats.runstats
        np.save(self._filename, runstats)

rsCallback = RunStatsCallback()
cbstack = [
    StopFileInterrupt(),
    ModelCheckpoint(file_best, save_best_only=True),
    TensorBoard(log_dir=path_tensorboard, histogram_freq=config_histogram_freq, write_graph=True, write_images=True),
    rsCallback
]
if config_checkpoint is True:
    cbstack.append(ModelCheckpoint(file_checkpoint, save_best_only=False))
    cbstack.append(CPRunStats(rsCallback, file_cp_runstats))

model.fit_generator(
    gen,
    steps_per_epoch=gen_count,
    epochs=config_epochs,
    callbacks=cbstack,
    max_q_size=15,
    validation_data=val_gen,
    validation_steps=val_gen_count,
    verbose=config_verbose,
    initial_epoch=config_initial_epoch,
)
runstats = rsCallback.runstats
model.save(file_final)



# Assemble best metrices and save the runstats

print('')
print('################################################')
print('###   Assembling stats and saving runstats   ###')
print('################################################')
print('')

print('Statistics of best model (best validation accuracy):')
best_stats = {}
names = model.metrics_names
model.load_weights(file_best)

best_list = [
    {'dataset': 'training', 'key': 'training', 'label': 'Training', 'listkey': 'traininglist'},
    {'dataset': 'validation', 'key': 'validation', 'label': 'Validation', 'listkey': 'validationlist'},
    {'dataset': 'testing', 'key': 'testing', 'label': 'Testing', 'listkey': 'testinglist'},
]
for best_entry in best_list:
    eval_dict = {key: [] for key in names}
    best = dg.generator(
        dataset=best_entry['dataset'],
        batch_size=None,
        shuffle=False,
        prepend_idx=False,
        append_config=False,
        preparation=(prepare),
        selection=(select),
        enhancement=(enhance),
    )
    best_count = dg.batches(dataset=best_entry['dataset'], batch_size=None)
    if best_count > 0:
        batch_sizes = []
        for _ in range(best_count):
            data, answer = next(best)
            batch_sizes.append(data[0].shape[0])
            stats = model.evaluate(data, answer, verbose=0)
            for key, value in zip(names, stats):
                eval_dict[key].append(value)
        eval_avg = {}
        for key in names:
            eval_avg[key] = np.average(np.asarray(eval_dict[key]), weights=batch_sizes)
        best_stats[best_entry['key']] = eval_avg
        best_stats[best_entry['listkey']] = eval_dict
        print('{0:25s} = {1:6.2f}%'.format(best_entry['label']+' accuracy', 100.0*eval_avg['binary_accuracy']))

runstats['best'] = best_stats
np.save(file_runstats, runstats)



print('')
print('###############################')
print('###   End training script   ###')
print('###############################')
print('')
