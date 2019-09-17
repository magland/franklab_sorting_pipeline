#!/usr/bin/env python
# coding: utf-8

# Spike sorting of one animal-day
import os
from mountaintools import client as mt
import spikeextractors as se
import spikeforest as sf
import spikeforestsorters as sorters
import ml_ms4alg
import numpy as np
import mlprocessors as mlpr

def main():
    # animal_day_path = '/vortex2/jason/kf19/preprocessing/20170913'
    animal_day_path = '20170913_kf19'
    animal_day_output_path = 'test_animal_day_output'

    epochs = []
    for name in sorted(os.listdir(animal_day_path)):
        if name.endswith('.mda'):
            epochs.append(load_epoch(animal_day_path +
                                    '/' + name, name=name[0:-4]))

    mkdir2(animal_day_output_path)
    intermediate_path = animal_day_output_path + '/intermediate'
    mkdir2(intermediate_path)

    # Start the job queue
    job_handler = mlpr.ParallelJobHandler(3)
    # for now the parallel job queue seems to have an issue
    # with mlpr.JobQueue(job_handler=job_handler) as JQ:
    for epoch in epochs:
        print('PROCESSING EPOCH: {}'.format(epoch['path']))
        mkdir2(animal_day_output_path + '/' + epoch['name'])
        mkdir2(intermediate_path + '/' + epoch['name'])
        for ntrode in epoch['ntrodes']:
            print('PROCESSING NTRODE: {}'.format(ntrode['path']))
            mkdir2(animal_day_output_path + '/' + epoch['name'] + '/' + ntrode['name'])
            recording_dir = intermediate_path + '/' + epoch['name'] + '/' + ntrode['name']
            firings_out = animal_day_output_path + '/' + epoch['name'] + '/' + ntrode['name'] + '/firings.mda'
            X = sf.mdaio.readmda(mt.realizeFile(ntrode['ephys']))
            geom = np.zeros((X.shape[0], 2))
            recording = se.NumpyRecordingExtractor(X, samplerate=30000, geom=geom)
            sf.SFMdaRecordingExtractor.write_recording(recording=recording, save_path=recording_dir)
            print('Sorting...')
            spike_sorting(
                recording_dir,
                firings_out
            )
        #JQ.wait()

def load_ntrode(path, *, name):
    return dict(
        name=name,
        path=path,
        ephys=mt.createSnapshot(path=path)
    )

def load_epoch(path, *, name):
    ntrodes = []
    for name2 in sorted(os.listdir(path)):
        if name2.endswith('.mda'):
            ntrodes.append(
                load_ntrode(path + '/' + name2, name=name2[0:-4])
            )
    return dict(
        path=path,
        name=name,
        ntrodes=ntrodes
    )




# Start the job queue
def mkdir2(path):
    if not os.path.exists(path):
        os.mkdir(path)

# See: https://github.com/flatironinstitute/spikeforest/blob/master/spikeforest/spikeforestsorters/mountainsort4/mountainsort4.py
class CustomSorting(mlpr.Processor):
    NAME = 'CustomSorting'
    VERSION = '0.1.0'

    recording_dir = mlpr.Input('Directory of recording', directory=True)
    firings_out = mlpr.Output('Output firings file')

    detect_sign = mlpr.IntegerParameter(
        'Use -1, 0, or 1, depending on the sign of the spikes in the recording')
    adjacency_radius = mlpr.FloatParameter(
        'Use -1 to include all channels in every neighborhood')
    freq_min = mlpr.FloatParameter(
        optional=True, default=300, description='Use 0 for no bandpass filtering')
    freq_max = mlpr.FloatParameter(
        optional=True, default=6000, description='Use 0 for no bandpass filtering')
    whiten = mlpr.BoolParameter(optional=True, default=True,
                                description='Whether to do channel whitening as part of preprocessing')
    clip_size = mlpr.IntegerParameter(
        optional=True, default=50, description='')
    detect_threshold = mlpr.FloatParameter(
        optional=True, default=3, description='')
    detect_interval = mlpr.IntegerParameter(
        optional=True, default=10, description='Minimum number of timepoints between events detected on the same channel')
    noise_overlap_threshold = mlpr.FloatParameter(
        optional=True, default=0.15, description='Use None for no automated curation')

    def run(self):
        # Replace this function with system calls, etc to do
        # mask_out_artifactrs, ml_ms4alg, curation, etc.
        print('================================================= test 1')
        sorters.MountainSort4.execute(
            recording_dir=self.recording_dir,
            firings_out=self.firings_out,
            detect_sign=self.detect_sign,
            clip_size=self.clip_size,
            adjacency_radius=self.adjacency_radius,
            detect_threshold=self.detect_threshold,
            detect_interval=self.detect_interval,
            num_workers=1,
            _use_cache=False
        )
        print('================================================= test 2')


def spike_sorting(recording_dir, firings_out):
    CustomSorting.execute(
        recording_dir=recording_dir,
        firings_out=firings_out,
        detect_sign=-1,
        adjacency_radius=50
    )

def mkdir2(path):
    if not os.path.exists(path):
        os.mkdir(path)


        
if __name__ == '__main__':
    main()