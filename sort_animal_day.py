#!/usr/bin/env python

# prerequisites:
# pip install spikeforest
# pip install ml_ms4alg

# Spike sorting of one animal-day
import os
from mountaintools import client as mt
import spikeextractors as se
import spikeforest as sf
import spikeforestsorters as sorters
import ml_ms4alg
import numpy as np
import mlprocessors as mlpr
import spiketoolkit as st
import argparse
import tempfile
import shutil
from shellscript import ShellScript

def main():

    parser = argparse.ArgumentParser(description="Franklab spike sorting for a single animal day")
    parser.add_argument('--input', help='The input directory containing the animal day ephys data', )
    parser.add_argument('--output', help='The output directory where the sorting results will be written')
    parser.add_argument('--num_jobs', help='Number of parallel jobs', required=False, default=1)
    parser.add_argument('--force_run', help='Force the processing to run (no cache)', action='store_true')
    parser.add_argument('--test', help='Only run 2 epochs and 2 ntrodes in each', action='store_true')

    args = parser.parse_args()

    # animal_day_path = '/vortex2/jason/kf19/preprocessing/20170913'
    # animal_day_path = '20170913_kf19'
    # animal_day_output_path = 'test_animal_day_output'

    animal_day_path = args.input
    animal_day_output_path = args.output

    epoch_names = [name for name in sorted(os.listdir(animal_day_path)) if name.endswith('.mda')]
    if args.test:
        epoch_names = epoch_names[0:2]
    epochs = [
        load_epoch(animal_day_path + '/' + name, name=name[0:-4], test=args.test)
        for name in epoch_names
    ]

    mkdir2(animal_day_output_path)

    print('Num parallel jobs: {}'.format(args.num_jobs))

    # Start the job queue
    job_handler = mlpr.ParallelJobHandler(int(args.num_jobs))
    with mlpr.JobQueue(job_handler=job_handler) as JQ:
        for epoch in epochs:
            print('PROCESSING EPOCH: {}'.format(epoch['path']))
            mkdir2(animal_day_output_path + '/' + epoch['name'])
            for ntrode in epoch['ntrodes']:
                print('PROCESSING NTRODE: {}'.format(ntrode['path']))
                mkdir2(animal_day_output_path + '/' + epoch['name'] + '/' + ntrode['name'])
                firings_out = animal_day_output_path + '/' + epoch['name'] + '/' + ntrode['name'] + '/firings.mda'
                metrics_out = animal_day_output_path + '/' + epoch['name'] + '/' + ntrode['name'] + '/metrics.json'
                recording_file_in = ntrode['recording_file']
                geom_in = recording_file_in[0:-4] + '.geom.csv'
                if os.path.exists(geom_in):
                    print('Using geometry file: {}'.format(geom_in))
                else:
                    geom_in = None
                
                print('Sorting...')
                spike_sorting(
                    recording_file_in=recording_file_in,
                    geom_in=geom_in,
                    firings_out=firings_out,
                    metrics_out=metrics_out,
                    args=args
                )
        JQ.wait()

def load_ntrode(path, *, name):
    return dict(
        name=name,
        path=path,
        recording_file=mt.createSnapshot(path=path)
    )

def load_epoch(path, *, name, test=False):
    ntrode_names = [name for name in sorted(os.listdir(path)) if name.endswith('.mda')]
    if test:
        ntrode_names = ntrode_names[0:2]
    ntrodes = [
        load_ntrode(path + '/' + name2, name=name2[0:-4])
        for name2 in ntrode_names
    ]
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
    VERSION = '0.1.7'

    recording_file_in = mlpr.Input('Path to raw.mda')
    geom_in = mlpr.Input('Path to geom.csv', optional=True)
    firings_out = mlpr.Output('Output firings.mda file')
    # firings_curated_out = mlpr.Output('Output firings.curated.mda file')
    metrics_out = mlpr.Output('Metrics .json output')

    samplerate = mlpr.FloatParameter("Sampling frequency")

    mask_out_artifacts = mlpr.BoolParameter(optional=True, default=False,
                                description='Whether to mask out artifacts')
    freq_min = mlpr.FloatParameter(
        optional=True, default=300, description='Use 0 for no bandpass filtering')
    freq_max = mlpr.FloatParameter(
        optional=True, default=6000, description='Use 0 for no bandpass filtering')
    whiten = mlpr.BoolParameter(optional=True, default=True,
                                description='Whether to do channel whitening as part of preprocessing')

    detect_sign = mlpr.IntegerParameter(
        'Use -1, 0, or 1, depending on the sign of the spikes in the recording')
    adjacency_radius = mlpr.FloatParameter(
        'Use -1 to include all channels in every neighborhood')
    
    
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

        with TemporaryDirectory() as tmpdir:
            if self.mask_out_artifacts:
                print('Masking out artifacts...')
                rec_fname = tmpdir + '/raw.mda'
                _mask_out_artifacts(self.recording_file_in, rec_fname)
            else:
                rec_fname = self.recording_file_in

            X = sf.mdaio.readmda(rec_fname)
            if type(self.geom_in) == str:
                print('Using geom.csv from a file', self.geom_in)
                geom = _read_geom_csv(self.geom_in)
            else:
                # no geom file was provided as input
                num_channels = X.shape[0]
                if num_channels > 6:
                    raise Exception('For more than six channels, we require that a geom.csv be provided')
                # otherwise make a trivial geometry file
                print('Making a trivial geom file.')
                geom = np.zeros((X.shape[0], 2))
            recording = se.NumpyRecordingExtractor(X, samplerate=30000, geom=geom)
            recording = st.preprocessing.bandpass_filter(
                recording=recording,
                freq_min=self.freq_min, freq_max=self.freq_max
            )
            if self.whiten:
                recording = st.preprocessing.whiten(recording=recording)

            num_workers = 2

            sorting = ml_ms4alg.mountainsort4(
                recording=recording,
                detect_sign=self.detect_sign,
                adjacency_radius=self.adjacency_radius,
                clip_size=self.clip_size,
                detect_threshold=self.detect_threshold,
                detect_interval=self.detect_interval,
                num_workers=num_workers,
            )
            sf.SFMdaSortingExtractor.write_sorting(sorting=sorting, save_path=self.firings_out)

            # not sure why this is not working
            # result = sorters.MountainSort4.execute(
            #     recording_dir=recording_dir,
            #     firings_out=self.firings_out,
            #     detect_sign=self.detect_sign,
            #     adjacency_radius=self.adjacency_radius,
            #     clip_size=self.clip_size,
            #     detect_threshold=self.detect_threshold,
            #     detect_interval=self.detect_interval,
            #     num_workers=num_workers,
            #     _use_cache=False
            # )

            
            print('Writing preprocessed data, preparing for automated curation...')
            recording_dir = tmpdir + '/pre'
            sf.SFMdaRecordingExtractor.write_recording(recording=recording, save_path=recording_dir)

            print('Computing cluster metrics...')
            cluster_metrics_path = tmpdir +'/cluster_metrics.json'
            _cluster_metrics(recording_dir + '/raw.mda', self.firings_out, cluster_metrics_path)

            print('Computing isolation metrics...')
            isolation_metrics_path = tmpdir +'/isolation_metrics.json'
            pair_metrics_path = tmpdir +'/pair_metrics.json'
            _isolation_metrics(recording_dir + '/raw.mda', self.firings_out, isolation_metrics_path, pair_metrics_path)

            print('Combining metrics...')
            metrics_path = tmpdir +'/metrics.json'
            _combine_metrics(cluster_metrics_path, isolation_metrics_path, metrics_path)

            shutil.copy(metrics_path, self.metrics_out)
            

class TemporaryDirectory():
    def __init__(self):
        pass

    def __enter__(self):
        self._path = tempfile.mkdtemp()
        return self._path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self._path)

    def path(self):
        return self._path

def _mask_out_artifacts(timeseries_in, timeseries_out):
    script = ShellScript('''
    #!/bin/bash
    ml-run-process ms3.mask_out_artifacts -i timeseries:{} -o timeseries_out:{} -p threshold:6 interval_size:2000 --force_run
    '''.format(timeseries_in, timeseries_out))
    script.start()
    retcode = script.wait()
    if retcode != 0:
        raise Exception('problem running ms3.mask_out_artifacts')

def _cluster_metrics(timeseries, firings, metrics_out):
    script = ShellScript('''
    #!/bin/bash
    ml-run-process ms3.cluster_metrics -i timeseries:{} firings:{} -o cluster_metrics_out:{} -p samplerate:30000 --force_run
    '''.format(timeseries, firings, metrics_out))
    script.start()
    retcode = script.wait()
    if retcode != 0:
        raise Exception('problem running ms3.cluster_metrics')

def _isolation_metrics(timeseries, firings, metrics_out, pair_metrics_out):
    script = ShellScript('''
    #!/bin/bash
    ml-run-process ms3.isolation_metrics -i timeseries:{} firings:{} -o metrics_out:{} pair_metrics_out:{} --force_run
    '''.format(timeseries, firings, metrics_out, pair_metrics_out))
    script.start()
    retcode = script.wait()
    if retcode != 0:
        raise Exception('problem running ms3.isolation_metrics')

def _combine_metrics(metrics1, metrics2, metrics_out):
    script = ShellScript('''
    #!/bin/bash
    ml-run-process ms3.combine_cluster_metrics -i metrics_list:{} metrics_list:{} -o metrics_out:{} --force_run
    '''.format(metrics1, metrics2, metrics_out))
    script.start()
    retcode = script.wait()
    if retcode != 0:
        raise Exception('problem running ms3.combine_metrics')

def _read_geom_csv(path):
    geom = np.genfromtxt(path, delimiter=',')
    return geom

def spike_sorting(*, recording_file_in, geom_in, firings_out, metrics_out, args):
    params = dict(
        recording_file_in=recording_file_in,
        firings_out=firings_out,
        metrics_out=metrics_out,
        mask_out_artifacts=True,
        freq_min=300,
        freq_max=6000,
        whiten=True,
        samplerate=30000,
        detect_sign=-1,
        adjacency_radius=50,
        _force_run=args.force_run
    )
    if geom_in:
        params['geom_in'] = geom_in
    CustomSorting.execute(**params)

def mkdir2(path):
    if not os.path.exists(path):
        os.mkdir(path)


        
if __name__ == '__main__':
    main()