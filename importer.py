import os
import glob
import fnmatch
import time
import guitarpro
import numpy as np
import pickle

__author__ = 'Jakob Abesser'


class Importer:

    BASS_MIDI_CHANNEL_RANGE = np.arange(33, 41)
    TIME_SIGNATURE_NUMERATOR = 4

    def __init__(self,
                 dir_root,
                 dir_data,
                 valid_extensions=None):
        self.dir_root = dir_root
        self.dir_data = dir_data
        if valid_extensions is None:
            self.valid_extensions = ['.gp3',
                                     '.gp4',
                                     '.gp5']
        self.pitch_selection = [np.array((35, 36)),
                                np.array((37, 38, 40)),
                                np.array((42, 44, 46, 51, 53))]

    def load_file_metadata(self):
        """ Read list of guitar pro files in subfolds of root directory
        Returns:
            files (list of dict): Each file is a dictionary with keys
                'filename': Absolute filename
                'artist': Artist
                'title': Title
        """
        fn_list = []
        for root, dirnames, filenames in os.walk(self.dir_root):
            for filename in fnmatch.filter(filenames, '*.gp*'):
                fn_list.append(os.path.join(root, filename))

        num_files = len(fn_list)
        files = [{'name': _} for _ in fn_list]
        for f in range(num_files):
            parts = fn_list[f].split(os.sep)
            dummy = parts[-1].split('-')
            files[f]['artist'] = parts[-2]
            if len(dummy) >= 2:
                files[f]['title'] = dummy[1].strip()
            else:
                files[f]['title'] = parts[-1].strip()

        print('{} guitar pro files found!'.format(num_files))

        return files

    def import_guitar_pro_files(self, files, skip_extracted_files=True):

        num_files = len(files)

        valid_files = 0

        for f, file_ in enumerate(files):
            try:
                if f % 50 == 0:
                    print('Process file {}/{}'.format(f+1, num_files))

                gp_file = guitarpro.parse(file_['name'])

                files[f]['title_gp'] = gp_file.title
                files[f]['artist_gp'] = gp_file.artist
                files[f]['tempo_gp'] = None

                tracks = gp_file.tracks
                num_tracks = len(tracks)

                # identify drum & bass track
                drum_track_idx = []
                bass_track_idx = []
                for t in range(num_tracks):
                    if tracks[t].isPercussionTrack:
                        drum_track_idx.append(t)
                    if tracks[t].channel.instrument in self.BASS_MIDI_CHANNEL_RANGE:
                        bass_track_idx.append(t)
                if len(bass_track_idx) != 1 or len(drum_track_idx) != 1:
                    continue

                # process file
                fn_target = os.path.join(self.dir_data,
                                         '{}_score'.format(os.path.basename(file_['name']).replace('.', '_')))
                fn_target = fn_target.replace(' ', '')

                # skip?
                if skip_extracted_files and os.path.isfile(fn_target):
                    continue

                track_idx = [bass_track_idx[0], drum_track_idx[0]]

                # check time signature
                time_sig_numerator = tracks[track_idx[0]].measures[0].timeSignature.numerator
                if time_sig_numerator != self.TIME_SIGNATURE_NUMERATOR:
                    continue
                score = [None, None]
                for i, idx in enumerate(track_idx):
                    track = tracks[idx]

                    pitch = []
                    onset = []
                    for bar, measure in enumerate(track.measures):
                        if bar == 0: # todo try
                            continue
                        for voice in measure.voices:
                            # Tempo
                            if files[f]['tempo_gp'] is None:
                                files[f]['tempo_gp'] = measure.tempo.value
                            # Notes
                            for beat in voice.beats:
                                for note in beat.notes:
                                    if note.type.name == 'normal':
                                        # general note parameters
                                        pitch.append(note.realValue)
                                        quarter_duration = float(note.beat.duration.quarterTime)
                                        onset.append((note.beat.start / quarter_duration - 1))
                                        # print('{} - {} - {} - {}'.format(note.beat.start,
                                        #                             note.beat.start / quarter_duration - 1,
                                        #                             bar,
                                        #                             onset[-1]))
                    onset = np.array(onset)
                    # assert np.all(np.diff(onset) > 0)

                    pitch = np.array(pitch)
                    if i == 1:
                        # map drum instrument pitches to 3 classes:
                        for d in range(3):
                            pitch[np.in1d(pitch, self.pitch_selection[d])] = d
                        select = np.in1d(pitch, np.arange(3))
                        # remove all other notes
                        onset = onset[select]
                        pitch = pitch[select]

                    score[i] = np.vstack((onset, pitch))
                files[f]['score'] = score

                with open(fn_target, 'wb+') as fh:
                    pickle.dump(files[f], fh)

                valid_files += 1
            except:
                pass

        print('{} / {} files are used!'.format(valid_files, num_files))

        a = 1


    def run(self):
        t = time.time()
        files = self.load_file_metadata()
        print('Took {} s'.format(time.time()-t))

        data = self.import_guitar_pro_files(files)


