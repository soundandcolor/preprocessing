import torch
import torch.utils.data as data
import numpy as np
import midi
import os
from itertools import islice
from random import shuffle

DATA_DIR = './dataset/'

class PreprocessedLoader(object):
    def __init__(self, directory, size=51153):
        self.directory = directory
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in xrange(self.size):
            data = torch.load(os.path.join(self.directory, '{}.data'.format(i)))
            labels = torch.load(os.path.join(self.directory, '{}.labels'.format(i)))
            yield data, labels

class LabeledLoader(object):
    def __init__(self, unlabeled, labeler):
        self.unlabeled = unlabeled
        self.labeler = labeler

    def __len__(self):
        return len(self.unlabeled)

    def __iter__(self):
        for data in self.unlabeled:
            labels = self.labeler(data).data
            yield data, labels

# dataset + loader for lazy loading piano tracks
# batch size is 1, doesn't support shuffle
# undefined behavior if size > the total # of piano tracks
# in the 130k song dataset there are 51154 piano tracks
class PianoTrackLoader(object):
    def __init__(self, midi_root, size=51154):
        self.midi_root = midi_root
        self.size = size

    def __len__(self):
        return self.size

    def midi_files(self):
        for root, dirs, files in os.walk(self.midi_root):
            for filename in files:
                if filename.split('.')[-1] == 'mid':
                    try:
                        yield midi.read_midifile(os.path.join(root, filename))
                    except:
                        pass

    def piano_tracks(self):
        for pattern in self.midi_files():
            for track in pattern:
                for event in track:
                    if isinstance(event, midi.TrackNameEvent):
                        if 'piano' in event.text.lower():
                            yield track, pattern.resolution

    def piano_tracks_subset(self, size):
        for i, (track, res) in enumerate(self.piano_tracks()):
            if i >= size:
                break
            yield track, res

    def vectorize(self, track):
    	notes = {0: 128, 1:128}
    	running_tick = 2
    	for event in track:
    	    if isinstance(event, midi.NoteOnEvent):
    	        notes[event.tick + running_tick] = event.data[0]
    	    running_tick += event.tick
    	max_tick = max(notes.keys())
    	ticks = [notes[tick] if (tick in notes) else 128 for tick in xrange(max_tick)]
    	return ticks

    def resample(self, old_res, new_res, ticks):
	window_size = int(old_res / new_res)
        if window_size < 1:
            return ticks
        output = [128]
        for i in xrange(int(len(ticks) / window_size)):
            output += [min(ticks[i * window_size:(i+1) * window_size])]
        return output

    def __iter__(self):
        for track, res in self.piano_tracks_subset(self.size):
            ticks_tensor = torch.LongTensor(self.resample(res, 30, self.vectorize(track)))
            ticks_tensor.unsqueeze_(0)
            yield ticks_tensor

# flattens all the tracks in a midi file
# arbitarily chooses one of the notes when multiple notes are played at the same time
# returns a vector of size = # ticks of the midi pitch values for the noteoneevents in a midi file

class MidiFiles(data.Dataset):
    def __init__(self, midi_root, labeler, input_transform=None):
        super(MidiFiles, self).__init__()
        self.labeler = labeler
        self.midi_files = []
        for root, dirs, files in os.walk(midi_root):
            for filename in files:
                if filename.split('.')[-1] == 'mid':
                    self.midi_files += [os.path.join(root, filename)]

    def __getitem__(self, index):
        try:
            pattern = midi.read_midifile(self.midi_files[index])
            notes = {}
            for track in pattern:
                for event in track:
                    if isinstance(event, midi.NoteOnEvent):
                        notes[event.tick] = event.data[0]
            max_tick = max(notes.keys())
            ticks = [notes[tick] if (tick in notes) else 128 for tick in xrange(max_tick)]
            if len(ticks) == 0:
                ticks = [128, 128]
        except:
            ticks = [128, 128]
        ticks_tensor = torch.LongTensor(ticks)
        ticks_tensor.unsqueeze_(0)
        label_tensor = self.labeler(ticks_tensor).data
        return ticks_tensor[0], label_tensor[0]

    def __len__(self):
        return len(self.midi_files)

class MidiLoader(object):
    def __init__(self, data, batch_size=1, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        order = range(len(self.data))
        if self.shuffle:
            shuffle(order)
        i = 0
        while i < len(self.data):
            data = []
            target = []
            for s in xrange(i, i + self.batch_size):
                if s < len(self.data):
                    data_s, target_s = self.data[s]
                    data += [data_s]
                    target += [target_s]
            data = torch.cat([torch.unsqueeze(data[s], 0) for s in xrange(len(data))])
            target = torch.cat([torch.unsqueeze(target[s], 0) for s in xrange(len(target))])
            yield data, target
            i += self.batch_size

def get_midi_data(labeler):
    return MidiFiles(DATA_DIR, labeler)
