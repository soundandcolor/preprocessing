import torch
import torch.utils.data as data
import numpy as np
import midi
import os
from itertools import islice
from random import shuffle

DATA_DIR = './dataset/'

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
        pattern = midi.read_midifile(self.midi_files[index])
        notes = {}
        for track in pattern:
            for event in track:
                if isinstance(event, midi.NoteOnEvent):
                    notes[event.tick] = event.data[0]
        max_tick = max(notes.keys())
        ticks = [notes[tick] if (tick in notes) else -10000 for tick in xrange(max_tick)]
        ticks_tensor = torch.LongTensor(1, len(ticks))
        ticks_tensor[0] = torch.Tensor(ticks)
        label_tensor = self.labeler(ticks_tensor)
        return ticks_tensor[0], label_tensor[0]

    def __len__(self):
        return len(self.midi_files)

class MidiLoader(object):
    def __init__(self, data, batch_size=1, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

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
            data = torch.cat(\
                [torch.unsqueeze(data[s], 0)\
                 for s in xrange(i, i + self.batch_size)\
                 if s < len(self.data)])
            target = torch.cat(\
                [torch.unsqueeze(target[s], 0)\
                 for s in xrange(i, i + self.batch_size)\
                 if s < len(self.data)])
            yield data, target
            i += self.batch_size

def get_midi_data(labeler):
    return MidiFiles(DATA_DIR, labeler)
