import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np

# only works on batch size 1
class HandcraftedKeyModel(nn.Module):
    def mode_weights(self, weights, mode):
        # need to fill in the non-key notes with default value
        major_intervals = [2,2,1,2,2,2,1]
        ret = torch.ones(12) * 0.05
        offset = 0
        for i in xrange(self.num_modes):
            ret[offset] = 1. / weights[i]
            offset += major_intervals[(mode + i) % self.num_modes]
        return ret

    def mode_scale_weights(self, weights, mode, scale):
        mode_w = self.mode_weights(weights, mode)
        return torch.Tensor(np.concatenate((mode_w.numpy()[scale:], mode_w.numpy()[:scale])))

    def mode_scale_values(self, heats):
        flat_values = 10. / torch.mm(heats.unsqueeze(0), self.weights)
        _, sorted_ind = torch.sort(flat_values)
        flat_values.zero_()
        flat_values[0,sorted_ind[0,-1]] = .7
        flat_values[0,sorted_ind[0,-2]] = .2
        flat_values[0,sorted_ind[0,-3]] = .1
        return flat_values.view(self.num_modes, self.num_keys)

    def __init__(self, decay_rate=-0.0005, max_heat=5., weights=[1.5, 0.1, 0.6, 0.3, 0.8, 0.1, 0.2]):
        super(HandcraftedKeyModel, self).__init__()
        self.decay_rate = math.exp(decay_rate)
        self.max_heat = max_heat
        self.notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        self.num_modes = 7
        self.num_keys = 12
        weights = torch.Tensor(weights)
        self.weights = torch.Tensor(self.num_modes, self.num_keys, self.num_keys)
        for mode in xrange(self.num_modes):
            for key in xrange(self.num_keys):
                self.weights[mode, key] = self.mode_scale_weights(weights, mode, key)
        self.weights = self.weights.view(-1, self.num_keys).transpose(1, 0)

    def forward(self, x):
        x = x[0]
        output = Variable(torch.Tensor(1, len(x), self.num_modes, self.num_keys), requires_grad=False)
        key_heats = torch.zeros(12)
        for i in xrange(len(x)):
            key_heats = key_heats * self.decay_rate
            if x[i] > -10000:
                key_heats[x[i] % 12] += 1.
            key_heats[key_heats > self.max_heat] = self.max_heat
            values = self.mode_scale_values(key_heats)
            output[0, i] = values

        return output
