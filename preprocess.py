from data import LabeledLoader, PianoTrackLoader
from handcrafted import HandcraftedKeyModel
import torch

handcrafted = HandcraftedKeyModel().cuda()
unlabeled_loader = PianoTrackLoader('./dataset')
midi_loader = LabeledLoader(unlabeled_loader, handcrafted)

for i, (data, labels) in enumerate(midi_loader):
    print i
    torch.save(data, './preprocessed/{}.data'.format(i))
    torch.save(labels, './preprocessed/{}.labels'.format(i))
