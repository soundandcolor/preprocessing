import torch
#from handcrafted import HandcraftedKeyModel
#from data import get_midi_data, MidiLoader

#batch_size = 1
#handcrafted = HandcraftedKeyModel().cuda()
#midi_data = get_midi_data(handcrafted)
#midi_loader = MidiLoader(midi_data, batch_size=batch_size, shuffle=False)

avg_labels = torch.FloatTensor(7, 12).cuda()
avg_labels.zero_()
modes = [
  "Ionian",
  "Dorian",
  "Phrygian",
  "Lydian",
  "Mixolydian",
  "Aeolian",
  "Locrian"
]

notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

#for i, (data, labels) in enumerate(midi_loader):
for i in range(30000):
    data = torch.load('./preprocessed/{}.data'.format(i)).cuda()
    labels = torch.load('./preprocessed/{}.labels'.format(i)).cuda()
#    if i > 100:
#        break
    avg_labels += labels.mean(0).mean(1)
    if i % 100 == 99:
        print("made it to song: {}".format(i + 1))
    #if i % 1000 == 999:
    #    plt.bar(np.arange(len(modes)), avg_labels.sum(1).view(len(modes)).numpy() / (i + 1.))
    #    plt.xlabel('Mode')
    #    plt.ylabel('Probability')
    #    plt.title('Mode Probabilities')
    #    plt.xticks(np.arange(len(modes)), modes)
    #    plt.show()
    #    plt.bar(np.arange(len(notes)), avg_labels.sum(0).view(len(notes)).numpy() / (i + 1.))
    #    plt.xlabel('Tonic')
    #    plt.ylabel('Probability')
    #    plt.title('Tonic Probabilities')
    #    plt.xticks(np.arange(len(notes)), notes)
    #    plt.show()

#avg_labels /= len(midi_data)
avg_labels /= 30000
print(avg_labels)
