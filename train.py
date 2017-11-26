import os
import sys
import librosa
import tflearn
import tensorflow as tf
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms

# now put all of the mfccs into an array
#os.chdir('/home/cc/Data/' + path)
os.chdir('DATA/')
audio_files = os.listdir(os.getcwd())
mfccs = []
classes = []
for f in audio_files:
#    y, sr = librosa.load(f)
#    classes.append(f.split("_")[1])
#    mfccs.append(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

net = tflearn.input_data(shape=[None, 13, 22]) #Two wave chunks
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 15, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(mfccs, classes, n_epoch=500, show_metric=True, snapshot_step=100)


def mfcc_batch_generator(batch_size=10, source=Source.DIGIT_WAVES, target=Target.digits):
	if target == Target.speaker: speakers = get_speakers()
	batch_features = []
	labels = []
	files = os.listdir(path)
	while True:
		print("loaded batch of %d files" % len(files))
		shuffle(files)
		for file in files:
			if not file.endswith(".wav"): continue
			wave, sr = librosa.load(path+file, mono=True)
			mfcc = librosa.feature.mfcc(wave, sr)
			if target==Target.speaker: label=one_hot_from_item(speaker(file), speakers)
			elif target==Target.digits:  label=dense_to_one_hot(int(file[0]),10)
			elif target==Target.first_letter:  label=dense_to_one_hot((ord(file[0]) - 48) % 32,32)
			elif target == Target.hotword: label = one_hot_word(file, pad_to=max_word_length)  #
			elif target == Target.word: label=string_to_int_word(file, pad_to=max_word_length)
				# label = file  # sparse_labels(file, pad_to=20)  # max_output_length
			else: raise Exception("todo : labels for Target!")
			labels.append(label)
			# print(np.array(mfcc).shape)
			mfcc=np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
			batch_features.append(np.array(mfcc))
			if len(batch_features) >= batch_size:
				# if target == Target.word:  labels = sparse_labels(labels)
				# labels=np.array(labels)
				# print(np.array(batch_features).shape)
				# yield np.array(batch_features), labels
				# print(np.array(labels).shape) # why (64,) instead of (64, 15, 32)? OK IFF dim_1==const (20)
				yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
				batch_features = []  # Reset for next batch
				labels = []