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

os.chdir('DATA/')

learning_rate = 0.0001
training_iters = 25  # steps
epoch_length = 100  # steps
batch_size = 10000

width = 20  # mfcc features
height = 432
classes = 2  # digits


def mfcc_batch_generator(batch_size = 10):
    audio_files = os.listdir(os.getcwd())
    batch_features = []
    labels = []
    counter = 0
    for file in audio_files:
        counter+=1
        if not file.endswith(".wav"): continue
        wave, sr = librosa.load(file, mono=True)
        mfcc = librosa.feature.mfcc(wave, sr)
        if (len(mfcc[0]) != height):
            print("Loading " + str(file) + " with " + str(np.array(mfcc).shape) +
                    " reshaping " + str(np.shape(mfcc)) + " to ", end = '') # Debug
            mfcc.resize(width, height)
            print(str(np.shape(mfcc)))
        else:
            print("Loading " + str(file) + " with " + str(np.array(mfcc).shape)) # Debug


        # Append class
        label = [0]*classes;
        label[int(file.split("_")[1])] = 1
        labels.append(label);

        # Magic?
        #mfcc=np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
        batch_features.append(np.array(mfcc))
        if len(batch_features) >= batch_size or counter == len(audio_files):
            # if target == Target.word:  labels = sparse_labels(labels)
            # labels=np.array(labels)
            # print(np.array(batch_features).shape)
            # yield np.array(batch_features), labels
            # print(np.array(labels).shape) # why (64,) instead of (64, 15, 32)? OK IFF dim_1==const (20)
            yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
            batch_features = []  # Reset for next batch
            labels = []

print("Loading data... ")
batch = word_batch = mfcc_batch_generator(batch_size)
X, Y = next(batch)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


print("Creating network... ")
net = tflearn.input_data([None, width, height])
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 256, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax', regularizer='L2')
net = tflearn.fully_connected(net, classes, activation='softmax', regularizer='L2')
net = tflearn.fully_connected(net, classes, activation='softmax', regularizer='L2')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training
print("Creating model... ")
model = tflearn.DNN(net, tensorboard_verbose=0)
#model.load("tflearn.lstm.model")
print("Training... ")
for i in range(0, training_iters):
    model.fit(X_train, y_train, n_epoch=epoch_length, validation_set=0.1, show_metric=True,
          batch_size=batch_size)
    try:
        input("Press Enter to continue... [Ctrl+C if done]")
    except KeyboardInterrupt as identifier:
        break

print("\n\nSaving model... [tflearn.lstm.model]")
model.save("tflearn.lstm.model")

#import pickle
#print("Pickling model... [export_model.dat]")
#with open("export_model.dat", "wb+") as handle:
#	pickle.dump(model, handle)

###
### Evaluating performance
###
from sklearn import metrics

p_train = model.predict(X_train)
p_test = model.predict(X_test)

# Advanced performance analzsis
print("\n\nTraining data:")
print("\n\nTraining data:", file=open("evaluation.txt", "a"))
print(metrics.classification_report(np.argmax(y_train, axis=1), np.argmax(p_train, axis=1), target_names=["SD", "HD"]))
print(metrics.classification_report(np.argmax(y_train, axis=1), np.argmax(p_train, axis=1), target_names=["SD", "HD"]), file=open("evaluation.txt", "a"))
print("\n\nTest data:")
print("\n\nTest data:", file=open("evaluation.txt", "a"))
print(metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(p_test, axis=1), target_names=["SD", "HD"]))
print(metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(p_test, axis=1), target_names=["SD", "HD"]), file=open("evaluation.txt", "a"))
