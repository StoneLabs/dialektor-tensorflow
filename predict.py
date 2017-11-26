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
width = 20  # mfcc features
height = 432
classes = 2  # digits

print("-> Loading")
print("Creating network... ")
net = tflearn.input_data([None, width, height])
# net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training
print("Creating model... ")
model = tflearn.DNN(net, tensorboard_verbose=0)

print("Loading weights... [tflearn.lstm.model]")
model.load("tflearn.lstm.model")


os.chdir('..')
print("-> Loading file")
if len(sys.argv) < 2:
    print("Please insert wav file as first argument!")
    sys.exit()
file = sys.argv[1]
if not file.endswith(".wav"):
    print("Please insert wav file!")
    sys.exit()
wave, sr = librosa.load(file, mono=True)
mfcc = librosa.feature.mfcc(wave, sr)
if (len(mfcc[0]) != height):
    print("Loading " + str(file) + " with " + str(np.array(mfcc).shape) +
            " reshaping " + str(np.shape(mfcc)) + " to ", end = '') # Debug
    mfcc.resize(width, height)
    print(str(np.shape(mfcc)))
else:
    print("Loading " + str(file) + " with " + str(np.array(mfcc).shape)) # Debug

print("-> Predicting data")
print(model.predict([mfcc]))