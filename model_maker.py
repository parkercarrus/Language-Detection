from telnetlib import X3PAD
from numpy import array, loadtxt
from keras.models import Sequential
from keras.layers import Dense
import csv
from csv import writer

dataset = loadtxt('bigdata.csv', delimiter=',')

#intializing which parts of data are input and output
X = dataset[:,0:10]
Y = dataset [:,10]

model = Sequential() #sequential shape and general flow network

#creating the layers
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#initialize loss function and weights/biases optimization
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the model (toggle verbose for live results printing)
history = model.fit(X, Y, epochs=12, batch_size=64, verbose = True)
_, accuracy = model.evaluate(X,Y, verbose = False)

model.save('ann_updated')




        

