import os
from random import shuffle
import numpy as np
import wave
import tflearn

CHUNK = 4096
batch_size = 10

def dense_to_one_hot(labels_dense, num_classes=10):
	return np.eye(num_classes)[labels_dense]

def load_wav_file(name):
	f = wave.open(name, "rb")

	chunk = []
	data0 = f.readframes(CHUNK)
	while data0:

		data = np.fromstring(data0, dtype='uint8')
		data = (data + 128) / 255.
		chunk.extend(data)
		data0 = f.readframes(CHUNK)

	chunk = chunk[0:CHUNK * 2]
	chunk.extend(np.zeros(CHUNK * 2 - len(chunk)))
	return chunk

def create_batch(path):

	batch_waves = []
	labels = []
	files = os.listdir(path)

	while True:
		shuffle(files)

		for wav in files:
			labels.append(dense_to_one_hot(int(float(wav[0]))))

			chunk = load_wav_file(path+"/"+wav)
			batch_waves.append(chunk)

			if len(batch_waves) >= 10000:
				yield batch_waves, labels
				batch_waves = []
				labels = []

batch = create_batch("data_numbers")
X, Y = next(batch)

number_classes=10 # Digits
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192])
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(X, Y,n_epoch=3,show_metric=True,snapshot_step=100)

demo = load_wav_file("data_numbers/5_Vicki_260.wav")
result = model.predict([demo])
result = np.argmax(result)
print("predicted digit for %s : result = %d "%("5_Vicki_260.wav",result))
