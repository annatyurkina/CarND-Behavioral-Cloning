import os
import csv
import json
import argparse
import cv2
import datetime
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda, Activation
from keras.layers.core import Lambda

BATCH_SIZE = 128
NUMBER_EPOCHS = 2
VERBOSE = False

def load_image_paths_and_steering():
	image_paths, steering = [], []
	image_paths_val, steering_val = [], []
	first_row = True

	with open('data/driving_log.csv', 'r') as csvfile:
		logreader = csv.reader(csvfile, delimiter=',')

		for row in logreader:
			# skip header row
			if(first_row):
				first_row = False
				continue

			steering_lable = float(row[3].strip())

			# only leave 1/6th of zero values
			if(steering_lable == 0.0 and randint(0, 6) > 0):
				continue

			# validation set is 1/3rd of actual non-augmented data	
			if(randint(0, 2) == 0):
				validation = True
			else:
				validation = False
			
			# do not augment data for validation, only adjust 
			# left/right angle shift
			if(validation):
				image_paths_val.append(row[0].strip())
				steering_val.append(steering_lable)
				image_paths_val.append(row[1].strip())
				steering_val.append(steering_lable + .25) 
				image_paths_val.append(row[2].strip())
				steering_val.append(steering_lable - .25) 
			else:
				image_paths.append(row[0].strip())
				steering.append(steering_lable)
				image_paths.append(row[1].strip())
				steering.append(steering_lable + .25) 
				image_paths.append(row[2].strip())
				steering.append(steering_lable - .25) 

				# add augmented data for training
				# to balance larger absolute angles 
				# and to emphasize sharper turns when closer to the line 
				if(steering_lable > .0):
					image_paths.append(row[0].strip())
					steering.append(steering_lable + .1) 
					image_paths.append(row[1].strip())
					steering.append(steering_lable + .35) 
					image_paths.append(row[0].strip())
					steering.append(steering_lable + .15) 
					image_paths.append(row[1].strip())
					steering.append(steering_lable + .30) 
				if(steering_lable < .0):
					image_paths.append(row[0].strip())
					steering.append(steering_lable  - .1) 
					image_paths.append(row[2].strip())
					steering.append(steering_lable - .35) 
					image_paths.append(row[0].strip())
					steering.append(steering_lable - .15) 
					image_paths.append(row[1].strip())
					steering.append(steering_lable - .30) 

	X_train, X_validation, y_train, y_validation = np.array(image_paths), np.array(image_paths_val), np.array(steering), np.array(steering_val)	
	print(len(y_validation))
	print(len(y_train))	

	X_train, y_train = shuffle(X_train, y_train)
	X_validation, y_validation = shuffle(X_validation, y_validation)

	# see train and validation sets histograms
	if(VERBOSE):
		a = np.array(y_train)
		plt.hist(a, bins=60)
		plt.show()
		X_validation, y_validation = shuffle(X_validation, y_validation)
		a = np.array(y_validation)
		plt.hist(a, bins=60)
		plt.show()

	return (X_train, X_validation, y_train, y_validation)

# not used in the current model
def train_validation_split(X, y):
	X, y = shuffle(X, y)
	X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)

	return (X_train, X_validation, y_train, y_validation)

def batch_generator(X, y, isvalidation = False):
	example_count = len(X)

	while True:
		X, y = shuffle(X, y)

		for offset in range(0, example_count, BATCH_SIZE):
			batch_x, batch_y = X[offset:min(offset+BATCH_SIZE, example_count)], y[offset:min(offset+BATCH_SIZE, example_count)]   

			images, labels = [], []
			for i in range(0, len(batch_x)):
				image = mpimg.imread('data/' + batch_x[i])
				images.append(image)
				labels.append(batch_y[i])

			X_batch = np.array(images)
			y_batch = np.array(labels)

			if(VERBOSE):
				print(X_batch.shape)

			yield(X_batch, y_batch)


# NVIDIA architacture with minor changes
def get_model():
	ch, row, col = 160, 320, 3  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
	model.add(Activation('relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, border_mode="same"))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, border_mode="same"))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dropout(.5))
	model.add(Dense(1))
	model.compile(optimizer="adam", loss="mse")
	return model


if __name__ == "__main__":

	# parsing command line parameters
	parser = argparse.ArgumentParser(description='Behavior Cloning')
	parser.add_argument('--batch', type=int, default=128, help='Batch Size')
	parser.add_argument('--epoch', type=int, default=2, help='Number of Epochs')
	parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
	args = parser.parse_args()
	BATCH_SIZE = args.batch
	NUMBER_EPOCHS = args.epoch
	VERBOSE = args.verbose

	X_train, X_validation, y_train, y_validation = load_image_paths_and_steering()

	print(len(X_train))	
	print(datetime.datetime.utcnow())

	model = get_model()
	model.fit_generator(
		batch_generator(X_train, y_train),
		samples_per_epoch=len(X_train),
		nb_epoch=NUMBER_EPOCHS,
		validation_data=batch_generator(X_validation, y_validation),
		nb_val_samples=len(X_validation)
		)

	print(datetime.datetime.utcnow())
	print("Saving model weights and configuration file.")

	if not os.path.exists("output"):
		os.makedirs("output")

	model.save_weights("output/model.h5", True)
	with open('output/model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)



