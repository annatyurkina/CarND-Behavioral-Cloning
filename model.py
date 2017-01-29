import os
import csv
import json
import argparse
import cv2
import matplotlib.image as mpimg
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda, Activation
from keras.layers.core import Lambda

BATCH_SIZE = 128
NUMBER_EPOCHS = 2

def load_image_paths_and_steering():
	image_paths, steering = [], []
	first_row = True

	with open('data/driving_log.csv', 'r') as csvfile:
		logreader = csv.reader(csvfile, delimiter=',')
		for row in logreader:
			if(first_row):
				first_row = False
				continue
			steering_lable = float(row[3].strip())
			augmented_steering_lable = 0.0
			image_paths.append(row[0].strip())
			steering.append(steering_lable) 
			if(abs(steering_lable) > .1):
				image_paths.append(row[1].strip())
				steering.append(steering_lable + .25) 
				image_paths.append(row[2].strip())
				steering.append(steering_lable - .25) 
			# if(steering_lable > .1):
			# 	augmented_steering_lable = steering_lable + randint(1,10)/100
			# if(steering_lable < -.1):
			# 	augmented_steering_lable = steering_lable - randint(1,10)/100
			# if(abs(augmented_steering_lable) > .1):
			# 	image_paths.append(row[0].strip())
			# 	steering.append(augmented_steering_lable) 
			# 	image_paths.append(row[1].strip())
			# 	steering.append(augmented_steering_lable + .25) 
			# 	image_paths.append(row[2].strip())
			# 	steering.append(augmented_steering_lable - .25) 

	return (image_paths, steering)

def train_validation_split(X, y):
	X, y = shuffle(X, y)
	X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)

	return (X_train, X_validation, y_train, y_validation)

def batch_generator(X, y):
	first_time = True
	example_count = len(X)
	while True:
		for offset in range(0, example_count, BATCH_SIZE):
			print(offset)
			print(min(offset+BATCH_SIZE, example_count))
			batch_x, batch_y = X[offset:min(offset+BATCH_SIZE, example_count)], y[offset:min(offset+BATCH_SIZE, example_count)]        
			images, labels = [], []
			for i in range(0, len(batch_x)):
				image = mpimg.imread('data/' + batch_x[i])
				# if(randint(0,1) > 0):
				# 	image = cv2.flip(image, 1)
				# 	batch_y[i] = -batch_y[i]
				images.append(image)
				labels.append(batch_y[i])

			X_batch = np.array(images)
			if(first_time):
				first_time = True
				print(X_batch.shape)
			#X_batch = np.divide(np.add(X_batch, -128),128)
			y_batch = np.array(labels)
			yield(X_batch, y_batch)


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
	model.add(Activation('sigmoid'))
	model.add(Dense(1))
	model.compile(optimizer="adam", loss="mse")
	return model


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Behavior Cloning')
	parser.add_argument('--batch', type=int, default=128, help='Batch Size')
	parser.add_argument('--epoch', type=int, default=2, help='Number of Epochs')
	args = parser.parse_args()
	BATCH_SIZE = args.batch
	NUMBER_EPOCHS = args.epoch

	image_paths, steering = load_image_paths_and_steering()
	X_train, X_validation, y_train, y_validation = train_validation_split(np.array(image_paths), np.array(steering))
	print(len(X_train))

	model = get_model()
	model.fit_generator(
		batch_generator(X_train, y_train),
		samples_per_epoch=len(X_train),
		nb_epoch=NUMBER_EPOCHS,
		validation_data=batch_generator(X_validation, y_validation),
		nb_val_samples=len(X_validation)
		)
	print("Saving model weights and configuration file.")

	if not os.path.exists("output"):
		os.makedirs("output")

	model.save_weights("output/model.h5", True)
	with open('output/model.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)



