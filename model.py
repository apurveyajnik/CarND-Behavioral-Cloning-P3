
from keras.models import Sequential
from keras.layers import Cropping2D,Dense,Activation,Flatten,Dropout,MaxPooling2D,Lambda,Convolution2D
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import utils
import cv2
import matplotlib.pyplot as plt
import os.path
data = []

with open("./driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		data.append(line)

train_data , val_data = train_test_split(data,test_size=0.2)

del data
#reduce memory usage



def generator(data,batch_size=32):
	num = len(data)
	while 1:
		utils.shuffle(data)
		for offset in range(0,num,batch_size):
			batch_data = data[offset:offset+batch_size]
			imgs = []
			angs = []
			for batch_sample in batch_data:
				folder ="./IMG/"
				ang=[]
				ang.append(float(batch_sample[3]))
				corr = 0.2
				ang.append(ang[0] + corr)
				ang.append(ang[0] - corr)
				# for using multiple cameras	
				path = []
				path.append(folder + batch_sample[0].split('/')[-1])
				path.append(folder + batch_sample[1].split('/')[-1])	
				path.append(folder + batch_sample[2].split('/')[-1])
				# using 3 cameras
				for i in range(3):
					if os.path.exists(path[0]) and os.path.exists(path[1]) and os.path.exists(path[2]):	
						# images where car is off the road removed from data	
						imgs.append(cv2.imread(path[i]))
						angs.append(ang[i])
						imgs.append(cv2.flip(cv2.imread(path[i]),1))
						angs.append(ang[i]*-1.0)
						# augmenting images increase data and to generalize	
			if len(imgs)==0 or len(angs)==0:
				continue
			X = np.array(imgs)
			y = np.array(angs)

			yield utils.shuffle(X,y)

train_gen = generator(train_data)
val_gen = generator(val_data)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5 ,input_shape=(160,320,3)))
#normalising images
model.add(Cropping2D(cropping=((70,25),(0,0))))
# cropping
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('relu'))

model.add(Dropout(0.2))
# Dropout used for reducing overfitting
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
history_obj = model.fit_generator(train_gen,samples_per_epoch=len(train_data),validation_data=val_gen,nb_val_samples=len(val_data),nb_epoch=5)

print(history_obj.history.keys())

plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('mean sqr error loss')
plt.ylabel('mse loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc='upper right')
plt.show()

model.save('model.h5')
print("model saved")
