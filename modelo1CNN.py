import cv2
import os

from keras.models               import Sequential
from keras.layers               import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from sklearn.metrics            import classification_report,confusion_matrix
from sklearn.model_selection    import train_test_split

import tensorflow   as tf
import numpy        as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

labels = ['negativo', 'positivo']
img_size = 128

def get_data(data_dir):
    data = []
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)

        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1]
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)

data = get_data('/home/jerson/Downloads/Datasets/dataZuado')

x = []
y = []

for feature, label in data:
  x.append(feature)
  y.append(label)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

X_train = np.array(X_train)/255
X_test = np.array(X_test)/255

X_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

X_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(img_size,img_size,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = 'rmsprop'
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

history = model.fit(X_train,y_train,epochs = 120 , validation_data = (X_test, y_test))

model.save("/home/jerson/Downloads/Coddigos/modelo1CNN.h5")

from time import time


t0 = time()
predictions = model.predict(X_test, use_multiprocessing=True)
t1 = time()

pred_aux = []

for i in range(predictions.shape[0]):
    if predictions[i , 0] > predictions[i , 1]:
        pred_aux.append(0)
    else:
        pred_aux.append(1)

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, pred_aux)
print()
print(cm)

print(classification_report(y_test, pred_aux, target_names = ['negativo (Class 0)','positivo (Class 1)']))
print ('function vers1 takes %f segundos' %(t1-t0))
