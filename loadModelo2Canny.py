import cv2
import os

from time               import time
from keras.models       import load_model
from sklearn.metrics    import classification_report

import numpy as np

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
                img_blur = cv2.GaussianBlur(resized_arr, (5, 5), 1)
                img_sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
                img_sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
                img_sobelxy = cv2.addWeighted(cv2.convertScaleAbs(img_sobelx), 0.5, cv2.convertScaleAbs(img_sobely), 0.5, 0)
                edges = cv2.Canny(img_sobelxy,100,200)

                data.append([edges, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)

data = get_data('/home/rech/Documents/Datamining/trabalho/CrackClassification')

x = []
y = []

for feature, label in data:
  x.append(feature)
  y.append(label)

x = np.array(x)/255

x.reshape(-1, img_size, img_size, 1)
y = np.array(y)

model = load_model('/home/rech/Documents/Datamining/trabalho/Coddigos/modelo2Canny.h5')
model.summary()

t0 = time()
predictions = model.predict(x, use_multiprocessing=True)
t1 = time()

pred_aux = []

for i in range(predictions.shape[0]):
    if predictions[i , 0] > predictions[i , 1]:
        pred_aux.append(0)
    else:
        pred_aux.append(1)

from sklearn import metrics
cm = metrics.confusion_matrix(y, pred_aux)
print()
print(cm)

print(classification_report(y, pred_aux, target_names = ['negativo (Class 0)','positivo (Class 1)']))
print ('function vers1 takes %f segundos' %(t1-t0))