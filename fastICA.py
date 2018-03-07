import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import FastICA
from sklearn.svm import SVC

import matplotlib.pyplot as plt

image=os.listdir("5each_full_align")
data=[]
colormap=[]
for i in image:
	print(i)
	data.append(cv2.imread("5each_full_align/"+i,0))


a,b,c=np.array(data).shape
data=np.array(data).reshape(a,b*c)
ica = FastICA(20).fit(data)
components = ica.transform(data)
projected = ica.inverse_transform(components)
num,poo=projected.shape

for i in range(num):
	a=projected[i].reshape(b,c)
	cv2.imwrite("5each_full_blurred2/"+image[i],a)
	
	


