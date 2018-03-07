import os
import datetime
import cv2
import numpy as np
import pyssim.ssim.ssimlib as pyssim
from skimage.measure import compare_ssim as ssim
from sklearn.cluster import SpectralClustering, AffinityPropagation
from sklearn import metrics

face_dir = 'orl_full'
count=0
for file in os.listdir(face_dir):
	print "filename : ",file,"\n"
	
	for i in os.listdir(face_dir+'/'+ file):
		if int(i.split('.')[0])<=5:
			os.system("cp "+face_dir+'/'+file+"/"+i+" 5each_full/"+file+"_"+i)
		
	count=count+1

	if count==0:
		exit(0)
