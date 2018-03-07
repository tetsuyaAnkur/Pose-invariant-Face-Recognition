import os
import datetime
import cv2
import numpy as np
import pyssim.ssim.ssimlib as pyssim
from skimage.measure import compare_ssim as ssim
from sklearn.cluster import SpectralClustering, AffinityPropagation
from sklearn import metrics,decomposition

# Constant definitions
SIM_IMAGE_SIZE = (640, 480)
IMAGES_PER_CLUSTER = 5

#Returns the normalized similarity value (from 0.0 to 1.0) for the provided pair of images.

def get_image_similarity(img1, img2):
    	# Converting to grayscale and resizing
	i1 = cv2.resize(cv2.imread(img1, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)
	i2 = cv2.resize(cv2.imread(img2, cv2.IMREAD_GRAYSCALE), SIM_IMAGE_SIZE)

	similarity = 0.0

        # Default SSIM implementation of Scikit-Image
	similarity = ssim(i1, i2)
	#print("similarity by using ssim",similarity)

	return similarity

# Fetches all images from the provided directory and calculates the similarity value per image pair.
def build_similarity_matrix(dir_name):
	images = os.listdir(dir_name)
	num_images = len(images)
	sm = np.zeros(shape=(num_images, num_images), dtype=np.float64)
	#print(sm.size)
	np.fill_diagonal(sm, 1.0)

	print("Building the similarity matrix using SSIM algorithm for %d images" %
          (num_images))
	start_total = datetime.datetime.now()

    	# Traversing the upper triangle only - transposed matrix will be used later for filling the empty cells.
	k = 0
	print("sm.shape[0] here : ",sm.shape[0],"  ",sm.shape[1],"\n")
	for i in range(sm.shape[0]):
		for j in range(sm.shape[1]):
			j = j + k
			if i != j and j < sm.shape[1]:
				sm[i][j] = get_image_similarity('%s/%s' % (dir_name, images[i]),
                                                '%s/%s' % (dir_name, images[j]))
		k += 1

    	# Adding the transposed matrix and subtracting the diagonal to obtain
    	# the symmetric similarity matrix
	sm = sm + sm.T - np.diag(sm.diagonal())

	end_total = datetime.datetime.now()
	print("Done - total calculation time: %d seconds" % (end_total - start_total).total_seconds())
	return sm

# Executes spectral clustering algorithm for similarity-based clustering
def do_cluster(dir_name):
	matrix = build_similarity_matrix(dir_name)
	print("printing matrix",matrix,"\n\n")

	sc = SpectralClustering(n_clusters=int(matrix.shape[0]/IMAGES_PER_CLUSTER),
                            affinity='precomputed').fit(matrix)
	print("printing special cluster matrix",sc,"\n\n")	

	return sc.labels_
