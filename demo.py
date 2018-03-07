import os
import numpy as np
import cv2
import main as tv
from matplotlib import pyplot as plt

DIR_NAME = '5each_full_blurred'

if __name__ == "__main__":
	c= tv.do_cluster(DIR_NAME)
	num_clusters = len(set(c))
	images = os.listdir(DIR_NAME)

	clusters = []
	for n in range(num_clusters):
		print("\n --- Images from cluster #%d ---" % n)
		imgs = []
		for i in np.argwhere(c == n):
			if i != -1:
				inp = images[int(i)].split("_")
				inp2 = inp[0][1:]
				imgs.append(int(inp2))

		clusters.append(imgs)

	tp = 0
	tn = 0
	fp = 0
	fn = 0
	cluster2 = []
	for i in clusters:
		d = []
		for j in range(1,41):
			count = i.count(j)
			d.append([count,j])
		d.sort()
		d.reverse()
		cluster2.append(d)
		tp += d[0][0]
		for k in range(1,len(d)):
			fp += d[k][0]

	for i in range(1,41):
		for j in cluster2:
			if j[0][1]!=i:
				for k in j:
					if k[1]!=i:
						tn += k[0]
					else:
						fn += k[0]
			
	print("tp = ",tp)
	print("tn = ",tn)
	print("fp = ",fp)
	print("fn = ",fn)

	
	acc = (tp+tn)/(tp+tn+fp+fn)
	pre = tp/(tp+fp)
	rec = tp/(tp+fn)

	print("Printing the metric values")
	print("ACCURACY = ",acc)
	print("PRECISION = ",pre)
	print("RECALL = ",rec)
