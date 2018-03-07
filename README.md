# Pose-invariant-Face-Recognition
Face Recognition using Structural Similarity(SSIM) and Spectral Clustering.

# Dependencies 
* dlib==19.9
* pyssim==0.4

# Installing Dependencies
* Installation instructions for dlib can be found at https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf
* Download pyssim from https://pypi.python.org/pypi/pyssim

# Datasets
* ORL (Olivetti Research Laboratory) dataset - http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
* Caltech_10K_WebFaces dataset - http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/

# Steps used for Face Recognition
* Face Alignment - I have used dlib's Face Aligner to align the facial images. Face Alignment is crucial because it increases  accuracy of localizing and normalizing faces. The code for the same is present in align_faces.py.

* Dimensionality Reduction - Since the bottleneck in the entire task of face recognition is the preprocessing step, so dimensionality reduction is very essential. Dimensionality reduction removes the redundant and unwanted pixels and speeds up computation and in turn the whole process of training. I have used Independent Component Analysis(ICA) for dimensionality reduction. The code for the same is present in fastICA.py.

* Similarity Measure - This is a measure of how similar two images are. It can take values between 0 and 1. This similarity measure is used by spectral clustering. I have used Structural Similarity(SSIM) as the similarity measure. The code for the same is present in main.py.

* Spectral Clustering - Spectral Clustering uses the similarity measure of every pair of images in the dataset(got from SSIM) to cluster the facial images. The code for the same is present in main.py.

# Results
After the clustering step all the facial_images present in one cluster are classified to be of the same person. 
