from gen_train_data import *
from numpy import *
import scipy.cluster.vq as vq
import sys
from sklearn.cluster import KMeans
import os
import random
import cv2
import numpy as np
import pickle

K_THRESH = 1

class siftFeatures(object):

    train_patches = []
    siftfeatures = None
    siftclusters = None
    negpatches = None
    trainhistograms = None
    classlabels = []
    nclusters = None
    patchdict = {}

    # Clustering
    codebook = None
    # initialize the training data

    def __init__(self):
        gtp = GenTrainPointsSIFT()
        self.train_patches = gtp.trainboxpatches
        self.negpatches = gtp.negpatches
        self.gen_sift_features()
        self.gen_sift_clusters()
        self.gen_histograms()
        self.gen_data_for_classifier()
        
        # 
        


    def display_random_path(self):
        r = random.randint(0, len(self.train_patches))
        random_patch = self.train_patches[r]
        cv2.imshow('sample',random_patch)
        cv2.waitKey(0)

    def gen_sift_features(self):
        """
        Generates sift features for the points
        """

        # if not os.path.isfile('sift'):
        #     raise 'Sift binary not in path'

        # if self.train_patches == []:
        #     raise 'Training patches is empty'

        # Generate sift featurs

        # Save patches to image


        siftforimages = {}
        i = 0
        classlabels = []
        for p in self.train_patches:
            #cv2.imwrite(str(i) +'.jpg',p)
            self.patchdict[str(i) +'.jpg'] = p
            classlabels.append(1)
            i+=1

            # Write negative patches
            j = i
        for p in self.negpatches:
            classlabels.append(-1)
            #cv2.imwrite(str(j) +'.jpg',p)
            self.patchdict[str(j) +'.jpg'] = p
            j+=1

        self.classlabels = classlabels

        for i in range(len(self.train_patches) ):
            sift = cv2.xfeatures2d.SIFT_create()    
            #siftforimages[str(i)+'.jpg'] = sift.detect(cv2.imread(str(i) + '.jpg'))
            #siftforimages[str(i) + '.jpg'] = sift.detectAndCompute(cv2.imread(str(i) +'.jpg',cv2.IMREAD_GRAYSCALE), None)
            
            try:
                siftforimages[str(i)+'.jpg'] = sift.detectAndCompute(self.patchdict[str(i) + '.jpg'], None)
            except cv2.error:
                print 'The incorrect patch size was' + str(np.shape(self.patchdict[str(i) + '.jpg']))
                cv2.imshow('patch',self.patchdict[str(i) + '.jpg'])
                cv2.waitKey(0)
        self.siftfeatures = siftforimages
        
            #print np.shape(siftforimages['1.jpg'][1])

        # Computing sift features for negative examples

        for k in range(len(self.train_patches ),j ):
            print 'processing image ' + str(k) +' for sift'

            sift = cv2.xfeatures2d.SIFT_create()
            #siftforimages[str(k) + '.jpg'] = sift.detectAndCompute(cv2.imread(str(k) +'.jpg', cv2.IMREAD_GRAYSCALE), None)
            siftforimages[str(k)+'.jpg'] = sift.detectAndCompute(self.patchdict[str(k) + '.jpg'], None)
        self.siftfeatures.update(siftforimages)


    def gen_sift_clusters(self):
        """
        Generates Kmeans clusters
        based on keypoints
        """

        # Get keypoints matrix from siftfeaturs
        if self.siftfeatures == None:
            raise 'Sift features is null'

        fullmatrix = []
        
        # for each image, get the sift features
        for i in range(len(self.siftfeatures.keys()) ):
            kpointmat = self.siftfeatures[str(i)+'.jpg'][1]
            kpointmat = np.matrix(kpointmat)
            
            if np.shape(kpointmat)[1] == 1:
                print 'No interesting points found for ' + str(i)+'.jpg'
                print ' Deleting that entry from sift'
                del self.siftfeatures[str(i)+'.jpg']
                continue
            print np.shape(kpointmat)
            if fullmatrix == []:
                fullmatrix = kpointmat
                fullmatrix = np.matrix(fullmatrix)
            else:
                fullmatrix = np.vstack((fullmatrix, kpointmat))

        # Also including negative examples

        # full matrix contains the fully stacked keypoints matrix of size n x 128
                # where n could be any integer depending on the number of keypoints
                # each keypoint is a 128 diamensional vector        

        # Perform clustering on the full matrix
                # Clustering is supposed to generate something called codebook
                nfeatures = np.shape(fullmatrix)[0] #

        nclusters = int(sqrt(nfeatures))
        codebook, distortion = vq.kmeans(fullmatrix,
                                                 nclusters,
                                             thresh=K_THRESH)

        self.codebook = codebook
        self.nclusters = nclusters


    def gen_histograms(self):
        """
        Generates histograms 
        """

        allhistograms = {}
        #for i in range(len(self.train_patches) + len(self.negpatches) ):
        #for i in range(len(self.siftfeatures)):
        for k in self.siftfeatures.keys():
            #descriptors = self.siftfeatures[str(i) + '.jpg'][1]
            descriptors = self.siftfeatures[k][1]
            code, dist = vq.vq(descriptors, self.codebook)
            histogram_of_words, bin_edges = histogram(code,
                                                      bins=range(self.codebook.shape[0] + 1),
                                                      normed=True)
            allhistograms[k] = histogram_of_words

        self.trainhistograms = allhistograms

    def gen_data_for_classifier(self):

        # Some     

        data_rows = zeros(self.nclusters + 1)
        #for i in range(len(self.train_patches) + len(self.negpatches)):
        i = 0
        for k in self.siftfeatures.keys():
            #histogram = self.trainhistograms[str(i) + '.jpg']
            histogram = self.trainhistograms[k]
            if histogram.shape[0] != self.nclusters:
                nclusters = histogram.shape[0]
                data_rows = zeros(nclusters + 1)
                print 'nclusters ahve been reduced to ' + str(nclusters)

            data_row = hstack((self.classlabels[i], histogram))
            data_rows = vstack((data_rows, data_row))
            i += 1
        data_rows = data_rows[1:]
        self.classifier_train_data = data_rows
        np.savetxt('traindata.csv',self.classifier_train_data, delimiter = ',')
        


if __name__ == '__main__':
    sf = siftFeatures()
    #sf.display_random_path()
    #sf.gen_sift_features()
    #sf.gen_sift_clusters()
    #sf.gen_histograms()
    #sf.gen_data_for_classifier()
    pickle.dump(sf.codebook, open('codebook.pickle','w'))