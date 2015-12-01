import ConfigParser as cp
import numpy as np
import os
import cv2
from skimage.util.shape import view_as_windows
from sklearn.feature_extraction import image
from gen_sift_features import *
from numpy import genfromtxt
from PIL import Image, ImageFont, ImageDraw
import re
import pickle

class GenSiftForTest(object):


    testimage = ''
    siftfeatures = None
    testhistograms = None
    codebook = None
    nclusters = None
    classifier_test_data = None
    classlabels = []
    testpatches = []
    patch = None
    
    # for storing patches
    patchdict = {}
    patchinddict = {}
    predictionfile = 'Predictions.csv'
    processedhists = [] # Histograms for which descriptor was not None
    def __init__(self):
        config = cp.RawConfigParser()
        config.read('config.cfg')
        self.testimage = config.get('init', 'testimage')
        
        # Get the codebook from the training phasse
       
        if not os.path.isfile('codebook.pickle'):
            print 'Codebook not found. Generating..\n'
            sf = siftFeatures()
            self.codebook = sf.codebook
            
        else:
            self.codebook = pickle.load(open('codebook.pickle'))
   
        self.nclusters = self.codebook.shape[0]
        
        # Generate sift featurs, histograms 
        self.gen_test_patches()
        self.gen_sift_features()
        self.gen_histograms()
        self.gen_test_data()
        #self.view_results2()
        #self.view_results_knn()
        
    def gen_patch_dict(self,image, stepSize, windowSize):
        
        patchdict = {}
        patchinddict = {}
        
        windows = []
        for y in xrange(0, image.shape[0], stepSize):
            for x in xrange(0, image.shape[1], stepSize):
                window = (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
                print window[2].shape
                
                if np.shape(window[2]) == windowSize:
                    windows.append(window)
        # Make the patch dict
        for i in range(len(windows)):
            patchdict[i] = np.array(windows[i][2])
            print windows[i][2]
            
            ind = (windows[i][0], windows[i][1])
            patchinddict[i] = ind
            
        self.patchdict = patchdict
        self.patchinddict = patchinddict
        return patchdict
            
        
        
    def sub2ind(self,array_shape, rows, cols):
        ind = rows*array_shape[1] + cols
        ind[ind < 0] = -1
        ind[ind >= array_shape[0]*array_shape[1]] = -1
        return ind

    def ind2sub(self, array_shape, ind):
        #ind[ind < 0] = -1
        #ind[ind >= array_shape[0]*array_shape[1]] = -1
        rows = int(ind) / array_shape[1]
        cols = int(ind) % array_shape[1]
        return [rows, cols]
    
    def gen_test_patches(self):
        
        # read the teest image
        print self.testimage
        im = cv2.imread(self.testimage)
        #print np.shape(im)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        #Generate patches from test iage 
        #patch =  view_as_windows(im, (64,64,3), 65)
        #patch =  view_as_windows(im, (100,100))
        #patch =  view_as_windows(gray, (64,64), 65)
        #patch = image.extract_patches(im, (100,100))
        patchdict = self.gen_patch_dict(gray,  65, (64,64))
        patch = patchdict.values()
        # fdimlen = np.shape(patch)[0]
#         sdimlen = np.shape(patch)[1]
#         allpatches = []
        
        # for i in range(fdimlen):
        #     for j in range(sdimlen):
        #         #p = patch[i][j][0]
        #         p = patch[i][j]
        #         #p = cv2.resize(p,(64,64))
        #         allpatches.append(p)
    
        #write the patches to file
        #for i in range(len(allpatches)):
        for i in range(len(patch)):
            cv2.imwrite('testPatch' + str(i)+'.jpg', patchdict[i])
        
        #self.testpatches = allpatches
        self.testpatches = patch
        self.patch = patch # original patch for indexing
        
        # save the patches to a file
        pickle.dump(self.patch, open('patch.pickle','w'))
        pickle.dump(self.patchdict, open('patchdict.pickle','w'))
        pickle.dump(self.patchinddict, open('patchinddict.pickle','w'))
        
    def gen_sift_features(self):
        classlabels = []
        siftforimages = {}
        for i in range(len(self.testpatches)):
            sift = cv2.xfeatures2d.SIFT_create()    
            #siftforimages[str(i)+'.jpg'] = sift.detect(cv2.imread(str(i) + '.jpg'))
            siftforimages['testPatch' + str(i)+'.jpg'] = sift.detectAndCompute(cv2.imread('testPatch' + str(i)+'.jpg'), None)
            classlabels.append(-1)
        self.siftfeatures = siftforimages
        self.classlabels = classlabels
        
    
    def gen_histograms(self):
        allhistograms = {}
        for i in range(len(self.testpatches) ):
            print 'Processing patch ' + str(i)
            descriptors = self.siftfeatures['testPatch' + str(i)+'.jpg'][1]
            if descriptors == None:
                continue
            self.processedhists.append(i)
            code, dist = vq.vq(descriptors, self.codebook)
            histogram_of_words, bin_edges = histogram(code,
                                                      bins=range(self.codebook.shape[0] + 1),
                                                      normed=True)
            allhistograms['testPatch' + str(i)+'.jpg'] = histogram_of_words

        self.testhistograms = allhistograms
        
    def gen_test_data(self):
        data_rows = zeros(self.nclusters + 1)
        processedhists = []
        for i in range(len(self.testpatches) ):
            
            if 'testPatch' + str(i)+'.jpg' not in self.testhistograms.keys():
                continue
            histogram = self.testhistograms['testPatch' + str(i)+'.jpg']
            processedhists.append('testPatch' + str(i)+'.jpg')
            if histogram.shape[0] != self.nclusters:
                nclusters = histogram.shape[0]
                data_rows = zeros(nclusters + 1)
                print 'nclusters ahve been reduced to ' + str(nclusters)

            data_row = hstack((self.classlabels[i], histogram))
            data_rows = vstack((data_rows, data_row))
        data_rows = data_rows[1:]
        self.classifier_train_data = data_rows
        np.savetxt('testdata.csv',self.classifier_train_data, delimiter = ',')
        histfile = open('processedhists.csv','w')
        for h in processedhists:
            histfile.write(h)
            histfile.write('\n')
        histfile.close()
        
    def view_results(self):
        data = genfromtxt(self.predictionfile, delimiter=',')
        predictions = [x[1] for x in data][1:]
        predzip = zip(range(len(predictions)), predictions)
        posindex = [x[0] for x in predzip if x[1] == 1]
        
        # get the actual test patches from processedhists
        actpatchindices = [self.processedhists[x] for x in posindex]
        #cv2.imshow('Predctions', cv2.imread('testPatch' + str(actpatchindices[50]) +'.jpg'))
        
        # get the bounding box
        lowest = min(actpatchindices)
        highest = max(actpatchindices)
        
        #Get the indices
        img = cv2.imread(self.testimage)

        [lr,lc] = self.ind2sub(np.shape(img), lowest)
        [hr,hc] = self.ind2sub(np.shape(img), highest)

        # cropped image
        cropped = img[lc:hc , lr:hc]
        # print cropped
        
        #cv2.imshow('Predctions', cv2.imread('testPatch140.jpg'))
        # cv2.imshow('Final', cropped)
#         cv2.waitKey(0)

        # Draw rectangles
        # get akk
        im = Image.open(self.testimage)
        draw = ImageDraw.Draw(im)

        for p in actpatchindices:
            #[x,y] = self.ind2sub(np.shape(img), p)
            
            # Getting the actual patch coordinates
            i = np.unravel_index(p, np.shape(img)[:2])
            x = i[0]
            y = i[1]
            #print x,y
            draw.point((x,y), fill = 255)
        
        im.show() 
    def view_results2(self):
        
        # get the matched patches file
        
        mpfile = open('matchedpatches.csv')
        matchedpat = mpfile.readlines()
        
        # Matched test patches
        patchindices = []
        print 'There are '+ str(len(matchedpat))+ ' matched patches'
        for m in matchedpat:
            #print m
            #cv2.waitKey(1)
           
            p = int(re.findall('[0-9]+',m)[0])
            #cv2.imshow(m,self.testpatches[p])
            #cv2.waitKey(0)
            
            patchindices.append(p)
            
            # Get the coresponding 2d index from patch
       
        indices = []
        for p in patchindices:
            ind = self.ind2sub(self.patch.shape, p)
            indices.append(ind)
   
            lowest = indices[0]
            highest = indices[-1]
            [x,y] = lowest
            [k,l] = highest
            finalpatch = self.patch[y:l, x:k]
            stack1 = np.hstack(finalpatch)
            finalstack = np.hstack(stack1)
            print np.shape(finalstack)
            cv2.imshow('sample', finalstack)
            cv2.waitKey(0)
   
            print indices
        
    
    def view_results_knn(self):
        
        mpfile = open('matchedpatchesknn.csv')
        matchedpat = mpfile.readlines()
        
        # Matched test patches
        patchindices = []
        print 'There are '+ str(len(matchedpat))+ ' matched patches'
        for m in matchedpat:
            print m
            #cv2.waitKey(1)
           
            p = int(re.findall('[0-9]+',m)[0])
            #cv2.imshow(m,self.testpatches[p])
            #cv2.waitKey(0)
            
            patchindices.append(p)
            
        # Get the coresponding 2d index from patch
        patchindSorted = sorted(patchindices)
        lowest = patchindSorted[0]
        highest = patchindSorted[-1]

        indices = []
        for p in patchindices:
            ind = self.ind2sub(self.patch.shape, p)
            indices.append(ind)

        # finalstack = self.patch[indices[1]]
#         for ip in indices:
#             p = self.patch[ip]
#             finalstack = np.hstack((finalstack, p))
#
#         finalstack = np.hstack(finalstack)
        [x,y] = lowest
        [k,l] = highest
        finalpatch = self.patch[y:l, x:k]
        print np.shape(finalpatch)
        stack1 = np.hstack(finalpatch)
        finalstack = np.hstack(stack1)
        print np.shape(finalstack)
        cv2.imshow('sample', finalstack)
        cv2.waitKey(0)
        
        print indices
        
if __name__ == '__main__':
    gst = GenSiftForTest()
    #gst.gen_test_patches()
    
        
        
        