import pickle
import ConfigParser as cp
import numpy
import re
import numpy as np
import cv2

def ind2sub(array_shape, ind):
    #ind[ind < 0] = -1
    #ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = int(ind) / array_shape[1]
    cols = int(ind) % array_shape[1]
    return [rows, cols]
patch = pickle.load(open('patch.pickle'))
config = cp.RawConfigParser()
config.read('config.cfg')
algo = config.get('init', 'algo')
testimage = config.get('init','testimage')

# Read the prediction file
mpfile = ''
if algo == 'svm':
    mpfile = 'matchedpatchesknn.csv'
else:
    mpfile = 'matchedpatches.csv'
    
# Process the matched indices

matchedpat = open(mpfile).readlines()

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

allpatches = []
fdimlen = patch.shape[0]
sdimlen = patch.shape[1]

for i in range(fdimlen):
    for j in range(sdimlen):
        #p = patch[i][j][0]
        p = patch[i][j]
        #p = cv2.resize(p,(64,64))
        allpatches.append(p)
        
matchedpatches = [allpatches[x] for x in range(len(allpatches)) if x in patchindices]
matchedpatches = np.array(matchedpatches)
print matchedpatches.shape

finalpatch = np.array([])

while len(finalpatch.shape)  != 2:
    finalpatch = np.hstack(matchedpatches)

finalpatch.shape
#
# cv2.imshow('finalpatch',finalpatch)
# cv2.waitKey(0)

    
dim1 = finalpatch.shape[0]
patch = np.transpose(patch)
allfinalpatch = np.array([])
allfinalpatch = np.vstack(finalpatch)
print allfinalpatch.shape
# patch = np.transpose(patch)
# for i in range(dim1 - 1):
#     allfinalpatch = np.hstack(finalpatch)
# print allfinalpatch.shape
cv2.imshow('allfinalpatch',allfinalpatch)
cv2.waitKey(0)




#
# # Get the coresponding 2d index from patch
# patchindSorted = sorted(patchindices)
# lowest = patchindSorted[0]
# highest = patchindSorted[-1]
# print patch.shape
#
# indices = []
# for p in patchindices:
#     ind = ind2sub(patch.shape, p)
#     #ind = np.unravel_index(p, patch.shape)
#     indices.append(ind)
#
# # finalstack = self.patch[indices[1]]
# #         for ip in indices:
# #             p = self.patch[ip]
# #             finalstack = np.hstack((finalstack, p))
# #
# #         finalstack = np.hstack(finalstack)
# lowest = indices[0]
# highest = indices[-1]
#
# print lowest
# [x,y] = lowest
# [k,l] = highest
# finalpatch = patch[y:l, x:k]
#
# # find the finalpatch in the original image
#
# image = cv2.imread(testimage,0) # reading the test image
# stack1 = np.hstack(patch[highest])
# stack2 = np.hstack(stack1)
# result = cv2.matchTemplate(image,stack2,cv2.TM_CCOEFF_NORMED)
# print np.unravel_index(result.argmax(),result.shape)
# cv2.imshow('sample', image[result[1]: (64 + result[1]), result[0]: (64 +result[0])])
# cv2.waitKey(0)
# # print np.shape(finalpatch)
# # stack1 = np.hstack(finalpatch)
# # finalstack = np.hstack(stack1)
# # print np.shape(finalstack)
# # cv2.imshow('sample', finalstack)
# # cv2.waitKey(0)
# #
# # print indices