from gen_train_data import *
from PIL import Image
import sys
import os
import random
import cv2

class siftFeatures(object):

    train_patches = []
    siftfeaturs = []
    # initialize the training data

    def __init__(self):
        gtp = GenTrainPointsSIFT()
        self.train_patches = gtp.trainboxpatches


    def display_random_path(self):
        r = random.randint(0, len(self.train_patches))
        random_patch = self.train_patches[r]
        cv2.imshow('sample',random_patch)
        cv2.waitKey(0)

    def gen_sift_features(self):
        """
        Generates sift features for the points
        """

        if not os.path.isfile('sift'):
            raise 'Sift binary not in path'

        if self.train_patches == []:
            raise 'Training patches is empty'

            
        # Write train patches to file
        i = 0
        for p in self.train_patches:
            im = Image.fromarray(p)
            im = im.convert('RGB')
            #im.save(str(i) + '.pgm')
            cv2.imwrite(str(i)+'.pgm',p)
            i += 1

        # Compute sift for images
        siftforimg = {}
        for i in range(len(self.train_patches)):
            imgname = str(i)+'.jpg'

            # Command for generating sift
            command = './sift < '+str(i)+'.jpg'
            print command
            sift = os.popen( command)
            siftforimg[str(i) + '.jpg'] = sift

        print siftforimg['1.jpg']
            

if __name__ == '__main__':
    sf = siftFeatures()
    sf.display_random_path()
    sf.gen_sift_features()
        
    