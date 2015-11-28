import ConfigParser as cp
from skimage.util.shape import view_as_windows
import random
from sklearn.feature_extraction import image
import cv2
import numpy as np
import os


class GenTrainPointsSIFT(object):
    posdatadir= ""
    negdatadir= ""
    trainfiles = []
    boundaryboxes = []
    imageseriesname = ""
    trainboxpatches = []
    bbfilenames = []
    negpatches = [] # negative training examples
    
    def __init__(self):
        config = cp.RawConfigParser()
        config.read('config.cfg')
        self.datadir = config.get('init', 'posdatadir')
        self.negdatadir = config.get('init', 'negdatadir')
        
        # Set the image series name
        temp = self.datadir.split('/')[-1]
        temp = temp.strip(" \r\n\"")
        self.imageseriesname = temp
        
        # Get the traiing and boundary boxes
        self.gen_train_files()
        self.get_boundary_boxes()
        self.get_trainbox_patches()
        self.gen_neg_examples()
        
        
    def gen_train_files(self):
        """
        Gets the training data file
        
        """
        files = os.listdir(self.datadir)
        self.imgfiles = [x for x in files if 'jpg' or 'png' or 'bmp' in files]
        
    
    def get_boundary_boxes(self):
        """
        Reads the boundary box files
        
        """
        files = os.listdir(self.datadir)
        bbfiles = [x for x in files if 'groundtruth' in x]
        
        
        # Get the bounding boxes 
        allboxes = []
        bbfilenames = []
        
        for b in bbfiles:
            s = open(self.datadir +'/' +b).read()
            boxes = s.split('\n')
            boxes = [x.split(' ')[ :4] for x in boxes]
            boxes = [x for x in boxes if x !=['']]
            allboxes += boxes
            
            # Adding file names corresponding to boundary boxes
            fname = b.replace('_'+self.imageseriesname.lower() + '.groundtruth', '.jpg')
            for i in range(len(boxes)):
                bbfilenames.append(fname)
        
       
        # Setting boundary boxes and associated filenames
        self.boundaryboxes = allboxes
        self.bbfilenames = bbfilenames
    
    def get_trainbox_patches(self):
        
        if self.boundaryboxes == []:
            raise ' Boundary box is empty. call get_boundary_boxes to fill in this array!\n'
        
        
        # For each bounding box read the image and generate the boxes
        
        for i in range(len(self.bbfilenames)):
            
            fname = self.bbfilenames[i]
            bbox = self.boundaryboxes[i]
            print bbox
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = float(bbox[2])
            y2 = float(bbox[3])
            # read the image 
            im = cv2.imread(self.datadir + '/' + fname)
            
            # resize the trainboxpatches
            
            patch = im[y1:y2, x1:x2, :]
            patch = cv2.resize(patch, (64,64))
            self.trainboxpatches.append(im[y1:y2, x1:x2, :])


    def gen_neg_examples(self):
        """
        Generates negative training patches
        from the dataset specified in the configuration
        """

        negfiles = os.listdir(self.negdatadir)
        negpatches = []

        for n in negfiles[:10]:
            f = self.negdatadir + '/' +  n

            if 'jpg' not in f:
                continue
            print 'Processing image'+ f
            im = cv2.imread(f)
            #im =cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)


            # Generate the negative patch
            #patch = image.extract_patches_2d(im, (256, 256)) # This generalates nultiple patches

            print np.shape(im)
            patch =  view_as_windows(im, (64,64,3),1)
            
            patch = [cv2.resize(x,(64,64)) for x in patch]

            # randomly choose 10 patches from all patches for this image

            patches = []
            for i in range(20):
                
                randompatch = random.choice(patch)
                patches.append(randompatch)

            negpatches += patches
            self.negpatches = negpatches

        
if __name__ == '__main__':
    gt = GenTrainPointsSIFT()
    #print gt.datadir
    #print gt.imageseriesname
    
    gt.get_boundary_boxes()
    #print gt.boundaryboxes
    #print gt.trainboxpatches
    gt.gen_neg_examples()
    cv2.imshow("sample",gt.trainboxpatches[1])

    # display a negative training examples
    cv2.imshow('negative example', random.choice(gt.negpatches))
    cv2.waitKey(0)    