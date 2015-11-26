import ConfigParser as cp
import cv2
import numpy as np
import os


class GenTrainPointsSIFT(object):
    datadir= ""
    trainfiles = []
    boundaryboxes = []
    imageseriesname = ""
    trainboxpatches = []
    bbfilenames = []
    
    def __init__(self):
        config = cp.RawConfigParser()
        config.read('config.cfg')
        self.datadir = config.get('init', 'datadir')
        
        # Set the image series name
        temp = self.datadir.split('/')[-1]
        temp = temp.strip(" \r\n\"")
        self.imageseriesname = temp
        
        # Get the traiing and boundary boxes
        self.gen_train_files()
        self.get_boundary_boxes()
        self.get_trainbox_patches()
        
        
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
            print fname
            print x1,y1,x2,y2
            im = cv2.imread(self.datadir + '/' + fname)
            print np.shape(im)
            self.trainboxpatches.append(im[y1:y2, x1:x2, :])
            
        
        
        
        
    
if __name__ == '__main__':
    gt = GenTrainPointsSIFT()
    #print gt.datadir
    #print gt.imageseriesname
    
    gt.get_boundary_boxes()
    #print gt.boundaryboxes
    #print gt.trainboxpatches
    cv2.imshow("sample",gt.trainboxpatches[1])
    cv2.waitKey(0)    