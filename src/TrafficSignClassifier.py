'''
Created on 13.06.2017

@author: Michael Scharf
@email: mitschen@gmail.com
'''

#used for the import of the TrafficSignData
import pickle
#read the csv file
import csv
from sklearn.utils import shuffle 
import os
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import LinearLocator
from tensorflow.contrib.layers import flatten
import tensorflow as tf

import threading     
import time
import math
import datetime


import scipy.ndimage  
import scipy.misc

C_TESTFILENAME = "test.p"
C_TRAINFILENAME = "train.p"
C_VALIDATIONFILENAME = "valid.p"
C_CLASSFILENAME = "signnames.csv"


class DefaultLoggerClient():
    def log(self, txt = None):
        if txt is None:
            return
        print (txt)

class LoggerClient():
    def __init__(self, _id, theLogger):
        assert(isinstance(theLogger, Logger))
        self.__id = _id
        self.__logger = theLogger
        
    def log(self, txt = None):
        if txt is None:
            return
        self.__logger._Logger__log(self.__id, txt)
        
        
class Logger():
    __mutexLock = threading.Lock()
    def __init__(self, outfile, toConsole = False):
        self.__filepath = outfile
        self.__content = {}
        self.__toConsole = toConsole
        
    def __log(self, _id, data):
        with Logger.__mutexLock:
            if (_id in self.__content.keys()):
                self.__content[_id].append(data)
                if(self.__toConsole == True):
                    print(data)
            else:
                self.__content[_id] = [data]
                
    def getLogger(self, _id):
        return LoggerClient(_id, self)
                
    def dump(self):
        toWrite = "\n\n"+str(datetime.datetime.now())+"\n";
        for key, val in self.__content.items():
            toWrite += "Entity {0:d}:\n".format(key)
            toWrite += "\n".join(val)
            #toWrite += string.join((list( map(lambda x : x+"\n", val))))
        with open(self.__filepath, "a") as f:
            f.write(toWrite)
        self.__content = {}
                

class TrafficSignClassifier():
    #static section
    #data we're reading from the traffic-signs data
    #This members are the input variables for any TrafficSignClassifier instance
    #    which might be configured with different values
    X_train = None
    Y_train = None
    X_valid = None
    y_valid = None
    X_test = None
    y_test = None
    X_custom = None
    y_custom = None
    classes = None
    #values are getting initializes during __preAnalyzeData
    #for each label, a list of index to the corresponding trainingInput
    trainingLabelSetIndex = None
    #unique id of each instance
    id = 0
    
    def __getId():
        #get a unique id for an instance
        #Used for multithreaded runtime to allow to associate the results
        #to a certain configuration
        TrafficSignClassifier.id+=1
        return TrafficSignClassifier.id
    
    def importData(filepath):
        #import the set of testdata once
        #This method is called once. All information are read into the
        #static member variables and then are shared between all
        #instances 
        training_file = os.path.join(filepath, C_TRAINFILENAME)
        validation_file= os.path.join(filepath, C_VALIDATIONFILENAME)
        testing_file = os.path.join(filepath, C_TESTFILENAME)
        csv_file = os.path.join(filepath, C_CLASSFILENAME)
        TrafficSignClassifier.classes = {}
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                TrafficSignClassifier.classes[row['ClassId']] = [row['SignName'], 0]

        assert('features' in train.keys())
        assert('labels' in train.keys())
        assert('features' in valid.keys())
        assert('labels' in valid.keys())
        assert('features' in test.keys())
        assert('labels' in test.keys())
        
            
        TrafficSignClassifier.X_train, TrafficSignClassifier.y_train = train['features'].astype(np.float32), train['labels']
        TrafficSignClassifier.X_valid, TrafficSignClassifier.y_valid = valid['features'].astype(np.float32), valid['labels']
        TrafficSignClassifier.X_test, TrafficSignClassifier.y_test = test['features'].astype(np.float32), test['labels']
        
        #the number of classes should be given by the different classIdentifiers 
        #provided in the set of train, test and validation results
        noClasses = set(TrafficSignClassifier.y_valid)
        noClasses.update(set(TrafficSignClassifier.y_train))
        noClasses.update(set(TrafficSignClassifier.y_test))
        assert(len(noClasses) == len(TrafficSignClassifier.classes.keys()))
        print("Files under ", os.path.abspath(filepath), "read\nwith size of (",TrafficSignClassifier.X_train.shape, TrafficSignClassifier.X_valid.shape, TrafficSignClassifier.X_test.shape,")\n")
        TrafficSignClassifier.__printSummary()
    
    def importCustomImages(filepath, fileList):
        #import custom bmp images and store them as
        files = []
        TrafficSignClassifier.y_custom = []
        TrafficSignClassifier.X_custom = [] 
        #compose fqnames and store the labels
        for val in fileList:
            files.append(os.path.join(filepath, val[0]))
            TrafficSignClassifier.y_custom.append(val[1])
        for f in files:
            TrafficSignClassifier.X_custom.append(scipy.misc.imread(f))
        TrafficSignClassifier.X_custom = np.array(TrafficSignClassifier.X_custom).astype(np.float32)
        TrafficSignClassifier.y_custom = np.array(TrafficSignClassifier.y_custom)
        
    def __printSummary(header = "Basic Summary of the DataSet"):
        #print a summary according to Jupyter Notebook requirements
        print(header)
        print("\tNumber of training examples = ", TrafficSignClassifier.X_train.shape[0])
        print("\tNumber of validation examples = ", TrafficSignClassifier.X_valid.shape[0])
        print("\tNumber of testing examples = ", TrafficSignClassifier.X_test.shape[0])
        print("\tImage data shape = ", (TrafficSignClassifier.X_train.shape[1], TrafficSignClassifier.X_train.shape[2]))
        print("\tNumber of classes = ", len(TrafficSignClassifier.classes.keys()))
    
    
    def preAnalyzeData():
        #make a statistic of how the training data is distributed
        #this method provides a graphical overview of the number of 
        #samples for each label
        TrafficSignClassifier.__preAnalyzeData()
        distribution = TrafficSignClassifier.trainingLabelSetIndex
        
        distribution_array = np.empty(shape=(2, len(distribution.keys())), dtype=int)
        distribution_array[0] = list(map(int,distribution.keys()))
        distribution_array[1] = list(map(lambda x : len(x),distribution.values()))

        plt.title("Number of testsamples")
        plt.bar(distribution_array[0], distribution_array[1])
        plt.xlabel('ClassId')
        plt.xticks(np.linspace(0, 42, 22, endpoint=True),rotation='vertical')
        plt.ylabel('number of labels')
        
        plt.show()
    
    def __preAnalyzeData(force = False):
        #count for each label the number of occurences
        #force is used to reread all the parameters from static members
        #in other cases, we'll immediately return
        #    this allows us to call this function in advance to any 
        #    other static function without doing extra effort
        trainIndex = TrafficSignClassifier.trainingLabelSetIndex
        
        if( force == True):
            trainIndex = None
            TrafficSignClassifier.trainingLabelSetIndex = None
        
        if(not (trainIndex is None)):
            return
        TrafficSignClassifier.trainingLabelSetIndex = {}
        trainIndex = TrafficSignClassifier.trainingLabelSetIndex
        for index, trainEntry in enumerate(TrafficSignClassifier.y_train):
            if trainEntry in trainIndex.keys():
                trainIndex[trainEntry].append(index)
            else:
                trainIndex[trainEntry] = [index]
    
    def simpleDataAugmentation():
        #simply adjust the testdata in a way that each label occuring the same time
        TrafficSignClassifier.__preAnalyzeData()
        distribution = TrafficSignClassifier.trainingLabelSetIndex
        
        #what is the max number of occurences for a certain label
        maxLabel = max(list(map(lambda x : len(x),distribution.values())))
        for key, val in distribution.items() :
            addSets = maxLabel - len(val);
            if(addSets == 0):
                continue
            addTraining = np.random.randint(min(val), max(val)+1, size = addSets)
            add_x_train = np.array(list(map(lambda x : np.copy(TrafficSignClassifier.X_train[x]), addTraining )))
            add_y_train = np.full((addSets, ), key )
            TrafficSignClassifier.y_train = np.append(TrafficSignClassifier.y_train, add_y_train)
            TrafficSignClassifier.X_train = np.append(TrafficSignClassifier.X_train, add_x_train, axis=0)
        TrafficSignClassifier.__preAnalyzeData(True)
        
    def dataAugmentation(size = None):
        #adjust the testdata in the following way:
        # occording to size, pick out size examples from each label
        # if no size is given, take the max available number of samples as size
        # for each set, take an image as it is, take an image shifted by 2 pixels on x/y axis
        #               take an image and rotate it for 15 degrees
        # Please note: shifting and rotation is always in same direction
        # Replace the readed training-dataset with the adjusted one
        TrafficSignClassifier.__preAnalyzeData()
        distribution = TrafficSignClassifier.trainingLabelSetIndex
         
        #what is the max number of occurences for a certain label
        maxLabel = size if not (size is None) else max(list(map(lambda x : len(x),distribution.values())))
        maxLabel = math.ceil(maxLabel / 3)*3
        X_newTrain = None
        y_newLabel = None
        for key, val in distribution.items() :
            #take maxLabel training elements from the set of given training elements
            addTraining = np.random.randint(min(val), max(val)+1, size = maxLabel)
            
            for x in range(0,maxLabel,3):
                samples = (TrafficSignClassifier.X_train[addTraining[x]], \
                           TrafficSignClassifier.X_train[addTraining[x+1]], \
                           TrafficSignClassifier.X_train[addTraining[x+2]])
                if(X_newTrain is None):
                    X_newTrain = [samples[0]]
                else:
                    X_newTrain.append(samples[0])
                X_newTrain.append(scipy.ndimage.interpolation.rotate(samples[1], 15., reshape=False, mode='nearest'))
                X_newTrain.append(scipy.ndimage.interpolation.shift(samples[2],  (2., 2., 0.), mode='nearest'))
                
            y_newLabel = np.append(y_newLabel, np.full((maxLabel, ), key )) if not (y_newLabel is None) else np.full((maxLabel, ), key )
        TrafficSignClassifier.X_train = np.array(X_newTrain)
        TrafficSignClassifier.y_train = y_newLabel
        TrafficSignClassifier.__preAnalyzeData(True)
        TrafficSignClassifier.__printSummary("Summary of DataSet after applying Augmentation")
        
    def __drawDataSet(samples, labels):
        #make a matplot output of samples/lables
        dim1 = math.ceil(len(labels) / 10.)
        dim2 = 10
        
        figure = plt.figure()
        index = 1
        for label, sample in zip(labels, samples):
            sample = sample.astype(np.uint8)
            image = sample.squeeze()
            a = figure.add_subplot(dim1,dim2, index)
            a.set_title(str(label[0]))
            a.axis('off')
            index+=1
            plt.imshow(image)
        plt.subplots_adjust(hspace = 0.5)            
        plt.show()
    
    def drawCustomDataSet():
        if( not(TrafficSignClassifier.X_custom is None)):
            samples = TrafficSignClassifier.X_custom
            labels = list(map(lambda x : (x, TrafficSignClassifier.classes[str(x)][0]), TrafficSignClassifier.y_custom))
            TrafficSignClassifier.__drawDataSet(samples, labels)
            return
        print("No custom data given")
        
    def drawDataSetExample():
        #pick out one sample from the set of given labels and
        #draw them once in a 3 x 43/3 table
        TrafficSignClassifier.__preAnalyzeData()
        distribution = TrafficSignClassifier.trainingLabelSetIndex
        
        samples = []
        labels = []
        for key in distribution:
            assert(0 < len(distribution[key]))
            index= distribution[key][0];
            samples.append(TrafficSignClassifier.X_train[index, :, :, :])
            labels.append((key, TrafficSignClassifier.classes[str(key)][0]) )
        assert(len(labels) == len(samples))
        
        TrafficSignClassifier.__drawDataSet(samples, labels)
        
    __getId = staticmethod(__getId)
    importData = staticmethod(importData)
    preAnalyzeData = staticmethod(preAnalyzeData)
    __preAnalyzeData = staticmethod(__preAnalyzeData)
    simpleDataAugmentation = staticmethod(simpleDataAugmentation)
    dataAugmentation = staticmethod(dataAugmentation)
    __printSummary = staticmethod(__printSummary)
    drawDataSetExample = staticmethod(drawDataSetExample)
    importCustomImages = staticmethod(importCustomImages) 
    drawCustomDataSet = staticmethod(drawCustomDataSet)
    __drawDataSet = staticmethod(__drawDataSet)

#/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
#/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    
    def __init__(self, cfg, logger = DefaultLoggerClient()):
        #plausibility check - guarantee that we've already imported some data
        self.X_train = None 
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.X_custom = None
        self.y_custom = None
        self.classes = None
        #some flags used for further processing
        self.flag_isGrayScaled = False
        self.flag_isNormalized = False
        self.flag_isCustomNormalized = False
        self.reloadData()
        self.id = TrafficSignClassifier.__getId()
        self.logger = logger
        self.logger.log("{0:d} Create Instance".format(self.id))
        self.cfg = cfg
        #remember if the tensor was already initialized
        self.__tensorInit = False
        self.__markdown_tableContent = ""
        
        #tensorflow placeholder
        #otherwise we're running into the non initialized variables problem
        self.phX = None
        self.phY = None
        self.logitsCB = None
        self.one_hot_yCB = None
        self.fc1_kp = None
        self.fc2_kp = None
        #tensor variables
        self.fc1_W = None
        self.fc2_W = None
        self.fc3_W = None
        self.fc1_b = None
        self.fc2_b = None
        self.fc3_b = None
        self.conv1_W = None
        self.conv1_b = None
        self.conv2_W = None
        self.conv2_b = None
        self.conv3_W = None
        self.conv3_b = None
        
        
    def reloadData(self):
        #in case that user adds custom data after instantiating the 
        assert(not (TrafficSignClassifier.X_train is None))
        self.X_train = np.copy(TrafficSignClassifier.X_train) 
        self.y_train = np.copy(TrafficSignClassifier.y_train)
        self.X_valid = np.copy(TrafficSignClassifier.X_valid)
        self.y_valid = np.copy(TrafficSignClassifier.y_valid)
        self.X_test = np.copy(TrafficSignClassifier.X_test)
        self.y_test = np.copy(TrafficSignClassifier.y_test)
        if(not (TrafficSignClassifier.X_custom is None) ):
            self.X_custom = np.copy(TrafficSignClassifier.X_custom)
            self.y_custom = np.copy(TrafficSignClassifier.y_custom)
        else:
            self.X_custom = None
            self.y_custom = None
        self.classes = TrafficSignClassifier.classes
        self.flag_isGrayScaled = False
        self.flag_isNormalized = False
        self.flag_isCustomNormalized = False
    
    def normalize_zeroMeanData(self):
        #adjust the samples so that they have a zero mean
        if(False == self.flag_isNormalized):
            self.X_test -= 128
            self.X_test /= 128.
            
            self.X_train -= 128
            self.X_train /= 128.
            
            self.X_valid -= 128
            self.X_valid /= 128.
        if not (self.X_custom is None) and (False == self.flag_isCustomNormalized):
            self.X_custom -= 128
            self.X_custom /= 128.
            self.flag_isCustomNormalized = True
        self.flag_isNormalized = True
        
    def convertToGrayScale_luminosity(self):
        #convert the data into grayscaled image. 
        #Please note: the user does not have to change the 
        #filter paremters in configuration - this is done automatically
        #during the training
        if(self.flag_isNormalized == True):
            print ("Please consider order of operations Grayscale => normalize allowed only")
            return
        
        c_redFraction = 0.21
        c_greenFraction = 0.72
        c_blueFraction = 0.07
        
        self.X_train = self.X_train[:,:,:,0] * c_redFraction \
                     + self.X_train[:,:,:,1] * c_greenFraction \
                     + self.X_train[:,:,:,2] * c_blueFraction
        self.X_valid = self.X_valid[:,:,:,0] * c_redFraction \
                     + self.X_valid[:,:,:,1] * c_greenFraction \
                     + self.X_valid[:,:,:,2] * c_blueFraction
        self.X_test  = self.X_test[:,:,:,0] * c_redFraction \
                     + self.X_test[:,:,:,1] * c_greenFraction \
                     + self.X_test[:,:,:,2] * c_blueFraction                                                   
        
        #correct the dimensions
        self.X_train = self.X_train[..., np.newaxis]
        self.X_valid = self.X_valid[..., np.newaxis]
        self.X_test = self.X_test[..., np.newaxis]
        
        self.flag_isGrayScaled = True
          
    def drawImage(self):
        example = self.X_train[0].astype(np.uint8)
        image = example.squeeze()

        plt.figure()
        if self.flag_isGrayScaled == True:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.show() 
    def __TensorInit(self, cfg):
        if(True == self.__tensorInit):
            return False
        
        #initialize all tensors
        
        #input data is 32x32 x 3 (depth)
        if(self.flag_isGrayScaled == True):
            self.phX = tf.placeholder(tf.float32, (None, self.X_train.shape[1], self.X_train.shape[2], 1))
        else:
            self.phX = tf.placeholder(tf.float32, (None, self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3]))
        self.phY = tf.placeholder(tf.int32, (None))
        #one-hot result out of 43
        self.one_hot_y = tf.one_hot(self.phY, len(self.classes.keys()))
        #setup the dropout tensor
        self.fc1_kp = tf.placeholder(tf.float32)
        self.fc2_kp = tf.placeholder(tf.float32)
        #specify the logits tensor
        self.logits = self.__LeNet(cfg, self.phX)
        
        
        self.__tensorInit = True
        return True
        pass
        
    def __LeNet(self,cfg, x):
        #some constants to make the code better readable
        c_filter = 0
        c_filter_in = 2
        c_filter_out = 3
        c_strides = 1
        
        mu = 0 if c_mu not in cfg.keys() else cfg[c_mu]
        sigma = 0.1 if c_sigma not in cfg.keys() else cfg[c_sigma]
        #convolutional layer 1
        inputFilter = cfg["cv1"][c_filter]
        if (self.flag_isGrayScaled == True):
            #reduce the input channels to 1
            inputFilter[c_filter_in] = 1
        self.conv1_W = tf.Variable(tf.truncated_normal(shape=(inputFilter), mean = mu, stddev = sigma), name='conv1_W')
        self.conv1_b = tf.Variable(tf.zeros(cfg["cv1"][c_filter][c_filter_out]), name='conv1_b')
        conv1   = tf.nn.conv2d(x, self.conv1_W, strides=cfg["cv1"][c_strides], padding='VALID') + self.conv1_b
        #activation - always using relu
        conv1 = tf.nn.relu(conv1)
        #pooling
        conv1 = tf.nn.max_pool(conv1, ksize=cfg["p1"][c_filter], strides=cfg["p1"][c_strides], padding='VALID')
        
        #convolutional layer 2
        self.conv2_W = tf.Variable(tf.truncated_normal(shape=(cfg["cv2"][c_filter]), mean = mu, stddev = sigma), name='conv2_W')
        self.conv2_b = tf.Variable(tf.zeros(cfg["cv2"][c_filter][c_filter_out]), name='conv2_b')
        conv2   = tf.nn.conv2d(conv1, self.conv2_W, strides=cfg["cv2"][c_strides], padding='VALID') + self.conv2_b
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=cfg["p2"][c_filter], strides=cfg["p2"][c_strides], padding='VALID')
        
        #convolutional layer 3
        cv3_filter = ([5,5,int(conv2.get_shape()[3]),1], [1,1,1,1])
        cv3_pool = ([1,1,1,1], [1,1,1,1] )
        CV3_avail = False
        if "cv3" in cfg.keys():
            assert("p3" in cfg.keys())
            CV3_avail = True
            cv3_filter = cfg["cv3"]
            cv3_pool = cfg["p3"]
            
        self.conv3_W = tf.Variable(tf.truncated_normal(shape=(cv3_filter[c_filter]), mean = mu, stddev = sigma), name='conv3_W')
        self.conv3_b = tf.Variable(tf.zeros(cv3_filter[c_filter][c_filter_out]), name='conv3_b')
        conv3   = tf.nn.conv2d(conv2, self.conv3_W, strides=cv3_filter[c_strides], padding='VALID') + self.conv3_b
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, ksize=cv3_pool[c_filter], strides=cv3_pool[c_strides], padding='VALID')
        
        #flatten
        fc0 = None
        if(CV3_avail == True):
            fc0 = flatten(conv3)
        else:
            fc0   = flatten(conv2)
        noOut = int(fc0.get_shape()[1])
        
        #keep the ratio as in LeNet architecture in case that 
        #user didn't specify it
        if("labels" not in cfg.keys()):
            cfg["labels"] = 43
        if("fc1" not in cfg.keys()):
            cfg["fc1"] = int((noOut + cfg["labels"])*5 / 17)
        if("fc2" not in cfg.keys()):
            cfg["fc2"] = int(cfg["fc1"]*7 / 10);
        
        #Layer 3: Fully Connected
        self.fc1_W = tf.Variable(tf.truncated_normal(shape=(noOut, cfg["fc1"]), mean = mu, stddev = sigma), name='fc1_W')
        self.fc1_b = tf.Variable(tf.zeros(cfg["fc1"]), name='fc1_b')
        fc1   = tf.matmul(fc0, self.fc1_W) + self.fc1_b
        
        #activation.
        fc1    = tf.nn.relu(fc1)
        fc1    = tf.nn.dropout(fc1, self.fc1_kp)
        
        #Layer 4: Fully Connected
        self.fc2_W  = tf.Variable(tf.truncated_normal(shape=(cfg["fc1"], cfg["fc2"]), mean = mu, stddev = sigma), name='fc2_W')
        self.fc2_b  = tf.Variable(tf.zeros(cfg["fc2"]), name='fc2_b')
        fc2    = tf.matmul(fc1, self.fc2_W) + self.fc2_b
        fc2    = tf.nn.relu(fc2)
        fc2    = tf.nn.dropout(fc2, self.fc2_kp)
        
        # SOLUTION: Layer 5: Fully Connected.
        self.fc3_W  = tf.Variable(tf.truncated_normal(shape=(cfg["fc2"], cfg["labels"]), mean = mu, stddev = sigma), name='fc3_W')
        self.fc3_b  = tf.Variable(tf.zeros(cfg["labels"]), name='fc3_b')
        logits = tf.matmul(fc2, self.fc3_W) + self.fc3_b
        
        #print configuration in markdown table format
        tableCnt =  "\n\nConv1 : "+str(cfg["cv1"][0][0]) +"x"+str(cfg["cv1"][0][1])+" "+str(cfg["cv1"][0][2])+"->"+str(cfg["cv1"][0][3])+"|\n"
        tableCnt += "MPool1: "+str(cfg["p1"][0][1])+"x"+str(cfg["p1"][0][2])+"|\n"
        tableCnt += "Shape1: "+str(conv1.get_shape()[1]) +"x"+str(conv1.get_shape()[2])+"x"+str(conv1.get_shape()[3])+"|\n"
        tableCnt += "Conv2 : "+str(cfg["cv2"][0][0]) +"x"+str(cfg["cv2"][0][1])+" "+str(cfg["cv2"][0][2])+"->"+str(cfg["cv2"][0][3])+"|\n"
        tableCnt += "MPool2: "+str(cfg["p2"][0][1])+"x"+str(cfg["p2"][0][2])+"|\n"
        tableCnt += "Shape2: "+str(conv2.get_shape()[1]) +"x"+str(conv2.get_shape()[2])+"x"+str(conv2.get_shape()[3])+"|\n"
        if(CV3_avail == True):
            tableCnt += "Conv3 : "+str(cfg["cv3"][0][0]) +"x"+str(cfg["cv3"][0][1])+" "+str(cfg["cv3"][0][2])+"->"+str(cfg["cv3"][0][3])+"|\n"
            tableCnt += "MPool3: "+str(cfg["p3"][0][1])+" "+str(cfg["p3"][0][2])+"|\n"
            tableCnt += "Shape3: "+str(conv3.get_shape()[1]) +"x"+str(conv3.get_shape()[2])+"x"+str(conv3.get_shape()[3])+"|\n"
            tableCnt += "FC1   : "+str(conv3.get_shape()[1]*conv3.get_shape()[2]*conv3.get_shape()[3])+"->"+str(cfg["fc1"])+", dropOut = "+str(cfg[c_keep_prop1])+"\n"
        else:
            tableCnt += "Conv3 : -|\nMPool3: -|\nShape3: -|\n"
            tableCnt += "FC1   : "+str(conv2.get_shape()[1]*conv2.get_shape()[2]*conv2.get_shape()[3])+"->"+str(cfg["fc1"])+", dropOut = "+str(cfg[c_keep_prop1])+"\n"
        tableCnt += "FC2   : "+str(cfg["fc1"])+"->"+str(cfg["fc2"])+", dropOut = "+str(cfg[c_keep_prop2])+"\n"
        tableCnt += "FC3   : "+str(cfg["fc2"])+"->"+str(cfg["labels"])+"\n"
        self.__markdown_tableContent +=tableCnt
        return logits    
    
        
    def TrainCNN(self, storeNet = False):
        cfg = self.cfg
        
        rate = 0.001 if c_learningrate not in cfg.keys() else cfg[c_learningrate]
        EPOCHS = 10 if c_epoch not in cfg.keys() else cfg[c_epoch]
        BATCH_SIZE = 128 if c_batchsize not in cfg.keys() else cfg[c_batchsize]
        keep_prop1 = 0.5 if c_keep_prop1 not in cfg.keys() else cfg[c_keep_prop1]
        keep_prop2 = 0.5 if c_keep_prop2 not in cfg.keys() else cfg[c_keep_prop2]
        cfg[c_keep_prop1] = keep_prop1
        cfg[c_keep_prop2] = keep_prop2

        #initialize the tensor if not already the case
        self.__TensorInit(cfg)

        #write some informations about the testrun
        self.__markdown_tableContent += "LearnRate {0:f}\nEpochCnt {1:d}\nBatchSize {2:d}".format(rate, EPOCHS, BATCH_SIZE)
        self.__markdown_tableContent += "\nKeep Prop FC1 "+str(keep_prop1)+"\nKeep Prop FC2 "+str(keep_prop2)+"\n\n"
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(self.X_train)
            
            self.logger.log("Training...")
            self.logger.log()
            
            for i in range(EPOCHS):
                X_train, y_train = shuffle(self.X_train, self.y_train)
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(training_operation, feed_dict={self.phX: batch_x, self.phY: batch_y, self.fc1_kp:keep_prop1, self.fc2_kp:keep_prop2})
                
                validation_accuracy = self.__EvaluateCNN(self.X_valid, self.y_valid)
                self.logger.log("EPOCH {} ...".format(i+1))
                self.logger.log("Instance {0:d}: Validation Accuracy = {1:.3f}".format(self.id, validation_accuracy))
                self.logger.log()
            self.__markdown_tableContent+="Validating {:.3f}".format(validation_accuracy)
            if (validation_accuracy > 0.93):
                #let's have a try on the testset
                testset_accuracy = self.__EvaluateCNN( self.X_test, self.y_test)
                self.logger.log("Instance {0:d}: On testdata we're achieving accuracy = {1:.3f}".format(self.id, testset_accuracy))
                self.__markdown_tableContent+="Testing {:.3f}".format(testset_accuracy)
            
            if(storeNet == True):
                tf.train.Saver().save(sess, './tsc_cfg_'+str(self.id))
                self.logger.log("Saved session in ./tsc_cfg_"+str(self.id))
        self.logger.log(self.__markdown_tableContent);
            

    def __EvaluateCNN(self, X_data, y_data):
        BATCH_SIZE = 128# if c_batchsize not in cfg.keys() else cfg[c_batchsize]
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={self.phX: batch_x, self.phY: batch_y, self.fc1_kp : 1.0, self.fc2_kp : 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
        
        
    def analyzeCustomData(self, _type='custom'):
        X = None
        Y = None
        accuracy = None
        top_5 = ""
        if _type == 'train':
            X = self.X_test
            Y = self.y_test
        elif _type == 'valid':
            X = self.X_valid
            Y = self.y_valid
        elif _type == 'test':
            X = self.X_train
            Y = self.y_train
        else:
            _type = 'custom'
            X = self.X_custom
            Y = self.y_custom
            
        with tf.Session() as sess:
            self.__TensorInit(self.cfg)
            fileToRestore = './tsc_cfg_'+str(self.id)
            tf.train.Saver().restore(sess, fileToRestore)
            self.logger.log("Restore session in {}".format(fileToRestore))
            accuracy = self.__EvaluateCNN(X, Y)
            self.logger.log("analyzeCustomData ({0}) = {1:.3f}".format(_type, accuracy))
            if _type=='custom':
                self.logger.log("Top K evaluation")
                propability = tf.nn.softmax(logits=self.logits)
                bestMatch = tf.nn.top_k(propability,5)
                val = sess.run(bestMatch, feed_dict={self.phX: X, self.phY: Y, self.fc1_kp : 1.0, self.fc2_kp : 1.0})
                for prop, match, label in zip(val[0], val[1], Y):
                    top_5 += "Label {} identified as:\n{} with {:.2f}, {} with {:.2f}\n"\
                                                "{} with {:.2f}, {} with {:.2f}\n{} with {:.2f}\n"\
                            .format(label, match[0], prop[0], match[1], prop[1], match[2], prop[2]\
                                    , match[3], prop[3], match[4], prop[4])
                self.logger.log(top_5)
        return (accuracy, top_5)
                    



class aThread(threading.Thread):
    def __init__(self, TSC, cfg):
        threading.Thread.__init__(self) 
        self.__tsc = TSC
        self.__cfg = cfg
    def run(self):
        #self.__tsc.convertToGrayScale_luminosity()
        self.__tsc.normalize_zeroMeanData()
        self.__tsc.TrainCNN(self.__cfg)


#some globals,
#use these values to configure the lenet_configuration
c_learningrate = "learningrate"   #default is 0.001
c_mu = "mu"                       # default is 0
c_sigma = "sigma"                 #default is 0.1
c_epoch = "epoch"                 #default is 10
c_batchsize = "batch"             #default is 128
c_keep_prop1 = "kp1"              #default is 0.5
c_keep_prop2 = "kp2"              #default is 0.5


if __name__ == '__main__':
    lenet_configuration = [ 
                            { 
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([5,5,3,32], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,32,43], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              # dimension of the hidden neurons - outputsize
                              "fc1" : 120,
                              "fc2" : 84,
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.001
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([4,4,3,16], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,16,43], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              # dimension of the hidden neurons - outputsize
                              "fc1" : 120,
                              "fc2" : 84,
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.001
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([5,5,3,108], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,108,108], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              # dimension of the hidden neurons - outputsize
                              "fc1" : 120,
                              "fc2" : 84,
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.001
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([5,5,3,43], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,43,108], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.001
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([4,4,3,43], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([4,4,43,108], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              "cv3" : ([3,3,108,43], [1,1,1,1]), "p3" : ([1,2,2,1], [1,1,1,1]),
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.001
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([4,4,3,108], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([4,4,108,43], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              "cv3" : ([2,2,43,43], [1,1,1,1]), "p3" : ([1,2,2,1], [1,1,1,1]),
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.001
                            } , 
                            { 
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([5,5,3,6], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,6,16], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              # dimension of the hidden neurons - outputsize
                              "fc1" : 120,
                              "fc2" : 84,
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.0005
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([4,4,3,16], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,16,43], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              # dimension of the hidden neurons - outputsize
                              "fc1" : 120,
                              "fc2" : 84,
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.0005
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([5,5,3,108], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,108,108], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              # dimension of the hidden neurons - outputsize
                              "fc1" : 120,
                              "fc2" : 84,
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.0005
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([5,5,3,43], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,43,108], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.0005
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([5,5,3,108], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,108,43], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.0005
                            },
                            {  
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([4,4,3,108], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([4,4,108,43], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              "cv3" : ([2,2,43,43], [1,1,1,1]), "p3" : ([1,2,2,1], [1,1,1,1]),
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.0005
                            },
                            { 
                              # filter shape + stride          maxpooling filter shape and stride
                              "cv1" : ([5,5,3,32], [1,1,1,1]), "p1" : ([1,2,2,1], [1,2,2,1]),
                              "cv2" : ([5,5,32,43], [1,1,1,1]), "p2" : ([1,2,2,1], [1,2,2,1]),
                              "labels" : 43,
                              c_epoch : 15,
                              c_learningrate : 0.001
                            }, 
                           ]
    myTestimages = [ ("13_yield.bmp", 13), ("14_stop.bmp", 14), 
                     ("15_noEntry.bmp", 15), ("17_oneway.bmp", 17),
                     ("33_turnRight.bmp", 33)]
    
    myThreads = []
    
    start = time.clock()
    TrafficSignClassifier.importData("../")
    TrafficSignClassifier.drawDataSetExample()
    
    #TrafficSignClassifier.preAnalyzeData()
    #TrafficSignClassifier.drawDataSetExample()
    #TrafficSignClassifier.dataAugmentation(1000)
    
    TrafficSignClassifier.dataAugmentation(15)
    log = Logger("../results.txt", True)
    finalCfg = lenet_configuration[len(lenet_configuration)-1]
    #tsc = TrafficSignClassifier(finalCfg, log.getLogger(1));
    tsc = TrafficSignClassifier(finalCfg);
    tsc.normalize_zeroMeanData()
    tsc.TrainCNN(True)
    #tsc.TrainCNN(False)
    TrafficSignClassifier.importCustomImages("../examples", myTestimages)
    TrafficSignClassifier.drawCustomDataSet()
    tsc.reloadData()
    tsc.normalize_zeroMeanData()
    result = tsc.analyzeCustomData()
    print("LALA", result)
    log.dump()
    print ("Time elapsed ", math.ceil((time.clock() - start)/60) )
    exit(0)
    
    for config in range(0, len(lenet_configuration)):
        start = time.clock()
        tsc = TrafficSignClassifier(log.getLogger(config));
        tsc.normalize_zeroMeanData()
        tsc.TrainCNN(lenet_configuration[config])
# Disable multithreading procedure - my machine is to slow for that         
#         tsc1 = aThread(TrafficSignClassifier(log.getLogger(config)), lenet_configuration[config])
#         tsc2 = aThread(TrafficSignClassifier(log.getLogger(config)), lenet_configuration[config+1])
#         myThreads.append(tsc1)
#         myThreads.append(tsc2)
#         tsc1.start()
#         tsc2.start();
#         for t in myThreads:
#             t.join()
        elapsed = (time.clock() - start)
        log.dump()
        print ("Time elapsed ", math.ceil(elapsed/60) , " Round " , config)
    pass


