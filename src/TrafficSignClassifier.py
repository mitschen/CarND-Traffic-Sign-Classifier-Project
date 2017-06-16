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
        TrafficSignClassifier.X_custom = np.array(TrafficSignClassifier.X_custom)
        TrafficSignClassifier.y_custom = np.array(TrafficSignClassifier.y_custom)
        
    def __printSummary():
        #print a summary according to Jupyter Notebook requirements
        print("Basic Summary of the DataSet")
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
        TrafficSignClassifier.__printSummary()
        
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
        
        
    __getId = staticmethod(__getId)
    importData = staticmethod(importData)
    preAnalyzeData = staticmethod(preAnalyzeData)
    __preAnalyzeData = staticmethod(__preAnalyzeData)
    simpleDataAugmentation = staticmethod(simpleDataAugmentation)
    dataAugmentation = staticmethod(dataAugmentation)
    __printSummary = staticmethod(__printSummary)
    drawDataSetExample = staticmethod(drawDataSetExample)
    importCustomImages = staticmethod(importCustomImages) 

#/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
#_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
#/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/
    
    def __init__(self, logger = DefaultLoggerClient):
        #plausibility check - guarantee that we've already imported some data
        assert(not (TrafficSignClassifier.X_train is None))
        self.X_train = np.copy(TrafficSignClassifier.X_train) 
        self.y_train = np.copy(TrafficSignClassifier.y_train)
        self.X_valid = np.copy(TrafficSignClassifier.X_valid)
        self.y_valid = np.copy(TrafficSignClassifier.y_valid)
        self.X_test = np.copy(TrafficSignClassifier.X_test)
        self.y_test = np.copy(TrafficSignClassifier.y_test)
        self.classes = TrafficSignClassifier.classes
        self.id = TrafficSignClassifier.__getId()
        self.logger = logger
        self.logger.log("{0:d} Create Instance".format(self.id))
        
        #tensorflow placeholder
        #otherwise we're running into the non initialized variables problem
        self.phX = None
        self.phY = None
        self.logitsCB = None
        self.one_hot_yCB = None
        self.fc1_kb = None
        self.fc2_kb = None
        
        #some flags used for further processing
        self.flag_isGrayScaled = False
        self.flag_isNormalized = False
    
    def normalize_zeroMeanData(self):
        #adjust the samples so that they have a zero mean
        self.X_test -= 128
        self.X_test /= 128.
        
        self.X_train -= 128
        self.X_train /= 128.
        
        self.X_valid -= 128
        self.X_valid /= 128.
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
        conv1_W = tf.Variable(tf.truncated_normal(shape=(inputFilter), mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(cfg["cv1"][c_filter][c_filter_out]))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=cfg["cv1"][c_strides], padding='VALID') + conv1_b
        #activation - always using relu
        conv1 = tf.nn.relu(conv1)
        #pooling
        conv1 = tf.nn.max_pool(conv1, ksize=cfg["p1"][c_filter], strides=cfg["p1"][c_strides], padding='VALID')
        
        #convolutional layer 2
        conv2_W = tf.Variable(tf.truncated_normal(shape=(cfg["cv2"][c_filter]), mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(cfg["cv2"][c_filter][c_filter_out]))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=cfg["cv2"][c_strides], padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=cfg["p2"][c_filter], strides=cfg["p2"][c_strides], padding='VALID')
        
        print (conv2.get_shape())
        #convolutional layer 3
        cv3_filter = ([5,5,int(conv2.get_shape()[3]),1], [1,1,1,1])
        print (cv3_filter)
        cv3_pool = ([1,1,1,1], [1,1,1,1] )
        CV3_avail = False
        if "cv3" in cfg.keys():
            assert("p3" in cfg.keys())
            CV3_avail = True
            cv3_filter = cfg["cv3"]
            cv3_pool = cfg["p3"]
            
        conv3_W = tf.Variable(tf.truncated_normal(shape=(cv3_filter[c_filter]), mean = mu, stddev = sigma))
        conv3_b = tf.Variable(tf.zeros(cv3_filter[c_filter][c_filter_out]))
        conv3   = tf.nn.conv2d(conv2, conv3_W, strides=cv3_filter[c_strides], padding='VALID') + conv3_b
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
        self.fc1_kp = tf.placeholder(tf.float32)
        fc1_W = tf.Variable(tf.truncated_normal(shape=(noOut, cfg["fc1"]), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(cfg["fc1"]))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
        
        #activation.
        fc1    = tf.nn.relu(fc1)
        fc1    = tf.nn.dropout(fc1, self.fc1_kp)
        
        #Layer 4: Fully Connected
        self.fc2_kp = tf.placeholder(tf.float32)
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(cfg["fc1"], cfg["fc2"]), mean = mu, stddev = sigma))
        fc2_b  = tf.Variable(tf.zeros(cfg["fc2"]))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
        fc2    = tf.nn.relu(fc2)
        fc2    = tf.nn.dropout(fc2, self.fc2_kp)
        
        # SOLUTION: Layer 5: Fully Connected.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=(cfg["fc2"], cfg["labels"]), mean = mu, stddev = sigma))
        fc3_b  = tf.Variable(tf.zeros(cfg["labels"]))
        logits = tf.matmul(fc2, fc3_W) + fc3_b
        
        #write some details about testrun
        self.logger.log(" CV1 "+str(cfg["cv1"])+"MaxP "+str(cfg["p1"]))
        self.logger.log(" CV1 - Shape "+str(conv1.get_shape()))
        self.logger.log(" CV2 "+str(cfg["cv2"])+"MaxP "+str(cfg["p2"]))
        self.logger.log(" CV2 - Shape "+str(conv2.get_shape()))
        if(CV3_avail == True):
            self.logger.log(" CV3 "+str(cfg["cv3"])+"MaxP "+str(cfg["p3"]))
            self.logger.log(" CV3 - Shape "+str(conv3.get_shape()))
        self.logger.log(" FC1 "+str(cfg["fc1"]))
        self.logger.log(" FC2 "+str(cfg["fc2"]))
        self.logger.log(" FC3 "+str(cfg["labels"]))
        
        return logits    
    
        
    def TrainCNN(self, cfg, storeNet = False):
        #input data is 32x32 x 3 (depth)
        if(self.flag_isGrayScaled == True):
            self.phX = tf.placeholder(tf.float32, (None, self.X_train.shape[1], self.X_train.shape[2], 1))
        else:
            self.phX = tf.placeholder(tf.float32, (None, self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3]))
        self.phY = tf.placeholder(tf.int32, (None))
        #one-hot result out of 43
        self.one_hot_y = tf.one_hot(self.phY, len(self.classes.keys()))

        #readout configuration
        #learning rate
        
        rate = 0.001 if c_learningrate not in cfg.keys() else cfg[c_learningrate]
        EPOCHS = 10 if c_epoch not in cfg.keys() else cfg[c_epoch]
        BATCH_SIZE = 128 if c_batchsize not in cfg.keys() else cfg[c_batchsize]
        keep_prop1 = 0.5 if c_keep_prop1 not in cfg.keys() else cfg[c_keep_prop1]
        keep_prop2 = 0.5 if c_keep_prop2 not in cfg.keys() else cfg[c_keep_prop2]


        #write some informations about the testrun
        self.logger.log("LearnRate {0:f}\nEpochCnt {1:d}\BatchSize {2:d}\n".format(rate, EPOCHS, BATCH_SIZE))
        self.logger.log("Keep Prop FC1 "+str(keep_prop1)+" Keep Prop FC2 "+str(keep_prop2))
        
        self.logits = self.__LeNet(cfg, self.phX)
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
            if (validation_accuracy > 0.93):
                #let's have a try on the testset
                testset_accuracy = self.__EvaluateCNN( self.X_test, self.y_test)
                self.logger.log("Instance {0:d}: On testdata we're achieving accuracy = {1:.3f}".format(self.id, testset_accuracy))
            
            if(storeNet == True):
                tf.train.Saver().save(sess, './tsc_cfg_'+str(self.id))
                self.logger.log("Saved session in ./tsc_cfg_"+str(self.id))
            

    def __EvaluateCNN(self, X_data, y_data):
        BATCH_SIZE = 128# if c_batchsize not in cfg.keys() else cfg[c_batchsize]
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #saver = tf.train.Saver()
        
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={self.phX: batch_x, self.phY: batch_y, self.fc1_kp : 1.0, self.fc2_kp : 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples        
        
    def analyzeCustomData(self, X_data, y_data):
        with tf.Session() as sess:
            fileToRestore = './tsc_cfg_'+str(self.id)
            print (fileToRestore)
            new_saver = tf.train.import_meta_graph(fileToRestore+'.meta')
            #tf.train.Saver().restore(sess, fileToRestore)
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))
            #tf.train.Saver().restore(tf.train.latest_checkpoint('.'))
            test_accuracy = self.__EvaluateCNN(X_data, y_data)
            print ("Test Accuracy = {:.3f}".format(test_accuracy))



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
                            } 
                           ]
    myTestimages = [ ("13_yield.bmp", 13), ("14_stop.bmp", 14), 
                     ("15_noEntry.bmp", 15), ("17_oneway.bmp", 17),
                     ("33_turnRight.bmp", 33)]
    
    myThreads = []
    
    start = time.clock()
    TrafficSignClassifier.importData("../")
    TrafficSignClassifier.importCustomImages("../examples", myTestimages)
    
    #TrafficSignClassifier.preAnalyzeData()
    #TrafficSignClassifier.drawDataSetExample()
    #TrafficSignClassifier.dataAugmentation(1000)
    
    TrafficSignClassifier.dataAugmentation(100)
    log = Logger("../results.txt", True)
    finalCfg = lenet_configuration[0]
    tsc = TrafficSignClassifier(log.getLogger(1));
    tsc.normalize_zeroMeanData()
    tsc.TrainCNN(finalCfg, True)
    tsc.analyzeCustomData(TrafficSignClassifier.X_custom, TrafficSignClassifier.y_custom)
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


