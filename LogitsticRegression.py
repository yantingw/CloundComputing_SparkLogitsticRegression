
import sys
import os
SPARK_HOME = "/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path
sys.path.append( SPARK_HOME + "/python/lib/py4j-0.10.9-src.zip") # Add python files to Python Path


import numpy as np
from pyspark import SparkConf, SparkContext
from  pyspark.mllib.regression import LabeledPoint
import math 
def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1] 
    _feats = feats[: len(feats) - 1]
    _feats.insert(0,label)
    features = [ float(feature) for feature in _feats ] # need floats
    #return np.array(features)
    
    ff = LabeledPoint(features[0],features[1:] )
    return ff


sc = getSparkContext()

# Load and parse the data

data = sc.textFile("data_banknote_authentication.txt")
parsedData = data.map(mapper)

# split whole data into training & testing set
train,test = parsedData.randomSplit((0.9,0.1))

# Train model

#model = LogisticRegressionWithSGD.train(train)
iteration = 1300
length = train.count()
update_rate = 0.02


def sigmoid(num):
        return float(1/(1+math.exp(-1*num)))

def calculate(each):
        """
        wx = (w * each.features)
        _z = wx.sum()
        yhat = sigmoid(_z)
        """
        yhat = predict(each)
        loss = yhat - each.label
        #grad.append(loss*yhat*(1-each.label))
        grad =loss*each.features #loss*yhat*(1-each.label)*each.features
        grad = np.insert(grad,0,loss*yhat*(1-each.label))
        return grad

def predict(each):
        wx = (w[1:] * each.features)
        _z = wx.sum() + w[0]
        yhat = sigmoid(_z)
        return yhat

def test_predict(each):
        num = predict(each)
        if num>0.5:
            yhat= 1
        else:
            yhat = 0
        return (yhat,each.label)

w  =  [0.0] * (len(parsedData.first().features)+1) #with bias
for i in range(iteration):
    gradRDD = train.map(calculate)
    sum_grad = gradRDD.reduce(lambda a,b:a+b)
    w = w - update_rate* sum_grad/length
    update_rate =  update_rate* 0.999
    #print(i,'w ',w)

    
# Predict the first elem will be actual data and the second 
# item will be the prediction of the modeli

labelsAndPreds = test.map(test_predict)

testErr = labelsAndPreds.filter(lambda (p,t): t!=p ).count()
one = test.filter(lambda each: each.label==1 ).count()
test_len =labelsAndPreds.count()
# Print some stuff
#print("total sample in training set ",test_len)
#print("training Error item = " , testErr )
print("training Error rate = " ,float( 1.0*testErr/test_len ))

labelsAndPreds = train.map(test_predict)

testErr = labelsAndPreds.filter(lambda (p,t): t!=p ).count()
one = test.filter(lambda each: each.label==1 ).count()
test_len =labelsAndPreds.count()
# Print some stuff
#print("total sample in training set ",test_len)
#print("training Error item = " , testErr )
print("training Error rate = " ,float( 1.0*testErr/test_len ))
