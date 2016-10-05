import numpy as np
import sys,os,csv
import jubatus
from jubatus.common import Datum

host = '127.0.0.1'
port = 9199
name = 'test'
client = jubatus.Classifier(host,port,name)

n_train = 5000
n_test = 2000
dir = "mldata/"

def gen_datum(filename):
    with open(filename,"rb") as f:
        binary = f.read()
        d = Datum()
        d.add_binary("image", binary)
    return d

def get_labels(path):
	labels = []
	with open(path,"rb") as f:
		reader = csv.reader(f)
		for row in reader:
			labels.append(row)
	return labels

def get_traindata(labels):
	traindata = []
	for index in range(n_train):
		img = dir+str(index)+".jpg"
		with open(img,"rb") as f:
			binary = f.read()
			label = labels[index][1]
			print label,index
			d = Datum()
			d.add_binary("image",binary)
			traindata.append([label,d])
	print "num of train data :",len(traindata)
	return traindata

def get_testdata(labels):
	testdata = []
	testlabels = []
	for index in range(n_train,(n_train+n_test)):
		img = dir+str(index)+".jpg"
		with open(img,"rb") as f:
			binary = f.read()
			d = Datum()
			d.add_binary("image",binary)
			testdata.append([d])
			testlabels.append(labels[index][1])
	print "num of test  data :",len(testdata)
	return testdata


def train(traindata):
	for entry in traindata:
		client.train([(entry[0],entry[1])])
		sys.stdout.write(".")
		sys.stdout.flush()

def test(testdata,labels):
	OK = NG = 0
	for i,test in enumerate(testdata):
		label = labels[i + n_train]
		predict = client.classify(test)
		predict = predict[0]
		predict.sort(key=lambda x:x.score, reverse=True)
		if predict[0].label == label[1]:
			OK += 1
			sys.stdout.write("o")
		else:
			NG += 1
			sys.stdout.write("x")
		sys.stdout.flush()
	print ""
	Accuracy = OK*1.0/(OK+NG)
	print "OK : %d, NG :%d, Accuracy : %f"%(OK,NG,Accuracy)





if __name__ == '__main__':
	labels = get_labels("mldata/labels.csv")
	train_data = get_traindata(labels)
	test_data = get_testdata(labels)
	train(train_data)
	test(test_data,labels)
