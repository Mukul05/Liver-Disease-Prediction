import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import linear_model
#from sklearn.model_selection cross_validation
from scipy.stats import norm

from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier



def process(path):
	df = pd.read_csv(path).values
	print(df)

	X=df[:,0:9]
	Y=df[:,10] 

	# Splitting dataset into training and test set
	train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size =.20)
	print('Training Features Shape:', train_features.shape)
	print('Training Labels Shape:', train_labels.shape)
	print('Testing Features Shape:', test_features.shape)
	print('Testing Labels Shape:', test_labels.shape)

	# Random Forest Classifier
	rf = RandomForestClassifier()
	rf.fit(train_features, train_labels.ravel()); # Build a forest of trees from training set
	y_pred = rf.predict(test_features) 
	print(y_pred)

	result2=open("results/resultRF.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()
	
	mse=mean_squared_error(test_labels, y_pred)
	mae=mean_absolute_error(test_labels, y_pred)
	
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR RandomForest IS %f "  % mse)
	print("MAE VALUE FOR RandomForest IS %f "  % mae)
	
	rms = np.sqrt(mean_squared_error(test_labels, y_pred))
	print("RMSE VALUE FOR RandomForest IS %f "  % rms)
	ac=accuracy_score(test_labels,y_pred)
	print ("ACCURACY VALUE RandomForest IS %f" % ac)
	print("---------------------------------------------------------")
	

	result2=open('results/RFMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/RFMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#d62728", "#8c564b"]
	explode = (0.1,0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title(' Random Forest Metrics Value')
	fig.savefig('results/RFMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

#process("data.csv")