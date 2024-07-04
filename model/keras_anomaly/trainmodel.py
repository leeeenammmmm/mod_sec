import pandas as pd
import csv
import numpy as np
import pandas as pd
import math
from time import gmtime, strftime
from sklearn.preprocessing import MinMaxScaler
from keras_anomaly_detection.library.plot_utils import visualize_reconstruction_error
from keras_anomaly_detection.library.recurrent import BidirectionalLstmAutoEncoder
from keras.preprocessing import sequence

DO_TRAINING = 0
training = 100000
def get_train_data(dataset): 
	data= []
	with open(dataset, "r") as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row) 
		data =  np.array(data)
	return data

def main():
	model_dir_path = './models'
	dataset = "data.csv"

	# Read data from training set
	input_data = get_train_data(dataset)

	# Start time
	print strftime("%Y-%m-%d %H:%M:%S", gmtime())
	# print([x[0] for x in input_data])
	# Extract data and labels
	X = [x[0] for x in input_data]

	# Generate a dictionary of valid characters
	valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(X)))}
	max_features = len(valid_chars) + 1
	maxlen = np.max([len(x) for x in X])

	# Convert characters to int and pad
	X = [[valid_chars[y] for y in x] for x in X]
	print(valid_chars)
	X = sequence.pad_sequences(X, maxlen=maxlen)
	print(input_data)
	scaler = MinMaxScaler()
	# input_data = scaler.fit_transform(input_data)

	ae = BidirectionalLstmAutoEncoder()

	# fit the data and save model into model_dir_path
	if DO_TRAINING:
		ae.fit(X[:training, :], model_dir_path=model_dir_path, estimated_negative_sample_ratio=0.9)

	# load back the model saved in model_dir_path detect anomaly
	ae.load_model(model_dir_path)
	print(ae.threshold)
	# anomaly_information = ae.anomaly(X[training:, :])
	test ="login.php"
	print(X[-1:, :])
	dt = anomaly_information = ae.anomaly(X[-2:, :])
	print(dt)
	# reconstruction_error = []
	# count = 0
	# anomaly_count =0
	# for idx, (is_anomaly, dist) in enumerate(anomaly_information):
	# 	count = count+1
	# 	print('# ' + str(idx) + ' is ' + ('abnormal' if is_anomaly else 'normal') + ' (dist: ' + str(dist) + ')')
	# 	reconstruction_error.append(dist)
	# 	if is_anomaly:
	# 		anomaly_count = anomaly_count + 1
	# visualize_reconstruction_error(reconstruction_error, ae.threshold)
	# print(str(1-float(anomaly_count)/count))

if __name__ == '__main__':
	main()