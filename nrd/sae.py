import numpy as np
import scipy
import scipy.io
import argparse
import json

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ld', type=float, default=500000) # lambda
	return parser.parse_args()


def normalizeFeature_old(x):
	# x = d x N dims (d: feature dimension, N: the number of features)
	x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide
	feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	feat = x / feature_norm[:, np.newaxis]
	return feat


def normalizeFeature(x):
	# x = d x N dims (d: feature dimension, N: the number of features)
	feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	feat = x / feature_norm[:, np.newaxis]
	return (2**0.5) * feat


def SAE(x, s, ld):
	# SAE is Semantic Autoencoder
	# INPUTS:
	# 	x: d x N data matrix
	#	s: k x N semantic matrix
	#	ld: lambda for regularization parameter
	#
	# OUTPUT:
	#	w: kxd projection matrix

	A = np.dot(s, s.transpose())
	B = ld * np.dot(x, x.transpose())
	C = (1 + ld) * np.dot(s, x.transpose())
	w = scipy.linalg.solve_sylvester(A, B, C)
	return w

def distCosine(x, y):
	xx = np.sum(x**2, axis=1)**0.5
	x = x / xx[:, np.newaxis]
	yy = np.sum(y**2, axis=1)**0.5
	y = y / yy[:, np.newaxis]
	dist = 1 - np.dot(x, y.transpose())
	return dist



def zsl_acc(semantic_predicted, semantic_gt, opts):
	# zsl_acc calculates zero-shot classification accuracy
	#
	# INPUTS:
	#	semantic_prediced: predicted semantic labels
	# 	semantic_gt: ground truth semantic labels
	# 	opts: other parameters
	#
	# OUTPUT:
	# 	zsl_accuracy: zero-shot classification accuracy (per-sample)

	dist = 1 - distCosine(semantic_predicted, normalizeFeature(semantic_gt.transpose()).transpose())
	y_hit_k = np.zeros((dist.shape[0], opts.HITK))
	for idx in range(0, dist.shape[0]):
		sorted_id = sorted(range(len(dist[idx,:])), key=lambda k: dist[idx,:][k], reverse=True)
		y_hit_k[idx, :] = opts.test_classes_id[sorted_id[0:opts.HITK]]
		
	n = 0
	for idx in range(0, dist.shape[0]):
		if opts.test_labels[idx] in y_hit_k[idx, :]:
			n = n + 1
	zsl_accuracy = float(n) / dist.shape[0] * 100
	return zsl_accuracy, y_hit_k


def main():
	train_data_lst = json.load(open('retacred_train_half_rel_representations.json', 'r'))

	# do not need to change
	test_data_lst = json.load(open('retacred_test_half_rel_representations.json', 'r'))


	train_data = np.array(train_data_lst)
	print(len(train_data))
	test_data = np.array(test_data_lst)
	print(len(test_data))

	train_class_attribute_vectors_lst = json.load(open('retacred_mix_half/retacred_train_S_vectors_half.json', 'r'))
	train_class_attribute_vectors = np.array(train_class_attribute_vectors_lst)
	print(len(train_class_attribute_vectors))


	lambda_lst = [100]
	for cur_lambda in lambda_lst:
		##### Training
		# SAE
		print("SAE begin")
		W = SAE(train_data.transpose(), train_class_attribute_vectors.transpose(), cur_lambda)
		print("SAE end")
		##### Test
		#opts.HITK = 1

		# [F --> S], projecting data from feature space to semantic space
		semantic_predicted = np.dot(test_data, W.transpose())
		semantic_predicted_out = normalizeFeature(semantic_predicted).tolist()
		with open('retacred_test_S_vectors_half_ld{}.json'.format(cur_lambda), 'w') as outF:
			json.dump(semantic_predicted_out, outF)

		train_predicted = np.dot(train_data, W.transpose())
		train_predicted_out = normalizeFeature(train_predicted).tolist()
		with open('retacred_train_S_vectors_half_ld{}.json'.format(cur_lambda), 'w') as outF:
			json.dump(train_predicted_out, outF)
			
	
if __name__ == '__main__':
	main()
