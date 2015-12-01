from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import svm
###Load data
data_file = open('traindata')
data = []
count = 0
print 'Loading data & label...'
for line in data_file:
	count += 1
	if count % 6 ==0:
		features = line.split()
		temp = {}
		for i in range(len(features)):
			temp[i] = float(features[i])
		data.append(temp)  
data_file.close()


###load label
label_file = open('trueclass.txt')
labels = []
count = 0
for line in label_file:
	count += 1
	if count % 6 == 0:
		params = line.split()
		labels.append(int(params[0]))
label_file.close()

print('Data loaded, start feature filtering......')


###F test
def value_cmpf(x):
	return (-x[1]);

def cal_Fscore(labels,samples):

	data_num=float(len(samples))
	p_num = {} #key: label;  value: data num
	sum_f = [] #index: feat_idx;  value: sum
	sum_l_f = {} #dict of lists.  key1: label; index2: feat_idx; value: sum
	sumq_l_f = {} #dict of lists.  key1: label; index2: feat_idx; value: sum of square
	F={} #key: feat_idx;  valud: fscore
	max_idx = -1

	### pass 1: check number of each class and max index of features
	for p in range(len(samples)): # for every data point
		label=labels[p]
		point=samples[p]

		if label in p_num: p_num[label] += 1
		else: p_num[label] = 1

		for f in point.keys(): # for every feature
			if f>max_idx: max_idx=f
	### now p_num and max_idx are set

	### initialize variables
	sum_f = [0 for i in range(max_idx)]
	for la in p_num.keys():
		sum_l_f[la] = [0 for i in range(max_idx)]
		sumq_l_f[la] = [0 for i in range(max_idx)]

	### pass 2: calculate some stats of data
	for p in range(len(samples)): # for every data point
		point=samples[p]
		label=labels[p]
		for tuple in point.items(): # for every feature
			f = tuple[0]-1 # feat index
			v = tuple[1] # feat value
			sum_f[f] += v
			sum_l_f[label][f] += v
			sumq_l_f[label][f] += v**2
	### now sum_f, sum_l_f, sumq_l_f are done

	### for each feature, calculate f-score
	eps = 1e-12
	for f in range(max_idx):
		SB = 0
		for la in p_num.keys():
			SB += (p_num[la] * (sum_l_f[la][f]/p_num[la] - sum_f[f]/data_num)**2 )

		SW = eps
		for la in p_num.keys():
			SW += (sumq_l_f[la][f] - (sum_l_f[la][f]**2)/p_num[la]) 

		F[f+1] = SB / SW

	return F

def cal_feat_imp(label,sample):

	print("calculating fsc...")

	score_dict=cal_Fscore(label,sample)

	score_tuples = list(score_dict.items())
	score_tuples.sort(key = value_cmpf)

	feat_v = score_tuples
	for i in range(len(feat_v)): feat_v[i]=score_tuples[i][0]
	print("fsc done")
	return score_dict,feat_v

def select(sample, feat_v):
	new_samp = []
	feat_v.sort()

	#for each sample
	for s in sample:
		point={}
		#for each feature to select

		for f in feat_v:
			if f in s: point[f]=s[f]

		new_samp.append(point)
	return new_samp
print 
###Transform the data by Selecting Top K features
##F = cal_Fscore(labels, data)
print 'Start doing F-test....'
f_dict, rank = cal_feat_imp(labels, data)
data = select(data, rank[:1000])

###Output filtered data, into .csv format
def writedata(samples,labels,filename):
	if filename:
		fp=open(filename,"w")

	num=len(samples)
	for i in range(num):
		kk=list(samples[i].keys())
		kk.sort()
		last = kk[-1]
		for k in kk:
			fp.write(str(samples[i][k]))
			if k != last:
				fp.write(',')
		fp.write("\n")

	fp.flush()
	fp.close()
print 'Writing to ftest.data...'
writedata(data, labels, './kpca-eigen/data/ftest.data')
