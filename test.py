from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import svm
data_file = open('./kpca-eigen/data/transformed_KPCA.data')
data = []
count = 0
print 'Loading data & label...'
for line in data_file:
	features = line.split(',')
	temp = []
	for i in range(len(features)):
		temp.append(float(features[i]))
	data.append(temp)  
data_file.close()


###load label
label_file = open('trueclass.txt')
labels = []
count = 0
for line in label_file:
	count += 1
	if count %6 == 0:
		params = line.split()
		labels.append(int(params[0]))
label_file.close()

print('Data loaded, start feature filtering......')

clf = RandomForestClassifier(n_estimators=400)

scores = cross_validation.cross_val_score(clf, data, labels, cv =10)
print scores