import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split



bc_data = pd.read_csv('breast-cancer-wisconsin.csv', header=None, names =["id",'clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei', 'bland_chrom','norm_nucleoli', 'mitoses', 'class'] )
bc_data.replace("?", -99999, inplace = True)
bc_data.drop(['id'], axis = 1, inplace = True)

X = np.array(bc_data.drop(['class'],axis=1))
y = np.array(bc_data['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
print(prediction)


















