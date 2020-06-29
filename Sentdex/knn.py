import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random
style.use("fivethirtyeight")

dataset = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]



def knn(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups!")
    
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )#2 dimension
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    
    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    #print(vote_result, confidence)
    
    return vote_result, confidence

accuracies = []

for i in range(25):

    result = knn(dataset, new_features, k=3)
    print(result)
    #[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
    #plt.scatter(new_features[0], new_features[1], s = 100, color = result)
    #plt.show()

    df = pd.read_csv('breast-cancer-wisconsin.csv', header=None, names =["id",'clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei', 'bland_chrom','norm_nucleoli', 'mitoses', 'class'] )
    df.replace("?", -99999, inplace = True)
    df.drop(['id'], axis = 1, inplace = True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = .2
    train_set = {2:[],4:[]}
    test_set ={2:[],4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]#first 80% of the data
    test_data = full_data[-int(test_size*len(full_data)):]#last 20% of the data

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct =0
    total=0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = knn(train_set, data, k=5)
            if group == vote:
                correct+=1
            else:
                print(confidence)
            total+=1
    accuracies.append(correct/total)


print("Accuracy:", correct/total)
print(sum(accuracies)/len(accuracies))
 # k increases doesn't mean more accurate
# accuracy is pretty abt identical for this knn algo compared to sklearn knn







