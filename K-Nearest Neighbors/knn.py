import requests
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

res = requests.get("https://people.cs.umass.edu/~bsilva/courses/CMPSCI_589/Spring2024/homeworks/datasets/iris.csv")
content = res.content
csv_file = open('iris.csv','wb')
csv_file.write(content)
csv_file.close()

results = []
with open("iris.csv") as csvfile:
    for row in csvfile:
        if row.strip().split(",") != ['']:
            results.append(row.strip().split(","))

def normalize_features(sample,set):
    normalized = []
    for i in range(len(sample)-1):
        max = float(set[0][i])
        min = float(set[0][i])
        for j in set:
            if float(j[i]) > max:
                max = float(j[i])
            if float(j[i]) < min:
                min = float(j[i])
        normalized_feature = (float(sample[i])-min)/(max-min)
        normalized.append(str(normalized_feature))
    normalized.append(sample[-1])
    return normalized

normalized_results = [normalize_features(i,results) for i in results]

def euclidian_dist(sample1,sample2):
    sum = 0
    for i in range(len(sample1)-1):
        sum += (float(sample1[i]) - float(sample2[i]))**2
    return math.sqrt(sum)

def predict_knn(input, k, training_set):
    results = []
    neighbors = []
    count = {}
    prediction = ''
    for sample in training_set:
        neighbors.append([sample,euclidian_dist(input,sample)])
    neighbors.sort(key=lambda x:x[1])
    for i in range(k):
        results.append(neighbors[i][0])
    for neighbor in results:
        if neighbor[-1] not in list(count.keys()):
            count[neighbor[-1]] = 0
        count[neighbor[-1]] += 1
    for iris in count.keys():
        if count[iris] == max(list(count.values())):
            prediction = iris
    return prediction

def compute_accuracy(set, k, training_set):
    correct = 0
    for input_data in set:
        if predict_knn(input_data,k,training_set) == input_data[-1]:
            correct += 1
    return correct/len(set)

x1 = []
y1 = []
x2 = []
y2 = []
for k in range(1,53,2):
    x1.append(k)
    x2.append(k)
    temp1 = []
    temp2 = []
    for i in range(20):
        shuffled_results = shuffle(normalized_results)
        training_set, testing_set = train_test_split(shuffled_results,train_size = 0.8,test_size=0.2)
        temp1.append(compute_accuracy(training_set,k,training_set))
        temp2.append(compute_accuracy(testing_set,k,training_set))
    y1.append(sum(temp1)/len(temp1))
    y2.append(sum(temp2)/len(temp2))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axes[0].plot(x1,y1,label = "Training Set")
axes[0].errorbar(x1,y1,yerr = np.std(y1),fmt = 'o')
axes[1].plot(x2,y2,label = "Testing Set")
axes[1].errorbar(x2,y2,yerr = np.std(y2),fmt = 'o')
axes[0].set_xlabel("Number of Neighbors K")
axes[1].set_xlabel("Number of Neighbors K")
axes[0].set_ylabel("Average Accuracy")
axes[1].set_ylabel("Average Accuracy")
axes[0].set_title("Accuracy of K-NN Algorithm on Training Set")
axes[1].set_title("Accuracy of K-NN Algorithm on Testing Set")
plt.show()