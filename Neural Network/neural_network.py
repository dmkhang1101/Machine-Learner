import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from backprop import *

wine = []
house_votes = []

with open('./wine.csv') as csvfile:
    for row in csvfile:
        wine.append(row.strip().split('\t'))

with open('./house_votes_84.csv') as csvfile:
    for row in csvfile:
        house_votes.append(row.strip().split(','))

wine = wine[1:]
attributes_house = house_votes[0][:-1]
attributes_house[0] = '#handicapped-infants'

ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform = 'pandas')
df = pd.read_csv('./datasets/hw3_house_votes_84.csv')
for attr in attributes_house:
    ohetransform = ohe.fit_transform(df[[attr]])
    df = pd.concat([df,ohetransform],axis = 1).drop(columns = attr)
df.to_csv('./datasets/house_votes_84_encoded.csv')

house_votes_encoded = []

with open('./datasets/house_votes_84_encoded.csv') as csvfile:
    not_data = True
    for row in csvfile:
        if not_data:
            not_data = False
            continue
        new_row = row.strip().split(',')
        new_row = new_row[1:]
        target = new_row.pop(0)
        new_row_float = [float(i) for i in new_row]
        new_row_float.append(target) # type: ignore
        house_votes_encoded.append(new_row_float)

def normalize_features(sample,set):
    normalized = []
    for i in range(1,len(sample)):
        max = float(set[0][i])
        min = float(set[0][i])
        for j in set:
            if float(j[i]) > max:
                max = float(j[i])
            if float(j[i]) < min:
                min = float(j[i])
        normalized_feature = (float(sample[i])-min)/(max-min)
        normalized.append(normalized_feature)
    normalized.append(sample[0])
    return normalized

normalized_wine = [normalize_features(i,wine) for i in wine]

def construct_10_folds(data,classes,index_of_classes):
    folds = []
    sub_data = {}
    size_of_each_fold = int(len(data)/10)
    for i in classes:
        sub_list = list(filter(lambda x: x[index_of_classes] == i,data))
        sub_data[i] = [sub_list,len(sub_list)/(len(data))]
    for i in range(10):
        fold = []
        for j in classes:
            sample = random.sample(sub_data[j][0],int(size_of_each_fold*sub_data[j][1]))
            fold.extend(sample)
            for element in sample:
                sub_data[j][0].remove(element)
        folds.append(fold)
    for i in classes:
        while len(sub_data[i][0]) != 0:
            folds[random.randint(0,9)].append(sub_data[i][0].pop())
    return folds

def prepare_training(training_set,classes):
    result = []
    for instance in training_set:
        label_prob = [0]*len(classes)
        attr = instance[:-1]
        label = instance[-1]
        index_label = classes.index(label)
        label_prob[index_label] = 1
        result.append([attr,label_prob])
    return result

def initialize_weights(structure):
    weights = []
    for i in range(1,len(structure)):
        theta = []
        for j in range(structure[i]):
            row = []
            for m in range (structure[i-1]+1):
                row.append(np.random.normal(0,1))
            theta.append(row)
        weights.append(theta)
    return weights

def update_weights(old_weights,gradients):
    new_weights = []
    for i in range(len(old_weights)):
        theta_old = np.matrix(old_weights[i])
        gradient = gradients[i]
        theta_new = theta_old-0.1*gradient #Step value
        new_weights.append(theta_new.tolist())
    return new_weights

def neural_network(lamb,structure,training_set,classes):
    training = prepare_training(training_set,classes)
    weights = initialize_weights(structure)
    costs = []

    for i in range(10): #Stopping criteria
        cost = []
    
        for instance in training:
            res = propagation(weights,lamb,[instance])
            weights = update_weights(weights,res[0])
            cost.append(res[1])

        costs.append(sum(cost))

    return weights,costs

ten_folds_house = construct_10_folds(house_votes_encoded,['0','1'],-1)
ten_folds_wine = construct_10_folds(normalized_wine,['1','2','3'],-1)

def compute(weights,testing_set,classes):
    matrix = []
    costs = []
    for i in range(len(classes)):
        matrix.append([0]*len(classes))
    for input in testing_set:
        prediction = forward_propagation(weights,input)
        costs.append(prediction[0])
        probability = prediction[1].tolist()[0]

        prediction_index = probability.index(max(probability))
        actual_index = input[1].index(1)

        matrix[actual_index][prediction_index] += 1

    accuracy = 0
    precision = 0
    recall = 0
    for i in range(len(classes)):
        accuracy += matrix[i][i]
        recall += (matrix[i][i]/sum(matrix[i]))
        precision_denom = 0
        for j in range(len(classes)):
            precision_denom += matrix[j][i]
        if precision_denom == 0:
            continue
        precision += (matrix[i][i]/precision_denom)
    accuracy = accuracy/len(testing_set)
    recall = recall/len(classes)
    precision = precision/len(classes)
    f1 = 2*((precision*recall)/(precision+recall))
    average_cost = sum(costs)/len(costs)
    return [accuracy, f1, average_cost]

def compute_performance(folds,lamb,structure,classes):
    accuracy = 0
    f1 = 0 
    for fold in folds:
        testing_set = fold
        training_set = []
        for train_fold in folds:
            if train_fold != fold:
                training_set.extend(train_fold)
        
        weights = neural_network(lamb,structure,training_set,classes)[0]
        result = compute(weights,prepare_training(testing_set,classes),classes)
        accuracy += result[0]
        f1 += result[1]

    print("Structure used: " + str(structure))
    print("Regularization paremeter used: " + str(lamb))
    print("Accuracy is: " + str(accuracy/10))
    print("F1 is: " + str(f1/10))

    return accuracy/10, f1/10

# Stopping Criteria for both : Stop after 7 iterations.
# Step cost for both : 0.1
# Performance on the wine dataset:
# compute_performance(ten_folds_wine,0.001,[13,8,3],['1','2','3']) #Accuracy : 0.9281, F1: 0.9440
# compute_performance(ten_folds_wine,0.001,[13,10,8,3],['1','2','3']) #Accuracy : 0.9553, F1: 0.9605
# compute_performance(ten_folds_wine,0.005,[13,10,7,3],['1','2','3']) #Accuracy : 0.9318, F1: 0.9389
# compute_performance(ten_folds_wine,0.001,[13,6,4,3,3],['1','2','3']) #Accuracy : 0.5228, F1: 0.4903
# compute_performance(ten_folds_wine,0.01,[13,12,8,7,5,3],['1','2','3']) #Accuracy : 0.36695, F1 : 0.17819
# compute_performance(ten_folds_wine,0.001,[13,12,8,6,4,3],['1','2','3']) #Accuracy : 0.5162, F1: 0.4215

# Performance on the house votes dataset:
# compute_performance(ten_folds_house,0.001,[48,36,2],['0','1']) #Accuracy: 0.9564, F1: 0.9549
# compute_performance(ten_folds_house,0.001,[48,36,16,2],['0','1']) #Accuracy : 0.9528, F1: 0.9511
# compute_performance(ten_folds_house,0.005,[48,36,15,2],['0','1']) #Accuracy : 0.9149, F1: 0.9193
# compute_performance(ten_folds_house,0.001,[48,12,10,8,2],['0','1']) #Accuracy : 0.94944, F1: 0.9482
# compute_performance(ten_folds_house,0.01,[48,36,19,12,2],['0','1']) #Accuracy : 0.8389, F1: 0.8629
# compute_performance(ten_folds_house,0.001,[48,36,20,10,2],['0','1']) #Accuracy: 0.9522, F1: 0.9520

#Plot J against number of training instances x: 
def plot(test,training,weights,lamb,classes):
    x = []
    J = []
    for i in range(len(training)+1):
        x.append(i)
        if i == 0:
            J.append(compute(weights,test,classes)[2].item())
            continue
        res = propagation(weights,lamb,[training[i-1]])
        weights = update_weights(weights,res[0])
        average_cost = compute(weights,test,classes)[2].item()
        regularized_cost = average_cost + (lamb/(2*len(test))) * sum_squared_thetas(weights)
        J.append(average_cost)
   
    return x,J

training_wine = []
for fold in ten_folds_wine[1:]:
    training_wine.extend(fold)
test_wine = prepare_training(ten_folds_wine[0],['1','2','3'])
training_wine = prepare_training(training_wine,['1','2','3'])
weights_wine = initialize_weights([13,10,8,3])
lamb_wine = 0.001

training_house = []
for fold in ten_folds_house[1:]:
    training_house.extend(fold)
test_house = prepare_training(ten_folds_house[0],['0','1'])
training_house = prepare_training(training_house,['0','1'])
weights_house = initialize_weights([48,36,2])
lamb_house = 0.001

# result_wine = plot(test_wine,training_wine,weights_wine,lamb_wine,['1','2','3'])
# x_wine = result_wine[0]
# J_wine = result_wine[1]
# plt.plot(x_wine,J_wine)
# plt.xlabel("Number of instances trained")
# plt.ylabel("Cost J")
# plt.title("Cost J against number of training instance for wine dataset")
# plt.show()

# result_house = plot(test_house,training_house,weights_house,lamb_house,['0','1'])
# x_house = result_house[0]
# J_house= result_house[1]
# plt.plot(x_house,J_house)
# plt.xlabel("Number of instances trained")
# plt.ylabel("Cost J")
# plt.title("Cost J against number of training instance for house votes dataset")
# plt.show()