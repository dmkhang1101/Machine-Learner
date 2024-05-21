import math
import random
import matplotlib.pyplot as plt

wine = []
house_votes = []

with open('wine.csv') as csvfile:
    for row in csvfile:
        wine.append(row.strip().split('\t'))

with open('house_votes_84.csv') as csvfile:
    for row in csvfile:
        house_votes.append(row.strip().split(','))

attributes_wine = wine[0][1:]
wine = wine[1:]
attributes_house = house_votes[0][:-1]
attributes_house[0] = 'handicapped-infants'
house_votes = house_votes[1:]
attributes_wine_type = [True]*len(attributes_wine)
attributes_house_type = [False]*len(attributes_house)
attributes_wine_value = ["numerical"]*len(attributes_wine)
attributes_house_value = [['0','1','2']]*len(attributes_house)

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

def is_numerical(attr,attributes,attributes_type):
    return attributes_type[attributes.index(attr)]

def entropy(dataset,classes,index_of_class):
    if len(dataset) == 0:
        return 0
    prob = []
    for i in classes:
        sub_class = list(filter(lambda x: x[index_of_class] == i,dataset))
        if (len(sub_class)/len(dataset)) == 0:
            continue
        prob.append(len(sub_class)/len(dataset))
    entropy = 0
    for p in prob:
        entropy -= p*math.log2(p)
    return entropy

def best_attribute(dataset,attributes,attributes_type,attributes_value,random_attributes,classes,index_of_class):
    original_entropy = entropy(dataset,classes,index_of_class)
    results = []
    thresholds = []
    for attr in random_attributes:
        if is_numerical(attr,attributes,attributes_type):
            index = attributes.index(attr)+1
            copy_of_dataset = dataset.copy()
            copy_of_dataset.sort(key = lambda x: x[index])
            sub_conditions = [] 
            sub_results = []
            for i in range(len(copy_of_dataset)-1):
                sub_conditions.append((float(copy_of_dataset[i][index]) + float(copy_of_dataset[i+1][index]))/2)
            for condition in sub_conditions:
                dataset_less_than_or_equal = list(filter(lambda x: float(x[index]) <= condition,dataset)) 
                dataset_greater = list(filter(lambda x: float(x[index]) > condition,dataset)) 
                average_entropy = entropy(dataset_less_than_or_equal,classes,index_of_class)*(len(dataset_less_than_or_equal)/len(dataset)) + entropy(dataset_greater,classes,index_of_class)*(len(dataset_greater)/len(dataset))
                sub_results.append(original_entropy-average_entropy)
            results.append(max(sub_results))
            thresholds.append(sub_conditions[sub_results.index(max(sub_results))])
        else:
            attribute_values = attributes_value[attributes.index(attr)]
            average_entropy = 0
            for attribute_value in attribute_values:
                sub_dataset = list(filter(lambda x: x[attributes.index(attr)] == attribute_value,dataset))
                average_entropy += entropy(sub_dataset,classes,index_of_class)*(len(sub_dataset)/len(dataset))
            info_gain = original_entropy - average_entropy
            results.append(info_gain)
            thresholds.append(None)
    best_attribute_index = results.index(max(results))
    if is_numerical(random_attributes[best_attribute_index],attributes,attributes_type):
        return [random_attributes[best_attribute_index],attributes.index(random_attributes[best_attribute_index])+1,thresholds[best_attribute_index]]
    else:
        return [random_attributes[best_attribute_index],attributes.index(random_attributes[best_attribute_index])]
        
class Node:
    def __init__(self,type,label):
        self.type = type
        self.label = label
        self.children = {}
    
    def addChild(self,node,choice):
        self.children[choice] = node

    def getChild(self,choice):
        return self.children[choice]
    
    def getType(self):
        return self.type
    
    def getLabel(self):
        return self.label

def majority_class(dataset,classes,index_of_classes):
    sub_data = []
    for i in classes:
        sub_data.append(len(list(filter(lambda x: x[index_of_classes] == i, dataset))))
    max_index = sub_data.index(max(sub_data))
    return classes[max_index]

def stopping_criteria(depth,dataset):
    return len(dataset) <= 20 or depth >= 7

def decision_tree(dataset,attributes,attributes_type,attributes_value,classes,index_of_class,depth):
    if stopping_criteria(depth,dataset):
        return Node('leaf',majority_class(dataset,classes,index_of_class))
    else:
        random_attributes = random.sample(attributes,k=math.ceil(math.log2(len(attributes)))) 
        best = best_attribute(dataset,attributes,attributes_type,attributes_value,random_attributes,classes,index_of_class)
        attribute_to_test = best[0]
        index_of_attr = best[1]
        node = Node('decision',attribute_to_test)
        depth = depth + 1
        if is_numerical(attribute_to_test,attributes,attributes_type):
            threshold_to_split = best[2]
            for i in range(2):
                sub_dataset = list(filter(lambda x : float(x[index_of_attr]) <= threshold_to_split,dataset)) if i == 0 else list(filter(lambda x : float(x[index_of_attr]) > threshold_to_split,dataset)) 
                if len(sub_dataset) == 0:
                    child = Node('leaf',majority_class(dataset,classes,index_of_class))
                else:
                    child = decision_tree(sub_dataset,attributes,attributes_type,attributes_value,classes,index_of_class,depth)
                node.addChild(child,"<= " + str(threshold_to_split)) if i == 0 else node.addChild(child,"> " + str(threshold_to_split))
            return node
        else: 
            attribute_values = attributes_value[attributes.index(attribute_to_test)]
            for i in attribute_values:
                sub_dataset = list(filter(lambda x : x[index_of_attr] == i,dataset))
                if len(sub_dataset) == 0:
                    child = Node('leaf',majority_class(dataset,classes,index_of_class))
                else:
                    child = decision_tree(sub_dataset,attributes,attributes_type,attributes_value,classes,index_of_class,depth)
                node.addChild(child,i)
            return node

def predict_decision_tree(input,tree,attributes,attributes_type):
    while tree.getType() != 'leaf':
        label = tree.getLabel()
        if is_numerical(label,attributes,attributes_type):
            index_of_attr = attributes.index(label)+1
            data = float(input[index_of_attr])
            threshold = float(list(tree.children.keys())[0].split(" ")[1])
            if data <= threshold:
                tree = tree.getChild('<= ' + str(threshold))
            else:
                tree = tree.getChild('> ' + str(threshold))
        else: 
            index_of_attr = attributes.index(label)
            choice = input[index_of_attr]
            tree = tree.getChild(choice)
    return tree.getLabel()

def compute(testing_set,forest,attributes,attributes_type,classes,index_of_class):
    matrix = []
    for i in range(len(classes)):
        matrix.append([0]*len(classes))
    for input in testing_set:
        result = []
        for tree in forest:
            prediction = predict_decision_tree(input,tree,attributes,attributes_type)
            result.append(prediction)
        final_prediction = max(set(result),key = result.count)
        prediction_index = classes.index(final_prediction)
        actual = input[index_of_class]
        actual_index = classes.index(actual)
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
    return [accuracy, precision, recall, f1]

def random_forest(ntree,folds,attributes,attributes_type,attributes_value,classes,index_of_class):
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    for fold in folds:
        testing_set = fold
        training_set = []
        for train_fold in folds:
            if train_fold != fold:
                training_set.extend(train_fold)
        forest = []
        for i in range(ntree):
            bootstrap = random.choices(training_set, k=len(training_set))
            forest.append(decision_tree(bootstrap,attributes.copy(),attributes_type,attributes_value,classes,index_of_class,0))
        result = compute(testing_set,forest,attributes,attributes_type,classes,index_of_class)
        accuracy += result[0]
        precision += result[1]
        recall += result[2]
        f1 += result[3]
    return  accuracy/10, precision/10, recall/10, f1/10

ten_folds_house = construct_10_folds(house_votes,['0','1'],-1)
ten_folds_wine = construct_10_folds(wine,['1','2','3'],0)

ntree = [1,5,10,20,30,40,50]

resulting_accuracy_house = []
resulting_precision_house = []
resulting_recall_house = []
resulting_f1_house = []

resulting_accuracy_wine = []
resulting_precision_wine = []
resulting_recall_wine = []
resulting_f1_wine = []

for value in ntree:
    result_house = random_forest(value,ten_folds_house,attributes_house,attributes_house_type,attributes_house_value,['0','1'],-1)
    resulting_accuracy_house.append(result_house[0])
    resulting_precision_house.append(result_house[1])
    resulting_recall_house.append(result_house[2])
    resulting_f1_house.append(result_house[3])
    print("accuracy house " + str(result_house[0]))

    result_wine = random_forest(value,ten_folds_wine,attributes_wine,attributes_wine_type,attributes_wine_value,['1','2','3'],0)
    resulting_accuracy_wine.append(result_wine[0])
    resulting_precision_wine.append(result_wine[1])
    resulting_recall_wine.append(result_wine[2])
    resulting_f1_wine.append(result_wine[3])
    print("accuracy wine " + str(result_wine[0]))
    print("finish " + str(value) + " trees" )

figure, axis = plt.subplots(nrows = 2, ncols = 4, figsize = (14,10))

axis[0,0].plot(ntree,resulting_accuracy_house)
axis[0,0].set_title("Accuracy of House Votes 84 Dataset")
axis[0,0].set_xlabel("Value of Ntree")
axis[0,0].set_ylabel("Accuracy of House Votes")

axis[0,1].plot(ntree,resulting_precision_house)
axis[0,1].set_title("Precision of House Votes 84 Dataset")
axis[0,1].set_xlabel("Value of Ntree")
axis[0,1].set_ylabel("Precision of House Votes")

axis[0,2].plot(ntree,resulting_recall_house)
axis[0,2].set_title("Recall of House Votes 84 Dataset")
axis[0,2].set_xlabel("Value of Ntree")
axis[0,2].set_ylabel("Recall of House Votes")

axis[0,3].plot(ntree,resulting_f1_house)
axis[0,3].set_title("F1 of House Votes 84 Dataset")
axis[0,3].set_xlabel("Value of Ntree")
axis[0,3].set_ylabel("F1 of House Votes")

axis[1,0].plot(ntree,resulting_accuracy_wine)
axis[1,0].set_title("Accuracy of Wine Dataset")
axis[1,0].set_xlabel("Value of Ntree")
axis[1,0].set_ylabel("Accuracy of Wine")

axis[1,1].plot(ntree,resulting_precision_wine)
axis[1,1].set_title("Precision of Wine Dataset")
axis[1,1].set_xlabel("Value of Ntree")
axis[1,1].set_ylabel("Precision of Wine")

axis[1,2].plot(ntree,resulting_recall_wine)
axis[1,2].set_title("Recall of Wine Dataset")
axis[1,2].set_xlabel("Value of Ntree")
axis[1,2].set_ylabel("Recall of Wine")

axis[1,3].plot(ntree,resulting_f1_wine)
axis[1,3].set_title("F1 of Wine Dataset")
axis[1,3].set_xlabel("Value of Ntree")
axis[1,3].set_ylabel("F1 of Wine")

figure.tight_layout()
plt.show()