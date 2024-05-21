from utils import *
import matplotlib.pyplot as plt
from collections import Counter
import math
import random

def count_occurrences(training_set):
    result = Counter()
    for i in range(len(training_set)):
        result.update(Counter(training_set[i]))
    return result

def classify_with_laplace(input,occurrences,vocab,alpha):
    sum_of_frequency = sum(list(occurrences.values()))
    result = 0
    discovered = []
    for word in input:
        if word in discovered:
            continue
        if word not in occurrences.keys():
            pr_word_given_class = math.log((alpha/(sum_of_frequency+alpha*len(vocab))))
        else:
            pr_word_given_class = math.log((occurrences[word]+alpha)/(sum_of_frequency+alpha*len(vocab)))
        result = result + pr_word_given_class
        discovered.append(word)
    return result

def classify_MNB_with_laplace(input,training_set_positive,training_set_negative,occurrences_positive,occurrences_negative,vocab,alpha):
    probability_negative = len(training_set_negative)/(len(training_set_positive) + len(training_set_negative))
    probability_positive = 1 - probability_negative

    probability_input_negative = math.log(probability_negative) + classify_with_laplace(input,occurrences_negative,vocab,alpha)
    probability_input_positive = math.log(probability_positive) + classify_with_laplace(input,occurrences_positive,vocab,alpha)
    
    if probability_input_negative > probability_input_positive:
        return 'Negative'
    elif probability_input_positive > probability_input_negative:
        return 'Positive'
    else:
        return random.choice(['Positive','Negative'])

def compute_quantities_with_laplace(alpha):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in testing_set_positive:
        prediction = classify_MNB_with_laplace(i,training_set_positive,training_set_negative,occurrences_positive,occurrences_negative,vocab,alpha)
        if prediction == 'Positive':
            true_positive += 1
        else:
            false_negative += 1

    for i in testing_set_negative:
        prediction = classify_MNB_with_laplace(i,training_set_positive,training_set_negative,occurrences_positive,occurrences_negative,vocab,alpha)
        if prediction == 'Negative':
            true_negative += 1
        else:
            false_positive += 1

    accuracy = (true_positive+true_negative)/(len(testing_set_positive) + len(testing_set_negative))
    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    cf_matrix = [["Labels         ","Predicted Positive","Predicted Negative"],
                ["Actual Positive","True Positive: " + str(true_positive), "False Negative: " + str(false_negative)],
                ["Actual Negative","False Positive: " + str(false_positive), "True Negative: "  + str(true_negative)]]

    print("Accuracy recorded is: " + str(accuracy))
    print("Precision recorded is: " + str(precision))
    print("Recall recorded is: "  + str(recall))
    print("Confusion Matrix is: ")
    print(cf_matrix[0])
    print(cf_matrix[1])
    print(cf_matrix[2])
    return accuracy,precision,recall,cf_matrix

training_set_positive, training_set_negative, vocab = load_training_set(0.2,0.2)
testing_set_positive, testing_set_negative = load_test_set(0.2,0.2)

occurrences_negative = count_occurrences(training_set_negative)
occurrences_positive = count_occurrences(training_set_positive)

compute_quantities_with_laplace(1)
x = []
y = []
alpha = 0.0001
while alpha <= 1000:
    x.append(alpha)
    y.append(compute_quantities_with_laplace(alpha)[0])
    alpha = alpha*10
plt.plot(x,y,label='Accuracy of MNB with different Alpha values')
plt.xscale('log')
plt.xlabel('Value of Alpha (log scaled)')
plt.ylabel('Accuracy on Test Set')
plt.show()

compute_quantities_with_laplace(10)
