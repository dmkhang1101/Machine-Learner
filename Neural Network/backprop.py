import numpy as np
import math

structure = [1,2,1]
theta1 = [[0.4,0.1],[0.3,0.2]]
theta2 = [[0.7,0.5,0.6]]
weights = [theta1,theta2]
lamb = 0
training_input_1 = [[0.13],[0.9]]
training_input_2 = [[0.42],[0.23]]
training_1 = [training_input_1,training_input_2]

structure2 = [2,4,3,2]
theta1_2 = [[0.42,0.15,0.4],
          [0.72,0.1,0.54],
          [0.01,0.19,0.42],
          [0.3,0.35,0.68]]
theta2_2 = [[0.21,0.67,0.14,0.96,0.87],
            [0.87,0.42,0.2,0.32,0.89],
            [0.03,0.56,0.8,0.69,0.09]]
theta3_2 = [[0.04,0.87,0.42,0.53],
            [0.17,0.1,0.95,0.69]]
weights2 = [theta1_2,theta2_2,theta3_2]
lamb2 = 0.25
training_input_1_2 = [[0.32,0.68],[0.75,0.98]]
training_input_2_2 = [[0.83,0.02],[0.75,0.28]]
training_2 = [training_input_1_2, training_input_2_2]

def sum_squared_thetas(weights):
    sum_squared = lambda x: sum([i**2 for i in x])
    removed_bias = []
    for i in range(len(weights)):
        removed_bias.append(list(map(lambda x: x[1:],weights[i])))
    total = 0
    for i in range(len(removed_bias)):
        total += sum([sum_squared(w) for w in removed_bias[i]])

    return total

def lamb_weight(weights,lamb):
    result = []
    for theta in weights:
        new_theta = []
        for w in theta:
            res = [0]
            rest = list(map(lambda x: x*lamb, w[1:]))
            res.extend(rest)
            new_theta.append(res)
        result.append(new_theta)
    
    return result

def activate(x):
    return (1/(1+np.exp(-x))).round(5)

def cost(actual,predicted):
    return -actual*math.log(predicted)-(1-actual)*math.log(1-predicted)

def forward_propagation(weights,inputs,instance=1):
    print("     Processing training instance "+str(instance))
    print("     Forward propagating the input " + str(inputs[0]))
    current_layer = 1
    layer_outputs = np.append(np.matrix(1.00000),np.matrix(inputs[0]),axis = 1)
    z_values = np.transpose(np.matrix(inputs[0]),axes=None)
    print("         a"+str(current_layer)+": "+str(np.array(layer_outputs)[0])+"\n")
    keep_track_a = []
    keep_track_z = []
    keep_track_w = []

    for i in range(len(weights)):
        current_layer += 1
        thetas = weights[i]
        weight = list(map(lambda x:x[1:],thetas))
        a_values = layer_outputs

        keep_track_a.append(a_values)
        keep_track_z.append(np.vectorize(activate)(z_values.flatten()))
        keep_track_w.append(np.matrix(weight))
        
        z_values = np.matrix(a_values).dot(np.transpose(np.matrix(thetas),axes = None)) 
        print("         z"+str(current_layer)+": "+str(np.array(z_values.flatten())[0]))
        if i != len(weights)-1:
            layer_outputs = np.append(np.matrix(1.00000),(np.vectorize(activate)(z_values.flatten())),axis=1)
        else:
            layer_outputs = np.vectorize(activate)(z_values.flatten())
        print("         a"+str(current_layer)+": "+str(np.array(layer_outputs)[0])+"\n")

    print("         f(x): "+str(np.array(layer_outputs)[0]))
    print("     Predicted output for instance "+str(instance)+": "+str(np.array(layer_outputs[0])[0]))
    print("     Expected output for instance " +str(instance)+": "+str(np.array(inputs[1])))

    j = (np.vectorize(cost)(np.matrix(inputs[1]),np.matrix(layer_outputs[0]))).sum(axis=1)

    print("     Cost, J, associated with instance "+str(instance)+": "+str(np.array(j)[0][0])+"\n")
    return [j,layer_outputs[0],keep_track_a,keep_track_z,keep_track_w]

def back_propagation(weights,inputs,instance,predicted,keep_track_a,keep_track_z,keep_track_w):
    current_layer = len(weights)+1
    expected = np.matrix(inputs[1])
    deltas = []
    curr_delta = np.matrix(predicted - expected)
    deltas.append(curr_delta)
    
    print("     Computing gradients based on training instance "+str(instance))
    print("         delta"+str(current_layer)+": "+str(np.array(curr_delta)[0]))
    for i in range(len(keep_track_z)-1,0,-1):
        current_layer = current_layer-1
        a = keep_track_z[i] 
        theta = keep_track_w[i] 
        curr_delta = np.array(curr_delta.dot(theta))*np.array(a)*np.array(1-a)
        deltas.append(np.matrix(curr_delta))
        print("         delta"+str(current_layer)+": "+str(np.array(curr_delta)[0]))
    print("")

    keep_track_a.reverse()
    grad = [[] for _ in range(len(weights))]
    for i in range(len(deltas)):
        delta = deltas[i] 
        a = keep_track_a[i] 
        gradients = np.transpose(delta,axes = None).dot(a)
        grad[i].append(gradients)
        print("         Gradients of Theta"+str(len(deltas)-i)+" based on training instance "+str(instance))
        for j in range(len(gradients)):
            print("             "+ str(np.array(gradients[j])[0]))
        print("")
    
    return grad


def propagation(weights,lamb,training):
    print("Computing the error/cost, J, of the network")
    resources = []
    average_j = 0
    for inputs in training:
        instance = training.index(inputs)+1
        result = forward_propagation(weights,inputs,instance)
        average_j += result[0]
        resources.append(result[1:])
    average_j = average_j/len(training)
    final_regularized_cost = average_j + (lamb/(2*len(training))) * sum_squared_thetas(weights)
    print("Final (regularized) cost, J, based on the complete training set: "+str(np.array(final_regularized_cost)[0][0])+"\n")

    print("")

    print("Running backpropagation")
    gradients = [[] for _ in range(len(weights))]
    for inputs in training:
        index = training.index(inputs)
        resource = resources[index]
        res = back_propagation(weights,inputs,index+1,resource[0],resource[1],resource[2],resource[3])
        for i in range(len(gradients)):
            gradients[i].append(res[i])

    print("     The entire training set has been processes. Computing the average (regularized) gradients: "+"\n")
    regularized_gradient = []
    lambda_weights = lamb_weight(weights,lamb)
    gradients.reverse()
    for i in range(len(gradients)):
        gradient = gradients[i]
        lambda_weight = np.matrix(lambda_weights[i])
        final = sum(np.array(gradient))
        result = list(map(lambda x: np.array(x) + np.array(lambda_weight), final))
        final_result = list(map(lambda x: np.array(x)*(1/len(training)), result))
        print("         Final regularized gradients of Theta"+str(i+1)+": ")
        for g in final_result[0]:
            print("             " + str(g))
        print("")
        regularized_gradient.append(final_result[0])
    
    return regularized_gradient, final_regularized_cost

# propagation(weights,lamb,training_1)
# propagation(weights2,lamb2,training_2)