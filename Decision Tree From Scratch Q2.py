# importing all the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.utils import shuffle 

dataset = pd.read_csv('tic-tac-toe.data',sep=',',names=['f1','f2','f3','f4','f5','f6','f7','f8','f9','class'])

# creating function of entropy
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True) # get different elements and counts of the target_column
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))]) # calculate the entropy
    return entropy # return the entropy

# calculating information gain of the entire dataset
def InfoGain(data,split_attribute_name,target_name="class"): 
    total_entropy = entropy(data[target_name])  # calculate the total entropy of the target_column
    vals,counts= np.unique(data[split_attribute_name],return_counts=True) # get different elements and counts of the target_column
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))]) #calculating the weighted entropy
    Information_Gain = total_entropy - Weighted_Entropy # calculate information gain
    return Information_Gain # return information gain

# calculating the gain ratio
def split_information(data,split_attribute_name,target_name="class"):
    vals,counts = np.unique(data[split_attribute_name],return_counts=True) # get different elements and counts of the target_column
    split_info = np.sum([(counts[i]/np.sum(counts))*(np.log2(counts[i]/np.sum(counts))) for i in range(len(vals))]) # calculate the split information 
    split_info = -(split_info) 
    IG = InfoGain(data,split_attribute_name,target_name='class') # calculating the information gain for the gain ratio
    if(split_info == 0):
        split_info = 0.000000000000000000000000000000000000000001
    Gain_ratio = IG / split_info # calculating the final gain ratio 
    return Gain_ratio # returning the final gain ratio

# creating ID3 decision tree
def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None,approach = "IG"):
    # First stopping Condition: If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # second stopping condition: If length of the data is zero then, return the mode target feature value
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]

    # if the feature is empty then return the parent node class
    elif len(features) == 0:
        return parent_node_class
    
    # if none of the above stopping condition are true then grow the tree.
    else:
        # parent node class is the default value for the node, which is the mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]

        #find and select the feature which finds the best split for the dataset
        if (approach == "IG"):
            item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        elif (approach == "GR"):
            item_values = [split_information(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        
        # find the index of the best split
        best_feature_index = np.argmax(item_values)
        # best feature at the best feature index for the split        
        best_feature = features[best_feature_index]
        
        # create a tree structure in the dictionary and assign the best feature as the root node.
        tree = {best_feature:{}}
        
        # remove the feature which was selected, means remove best features
        features = [i for i in features if i != best_feature]
        
        # now grow the node under the best feature
        for value in np.unique(data[best_feature]):
            value = value
            
            # split the dataset where we found the best value of the information gain to create a sub_data
            sub_data = data.where(data[best_feature] == value).dropna()

            # now apply recursion on the ID3 function using sub_data
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class,approach=approach)
            
            # then on the best feature apply the subtree
            tree[best_feature][value] = subtree
        return(tree)    

# the function to predict the output of the test data using the tree that we have created
def predict(query,tree,default = 'positive'):
    # we will check for every feature if it exists in the query. 
    # If we find the feature name which exists in the dictionary then go inside the dictionary otherwise return the default value 
    for key in list(query.keys()):
        if key in list(tree.keys()):
            # In here if we will tackel the situation where we come across an unseen query. then we will return the default value
            try:
                result = tree[key][query[key]] 
            except:
                return default
            # save the result of the key which fits the value for the key
            result = tree[key][query[key]]
            # we implemented recursion in here because we have to go through the tree until we find the matching key and value in the our tree dictionary
            if isinstance(result,dict):
                return predict(query,result)
            else:
            # if we found the result then return the result.
                return result

def test(data,tree):
    # create queries simply by removing feature value column from the test dataset
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    # creating an empty dataset where the prediction made by the tree will be stored
    # we will use this to calculate the accuracy of the dataset
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    prediction_accuracy = (np.sum(predicted["predicted"] == data["class"])/len(data))*100
    # returning the accuracy and the predicted dataset
    return prediction_accuracy,predicted

# Confusion matrix function
def confusionmatrix(actual, predicted, normalize = False):
    unique = sorted(set(actual)) # find the unique values from the actual dataset
    matrix = [[0 for _ in unique] for _ in unique] # create a matrix
    imap   = {key: i for i, key in enumerate(unique)}
    # Now let's generate the confusion matrix
    for p, a in zip(predicted, actual): 
        matrix[imap[p]][imap[a]] += 1 # creating confusion matrix for each indexes
    # perform matrix normalization
    if normalize:
        sigma = sum([sum(matrix[imap[i]]) for i in unique])
        matrix = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]
    # return the final confusion matrix
    return matrix

# create a function for performing 10 folds cross validation 10 times
def cross_validation_score(dataset, num_of_folds = 10,IGorGR = 'IG'):
    confusion_expected = []  # an empty list for the actual values for the confusion matrix
    confusion_predicted = []  # an empty list for the predicted values for the confusion matrix
    num_of_rows = dataset.shape[0] # number of rows of the dataset
    fold_size = num_of_rows / num_of_folds # calculating the fold size of the dataset
    fold_size = int(fold_size) # converting it to int
    accuracy = [] # empty array for the accuacy
    tree_number_for_print = 0
    for z in range(10): # 10 times
        m = 0 # m and n are used for performing 10 fold cross validation
        n = 0 
        dataset_copy = dataset
        dataset_copy = shuffle(dataset_copy).reset_index(drop=True) # shuffle the dataset
        for i in range(10): # 10 fold cross validation
            tree_number_for_print = tree_number_for_print + 1
            #print('-----------------------------------------------------------------------Tree Number',tree_number_for_print,'------------------------------------------------------------')
            m = m + fold_size  # back limit for the cross validation split
            training_data = dataset_copy.drop(dataset_copy.index[n:m]).reset_index(drop=True) # training data
            testing_data = dataset_copy.iloc[n:m].reset_index(drop=True) # testing/ validation  data
            confusion_expected.append(testing_data) # for the confusion matrix
            n = n + fold_size   # front limit for the cross validation split
            tree = ID3(training_data,training_data,training_data.columns[:-1],approach=IGorGR) # find the tree using ID3 function
            #pprint(tree) # print the final tree
            prediction_accuracy,predicted = test(testing_data,tree) #getting the prediction accuracy
            #print("The accuracy of the tree number: ",tree_number_for_print," is: ",prediction_accuracy)
            confusion_predicted.append(predicted) # predicted dataset for the confusion matrix
            accuracy.append(prediction_accuracy) # store the accuracy value
    mean = np.mean(accuracy) # calculate the mean of all the accuracy
    variance = np.var(accuracy) # calculate the variance of all the accuracy
    confusion_mat_index = accuracy.index(max(accuracy)) # get the index for the confusion matrix
    con_fun_actual = list(confusion_expected[confusion_mat_index]['class']) # actual list for the confusion matrix
    con_fun_predicted = list(confusion_predicted[confusion_mat_index]['predicted']) # predicted list for the confusion matrix
    confusion_matrix = confusionmatrix(con_fun_actual,con_fun_predicted) # creating the confusion matrix based on the confusion matrix function
    confusion_matrix = np.matrix(confusion_matrix) # convert the confusion matrix to a matrix 
    return mean,variance,confusion_matrix # return the value of the mean, varaince and confusion matrix

# printing the final mean, variance and confusion matrix.
print("The decision tree of tic-tac-toe dataset using Information Gain Approach") 
final_mean_IG,final_variance_IG,final_confusion_matrix_IG = cross_validation_score(dataset,10,'IG')  
print("Final Mean of Information Gain Tree: ",final_mean_IG)
print("Final variance of Information Gain Tree: ", final_variance_IG)
print("Confusion matrix of Information Gain Tree: \n",final_confusion_matrix_IG)

print("The decision tree of tic-tac-toe dataset using Gain Ratio Approach") 
final_mean_GR,final_variance_GR,final_confusion_matrix_GR = cross_validation_score(dataset,10,'GR')
print("Final Mean of Gain Ratio Tree: ",final_mean_GR)
print("Final variance of Gain Ratio Tree: ", final_variance_GR)
print("Confusion matrix of Gain Ratio Tree: \n",final_confusion_matrix_GR)


#############################################################     WINE      #########################################################
# take the wine dataset and pre-process it.
dataset = pd.read_csv('wine.data',sep=',',names=['class','alcohol','malic acid','ash','Alcalinity of ash','magnesium','total phenols','flavanoids','Nonflavanoid phenols','proanthocyanins','color intensity','hue','OD280/OD315 of diluted wines','proline'])
a = dataset['class']
dataset = dataset.drop(["class"],axis=1)
dataset = ((dataset-dataset.min())/(dataset.max()-dataset.min()) * (1 + 1)) - 1
dataset['class'] = a   # dataset with class at the end
# feature names of the list.
feature_names = ['alcohol','malic acid','ash','Alcalinity of ash','magnesium','total phenols','flavanoids','Nonflavanoid phenols','proanthocyanins','color intensity','hue','OD280/OD315 of diluted wines','proline']

# entropy calculation function of the wine dataset
def entropy_wine(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    # entropy calculation of the target_column dataset
    entropy = np.sum(
        [
            (-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
            for i in range(len(elements))
        ]
    )
    # return the entropy
    return entropy

# calculating information gain
def infogain_wine(dataset,leftsplit,rightsplit,total_entropy,split_values):
    total_elements = dataset.shape[0] # total elements of the dataset
    entropy_leftsplit = entropy_wine(leftsplit) # entropy of the leftsplit
    entropy_rightsplit = entropy_wine(rightsplit) # entropy of the rightsplit
    weighted_entropy = (((leftsplit.shape[0]/total_elements)*entropy_leftsplit) + ((rightsplit.shape[0]/total_elements)*entropy_rightsplit)) # calculating the weighted entropy
    final_infogain = total_entropy - weighted_entropy # final information gain
    return final_infogain # returning final information gain

# calculating the gain ratio
def split_information_wine(dataset,leftsplit,rightsplit,total_entropy,split_values):
    total_elements = dataset.shape[0] # length of the dataset
    leftsplit_size = leftsplit.shape[0] # length of the leftsplit of the dataset
    rightsplit_size = rightsplit.shape[0] # length of the rightsplit of the dataset
    if (rightsplit.shape[0] == 0):
        rightsplit_size = 0.000000000000000000000000000000000000000001
    split_information = (((leftsplit_size/total_elements)*(np.log2(leftsplit_size/total_elements)))+((rightsplit_size/total_elements)*(np.log2(rightsplit_size/total_elements)))) # calculating the split information 
    split_information = -split_information
    IG = infogain_wine(dataset,leftsplit,rightsplit,total_entropy,split_values) # calculating the information gain
    if (split_information == 0):
        split_information = 0.000000000000000000000000000000000000000001
    Gain_Ratio = IG / split_information # calculating the gain ratio 
    return Gain_Ratio# returning the gain ratio

# calculating the best split index
def best_split_index(dataset,feature_to_split,feature_names,target_attribute_name = "class",IGGR = 'IG'):
    # finding the total entropy 
    total_entropy = entropy_wine(dataset[target_attribute_name])

    # sorting the dataset for a perticular feature
    sorted_dataset = dataset.sort_values([feature_to_split],axis = 0,ascending = True)

    # find the split_values
    split_feature_val = sorted_dataset[feature_to_split].values

    # finding information gain
    best_info_gain = []
    for i in range(len(split_feature_val)):
        # finding differet splits
        leftsplit = sorted_dataset.loc[sorted_dataset[feature_to_split] <= split_feature_val[i]]
        rightsplit = sorted_dataset.loc[sorted_dataset[feature_to_split] > split_feature_val[i]]
        if (IGGR == 'IG'): # for information gain 
            gain_information = infogain_wine(sorted_dataset,leftsplit['class'],rightsplit['class'],total_entropy,split_feature_val)
        elif (IGGR == 'GR'): # for gain ratio
             gain_information = split_information_wine(sorted_dataset,leftsplit['class'],rightsplit['class'],total_entropy,split_feature_val)
        # storing the best information gain
        best_info_gain.append(gain_information)
    # storing the index of the best split value
    index_of_best_split_val = best_info_gain.index(max(best_info_gain))
    # finding the best information gain 
    max_info_gain = max(best_info_gain)
    return max_info_gain

# creating the left and right split of the input dataset based on the Information Gain
def left_right_split(dataset,best_feature,target_attribute_name = 'class',IGGR = 'IG'):
    # finding the total entropy 
    total_entropy = entropy_wine(dataset[target_attribute_name])
    # sorting the dataset for a perticular feature
    sorted_dataset = dataset.sort_values([best_feature],axis = 0,ascending = True)
    # find the split_values
    split_feature_val = sorted_dataset[best_feature].values
    # finding information gain
    best_info_gain = []
    for i in range(len(split_feature_val)):
        # finding differet splits
        leftsplit = sorted_dataset.loc[sorted_dataset[best_feature] <= split_feature_val[i]]
        rightsplit = sorted_dataset.loc[sorted_dataset[best_feature] > split_feature_val[i]]
        if (IGGR == 'IG'):
            gain_information = infogain_wine(sorted_dataset,leftsplit['class'],rightsplit['class'],total_entropy,split_feature_val)
        elif (IGGR == 'GR'):
             gain_information = split_information_wine(sorted_dataset,leftsplit['class'],rightsplit['class'],total_entropy,split_feature_val)
        best_info_gain.append(gain_information)
    # finding the best split value index
    index_of_best_split_val = best_info_gain.index(max(best_info_gain))
    leftsplit = sorted_dataset.loc[sorted_dataset[best_feature] <= split_feature_val[index_of_best_split_val]] # leftsplit of the dataset
    rightsplit = sorted_dataset.loc[sorted_dataset[best_feature] > split_feature_val[index_of_best_split_val]] # rightsplit of the dataset
    if (IGGR == 'IG'):
        gain_information = infogain_wine(sorted_dataset,leftsplit['class'],rightsplit['class'],total_entropy,split_feature_val)
    elif (IGGR == 'GR'):
        gain_information = split_information_wine(sorted_dataset,leftsplit['class'],rightsplit['class'],total_entropy,split_feature_val)
    return leftsplit,rightsplit,split_feature_val[index_of_best_split_val]

# creating the ID3 decision tree
def ID3_wine(data,originaldata,features,target_attribute_name="class",parent_node_class = None,IGGR = 'IG'):
    # First stopping Condition: If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # second stopping condition: If length of the data is zero then, return the mode target feature value
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]

    # if the feature is empty then return the parent node class
    elif len(features) == 0:
        return parent_node_class

    # if none of the above stopping condition are true then grow the tree.
    else:
        # parent node class is the default value for the node, which is the mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        #find and select the feature which finds the best split for the dataset
        IG = []
        for i in features:
            IG.append(best_split_index(data,i,features,'class',IGGR=IGGR))
        # find the index of the best split
        index_best_feature = IG.index(max(IG))
        # best feature at the best feature index for the split 
        best_feature = features[index_best_feature]
        # create a tree structure in the dictionary and assign the best feature as the root node.
        tree = {best_feature:{}}
        # now grow the node under the best feature 
        # applying recursion on the ID3 tree function
        for i in range(2):
            data_copy = data
            if (i == 0):
                l,tp0,split_value_l = left_right_split(data_copy,best_feature,'class',IGGR=IGGR)
                subtree_l = ID3_wine(l,dataset,features,'class',parent_node_class,IGGR=IGGR)
                split_value_l = str(split_value_l)
                split_value_l = '<= ' + split_value_l
                tree[best_feature][split_value_l] = subtree_l
            elif (i == 1):
                tp0,r,split_value_r = left_right_split(data_copy,best_feature,'class',IGGR=IGGR)
                subtree_r = ID3_wine(r,dataset,features,'class',parent_node_class,IGGR=IGGR)
                split_value_r = str(split_value_r)
                split_value_r = '> ' + split_value_r
                tree[best_feature][split_value_r] = subtree_r
        return(tree)

# the function to predict the output of the test data using the tree that we have created
def predict_wine(queries,tree,default=2):
    # we will check for every feature if it exists in the query. 
    # If we find the feature name which exists in the dictionary then go inside the dictionary otherwise return the default value 
    for key in list(queries.keys()):
        if (type(tree) == dict):
            if key in list(tree.keys()):
                b = tree[key].keys()
                b = list(b)
                z = float(b[0][2:])
                if (queries[key] <= z):   
                    copy_tree = tree
                    copy_tree = copy_tree[key][b[0]]
                    a = predict_wine(queries,copy_tree)
                    return a
                # in here we are trying to predict the output based on the tree generated by us
                elif (queries[key] > z):
                    copy_tree = tree
                    copy_tree = copy_tree[key][b[1]]
                    b = predict_wine(queries,copy_tree)
                    return b
        else:
            return tree

# getting the prediction accuracy
def test_wine(data,tree):
    # create queries simply by removing feature value column from the test dataset
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict_wine(queries[i],tree,1.0) 
    prediction_accuracy = (np.sum(predicted["predicted"] == data["class"])/len(data))*100
    return prediction_accuracy,predicted

# A Simple Confusion Matrix Implementation
def confusionmatrix_wine(actual, predicted, normalize = False):
    unique = sorted(set(actual))
    matrix = [[0 for _ in unique] for _ in unique]
    imap   = {key: i for i, key in enumerate(unique)}
    # Generate Confusion Matrix
    for p, a in zip(predicted, actual):
        matrix[imap[p]][imap[a]] += 1
    # Matrix Normalization
    if normalize:
        sigma = sum([sum(matrix[imap[i]]) for i in unique])
        matrix = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]
    return matrix
 
# create a function for performing 10 folds cross validation 10 times
def cross_validation_score_wine(dataset, num_of_folds = 10,IGGR='IG'):
    confusion_expected = [] # an empty list for the actual values for the confusion matrix
    confusion_predicted = [] # an empty list for the predicted values for the confusion matrix
    num_of_rows = dataset.shape[0] # number of rows in the dataset
    fold_size = num_of_rows / num_of_folds # calculating the fold size
    fold_size = int(fold_size) # converting it to integer
    accuracy = [] # array for the accuracy
    tree_number_for_print = 0 # number of trees
    for z in range(10): # 10 times 10 fold cross validation
        m = 0
        n = 0 
        dataset_copy = dataset # copied dataset
        dataset_copy = shuffle(dataset_copy).reset_index(drop=True)
        for i in range(10): # 10 fold cross validation
            tree_number_for_print = tree_number_for_print + 1
            m = m + fold_size  # back limit
            training_data = dataset_copy.drop(dataset_copy.index[n:m]).reset_index(drop=True)
            testing_data = dataset_copy.iloc[n:m].reset_index(drop=True)
            confusion_expected.append(testing_data)
            n = n + fold_size   # front limit
            ## find tree and it's accuracy
            tree = ID3_wine(training_data,training_data,feature_names,IGGR=IGGR)
            #pprint(tree)
            prediction_accuracy,predicted = test_wine(testing_data,tree)
            #print("The accuracy of the tree number ",tree_number_for_print," is: ",prediction_accuracy)
            confusion_predicted.append(predicted)
            accuracy.append(prediction_accuracy)
    mean = np.mean(accuracy) # finding mean of the accuracy
    variance = np.var(accuracy) # finding variance of the accuracy
    confusion_mat_index = accuracy.index(max(accuracy)) 
    con_fun_actual = list(confusion_expected[confusion_mat_index]['class'])
    con_fun_predicted = list(confusion_predicted[confusion_mat_index]['predicted'])
    confusion_matrix = confusionmatrix_wine(con_fun_actual,con_fun_predicted)
    confusion_matrix = np.matrix(confusion_matrix)
    return mean,variance,confusion_matrix

print("The decision tree of Wine dataset using information gain")
final_mean_IG,final_variance_IG,final_confusion_matrix_IG = cross_validation_score_wine(dataset,num_of_folds=10,IGGR='IG')
print("Final Mean: ",final_mean_IG)
print("Final variance: ", final_variance_IG)
print("Confusion matrix: \n",final_confusion_matrix_IG)

print("The decision tree of Wine dataset using Gain Ratio")
final_mean_GR,final_variance_GR,final_confusion_matrix_GR = cross_validation_score_wine(dataset,num_of_folds=10,IGGR='GR')
print("Final Mean: ",final_mean_GR)
print("Final variance: ", final_variance_GR)
print("Confusion matrix: \n",final_confusion_matrix_GR)