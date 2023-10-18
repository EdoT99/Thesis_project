import os
import numpy as np
import pandas as pd
import glob
import random
import numbers
import sys
import matplotlib.pyplot as plt
import datetime
from sklearn import tree
from sklearn.neighbors import KernelDensity
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve

try:
  from imblearn.over_sampling import SMOTE

except ModuleNotFoundError:
  print("module 'SMOTE' is not installed")

#this function, given a dataset, provides number of attributes(columns) and classes
def attr_count(df):
    count = 0
    for col in df.columns:
        if col == 'Species' or col == 'type' or col == 'Class' or col =='Id' or col == 'Class':
            classe = col

            continue
        else: 
            count += 1

    return count, df[classe].value_counts()


def is_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def main():
    if len(sys.argv) < 3:
        print('This script takes 2 command line parameters in sequence: \n - Number of variables \n - CV folds')
        print('if number of variables is integer, takes those attributes; any other alphabetical letter takes all available attributes in dataset')
        user_input = input("Enter Number of variables: ")
        cv_input = input('Enter number of CV splits')

        if is_integer(user_input):
            value = int(user_input)
            print(f'Number of attributes selected: {value}')

        else:
            value = True
            print('Taking all attributes')
        
        n_splits = splits(cv_input)

        
    elif len(sys.argv) == 3:

        if is_integer(sys.argv[1]):
            value = int(sys.argv[1])
            print(f'Number of attributes selected: {value}')
        else:
            value = True
            print('Taking all attributes')

        n_splits = splits(sys.argv[2])
    else:
        print('Wrong parameters passed...run again the script along with an integer for attributes < total attributes OR any alphabetical letter; AND integer for cross validation')
        sys.exit()

    return value, n_splits

#this function asks user the number of splits in CV, default 10 splits!
def splits(split_number):

    if is_integer(split_number):
        n_splits = int(split_number)
        print(f'Number of CV splits:  ->  {n_splits}')
    else:
        n_splits = 10
        print('Erroneus number for cv-folds: Default number of CV splits  ->  10 ')
    return n_splits

 #this function asks the user to choose the model
def choose_classificator(): 
    user_input = int(input('Choose the classificator, press (1) for NB; \n press (2) for DecisionTrees; \n press  (3) for SVM: \n'))
    if user_input == 1:
        classifier = GaussianNB()
    elif user_input == 2:
        classifier = tree.DecisionTreeClassifier()
    elif user_input == 3:
        classifier = SVC(probability=True) 
    print('Model chosen: ', classifier)
    return classifier


def dataset_parser(dataset,attributes):
  df = dataset.iloc[: , 1:]    #only variable + class columns
  classes = df.iloc[:,0].unique()
  encode_class = [[clas,i] for i,clas in enumerate(classes)]
  df.rename(columns={df.columns[0]: 'Class'}, inplace=True)
  print(encode_class)
  # transforming class names to binary
  for classe in encode_class:
    df.loc[df['Class'] == str(classe[0]),'Class'] = str(classe[1])
  labels = df['Class'].astype('int')
  #only attributes (variables/ genes)

  if attributes == True:               #if user does not specify number of attributes, takes all the columns
    vars = [ var for var in df.columns]        
    data = df[vars[1:]].values
    print(len(vars))
  else:                                  #if users sets it, takes only the number of attributes typed
    vars = [ var for var in df.columns[:attributes+1]]  
    data = df[vars[1:]].values

  labels = df[vars[0]].astype('int')

  return data,labels, vars

#This function, search the MAX occurring class, for each minority class it calculates the N of examples to be oversampled,
#Returns a list of tuples: key class, integer examples
def max_occurrence(Y):
    lista = []
    counts = Counter(Y)
    max_occurring = max(counts, key=counts.get)
    #print(max_occurring)
    for key,value in counts.items():
        if key == max_occurring:
            max_value = value
            continue
        else:
            val = max_value - value
            lista.append((key,val))
    return lista

#this function  appends newly generated rows (examples) by KDE to existing original ones;
#Returns X data anD Y labels with new examples oversampled by KDE
def augmenting_df(list_a,X,Y):
  print('Raw training: ',X.shape)
  new_arrays, new_labels = [], []

  for clas in list_a:  #iterating over class symbol and array
      array = clas[1]
      labels = [ clas[0] for i in range(len(array))]
      new_labels.append(labels)
      new_arrays.append(array) 

  labels = np.concatenate(new_labels,axis= 0)
  arrays = np.concatenate(new_arrays, axis = 0)

  new_rows = np.vstack([labels,arrays.T]).T
  original_rows = np.vstack([Y,X.T]).T
  
#appending arryays is generally faster than stacking, do not stack while iterating, it increase memory usage
  oversamp = np.vstack([original_rows,new_rows])
  print('KDE final oversampled (classes + data): ', oversamp.shape)
  X_kde = oversamp[:,1:]
  Y_kde = oversamp[:,0]
  #print(Y_kde)
  #print(X_kde[:,1:])
  
  return  X_kde,Y_kde


#oversampling minority class (all the minority if dataset multiclass) 
def oversamp_KDE_definitive(X,Y):
    
    
    list_samples = []
    #call function to provide minority classes and missing examples for each to match majoirty class
    lista = max_occurrence(Y)
    #for every class in dataframe, oversamples a number of istances to match maj class
    for item in lista:
        
        classe, n_istances = item[0], item[1]
      #selecting minority examples BY INDEX
        
        indices = [i for i, class_value in enumerate(Y) if class_value == classe]
        data = X[indices,:]
        print('Selected minority:',data.shape)
      #creating density estimation
        kde = KernelDensity(kernel = 'gaussian', bandwidth= 'silverman').fit(data)
      #drawing samples from KDE
        examples = kde.sample(n_istances)
        print('KDE; Class: ', classe, ';istances generated:', len(examples))
        print('New examples: ',examples.shape)

      # INSERT THE ROWS HERE, DO NOT USE FUNCTION 
        list_samples.append((classe, examples)) 
        
    X,Y = augmenting_df(list_samples, X,Y)

    return X,Y
#this function oversamples raw data by SMOTE
#returns X data and Y labels with new added examples
def oversamp_SMOTE_definitive(X,Y):
  #matches number of istances in the maj class
  sm = SMOTE(random_state = 42, sampling_strategy = 'not majority') # resampling all mathcing the maj class (excluded)
  # generates new istances
  X_res , y_res = sm.fit_resample(X,Y)
  print('Oversampled with SMOTE:', Counter(y_res))
  
  return X_res,y_res

def cross_validation(X,Y,n_splits,model):
        
    #Provides the data in X,Y as np.array
    skf = StratifiedKFold(n_splits= n_splits, random_state=8, shuffle=True)
    skf.get_n_splits(X,Y)  #X is (n_samples; n_features) y (target variable)

    list_predict_prob_KDE, list_predict_prob_origi, lst_predict_SMOTE = [], [], []   # initilize empty lists for probabilities
    lst_y_test_labels = []
    lst_X_test = []

    for j,(train_index, test_index) in enumerate(skf.split(X,Y)):     #iterate over the number of splits, returns indexes of the train and test set
        
            x_train_fold, x_test_fold = X[train_index], X[test_index]      #selected data for train and test in j-fold
            y_train_fold, y_test_fold = Y[train_index], Y[test_index]      #selected targets
            
            print('\n')
            print('ITERATION --> ',j)
            print('Raw training data: ',x_train_fold.shape)
            print('Raw training labels: ',y_train_fold.shape)
            print('Testing data: ',x_test_fold.shape)

        #NOTA --> creating a dataframe during each iteration, is memeory consuming, therefore it is better to suppplpy difectly data arrays and albels series
            scaler = MinMaxScaler()
        #-----------------------------------------------------------------------
        # 1) STEP
        #call the method OversampKDE on the training partitioning, which
            #- given a list of class examples, oversamples the minority classes
            x1_fold,y1_fold= oversamp_KDE_definitive(x_train_fold,y_train_fold)
        #call the method OversampSMOTE on the training partitioning
            x2_fold,y2_fold = oversamp_SMOTE_definitive(x_train_fold,y_train_fold)

        #------------------------------------------------------------------------
        # 2) STEP
        #fit model on augmented dataset KDE
            X1_scaled = scaler.fit_transform(x1_fold)
            model.fit(X1_scaled,y1_fold)
            # --> test model on test fold; append predict_proba
            y_pred_kde = model.predict_proba(scaler.transform(x_test_fold))

            list_predict_prob_KDE.append(y_pred_kde)
            #----------------------------
            #fit model on augmented training SMOTE
            X2_scaled = scaler.fit_transform(x2_fold)
            model.fit(X2_scaled,y2_fold)
            # test --< model on test fold ; append predict_proba
            y_pred_smote = model.predict_proba(scaler.transform(x_test_fold))
            lst_predict_SMOTE.append(y_pred_smote)
            #-----------------------------------
            #fit model on train set (normal)
            ori_scaled = scaler.fit_transform(x_train_fold)
            model.fit(ori_scaled,y_train_fold)
            # --> test model on test fold; append predict_proba
            y_pred = model.predict_proba(scaler.transform(x_test_fold))
            list_predict_prob_origi.append(y_pred)
            #-------------------------------------
            # 3) STEP
            # append Y_test labels to lst_y_test_labels
            lst_y_test_labels.append(y_test_fold)

    d_proba = {'model_kde':list_predict_prob_KDE,'model_original':list_predict_prob_origi,'model_smote':lst_predict_SMOTE}  #storing overall probabilities 
    y_test_labels = np.concatenate(lst_y_test_labels, axis = 0)
    
    return y_test_labels, d_proba

#This function builds a table storing fpr,tpr and auc for each model
#the table is returned and fed to roc_curve_save
def result_render(d_probabilities,y_test):
  table = pd.DataFrame(columns = ['Classifier','fpr','tpr','auc'])

  for model_name, model_proba in d_probabilities.items():  #iterating over 3 probabilities of 3 models
    proba_ = np.concatenate(model_proba, axis = 0)
    fpr, tpr, _ = roc_curve(y_test,  proba_[:,1])  #probabilities for each class
    auc = roc_auc_score(y_test, proba_[:,1])
    row = {'Classifier': model_name, 'fpr': fpr,'tpr': tpr,'auc': auc}
    table.loc[len(table)] = row

  return table

def roc_curve_save(title_set,table, save_folder, model):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fig = plt.figure(figsize=(8,6))
    table.set_index('Classifier', inplace = True)

    for i in table.index:
        plt.plot(table.loc[i]['fpr'], 
            table.loc[i]['tpr'], 
            label="{}, AUC={:.3f}".format(i, table.loc[i]['auc']))
      
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC - {}'.format(title_set), fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    # Generate a timestamp as part of the filename
    timestamp = datetime.datetime.now().strftime("%Y,%m,%d,%H,%M,%S")

    file_name = os.path.join(save_folder, '{}_{}_{}_png'.format(title_set,model, timestamp))
    plt.savefig(file_name)


if __name__== '__main__':
   
    source_folder = 'C:/Users/acer/Desktop/Binary_dataset_selected'                     #source dataset folder
    save_folder = 'C:/Users/acer/Desktop/Binary_dataset_selected/ROC_curve/All_attributes' #destination ROC curves folder
    all_files = glob.glob(source_folder +'/*.csv')
    
    value, n_splits = main()    #returns either a boolean (True) or a integer value  #returns an integer
    classifier = choose_classificator() 

    for file in all_files:
        if file.endswith(".csv") :
            name = file.split('\\')[1]
            title = name[:-4]
            dataset = pd.read_csv(file)

            n_cols, classes = attr_count(dataset)
            print('--------------------- LOADIG NEW DATASET ------------------ LOADING NEW DATASET --------------------- LOADING NEW DATASET -------------------- LOADING NEW DATASET ---------------')
            print('{}: \n Variables: {}'.format(title,n_cols))
            print('Classes:')
            print(classes)
                

            #returns data.array ad labels.array, attributes only on the basis of col selected, 
            #if not given by the USER, all attributes in database are selected
            X,Y, variables = dataset_parser(dataset,value)
            print('Data: ',X.shape)
            print('Labels: ',Y.shape)
            labels, probabilities_final = cross_validation(X,Y,n_splits,classifier)  #added model choice
            table_results = result_render(probabilities_final,labels) #labels_name !!
            roc_curve_save(title, table_results, save_folder, classifier)        

            # the function outputS a ROC curve as file name the current script and
            # Save the figure as a PNG file
    
           


            






