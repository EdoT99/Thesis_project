import pandas as pd 
import numpy as np 
import sklearn
from sklearn.neighbors import KernelDensity
from collections import Counter
from datetime import datetime
import glob
import os
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
from statistics import mean
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef 
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from imcp import mcp_curve, mcp_score, plot_mcp_curve, imcp_curve, imcp_score, plot_imcp_curve, plot_curve


def dataset_parser_def(df):

  count = 0
  for col in df.columns:
        if col == 'species' or col == 'type' or col == 'Class' or col =='Id' or col == 'class':
            class_col = col
            continue
        else: 
            count += 1
  print('Number of attributes:',count)
  value_counts = df[class_col].value_counts()
  for index, value in value_counts.items():
     print(f'{index}: {value}')


  if class_col == 'type':        #CUMIDA
     df = df.iloc[: , 1:]
     data = df.iloc[:, 2:].values
     #print('DATA\n',data)

  elif class_col == 'class' or class_col == 'species':      #IRIS
     data = df.iloc[:, :4].values
     #print('DATA\n',data)

  elif class_col == 'Class':      #233_features data
      data = df.iloc[:,1:].values
      #print('DATA\n',data)

  print('data original:',data.shape,'\n')
  classi = df[class_col].unique()
  class_sorted = sorted(classi)

  class_mapping = {label: idx for idx, label in enumerate(class_sorted)}
  #print("Column names:", df.columns)
  #print('Class sorted',class_sorted)
  # Print the mapping of class labels to numeric values
  print("Class mapping:", class_mapping)

  # Transform class labels to numeric values in the DataFrame
  df[class_col] = df[class_col].map(class_mapping)
  labels = df[class_col].astype(int)

  
  #print('classes:',labels.shape)

  return data,labels


def occurrence_def(Y):
# Count occurrences of each element
    counts = Counter(Y)
    # Find the most frequently occurring element and its count
    max_occurring = max(counts, key=counts.get)
    min_occurring = min(counts,key=counts.get)
    max_value = counts[max_occurring]
    min_value = counts[min_occurring]
    IR = {}
    lista = []
    for classe,count in counts.items():
        value = max_value-count
        lista.append((classe,value))
        imb_ratio = count/min_value
        IR[classe] = round(imb_ratio,3)
        
    print(f'Imbalance Ratio (IR)  {IR}')

    return sorted(lista, key=lambda x: x[1], reverse=False), sorted(IR.items(), key=lambda pair: pair[1], reverse=True), max_value

'''
def check_range(array, ranges, label):
    if label not in ranges:
        ranges[label] = {}
    for i, col in enumerate(array.T):
        ranges[label][i] = {'min': float(col.min()), 'max': float(col.max()), 'mean': float(col.mean())}
    print('Sampled:', label, '\n', ranges)
        
    return ranges
'''


def kde_sampler(kernel,data,n_istances):
    kde = KernelDensity(kernel = kernel, algorithm = 'ball_tree', bandwidth = 'silverman').fit(data)
    examples = kde.sample(n_istances, random_state=0)

    return examples


def stacking_def(original_array,label,array):
    #print('New_data:',array.shape)
    labels = [ label for i in range(len(array))]
    #print('New labels:',len(labels))
    rows = np.vstack([labels,array.T]).T

    return np.vstack([original_array,rows])



def oversamp_KDE_definitive(X,Y):
    #ranges_kde, ranges_original = {},{} 
    #ranges_original = check_range(data,ranges_original,classe)
    #call function to provide minority classes and missing examples for each to match majoirty class
    lista, IR, max_value = occurrence_def(Y)
    print(f'Imbalance Ratio (IR)  {IR}')
    imbalance =  False
    for item in IR:
        if item[1] > 1:
                imbalance =  True
                break
    
    dataset = np.vstack([Y,X.T]).T
    print('Original dataset', dataset.shape)
    #if imbalance
    if imbalance:
        stacking_array =  dataset
        #print('There is imbalance')
        print('\n')
        #for every class in dataframe, oversamples a number of istances to match maj class
 
        print(f'Majority class num of istances: {max_value}')
        for item in lista[1:]:
          classe, n_istances = item[0], item[1]

        #selecting minority examples BY INDEX
          indices = [i for i, class_value in enumerate(Y) if class_value == classe]
          data = X[indices,:]
          print('Selected minority:',data.shape)

        #creating density estimation and sampling new data
          examples = kde_sampler('gaussian',data,n_istances)
          stacking_array = stacking_def(stacking_array,classe,examples)
          '''
          print('KDE - Class: ', classe, ';NEW istances generated:', len(examples))
          print('New exampels: ',examples.shape)
          #print('Sampled EXAMPLES:\n',examples)     
          print('Updated dataframe:',stacking_array.shape,'\n')
          '''
          
    #if NO imbalance
    else:
        stacking_array =  dataset
        #n_istances = max_value
        n_istances = 0
        print(f'No class imbalance detected, proceeding to sample {n_istances} for each class')
        print('\n')
        # for every class in dataframe, proceeds to add n_istances to each class
        for item in lista:
          classe = item[0]
        #selecting minority examples BY INDEX
          indices = [i for i, class_value in enumerate(Y) if class_value == classe] ##!!!!!!
          data = X[indices,:]
          print('Selected minority:',data.shape)

        #creating density estimation and sampling new data
          examples = kde_sampler('gaussian',data,n_istances)
          stacking_array = stacking_def(stacking_array,classe,examples)
          '''
          print('KDE - Class: ', classe, ';NEW istances generated:', len(examples))
          print('New exampels: ',examples.shape)
          #print('Sampled EXAMPLES:\n',examples)
          print('Updated dataframe',stacking_array.shape,'\n')
          '''
          

    #ranges_kde = check_range(examples,ranges_kde, classe)
    print('FINAL OVERSAMPLED dataframe:',stacking_array.shape)
    #tupling class num to new examples
    x = stacking_array[:,1:]
    y = stacking_array[:,0].astype(int)
    print('Oversampled with KDE:', Counter(y))

    return x,y



def oversamp_SMOTE_definitive(X,Y):
  #matches number of istances in the maj class
  sm = SMOTE(random_state = 0, sampling_strategy = 'not majority') # resampling all mathcing the maj class (excluded)
  # generates new istances
  x , y = sm.fit_resample(X,Y)
  print('Oversampled with SMOTE:', Counter(y))
  
  return x,y

def metric_scores(y_true, y_pred, averages):

    f1_scores = f1_score(y_true, y_pred, average =averages)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true,y_pred,average=averages,zero_division = 0)
    recall = recall_score(y_true, y_pred,average=averages,zero_division = 0)
    acc = accuracy_score(y_true,y_pred)

    return mcc, f1_scores, prec,recall, acc



def parser_metrics(dataset,dict):
  list_metrics = []
  for key in dict:
      for val in dict[key]:
          list_metrics.append(val)
  
  metrics_row = {'Dataset': dataset,'Acc(raw)':list_metrics[0],'Precision(raw)':list_metrics[1],'Recall(raw)':list_metrics[2],'MCC(raw)':list_metrics[3],'f1(raw)':list_metrics[4],'Acc(smote)':list_metrics[5],'Precision(smote)':list_metrics[6],
                'Recall(smote)':list_metrics[7],'MCC(smote)':list_metrics[8],'f1(smote)':list_metrics[9],'Acc(kde)':list_metrics[10],'Precision(kde)':list_metrics[11],'Recall(kde)':list_metrics[12],'MCC(kde)':list_metrics[13],'f1(kde)':list_metrics[14] }

  return  metrics_row

def cross_validation(X,Y,n_splits,model,averages):
    
    print(f'CROSS-VALIDATION: {n_splits} folds ; {model} algorithm')
    #Provides the data in X,Y as np.array
    skf = StratifiedKFold(n_splits= n_splits, random_state=0, shuffle=True)
    skf.get_n_splits(X,Y)  #X is (n_samples; n_features) y (target variable)

    predicted_KDE, predicted_origi, predicted_SMOTE = [], [], []   # initilize empty lists for probabilities
    list_acc_ori, list_acc_smote, list_acc_kde = [], [], []
    lst_y_test_labels = []
    pre_ori_lst, pre_smote_lst,pre_kde_lst =[],[],[]
    rec_ori_lst, rec_smote_lst, rec_kde_lst = [], [], []
    list_mcc_smote, list_mcc_kde, list_mcc_ori = [], [], []
    list_f1_kde, list_f1_smote, list_f1_ori = [], [], []

    for j,(train_index, test_index) in enumerate(skf.split(X,Y)):     #iterate over the number of splits, returns indexes of the train and test set
        
            x_train_fold, x_test_fold = X[train_index], X[test_index]      #selected data for train and test in j-fold
            y_train_fold, y_test_fold = Y[train_index], Y[test_index]      #selected targets
            
            print('\n')
            print('ITERATION --> ',j)
    
        #-----------------------------------------------------------------------
        # 1) STEP
        #call the method OversampKDE on the training partition, which
            #- given a list of class examples, oversamples the minority classes
            x1_fold,y1_fold = oversamp_KDE_definitive(x_train_fold,y_train_fold)
        #call the method OversampSMOTE on the training partitioning
            x2_fold,y2_fold = oversamp_SMOTE_definitive(x_train_fold,y_train_fold)

        #------------------------------------------------------------------------
        # 2) STEP
        #fit model on augmented dataset KDE
            model.fit(x1_fold,y1_fold)
            # --> test model on test fold; append predict_proba
            y_proba_kde = model.predict_proba(x_test_fold)
            y_pred_kde = model.predict(x_test_fold)
            predicted_KDE.append(y_proba_kde)

            mcc_kde,f1_kde,prec_kde,rec_kde,acc_kde = metric_scores(y_test_fold,y_pred_kde,averages)
            list_acc_kde.append(acc_kde)
            list_mcc_kde.append(mcc_kde)
            list_f1_kde.append(f1_kde)
            pre_kde_lst.append(prec_kde)
            rec_kde_lst.append(rec_kde)

            #----------------------------
            #fit model on augmented training SMOTE
            model.fit(x2_fold,y2_fold)
            # test --< model on test fold ; append predict_proba
          
            y_proba_smote = model.predict_proba(x_test_fold)
            y_pred_smote = model.predict(x_test_fold)
            predicted_SMOTE.append(y_proba_smote)

            mcc_smote,f1_smote,prec_smote,rec_smote,acc_smote = metric_scores(y_test_fold,y_pred_smote,averages)
            list_f1_smote.append(f1_smote)
            list_acc_smote.append(acc_smote)
            list_mcc_smote.append(mcc_smote)
            pre_smote_lst.append(prec_smote)
            rec_smote_lst.append(rec_smote)
            
            #-----------------------------------
            #fit model on train set (normal)
            #x_train_scaled = scaler.fit_transform(x_train_fold)
            model.fit(x_train_fold,y_train_fold)

            # --> test model on test fold; append predict_proba
            y_proba_ori = model.predict_proba(x_test_fold)
            y_pred_ori= model.predict(x_test_fold)
            predicted_origi.append(y_proba_ori)

            mcc_ori,f1_ori,prec_ori,rec_ori,acc_ori = metric_scores(y_test_fold,y_pred_ori,averages)
            list_acc_ori.append(acc_ori)
            list_mcc_ori.append(mcc_ori)
            list_f1_ori.append(f1_ori)
            pre_ori_lst.append(prec_ori)
            rec_ori_lst.append(rec_ori)

            #-------------------------------------
            # 3) STEP
            # append Y_test labels to lst_y_test_labels
            lst_y_test_labels.append(y_test_fold)

   
    metrics_overall = {'model_original': [mean(list_acc_ori),mean(pre_ori_lst),mean(rec_ori_lst),mean(list_mcc_ori),mean(list_f1_ori)],'model_smote':[mean(list_acc_smote),mean(pre_smote_lst),mean(rec_smote_lst),mean(list_mcc_smote),mean(list_f1_smote)],'model_kde':[mean(list_acc_kde),mean(pre_kde_lst),mean(rec_kde_lst),mean(list_mcc_kde),mean(list_f1_kde)]}
    metrics_precision = {'model_original':mean(pre_ori_lst),'model_smote':mean(pre_smote_lst),'model_kde':mean(pre_kde_lst)}
    metrics_recall = {'model_original':mean(rec_ori_lst),'model_smote':mean(rec_smote_lst),'model_kde':mean(rec_kde_lst)}
    #averaged accuracies over k-folds
    d_acc = {'model_original' : mean(list_acc_ori),'model_smote':mean(list_acc_smote),'model_kde':mean(list_acc_kde)}
    #aggregating k-fold probability estimates
    d_proba = {'model_original':np.concatenate(predicted_origi,axis = 0),'model_smote':np.concatenate(predicted_SMOTE,axis = 0),'model_kde':np.concatenate(predicted_KDE,axis = 0)}  
    y_test_labels = np.concatenate(lst_y_test_labels, axis = 0)
    
    return y_test_labels, d_proba, d_acc, metrics_recall, metrics_precision, metrics_overall


def result_render_binary(d_probabilities,d_accuracies,precision_d,recall_d,y_test,title_set,save_imcp,classifier):
  table_binary = pd.DataFrame(columns = ['Classifier','fpr','tpr','auc'])
  list_metrics, imcp_scores, auc_scores, accuracy, precisions, recalls = [],[],[],[],[],[]
  metrics_add = []
  scores = {}
  for model_name, y_pred in d_probabilities.items():
    scores[model_name] = y_pred
    #print('Predicted scores :',y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_pred[:,1])  #probabilities for each class  
    auc_binary = roc_auc_score(y_test,y_pred[:,1])
    imcp_scores.append(float(imcp_score(y_test,y_pred,abs_tolerance=0.0000001)))
    auc_scores.append(float(auc_binary))
    row = {'Classifier': model_name, 'fpr': fpr,'tpr': tpr,'auc': auc_binary}          
    table_binary.loc[len(table_binary)] = row  

  #output_path = os.path.join(save_imcp, '{}_{}_imcp'.format(title_set,classifier))
  #plot_imcp_curve(y_test,scores,abs_tolerance=0.0000001,output_fig_path=output_path)

  for rec in recall_d.values():
     recalls.append(rec)
  for pre in precision_d.values():
     precisions.append(pre)
  
  for prec,reca in zip(precisions,recalls):
     metrics_add.append(prec)
     metrics_add.append(reca)
     

  for acc in d_accuracies.values():  #appending average accuracies (over 10)  for raw,smote,kde to list:  3 accuracies
    accuracy.append(acc)
  #print('ACC:',accuracy)
  #print('AUC:',auc_scores)
  #print('IMCP:',imcp_scores)
  for acc_score,auc_s,imcp_s in zip(accuracy,auc_scores,imcp_scores):  #creating list containing acc,auc,imcp for each method sequentially
    list_metrics.append(float(f'{acc_score:.4f}'))
    list_metrics.append(float(f'{auc_s:.4f}'))
    list_metrics.append(float(f'{imcp_s:.4f}'))
  #print(list_metrics)
  return table_binary, list_metrics, metrics_add


def result_render_multiclass(d_probabilities,d_accuracies,precision_d,recall_d,y_test,title_set,save_folder_imcp,classifier):
  table_multi_micro = pd.DataFrame(columns = ['Classifier','fpr','tpr','auc'])
  table_multi_macro = pd.DataFrame(columns = ['Classifier','fpr','tpr','auc'])
  imcp_scores, auc_micro,auc_macro, accuracy= [], [], [],[]
  list_metrics = []
  metrics_add, precisions, recalls = [],[], []
  classes = sorted(list(np.unique(y_test)))
  n_classes = len(np.unique(y_test))

  y_test_binarize = label_binarize(y_test, classes=classes)
  #print('Binarized:',y_test_binarize)
  #y_test_binarize = label_binarize(y_test, classes=np.arange(classes))

  scores = {}

  for model_name, model_proba in d_probabilities.items():  #iterating over 3 probabilities of 3 models
    y_pred = model_proba
    #print('Predicted scores :',model_proba.shape)
    scores[model_name] = model_proba
    

    fpr ,tpr ,roc_auc ,thresholds = dict(), dict(), dict() ,dict() 
    # micro-average
    for i in range(n_classes):
      fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_binarize[:, i], y_pred[:, i], drop_intermediate=True)
      roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarize.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #aggregates all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    #mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    #print('All FPR:',all_fpr)
    tpr["macro"] = mean_tpr
    #print(mean_tpr)
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # storing average-micro fpr, tpr, auc for each method (original,smote,kde)
    row_micro = {'Classifier': model_name, 'fpr': fpr['micro'],'tpr':tpr['micro'],'auc':roc_auc['micro']}
    #row_micro = {'Classifier': model_name, 'fpr': fpr['micro'],'tpr':tpr['micro'],'auc':roc_auc['micro']}
    table_multi_micro.loc[len(table_multi_micro)] = row_micro

    # storing average-macro fpr, tpr, auc for each method (original,smote,kde)
    row_macro = {'Classifier': model_name,'fpr':fpr['macro'],'tpr':tpr['macro'],'auc':roc_auc['macro']}
    #row_macro = {'Classifier': model_name,'fpr':fpr['macro'],'tpr':tpr['macro'],'auc':roc_auc['macro']}
    table_multi_macro.loc[len(table_multi_macro)] = row_macro

    #appending AUC(ROC) for micro and macro average
    auc_micro.append(roc_auc_score(y_test, y_pred, multi_class='ovr',average = 'micro' ))
    auc_macro.append(roc_auc_score(y_test, y_pred, multi_class='ovr',average = 'macro' ))
    #appending aimcp for (raw,smote,kde)
    imcp_scores.append(imcp_score(y_test,y_pred,abs_tolerance=0.0000001))                                                     
    #plot_imcp_curve
    #output_path = os.path.join(save_folder_imcp, '{}_{}_imcp.png'.format(title_set,classifier))
  #plot_imcp_curve(y_test,scores,abs_tolerance=0.0000001,output_fig_path=output_path)

  for acc in d_accuracies.values():  #appending average accuracies (over 10)  for raw,smote,kde to list:  3 accuracies
      accuracy.append(acc)
  for acc_score,auc_micro,auc_macro, imcp_s in zip(accuracy,auc_micro,auc_macro,imcp_scores):  #creating list containing acc,auc,imcp for each method sequentially
      list_metrics.append(float(f'{acc_score:.4f}'))
      list_metrics.append(float(f'{auc_micro:.4f}'))
      list_metrics.append(float(f'{auc_macro:.4f}')) #auc micro  #inserted new auc !! macro
      list_metrics.append(float(f'{imcp_s:.4f}'))
  
  for rec in recall_d.values():
     recalls.append(rec)
  for pre in precision_d.values():
     precisions.append(pre)
  
  for prec,reca in zip(precisions,recalls):
     metrics_add.append(prec)
     metrics_add.append(reca)

  return list_metrics, table_multi_macro, table_multi_micro, metrics_add



# newly added function, plots in 2 separate files --> average-micro and average-macro Multi class ROC curves for all 3 methods
def multi_class_roc_save(title_set,table,model,save_folder,name = str()):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    #macro
    #plt.figure(dpi=600)
    plt.figure(figsize=(8,6))
    table.set_index('Classifier', inplace = True)
    #colors = ['navy','orange','green']
    colors = ['royalblue','saddlebrown','lightcoral']
    for i,color in zip(table.index,colors):
      plt.plot(table.loc[i]['fpr'], 
            table.loc[i]['tpr'], 
            label="{}, AUC={:.3f}".format(i, table.loc[i]['auc']),color = color)
    plt.xlim([-0.005, 1.01])
    plt.ylim([-0.005, 1.01])
    plt.xticks([i/10.0 for i in range(11)])
    plt.yticks([i/10.0 for i in range(11)])
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title('{}-average ROC curve  - {}'.format(name, title_set), fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    plt.grid(linestyle='--', linewidth=0.5)

    file_name_macro = os.path.join(save_folder, '{}_{}_{}'.format(title_set,model,name))
    plt.savefig(file_name_macro)
    plt.close()


def roc_curve_save(title_set,table, save_folder, model):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.figure(figsize=(8,6))
    table.set_index('Classifier', inplace = True)
    colors = ['royalblue','saddlebrown','lightcoral']
    for i,color in zip(table.index,colors):
        plt.plot(table.loc[i]['fpr'], 
            table.loc[i]['tpr'], 
            label="{}, AUC={:.3f}".format(i, table.loc[i]['auc']), color = color)
    plt.xlim([-0.005, 1.01])
    plt.ylim([-0.005, 1.01])
    plt.xticks([i/10.0 for i in range(11)])
    plt.yticks([i/10.0 for i in range(11)])
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title('ROC - {}'.format(title_set), fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    plt.grid(linestyle='--', linewidth=0.5)
    file_name = os.path.join(save_folder, '{}_{}_png'.format(title_set,model))
    plt.savefig(file_name)
    plt.close()

if __name__== '__main__':
   

   cv_splits = 10
   classifier = GaussianNB()
   #classifier = RandomForestClassifier(random_state=0)
   save_folder = '../experiment/results'
   save_folder_roc_micro = '../experiment/results/roc_curves/micro'
   save_folder_roc_macro = '../experiment/results/roc_curves/macro'
   save_folder_roc_binary = '../experiment/results/roc_curves/binary'   
   save_folder_imcp_m = '../experiment/results/imcp_curves/multiclass'
   save_folder_imcp_b = '../experiment/results/imcp_curves/binary'

   final_table_multiclass= pd.DataFrame(columns = ['Dataset','Acc(raw)','AUC(raw micro)','AUC(raw macro)','IMCP(raw)','Acc(smote)','AUC(smote micro)','AUC(smote macro)','IMCP(smote)','Acc(KDE)','AUC(KDE micro)','AUC(KDE macro)','IMCP(KDE)'])
   final_table_binary = pd.DataFrame(columns= ['Dataset','Acc(raw)','AUC(raw)','IMCP(raw)','Acc(smote)','AUC(smote)','IMCP(smote)','Acc(KDE)','AUC(KDE)','IMCP(KDE)'])
   precision_recall_binary = pd.DataFrame(columns = ['Dataset','Precision(raw)','Recall(raw)','Precision(smote)','Recall(smote)','Precision(KDE)','Recall(KDE)'])
   dataframe_metrics = pd.DataFrame(columns = ['Dataset','Acc(raw)','Precision(raw)','Recall(raw)','MCC(raw)','f1(raw)','Acc(smote)','Precision(smote)','Recall(smote)','MCC(smote)','f1(smote)','Acc(kde)','Precision(kde)','Recall(kde)','MCC(kde)','f1(kde)'])


   source_folder = "../experiment/UCI/numeric"
   all_files = glob.glob(os.path.join(source_folder, '*.csv'))
   all_files.sort()
   print(all_files)

   startTime = datetime.now()


   for file in all_files:
        if file.endswith(".csv") :
            name = file.split('/')[-1]
            title = name[:-4]
            #table_name = name[name.find('_')+1:-4]
            table_name = title
            print('─' * 100)
            print("NEW DATASET: ",title)
            dataset = pd.read_csv(file)
            print(dataset.head())
            
            x,y = dataset_parser_def(dataset)
            classes = len(np.unique(y))

            if classes == 2:
               averages = 'binary'
            else:
               averages = 'macro'
               #averages = 'micro'
               
            #oversampled_dataframe = oversamp_KDE_definitive(x,y)
            labels, probabilities_final, accuracies, precision, recall , dict_metrics= cross_validation(x,y,cv_splits,classifier,averages) 


            endTime = datetime.now()            
            result_date = endTime.strftime("%Y%m%d_%H%M")
            
            if classes ==  2: 

              table_results, list_metrics, metrics_add = result_render_binary(probabilities_final,accuracies,precision,recall,labels,title,save_folder_imcp_b,classifier)
              #roc_curve_save(title,table_results,save_folder_roc_binary,classifier)
              row_table_b = {'Dataset' : table_name ,'Acc(raw)':list_metrics[0],'AUC(raw)':list_metrics[1],'IMCP(raw)':list_metrics[2],'Acc(smote)':list_metrics[3],'AUC(smote)':list_metrics[4],'IMCP(smote)': list_metrics[5],'Acc(KDE)':list_metrics[6],'AUC(KDE)':list_metrics[7],'IMCP(KDE)':list_metrics[8]}
              #precision_recall_row= {'Dataset':table_name,'Precision(raw)': metrics_add[0], 'Recall(raw)': metrics_add[1], 'Precision(smote)':  metrics_add[2], 'Recall(smote)': metrics_add[3],'Precision(KDE)':metrics_add[4],'Recall(KDE)':metrics_add[5]}
              final_table_binary.loc[len(final_table_binary)] = row_table_b
              #precision_recall_binary.loc[len(precision_recall_binary)] = precision_recall_row
              full_path = f'{save_folder}/{result_date}_metrics_binary_datasets.csv'
              final_table_binary.to_csv(full_path, index=False, sep=';', decimal=',')
              #final_table_binary.to_csv('../experiment/results/{result_date}_metrics_binary_datasets.csv',index = False)
              #precision_recall_binary.to_csv('../experiment/results/{result_date}_metrics_additional.csv',index = False)
              print('\n')
              print(final_table_binary)

            else:
              
              list_metrics, table_multi_macro, table_multi_micro ,metrics_add = result_render_multiclass(probabilities_final,accuracies,precision,recall,labels,title,save_folder_imcp_m,classifier) #labels_name !!
              #multi_class_roc_save(title,table_multi_macro,classifier,save_folder_roc_macro, name= 'Macro')
              #multi_class_roc_save(title,table_multi_micro,classifier,save_folder_roc_micro,name = 'Micro')
              row_table = {'Dataset' : table_name ,'Acc(raw)':list_metrics[0],'AUC(raw micro)':list_metrics[1],'AUC(raw macro)':list_metrics[2],'IMCP(raw)':list_metrics[3],'Acc(smote)':list_metrics[4],'AUC(smote micro)': list_metrics[5],'AUC(smote macro)':list_metrics[6],'IMCP(smote)':list_metrics[7],'Acc(KDE)':list_metrics[8],'AUC(KDE micro)':list_metrics[9],'AUC(KDE macro)':list_metrics[10],'IMCP(KDE)':list_metrics[11]}    
              #precision_recall_row= {'Dataset':table_name,'Precision(raw)': metrics_add[0], 'Recall(raw)': metrics_add[1], 'Precision(smote)':  metrics_add[2], 'Recall(smote)': metrics_add[3],'Precision(KDE)':metrics_add[4],'Recall(KDE)':metrics_add[5]}
              final_table_multiclass.loc[len(final_table_multiclass)] = row_table
              #precision_recall_binary.loc[len(precision_recall_binary)] = precision_recall_row
              #precision_recall_binary.to_csv('../experiment/results/metrics_additional.csv',index = False)
              full_path = f'{save_folder}/{result_date}_metrics_multiclass_datasets.csv'
              final_table_multiclass.to_csv(full_path, index=False, sep=';', decimal=',')
              #final_table_multiclass.to_csv('../experiment/results/f{result_date}_metrics_multiclass_datasets.csv',index = False)  
              print('\n')
              print(final_table_multiclass)
            
            print('\n')
            print('Date:',result_date)
            print('\n')
            print('Execution time:',endTime - startTime)
            print('\n')
          


       