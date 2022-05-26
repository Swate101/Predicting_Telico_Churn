from cgi import test
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import env
from pydataset import data
import scipy
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test

def prep_telco(df):
    #dropping duplicates
    #df.drop_duplicates(inplace=True)
    
    #drop collumns
    telco_columns_drop = ['contract_type','payment_type','internet_service_type','partner','phone_service','online_security','online_backup','device_protection','streaming_tv','streaming_movies']
    df = df.drop(columns= telco_columns_drop,axis=1)
    
    #strip spces from charges, turn to a float
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    df['total_charges'] = df.total_charges.astype('float')
    
    #creating numeric dummmy columns
    dummy_df = pd.get_dummies(df[['gender','dependents','multiple_lines','tech_support','paperless_billing','churn']], dummy_na=False, drop_first=[True, True])
    
    #concat
    df = pd.concat([df, dummy_df], axis = 1)
    
    #train validate test
    train, validate, test = split_telco_data(df)

    return train, validate, test

def p_t(df):
    telco_columns_drop = ['contract_type','payment_type','internet_service_type','partner','phone_service','online_security','online_backup','device_protection','streaming_tv','streaming_movies']
    df = df.drop(columns= telco_columns_drop,axis=1)
    
    # turn to a float
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    df['total_charges'] = df.total_charges.astype('float')
    
    # dummmy columns
    dummy_df = pd.get_dummies(df[['gender','dependents','multiple_lines','tech_support','paperless_billing','churn']], dummy_na=False, drop_first=[True, True])
    
    #concat
    df = pd.concat([df, dummy_df], axis = 1)
    
    #train validate test
    train, validate, test = split_telco_data(df)

    return train, validate, test

def count_percentage_subplots(features_list, rows, cols, huee, dataa, x_ticks_rotation = 0, figsize_row = 14, figsize_col = 9 , prcnt_color = 'white', prcnt_height = -100 ):
  fig = plt.figure(figsize = (figsize_row, figsize_col))
  ax_list = []
  for i in range(1,cols * rows+1):
    ax_list.append("ax"+str(i))
  for index,ax_name in enumerate(ax_list): # for features 
    ax_name = plt.subplot(rows, cols, index+1)
    feature = features_list[index]
    sns.countplot(x=feature , hue = huee, data= dataa, order = sorted(list(dataa[feature].unique())))
    plt.xticks(rotation= x_ticks_rotation)
    for index,p in enumerate(ax_name.patches):
      height = p.get_height()
      temp = list(round(dataa.groupby(huee)[feature].value_counts(sort = False)/len(dataa)*100,2))
      ax_name.text(p.get_x()+p.get_width()/2., height+prcnt_height, str(temp[index]) + "%", horizontalalignment='center', fontsize=11, color=prcnt_color, weight = 'heavy') 
      
  fig. tight_layout(pad=4.0)
  plt.show()
    
def model_metrics(table_name,model_name, y_test, pred, print_cm , print_cr):
  if print_cm == True:
    plot_confusion_matrix(y_test, pred)
    print("\n")
  if print_cr == True:
    print(classification_report(y_test, pred),"\n")

  acc = round( accuracy_score(y_test, pred),4)
  precision = round(precision_score(y_test, pred),4)
  recall = round(recall_score(y_test, pred),4)
  f1 = round(f1_score(y_test, pred),4)
  f2 = round(fbeta_score(y_test, pred, beta = 2),4)

  table_name.loc[table_name.shape[0]] = [model_name ,acc, precision, recall, f1, f2]


    
    