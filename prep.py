#!/usr/bin/env python
# coding: utf-8

# In[1]:


def prep_telco(df):
    #dropping duplicates
    #df.drop_duplicates(inplace=True)
    
    #drop collumns
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


# In[ ]:




