import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#takes in original dataframe and number, returns n(num) new df of random scaled, and random categorical 
def get_permutations(df, num):
    t_lists = []
    for r in range(0,num):
        t_list = []
        for c in df.columns.values:
            if df['{}' .format(c)].dtype == 'object':
                t_list.append(np.random.choice(df['{}' .format(c)].unique()))
            elif c == 'Input16':
                t_list.append(np.random.randint(0,2))
            elif c == 'Input14':
                t_list.append(np.random.choice(df['{}' .format(c)].unique()))
            else:
                t_list.append(random.random())
        t_lists.append(t_list)
        
    permutations_df = pd.DataFrame(t_lists)
    permutations_df.columns = df.columns.values
    permutations_df.drop(['Target'], axis = 1, inplace = True)
    permutations_df = pd.get_dummies(permutations_df, columns=['Input1', 'Input2', 'Input3'])
    return permutations_df

#gets log_Target, dummy 1-3, 
def xform_data(t_df, standard_scale = False):
    if standard_scale == True:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    t_df['log_Target'] = np.log(t_df.Target)
    t_df.drop('Target', axis = 1, inplace = True)
    df_dummied = pd.get_dummies(t_df, columns=['Input1', 'Input2', 'Input3'])
    continuous_cols = ['Input{}'.format(i) for i in range(4,16)]
    continuous_cols.append('log_Target')
    df_dummied[continuous_cols] = scaler.fit_transform(df_dummied[continuous_cols])
    return df_dummied, scaler, continuous_cols