import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tqdm

#takes in original dataframe and number, creates appropriate random vars, returns n(num) new df of random to scale
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

#input raw data, gets log_Target, dummy 1-3, scales continuous vars, returns df, scaler, and column names
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

#input copy of permutations, unscales, unlogs Target, renames column, returns df
def get_experiment_df(t_df, means, sigmas, wape, scale, cols):
    t_df['log_Target'] = means
    t_df['Certainty'] = (100 - (1960 * sigmas))
    t_df[cols] = scale.inverse_transform(t_df[cols])
    t_df.log_Target = np.exp(t_df.log_Target)
    t_df.rename(columns={'log_Target': "Target"}, inplace = True)
    return t_df

def get_preds(data, num):
    y_pred_list = []
    for i in tqdm.tqdm(range(len(data))):
        tensor = tf.constant([data.iloc[i]], dtype='float32')
        pred_list = []
        for j in range(num):
            y_pred = model.predict(tensor)
            pred_list.append(y_pred)
        y_pred_list.append(pred_list)
    return y_pred_list

def means_sigmas(pred_list):
    y_means = np.mean(y_pred_list, axis = 1)
    y_sigmas = np.std(y_pred_list, axis = 1)
    y_sigmas = y_sigmas.reshape(len(y_means),)
    y_means = y_means.reshape(len(y_sigmas),)
    return means, sigmas