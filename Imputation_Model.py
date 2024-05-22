from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import os

## Missing Values Generation
df_pems=pd.read_csv("/content/drive/MyDrive/MISSING_DATASET/vel.csv",header=None)

for node in range(df_pems.shape[1]):
  df_pems[node]=np.where(df_pems[node]==0,df_pems[node].mean(),df_pems[node])

TOTAL_ARR=[]
for node in tqdm(range(df_pems.shape[1])):
    arr=[]
    for j in range(119):
        frr=[]
        for k in range(288):
            frr.append(df_pems[node].values[288*j+k])
        arr.append(frr)
    TOTAL_ARR.append(arr)

TOTAL_ARR=np.array(TOTAL_ARR)




## RANDOM MISSING VALUES
dense_tensor=TOTAL_ARR
dim = dense_tensor.shape
missing_rate = 0.8
sparse_tensor = dense_tensor * np.round(np.random.rand(dim[0], dim[1], dim[2]) + 0.5 - missing_rate)
if np.isnan(sparse_tensor).any() == False:
    ind = sparse_tensor != 0
    pos_obs = np.where(ind)
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
elif np.isnan(sparse_tensor).any() == True:
    pos_test = np.where((dense_tensor != 0) & (np.isnan(sparse_tensor)))
    ind = ~np.isnan(sparse_tensor)
    pos_obs = np.where(ind)
    sparse_tensor[np.isnan(sparse_tensor)] = 0
num_obs = len(pos_obs[0])
dense_test = dense_tensor[pos_test]
i,j,k=pos_test




# ## NON-random missing values
# dense_tensor = TOTAL_ARR
# dim = dense_tensor.shape
# missing_rate = 0.8
# sparse_tensor = dense_tensor * np.round(np.random.rand(dim[0], dim[1])[:, :, np.newaxis] + 0.5 - missing_rate)
# ##
# if np.isnan(sparse_tensor).any() == False:
#     ind = sparse_tensor != 0
#     pos_obs = np.where(ind)
#     pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
# elif np.isnan(sparse_tensor).any() == True:
#     pos_test = np.where((dense_tensor != 0) & (np.isnan(sparse_tensor)))
#     ind = ~np.isnan(sparse_tensor)
#     pos_obs = np.where(ind)
#     sparse_tensor[np.isnan(sparse_tensor)] = 0
# num_obs = len(pos_obs[0])
# dense_test = dense_tensor[pos_test]

DATAPOINTS=[]
NODE=[]
for m in tqdm(range(len(i))):
  DATAPOINTS.append(288*j[m]+k[m])
  NODE.append(i[m])

d=pd.DataFrame([NODE,DATAPOINTS,dense_test])

d_transpose=d.T
d_transpose.to_csv("/content/drive/MyDrive/MTERLA_FINAL/ORIGINAL_DATAPOINTS_0.8.csv",index=False)

## FOR REPLACING MISSING VALUES
df_datapoints=pd.read_csv("/content/drive/MyDrive/MTERLA_FINAL/ORIGINAL_DATAPOINTS_0.8.csv")
df=pd.read_csv("/content/drive/MyDrive/MISSING_DATASET/vel.csv",header=None)
adj=np.load("/content/drive/MyDrive/MISSING_DATASET/MTRLA_ADJACENCY_MATRIX.npy")
I=np.identity(207)
adj=adj-I
adj=np.where(adj!=0,1,0)
adj=np.mat(adj)
i,j=np.where(adj!=0)
i=list(i)
d=np.sum(adj,axis=1)


## IMPUTATION OF THE MISSING VALUES
Result=pd.DataFrame([],columns=["ACTUAL_SPEED","PREDICTED_SPEED"])
SPEED=[]
ACTUAL=[]
for vod in tqdm(range(df_datapoints.shape[0])):
  if vod<Result.shape[0]:
    pass
  else:
    d_frame=validation_data(df_datapoints[str(0)].values[vod],df_datapoints[str(1)].values[vod])
    SPEED.append(d_frame['Replaced_speed'].values[0])
    ACTUAL.append(df_datapoints[str(2)].values[vod])
    r_dataframe=pd.DataFrame([[df_datapoints[str(2)].values[vod],d_frame['Replaced_speed'].values[0]]],columns=["ACTUAL_SPEED","PREDICTED_SPEED"])
    Result=pd.concat([Result,r_dataframe],ignore_index=True)
    Result.to_csv("/content/drive/MyDrive/MISSING_VALUE_RESULTS_MTERLA/NEW_RESULTS_0.2.csv",index=False)