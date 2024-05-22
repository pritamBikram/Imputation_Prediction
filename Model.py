import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import preprocess_data,load_sz_data,load_los_data
from tgcn import tgcnCell
import time
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
train_rate =  0.8
seq_len = 12
output_dim = pre_len = 12
batch_size =64
lr =0.001
training_epoch = 100
gru_units =64
heads=4
data=pd.read_csv("/content/drive/MyDrive/MTRLEA/DATA/MTERLA_DATA_0.2_ROOT.csv")
data1=np.mat(data)
adj=pd.read_csv("/content/drive/MyDrive/MTRLEA/MTERLA_1.csv")
adj=np.mat(adj)
adj=adj.astype(np.float32)
time_len = data1.shape[0]
num_nodes = data1.shape[1]
#### normalization
max_value = np.max(data1)
data1  = data1/max_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)
totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

def self_attention1(x, weight_att,bias_att):
    x1=tf.reshape(x,[-1,gru_units])
    h1=tf.matmul(x1,weight_att['w11'])+bias_att['b11']
    h2=tf.matmul(x1,weight_att['w12'])+bias_att['b12']
    h3=tf.matmul(x1,weight_att['w13'])+bias_att['b13']
    h4=tf.matmul(x1,weight_att['w14'])+bias_att['b14']
    head=tf.concat([h1, h2,h3,h4], 1)
    head=tf.matmul(head,weight_att['head'])+bias_att['bias_head']
    NODES_D=tf.reshape(head, [-1, num_nodes])
    f1 = tf.matmul(NODES_D, weight_att['w2']) + bias_att['b2']
    g1 = tf.matmul(NODES_D, weight_att['w2']) + bias_att['b2']
    h1 = tf.matmul(NODES_D, weight_att['w2']) + bias_att['b2']

    f11 = tf.reshape(f1, [-1,seq_len])
    g11 = tf.reshape(g1, [-1,seq_len])
    h11 = tf.reshape(h1, [-1,seq_len])
    s_out = g11 * f11
    beta_out = tf.nn.softmax(s_out, dim=-1)  # attention map
    context_out= tf.expand_dims(beta_out,2) * tf.reshape(head,[-1,seq_len,num_nodes])
    context_out = tf.transpose(context_out,perm=[0,2,1])
    return context_out, beta_out

def TGCN(_X, _weights, _biases):
    ###
    cell_1 = tgcnCell(w_g_1, b_g_1, w_g_2, b_g_2, w_g_3, b_g_3, w_g_4, b_g_4, w_g_5, b_g_5, w_c_1, b_c_1, w_c_2, b_c_2,w_c_3, b_c_3, w_c_4, b_c_4, w_c_5, b_c_5,gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    out = tf.concat(outputs, axis=0)
    out = tf.reshape(out, shape=[seq_len,-1,num_nodes,gru_units])
    out = tf.transpose(out, perm=[1,0,2,3])

    last_output,alpha = self_attention1(out, weight_att, bias_att)

    output = tf.reshape(last_output,shape=[-1,seq_len])
    output = tf.matmul(output, weights['out']) + biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])

    return output, outputs, states, alpha

tf.compat.v1.disable_eager_execution()
inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.compat.v1.placeholder(tf.float32, shape=[None, pre_len, num_nodes])


bias_candidate=0.0
bias_gate=1.0
# # Graph weights

weights = {
    'out': tf.Variable(tf.random.normal([seq_len, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random.normal([pre_len]),name='bias_o')}
weight_att={
    'w11':tf.Variable(tf.random.normal([gru_units,1], stddev=0.1),name='att_w11'),
    'w12':tf.Variable(tf.random.normal([gru_units,1], stddev=0.1),name='att_w12'),
    'w13':tf.Variable(tf.random.normal([gru_units,1], stddev=0.1),name='att_w13'),
    'w14':tf.Variable(tf.random.normal([gru_units,1], stddev=0.1),name='att_w14'),
    'w15':tf.Variable(tf.random.normal([gru_units,1], stddev=0.1),name='att_w15'),
    'w16':tf.Variable(tf.random.normal([gru_units,1], stddev=0.1),name='att_w16'),

    'head':tf.Variable(tf.random.normal([heads,1], stddev=0.1),name='att_head'),
    'w2':tf.Variable(tf.random.normal([num_nodes,1], stddev=0.1),name='att_w2')}
bias_att = {
    'b11': tf.Variable(tf.random.normal([1]),name='att_b11'),
    'b12': tf.Variable(tf.random.normal([1]),name='att_b12'),
    'b13': tf.Variable(tf.random.normal([1]),name='att_b13'),
    'b14': tf.Variable(tf.random.normal([1]),name='att_b14'),
    'b15': tf.Variable(tf.random.normal([1]),name='att_b15'),
    'b16': tf.Variable(tf.random.normal([1]),name='att_b16'),

    'bias_head': tf.Variable(tf.random.normal([1]),name='bias_head'),
    'b2': tf.Variable(tf.random.normal([1]),name='att_b2')}



w_g_1=tf.compat.v1.get_variable('weights1', [1, 2],initializer=tf.initializers.glorot_uniform())
b_g_1=tf.compat.v1.get_variable("biases1", [2], initializer=tf.constant_initializer(bias_gate))
w_g_2=tf.compat.v1.get_variable('weights2', [2,4],initializer=tf.initializers.glorot_uniform())
b_g_2=tf.compat.v1.get_variable("biases2", [4], initializer=tf.constant_initializer(bias_gate))
w_g_3=tf.compat.v1.get_variable('weights3', [5,8],initializer=tf.initializers.glorot_uniform())
b_g_3=tf.compat.v1.get_variable("biases3", [8], initializer=tf.constant_initializer(bias_gate))
w_g_4=tf.compat.v1.get_variable('weights4', [8, 16], initializer=tf.initializers.glorot_uniform())
b_g_4=tf.compat.v1.get_variable("biases4", [16], initializer=tf.constant_initializer(bias_gate))
w_g_5=tf.compat.v1.get_variable('weights5', [21+gru_units, 2*gru_units], initializer=tf.initializers.glorot_uniform())
b_g_5=tf.compat.v1.get_variable("biases5", [2*gru_units], initializer=tf.constant_initializer(bias_gate))

w_c_1=tf.compat.v1.get_variable('weights6', [1, 2],initializer=tf.initializers.glorot_uniform())
b_c_1=tf.compat.v1.get_variable("biases6", [2], initializer=tf.constant_initializer(bias_candidate))
w_c_2=tf.compat.v1.get_variable('weights7', [2,4], initializer=tf.initializers.glorot_uniform())
b_c_2=tf.compat.v1.get_variable("biases7", [4], initializer=tf.constant_initializer(bias_candidate))
w_c_3=tf.compat.v1.get_variable('weights8', [5,8], initializer=tf.initializers.glorot_uniform())
b_c_3=tf.compat.v1.get_variable("biases8", [8], initializer=tf.constant_initializer(bias_candidate))
w_c_4=tf.compat.v1.get_variable('weights9', [8, 16],initializer=tf.initializers.glorot_uniform())
b_c_4=tf.compat.v1.get_variable("biases9", [16], initializer=tf.constant_initializer(bias_candidate))
w_c_5=tf.compat.v1.get_variable('weights10', [21+gru_units, gru_units], initializer=tf.initializers.glorot_uniform())
b_c_5=tf.compat.v1.get_variable("biases10", [gru_units], initializer=tf.constant_initializer(bias_candidate))

model_name = 'tgcn'
if model_name == 'tgcn':
    pred,ttts,ttto,jj= TGCN(inputs, weights, biases)
y_pred = pred

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables())
label = tf.reshape(labels, [-1,num_nodes])
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables =tf.compat.v1.global_variables()
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
sess.run(tf.compat.v1.global_variables_initializer())
from tqdm import tqdm
import pandas as pd
df=pd.DataFrame([],columns=["TRAIN_RMSE",'TEST_RMSE',"TEST_MAE","ACCURACY"])
path="/content/drive/MyDrive/CONVERGENCE_STUDY"

def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    mape = mean_absolute_percentage_error(a,b)*100
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae,mape,1-F_norm, r2, var


x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]
test_mape=[]
high_acc=0
low_rmse=0
high_mape=999999999999999999999
batch_mae=[]
batch_mape=[]
for epoch in range(training_epoch):
    for m in tqdm(range(totalbatch)):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1,train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict = {inputs:mini_batch, labels:mini_label})

        train_label = np.reshape(mini_label,[-1,num_nodes])
        rmse1, mae1,mape1, acc1, r2_score1, var_score1 = evaluation(train_label, train_output )
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)
        batch_mae.append(mae1* max_value)
        batch_mape.append(mape1)

     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs:testX, labels:testY})
    test_label = np.reshape(testY,[-1,num_nodes])
    rmse, mae,mape, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)
    test_mape.append(mape)
    np.save("T.npy",test_mape)
    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'train_loss:{:.4}'.format(batch_loss[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse * max_value),
          'train_mae:{:.4}'.format(batch_mae[-1]),
          'test_mae:{:.4}'.format(mae * max_value),
          'train_mape:{:.4}'.format(batch_mape[-1]),
          'test_mape:{:.4}'.format(mape))
    df2=pd.DataFrame([[round(batch_loss[-1],4),round(loss2,4),round(batch_rmse[-1],4),round(rmse * max_value,4),round(mae * max_value,4),round(mape,4),round(batch_mae[-1],4),round(batch_mape[-1],4)]],columns=["TRAIN_LOSS","TEST_LOSS","TRAIN_RMSE",'TEST_RMSE',"TEST_MAE","TEST_MAPE","TRAIN_MAE","TRAIN_MAPE"])
    df=pd.concat([df,df2],ignore_index=True)
    df.to_csv(f"/content/drive/MyDrive/HYPERPARAMTER/atttention_heads/MTERLA_0.2_{pre_len}_{heads}.csv",index=False)
    if test_mape[-1]<high_mape:
      print(f"test mape improved from {high_mape} to {test_mape[-1]}")
      high_mape=test_mape[-1]
      saver.save(sess, path+'/model_100/TGCN_pre_%r'%epoch, global_step = epoch)
    else:
      print("NOT IMPROVED")
