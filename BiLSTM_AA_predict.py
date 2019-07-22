"""
Bi-LSTM_AA_predict
@author: Amin Aria

"""
#############################----------Library import and function Definition------#################################

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib import rnn
import time
import matplotlib.pyplot as plt

###Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)



def inputfile(address,columns,step):
    my_data = pd.read_csv(address)
    my_data = my_data[columns]
    my_data = my_data.values
    ###---Training range ----###
    trange  = range (N_tw*2,my_data.shape[0],step)
    threewayAE1= np.ndarray(shape=(len(trange),N_tw,N_ft), dtype="float32", order='F')
    y1= np.ndarray(shape=(len(trange),1), dtype="float32", order='F')
    yFCDN= np.ndarray(shape=(len(trange),2), dtype="float32", order='F')
    #counter
    c = 0
    for i in trange:
        threewayAE1[c,:,:]=my_data[range(i-N_tw*2,i,2),0:N_ft] #features Input Columns
        y1[c,0]=my_data[i,N_ft]#label Column
        yFCDN[c,0:2]=my_data[i,(N_ft+1):-1]#Ground truth and time corresponding to each prediction
        c=c+1
    return threewayAE1, y1,yFCDN


###########################----------Input variables, files, and hyperparameters------###############################
N_tw = 20 #number of time steps
N_ft = 2 #number of features  
cols= ['NCC','NCE','Size','FCDN Size','time']


n_hidden = 128 #number of nodes in each gate or unit of a LSTM cell
k_p = 0.7 # Keep Probability probability for LSTM cells
num_classes = 1 
#--------------------------------------------------------------------
x = tf.placeholder(tf.float32, shape=[None, N_tw,N_ft],name="x") 


###-----Address and Columns of the input files (Y value at the last column)-----######
x2,y2,yFCDN=inputfile('Exp 18.csv',cols,1)


#########----Data to be analyzed----######
LSTMx= x2 
LSTMy= y2


####---------if used for fusion
'''
cols= ['LSTM','FCDN','Real','FCDN','Time']
xfuse,yfuse,yFCDN=inputfile('LSTM_predict.csv',cols,3)
LSTMx= xfuse #np.load(file = 'LSTMx.npy') 
LSTMy= yfuse
'''

pre_set= np.sort(np.random.randint(0,len(LSTMx),int(len(LSTMx)*0.7)))
X_predict =LSTMx[pre_set,:,0:(N_ft)]
Y_real = LSTMy[pre_set,:]
Y_FCDN = yFCDN[pre_set,:]

###########################################--------Model Bulider--------##################################################

timesteps = N_tw # timesteps

def BiRNN(x, num_hidden,keep_prob):
    
	# Define weights
    weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
	    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
        }		
    # Prepare data shape to match `rnn` function requirements current data input shape: (batch_size, timesteps, n_input)
    #batch size = number of sample points, time step = number of sample points to be considered at each instance of time
    #num_input = number of features
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    x = tf.unstack(x, timesteps, 1) # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)

    # Define lstm cells with tensorflow
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn.BasicLSTMCell(num_hidden, forget_bias=0.4, reuse= tf.AUTO_REUSE)
    ,input_keep_prob=keep_prob, output_keep_prob=keep_prob,variational_recurrent=False) # Forward direction cell
    
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn.BasicLSTMCell(num_hidden, forget_bias=0.4, reuse = tf.AUTO_REUSE)
    ,input_keep_prob=keep_prob, output_keep_prob=keep_prob,variational_recurrent=False) # Backward direction cell

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    
    return (tf.matmul(outputs[-1], weights['out']) + biases['out']) # Linear activation, using rnn inner loop last output


#################-------Passing the model to this prediction file>>>
network = BiRNN(x, num_hidden=n_hidden,keep_prob=k_p)


sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, 'Checkpoints_AA/Model_1')


print("Testing Data-set ")


st = time.time()
Y_predict = sess.run(network,feed_dict={x:X_predict})

run_time = time.time()-st

## save to xlsx file
df = pd.DataFrame (Y_predict)
##adding FCDN Size
df = pd.concat([df,pd.DataFrame (Y_FCDN)], axis=1)
df = pd.concat([df,pd.DataFrame (Y_real)], axis=1)
df.columns=['predict','FCDN','Time','Real']

###


filepath = 'predict.xlsx'
df.to_excel(filepath, index=False)
plt.figure(figsize=(5,5))
plt.scatter(Y_real,Y_predict)
plt.xlabel("Real Crack Length (mm)")
plt.ylabel("Predicted Crack length (mm)")
plt.xlim(0,2)
plt.ylim(0,2)
plt.plot(Y_real,Y_real,color='red')

print("")
print("Finished!")
