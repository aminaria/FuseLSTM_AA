# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:01:04 2019

@author: Amin Aria, Sergio Cofre 
Bi-LSTM for fatigue crack length esitimation and prediction
"""
#############################----------Library import and function Definition------#################################
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
#from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm




# Define this at the beginning of the file
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self) :
        for f in self.files:
            f.flush()

def Next_Batch(X,y,batch,batch_size):
    #i = np.random.randint(0,len(X)-batch_size)
    batch_x = X[batch:batch+batch_size,:]
    batch_y = y[batch:batch+batch_size]
    return batch_x, batch_y



####################################----------DATA PREP------#########################################################

# A function to reshape and select the data, as needed to be fed to LSTM
def inputfile(address,columns,step):
    my_data = pd.read_csv(address)
    my_data = my_data[columns]
    my_data = my_data.values
    ###---Training range ----###
    
    #One can determin here if they want to use all data rows or just some of them.
    trange  = range (N_tw,my_data.shape[0],step)
    
    #The input of the LSTM model is a three way array with dimenstions for 
        #entries to be considered
        #temporal correlation range to be considered (N_tw) for predictions made for each entry
        #Number of features to considered (N_ft)
    threewayAE1= np.ndarray(shape=(len(trange),N_tw,N_ft), dtype="float32", order='F')
    
    #label vector
    y1= np.ndarray(shape=(len(trange),1), dtype="float32", order='F')
    
    #counter
    c = 0
    for i in trange:
        threewayAE1[c,:,:]=my_data[range(i-N_tw*2,i,2),0:N_ft]
        y1[c,0]=my_data[i,N_ft]
        c=c+1
        
    return threewayAE1, y1

tf.reset_default_graph() 

####################################----------Input variables/paramters------#########################################

#---------------------------Hyperparameters
N_tw = 20 #number of time steps (Temporal Correlation Range)
N_ft = 2 #number of features  
cols= ['NCC','NCE','Size'] #Features of the orginial dataset to be considered plus the label column (here 'Size' is label column)

training_steps = 400
batch_size = 10 # number of time windows to be considered at each step of training 
n_hidden = 128 #number of LSTM cells repeated
k_p =0.7 # dropout probability for LSTM cells



##########-----Address and Columns of the input data (Y value at the last column)-----###############

##Data reduction step, used for large datasets, is the last argument. In the case that it is 1, 
#all entries of the data set are considered in the anlaysis

x2,y2=inputfile('Exp 18.csv',cols,3)
np.save('LSTMy',y2)
np.save('LSTMx',x2)

###
#######------------if used for data fusion
###

'''
cols= ['LSTM','FCDN','Real']
xfuse,yfuse=inputfile('LSTM_predict.csv',cols,2)
np.save('LSTMy',yfuse)
np.save('LSTMx',xfuse)

'''

######################################################################################################################


#-----------------------------Train, Cross val and Test Files
LSTMx= np.load(file = 'LSTMx.npy') 
LSTMy= np.load(file = 'LSTMy.npy')  

#Determine training, cross-valdation, and testing portions of the data set in the following lines
train_set= np.random.randint(0,len(LSTMx),int(len(LSTMx)*0.7))
cr_set= np.random.randint(0,len(LSTMx),int(len(LSTMx)*0.2))
test_set= np.random.randint(0,len(LSTMx),int(len(LSTMx)*0.3))

X_train =LSTMx[train_set,:,0:(N_ft)]
X_crossval = LSTMx[cr_set,:,0:(N_ft)]
X_test = LSTMx[test_set,:,0:(N_ft)]

#label vectors
Y_train = LSTMy[train_set,:]
Y_crossval = LSTMy[cr_set,:]
Y_test = LSTMy[test_set,:]


num_classes = Y_train.shape[1] # number of prediction classes


#####---------------------- Placeholders--------------------------#####
x = tf.placeholder(tf.float32, shape=[None, N_tw,N_ft],name="x") 
y_true = tf.placeholder(tf.float32, shape=[None, num_classes ],name="y_true")
Pneg = tf.placeholder(tf.float32, shape=[],name="Pneg")
keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32, shape=[])
is_training=tf.placeholder(tf.bool)



###############################--- BILSTM ---#######################################################
# Network Parameters
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
    # batch size = number of sample points, time step = number of sample points to be considered at each instance of time
    # num_input = number of features
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    x = tf.unstack(x, timesteps, 1) # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
 
    
    # Define lstm cells with tensorflow
    
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn.BasicLSTMCell(num_hidden, forget_bias=0.4)
    ,input_keep_prob=keep_prob, output_keep_prob=keep_prob,variational_recurrent=False) # Forward direction cell
    
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(rnn.BasicLSTMCell(num_hidden, forget_bias=0.4)
    ,input_keep_prob=keep_prob, output_keep_prob=keep_prob,variational_recurrent=False) # Backward direction cell

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    
    return (tf.matmul(outputs[-1], weights['out']) + biases['out']) # Linear activation, using rnn inner loop last output


#############################################################################################################################

#Function for training the model

def main(training_steps,batch_size,n_hidden,k_p,save_path):
   
    # Training Parameters
    # learning_rate = 0.001 #0.001
    # training_steps = 4 
    # batch_size = 20#512#100
    # k_p keep probability
    
    batches = int(len(X_train)/batch_size)
    display_step = 2

    logits = BiRNN(x, n_hidden,k_p)
    prediction = tf.nn.relu(logits)

    
    
    # Define accuracy and optimization loss function
    
    learning_rate=tf.placeholder(tf.float32,shape=None)


    #accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true,prediction))))
    
    
    #Favoring overestimation over underestimation of the damage size
    ####-----ACCURACY metric defined based on Zhang et al. (2017) paper on RUL estimation using LSTM
    ###------ Negative length values are not desired (last line of accuracy)
    
    loss_op = tf.reduce_mean(tf.cast(tf.math.greater(prediction,y_true),tf.float32)*
                              (tf.exp(tf.subtract(y_true,prediction)/(-1.0))-1)+
                               tf.cast(tf.math.greater(y_true,prediction),tf.float32)*
                              (tf.exp(tf.subtract(y_true,prediction)/(0.5))-1)+
                              tf.cast(tf.math.greater(0.00001,prediction),tf.float32)*Pneg)
                             

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_op) #GradientDescent, RMSProp, Adam, AdadeltaOptimizer
    
    # Evaluate model (with test logits, for dropout to be disabled)
    
    
    ####-----ACCURACY metric defined based on Zhang et al. (2017) paper on RUL estimation using LSTM

    accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true,prediction))))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

 
    # Start training
    saver = tf.train.Saver()
    
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        start=time.time()
        l_rate=0.001
        
        #Early stopping Mechanism
        
        loss1=[0,1,0,1,0,1,0,1] 
        for step in range(1, training_steps+1):
            if abs(loss1[(step+5)]-loss1[(step+4)])+abs(loss1[(step+5)]-loss1[(step+2)])+abs(loss1[(step+5)]-loss1[(step)]) < 0.0000001:
                break
            else:
                if step> 50:
                    Pnegative =100
                else:
                    Pnegative = 0
                
                for i in range(batches):
                    batch = i*batch_size    
                    batch_x,batch_y = Next_Batch(X_train,Y_train,batch,batch_size)
                    sess.run(train_op, feed_dict={Pneg:Pnegative,x: batch_x, y_true: batch_y, learning_rate: l_rate,keep_prob:k_p, is_training: True})
            
                if step % display_step == 0:
                    
                # Calculate batch loss and accuracy
                    acc_cross = sess.run(accuracy, feed_dict={x: X_crossval,y_true: Y_crossval, learning_rate: l_rate,keep_prob:1.0, is_training: False})
                    loss_cross = sess.run(loss_op, feed_dict={Pneg:Pnegative,x: X_crossval,y_true: Y_crossval, learning_rate: l_rate,keep_prob:1.0, is_training: False})
                    acc_train = sess.run(accuracy, feed_dict={x: batch_x,y_true: batch_y, learning_rate: l_rate,keep_prob:1.0, is_training: False})
                    loss1.append(acc_cross)
                    loss1.append(acc_cross)

                
                    print("Step " + str(step)+"/"+str(training_steps) + ", Accuracy: Training = " + \
                      "{:.4f}".format(acc_train) + ", Cross Val = " + \
                      "{:.3f}".format(acc_cross) + ", Cross_loss = " + \
                      "{:.3f}".format(loss_cross))
            
        
        
        saver.save(sess,save_path = save_path)
        
        print("Optimization Finished!")
        
        stop=time.time()
        t=stop-start
        print('\n Total Time = %g [s] \n'%t)
        test_data = X_test #.reshape((-1, timesteps, num_input))
        test_label = Y_test
        acc = []
        for t in range(1000):
            test_acc=sess.run(accuracy, feed_dict={x: test_data, y_true: test_label,keep_prob:k_p, is_training: False})
            #print('Step {}/{}, Accuracy Metric = {}'.format((t+1),1000,test_acc))
            acc.append(test_acc)
        
        acctest=np.array(acc)
        print("Average Testing Accuracy:", \
            np.mean(acctest))
        
    return acc_cross,loss_cross,acctest

if __name__ == '__main__':
    
    Folder = 'Checkpoints_AA' 
    File = 'Model_1'
    mypath = os.path.join(os.getcwd(),Folder)
    if not os.path.isdir(mypath):
        os.makedirs(mypath)  
        
    save_path = os.path.join(mypath,File)
    


    acc_cross,loss_cross,rmse = main(training_steps,batch_size,n_hidden,k_p,save_path)
    mu, std = norm.fit(rmse)
    plt.hist(rmse,bins=10, alpha=0.3, color='red',normed=True)
    plt.xlabel('RMSE')
    plt.ylabel('Frequency')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.3f,  std = %.3f" % (mu, std)
    plt.title(title)
    