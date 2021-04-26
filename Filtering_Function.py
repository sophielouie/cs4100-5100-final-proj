#!/usr/bin/env python
# coding: utf-8

# In[1]:


from HMM_Functions import *
from matplotlib import cm, pyplot as plt
import seaborn
from matplotlib.dates import YearLocator, MonthLocator, DayLocator
import pandas as pd
import numpy as np
from datetime import datetime


# In[2]:


tpm = [[0.3, 0.3, 0.2, 0.1, 0.05, 0.05],
       [0.25, 0.3, 0.25, 0.1, 0.05, 0.05],
       [0.1, 0.1, 0.3, 0.2, 0.2, 0.1],
       [0.1, 0.1, 0.2, 0.3, 0.2, 0.1],
       [0.05, 0.05, 0.1, 0.25, 0.3, 0.25],
       [0.05, 0.05, 0.1, 0.2, 0.3, 0.3]]

#tpm = create_TPM(6)

pi = [.2, .1, .1, .2, .1, .1]

epm = create_EPM(6, 2)
emissions = ['Decreasing', 'Decreasing','Increasing','Decreasing',
 'Decreasing','Decreasing','Decreasing', 'Decreasing', 'Increasing','Decreasing','Decreasing',
 'Decreasing','Increasing','Increasing','Decreasing','Decreasing']


# In[28]:


'''
                            TPM          FWD PROB
P(Xt+k+1 | e1:t) = ∑t+k P(Xt+k+1|xt+k) P(xt+k|e1:t)


∑t+k P(Xt+k+1|xt+k) * P(xt+k|e1:t): Iterate through ALL the states and add up the TPM * most recent fwd probability

Outcome: matrix with probabilities for each of hte next states, prediction is the highest one:
          I    D
state 0: .05  .05  = .1  -> .5 .5 = 1
state 1: .05  .05  = .1  -> .5 .5 = 1
state 2: .1   .1  = .2   
state 3: .05  .25 = .3   -> .05 .95 = 1
state 4: .1.  .1  = .2
state 5: .05  .05 =.01

'''

def filtering(tpm, epm, emissions, pi, future_days=7):

    t = len(emissions)
    NUM_STATES = len(tpm)
    fwrd = forward(tpm, epm, pi, emissions)
    state_pred = []
    pred_list = []
    normalized_list = fwrd[t-1]
    
    #starting_fwd = fwd[T][state_i]
    #starting_fwd
    
    #∑t+k P(Xt+k+1|xt+k) * P(xt+k|e1:t): Iterate through ALL the states and add up the TPM * most recent fwd probability
    
    '''
    
   do we need to do this part?  
   EPM of I at state 0  * probability state 0| KNOW that we were just in 0 *  fwd of being in state 0 at previous time step
   EPM of D at state 0  * probability state 0| KNOW that we were just in 0 *  fwd of being in state 0 at previous time step
   
          probability state 0| KNOW that we were just in 1 *  fwd of state 1
          probability state 0| KNOW that we were just in 2 *  fwd of state 2
          probability state 0| KNOW that we were just in 3 *  fwd of state 3
          probability state 0| KNOW that we were just in 4 *  fwd of state 4
          probability state 0| KNOW that we were just in 5 *  fwd of state 5
            add all this up to get the probabiltiy of next state being state 0
            
          probability state 1| KNOW that we were just in 0 *  fwd of state 0
          probability state 1| KNOW that we were just in 1 *  fwd of state 1
          probability state 1| KNOW that we were just in 2 *  fwd of state 2
          probability state 1| KNOW that we were just in 3 *  fwd of state 3
          probability state 1| KNOW that we were just in 4 *  fwd of state 4
          probability state 1| KNOW that we were just in 5 *  fwd of state 5
            add all this up to get the probabiltiy of next state being state 1
            
          probability state 2| KNOW that we were just in 0 *  fwd of state 0
          probability state 2| KNOW that we were just in 1 *  fwd of state 1
          probability state 2| KNOW that we were just in 2 *  fwd of state 2
          probability state 2| KNOW that we were just in 3 *  fwd of state 3
          probability state 2| KNOW that we were just in 4 *  fwd of state 4
          probability state 2| KNOW that we were just in 5 *  fwd of state 5
            add all this up to get the probabiltiy of next state being state 2
        '''
    
    for iter in range(future_days):
        pred_list = []
        
        for state_i in range(NUM_STATES):
            pred_state_dist = tpm[state_i]
            total = 0

            for state_j in range(NUM_STATES):
                transition_prob = pred_state_dist[state_j]
                fwd_prob =  normalized_list[state_j]

                total = total + (transition_prob * fwd_prob)

            pred_list.append(total)

        denominator = 0
        normalized_list = []
        for i in range(len(pred_list)):
            denominator += pred_list[i]

        #print('denom is...', denominator)

        temp_list = []
        for i in range(len(pred_list)):
            value = pred_list[i]/denominator
            temp_list.append(value)

        normalized_list = normalized_list.clear()
        
        normalized_list = temp_list.copy()

        print('\nnormalized_list is..', normalized_list)

        checksum = sum(normalized_list)
        print('equals 1?',checksum)

        maxstate = np.argmax(normalized_list)
        state_pred.append(maxstate)
        print("max", maxstate)
    
    return(state_pred)
            
    


# In[29]:


pred = filtering(tpm, epm, emissions, pi)
pred

