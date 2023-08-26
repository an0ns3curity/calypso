# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:59:18 2023
@author: Kuheli

In this code, we generate multiple (min # of instances = 10) instances of a (m=4) LPPUF and then evaluate its uniformity,
                                    uniqueness and reliabilty. For easy visual representation, the same has also been
                                    plotted. Please refer to the code parameters for changing as per your needs. 

Variable Description:
    0. path : Change this variable to the main directory of pypuf where the LPPUF codes reside.
    1. chalLength: It indicates the length of the challenge given to the PUF instances. Standard challenge length
    are {64, 128, 192, 256} bits.
    2. noiseLevel1: It indicates the noise in the first layer of the LP-PUF architecture
    3. noiseLevel2: It indicates the noisiness in the final layer (m-XOR APUF) of the LP-PUF
    4. NumberOfChallenges: It denotes the number of challenges for which the PUF must be evaluated
    5. ChalGen_seed: It is a randomly generated seed for generating the set of 'NumberOfChallenges' challenges
    6. NumberOfInstances: It indicates the number of PUF instantiations being done for our evaluation
    7. NumberOfTrigger: This variable indicates the number of measurements for a given challenge set and the PUF instance
    under consideration. In this case, we consider NumberOfTrigger as 11 and then perform temporal majority voting to get
    the golden response for analysis. 
    8. PUF_seed_all: A list of "NumberOfInstances" randomly generated prime nos. for instanting the instance-specific PUF.
    9. LP_XOR_ind: This variables indicates the 'm' parameter of the LP-PUF. We consider m = 4 for our use.
"""
#Packages Used
#import lppuf2m
import random
from pypuf.io import random_inputs
import numpy as np
import copy
from scipy import stats
from scipy.spatial.distance import hamming
from pypuf.simulation import XORArbiterPUF

#Variables used as per the variable description above
#path = 'D:\pypuf-main'
#pathFiles = 'D:\pypuf-main\pypuf'    # 'E:/pypuf-main/LPPUF_Evaluation'
chalLength = 64
noiseLevel = 0.09
noiseLevel2 = 0.1
NumberOfChallenges = 100000 # 0.1M Challenges
ChalGen_seed = 101419 # a randomly generated prime number
#Number of Instances
NumberOfInstances = 4
NumberOfTrigger = 3
PUF_seed_all = [100787,51827,51001,81703,82139,122069,122039,342281,341597,341879]
LP_XOR_ind = 4

#Challenge Generation
#Not converting to parity vectors in this code
chal = random_inputs(n=chalLength, N=NumberOfChallenges, seed=ChalGen_seed)
#Creating a numpy array for saving the responses
responses_all = np.zeros([NumberOfChallenges,NumberOfInstances,NumberOfTrigger])

for pind in range(NumberOfInstances):
    PUF_seed = PUF_seed_all[pind]
    #crpFileName = pathFiles+'/CRPs_{LP_XOR_Ind}-LP-PUF_chal{chalSize}_nChal{nChal}_noise1{noise1}_noise2{noise2}.npz'.format(LP_XOR_Ind=LP_XOR_ind, chalSize = chalLength, nChal = NumberOfChallenges, noise1=noiseLevel,noise2=noiseLevel2)
    #rpFileName = pathFiles+'/RPs_{LP_XOR_Ind}-LP-PUF_chal{chalSize}_nChal{nChal}_noise1{noise1}_noise2{noise2}.npz'.format(LP_XOR_Ind=LP_XOR_ind, chalSize = chalLength, nChal = NumberOfChallenges, noise1=noiseLevel,noise2=noiseLevel2)
    puf = XORArbiterPUF(n=64, k=10, seed=random.randint(0, 1000), noisiness=noiseLevel)
    #lppuf2m.LPPUFv1(n=chalLength, m=LP_XOR_ind, seed=PUF_seed, noisiness_1=noiseLevel, noisiness_2=noiseLevel2)
    
    for tind in range(NumberOfTrigger):
        temp_resp = puf.eval(chal)
        responses_all[:,pind,tind] = temp_resp

#Converting responses to 0,1
resp01 = copy.deepcopy(responses_all)
resp01[resp01==1] = 0
resp01[resp01==-1] = 1

#np.savez(rpFileName, responses_all=responses_all)

#%%
#calculate uniformity for each instance as well as report the average uniformity value

Uniformity = np.zeros([NumberOfInstances])
resp_golden = np.zeros([NumberOfChallenges,NumberOfInstances])
#Calculate the Golden Response for noisy case

for i in range(NumberOfInstances):
    resp_golden[:,i] = stats.mode(resp01[:,i,:], axis = 1)[0].reshape(NumberOfChallenges)
    Uniformity[i] = (np.sum(resp_golden[:,i])/NumberOfChallenges)*100
        
print(Uniformity)
Uniformity_avg = np.mean(Uniformity)
print("the avarage uniformity for lppuf2m is ", Uniformity_avg)

#%%
#Calculate the reliability values for each instance as well as report the average reliability value across the 'NumberOfInstances' instances

Reliability = np.zeros([NumberOfInstances,NumberOfTrigger])

for idx in range(NumberOfInstances):
    for i in range(NumberOfTrigger):
        rel = 0
        for k in range(NumberOfChallenges):
            if(resp_golden[k,idx]==resp01[k,idx,i]):
                rel = rel+1
        Reliability[idx,i] = (rel/NumberOfChallenges)*100
       
Reliability_avg = np.average(Reliability,axis=1)
print("the reliability of lppuf2m is", Reliability_avg)

#%%
#Metric Calculations - Uniqueness

comb_array = np.array(np.meshgrid(np.arange(0,NumberOfInstances), np.arange(0,NumberOfInstances))).T.reshape(-1, 2)
Fractional_HD = np.zeros(len(comb_array))

for i in range(len(comb_array)):
    Fractional_HD[i] = hamming(resp_golden[:,comb_array[i][0]], resp_golden[:,comb_array[i][1]])
    
Uniqueness = np.mean(Fractional_HD[Fractional_HD!=0]*100)
print("the uniqueness of the lppuf2m is ", Uniqueness)
