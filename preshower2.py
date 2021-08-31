import sys
import math
import os
import pandas as pd
import uproot
import numpy as np
import csv
import random
from scipy import linalg

# Defining a combinatorics function for later use,
# stolen from stackoverflow
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


""" This code is meant to be ran from within the folder that 
 "unweighted_events.root" is contained in using, for 
 example, python ../../../preshower2.py"""

# This gets the current directory, and adds "unweighted_events.root"
# making is a little more general
directory = os.getcwd()
unweightedFile = "unweighted_events.root"
inputfile = directory + '/' + unweightedFile

# uproot opens the above file
file = uproot.open(inputfile)
weight = file["LHEF"].arrays(["Rwgt.Weight"]) # Empty 
particles = file["LHEF"].arrays(["Particle.PID", "Particle.E", "Particle.Px", "Particle.Py", "Particle.Pz",
                                 "Particle.M", "Particle.PT", "Particle.Eta", "Particle.Phi"])
pids = particles[b'Particle.PID']
all_weight = weight[b'Rwgt.Weight']

# Initializing params, will contain integers corresponding to 
# couplings we've turned on
params = []
with open('../../../wilsonParams.dat', 'r') as f:
    lines = f.read()
f.close()

lines = lines.split(',')
params = [int(x) for x in lines[:-1]]
print(params)

# Wilson Coeff by case number 1-6 (2 coefficients need 6 equations
# for 6 unknowns). Obtained currently by reading in 
# from reweight_card.dat
total = []
wilsonCo = [[] for _ in range(39)]

try:   
    with open('wilson_coef.dat', 'r') as f:
        lines = f.readlines()
        for i in range(0,len(lines)):
            lines[i] = [lines[i].strip('[').strip('\n').strip(']')]
        print('File existed! Reading in coefficients...')
        for i in range(0,len(lines)):
            for j in range(0, len(lines[i])):
                if ',' in lines[i][j]:
                    lines[i]=lines[i][j].split(',')
                else:
                    lines[i] = []
        for j in range(0,len(params)):
            item = params[j]-1
            for i in range(0,len(lines[item])):
                lines[item][i] = float(lines[item][i])
        wilsonCo = np.array(lines)
    for item in params:
        total.append(wilsonCo[item-1])
    total = np.array(total)
    total = np.transpose(total)


except FileNotFoundError:
    print("File DOES NOT EXIST! Reading from reweight_card.dat instead.")
    coefFile =  open('../../Cards/reweight_card.dat', 'r')
    lines = coefFile.readlines()[26:]

    for line in lines:
        line = line.strip()
        line = line.split(' ')
        if 'NEWCOUP' in line:
            if int(line[2]) in params:
                wilsonCo[int(line[2])-1].append(float(line[3]))
    coefFile.close()

    with open('wilson_coef.dat', 'w') as f:
        for item in wilsonCo:
            f.write(str(item)+'\n')
    f.close()
    for item in params:
        total.append(wilsonCo[item-1])
    total = np.array(total)
    total = np.transpose(total)


num_evnt = len(pids)

particle1_id = 11
particle2_id = -11

event_info = pd.DataFrame()

evnt_num = 0
weight_sum = 0


# Here we have begun extracting four vector information.
loc_part1 = np.where(pids[0] == particle1_id)[0][0]
loc_part2 = np.where(pids[0] == particle2_id)[0][0]


part1pT = particles[b"Particle.PT"][:,loc_part1]
part2pT = particles[b"Particle.PT"][:,loc_part2]

part1eta = particles[b"Particle.Eta"][:,loc_part1]
part2eta = particles[b"Particle.Eta"][:,loc_part2]

part1phi = particles[b"Particle.Phi"][:,loc_part1]
part2phi = particles[b"Particle.Phi"][:,loc_part2]


#for event in range(num_evnt):
#    
#    evnt_pid = pids[event]
#    loc_part1 = np.where(evnt_pid == particle1_id)
#    loc_part2 = np.where(evnt_pid == particle2_id)
#    print(evnt_pid)
    

#    
#    part1pT = particles[b"Particle.PT"][event][loc_part1][0]
#    part1eta = particles[b"Particle.Eta"][event][loc_part1][0]
#    print(part1eta)
#    part1phi = particles[b"Particle.Phi"][event][loc_part1][0]
    
    
#    part2pT = particles[b"Particle.PT"][event][loc_part2][0]
#    part2eta = particles[b"Particle.Eta"][event][loc_part2][0]
#    part2phi = particles[b"Particle.Phi"][event][loc_part2][0]
    
    
#    event_weight = all_weight[event]

    
    # event_info = event_info.append({'Event' : evnt_num, 
                                    # 'emPID' : particle1_id, 'epPID' : particle2_id,
                                    # 'pT': part1pT, 
                                    # 'emEta': part1eta, 
                                    # 'epEta': part2eta, 
                                    # 'emPhi': part1phi, 
                                    # 'epPhi': part2phi,
                                    # 'Weight_c1': event_weight[0], 
                                    # 'Weight_c2': event_weight[1],
                                    # 'Weight_c3': event_weight[2],
                                    # 'Weight_c4': event_weight[3], 
                                    # 'Weight_c5': event_weight[4],
                                    # 'Weight_c6': event_weight[5],
                                    # 'c2_cWW': cWW[1], 
                                    # 'c2_cA': cA[1], 
                                    # 'c3_cWW': cWW[2], 
                                    # 'c3_cA': cA[2],
                                    # 'c4_cWW': cWW[3], 
                                    # 'c4_cA': cA[3], 
                                    # 'c5_cWW': cWW[4], 
                                    # 'c5_cA': cA[4],
                                    # 'c6_cWW': cWW[5],
                                    # 'c6_cA': cA[5]}, 
                                    # ignore_index=True)
#    evnt_num = evnt_num + 1

    
# Calculating pz from different weights
#event_info['pzSM'] = event_info['Weight_c1']/(event_info['Weight_c1'].mean())
#event_info['pzdim6_c2'] = event_info['Weight_c2']/(event_info['Weight_c2'].mean())
#event_info['pzdim6_c3'] = event_info['Weight_c3']/(event_info['Weight_c3'].mean())
#event_info['pzdim6_c4'] = event_info['Weight_c4']/(event_info['Weight_c4'].mean())
#event_info['pzdim6_c5'] = event_info['Weight_c5']/(event_info['Weight_c5'].mean())

# Calculating rhat from different pz
#event_info['rhat_c2'] = event_info['pzdim6_c2']/event_info['pzSM']
#event_info['rhat_c3'] = event_info['pzdim6_c3']/event_info['pzSM']
#event_info['rhat_c4'] = event_info['pzdim6_c4']/event_info['pzSM']
#event_info['rhat_c5'] = event_info['pzdim6_c5']/event_info['pzSM']



#print(event_info.head(5))

""" matrix of Wilson Coefficients, ij. The first column is
    1 (corresponding to the SM).  The second column is the
    first Wilson Coefficient changed (i.e., 7 = cWW).  The third
    column is the second changed etc.  The 2+n through the 2n+1
    columns are the squares of the coefficients.  The remaining 
    columns are the mixing of the coefficients."""
# general variables for nCr and the total number of terms
n = int(len(params))
terms = int(2*n + 1 + nCr(n,2))

# constructing all of the different combinations of the 
# Wilson Coefficients, to quadratic order (theta1*theta2,
# theta1*theta3, etc.)
mixing = []
column = []
for i in range(0, len(total)):
    for j in range(0, len(total[0])):
        for k in range(0, j):
            mixing.append(total[i][k]*total[i][j])


# Constructing all of the linear and quadratic terms of 
# Wilson coefficients (theta1, theta1^2, etc)
ones = np.ones((len(total[0])*2+1, len(total)))
ones[1:1+n] = np.transpose(total[:,:])

ones[1+n:(2*n+1)] = np.transpose(total[:,:]**2)
ones = np.transpose(ones)

mixing = np.array(mixing)
mixing = np.reshape(mixing, (len(total), len(mixing)//len(total)))


# Combining the mixed and unmixed into single matrix
genMatrix = np.append(ones, mixing, axis = 1)

print('Generalized Matrix: \n', genMatrix, '\n\n\n')

# use numpy to invert the matrix.  This is a touchy step, 
# especially if the matrix is ill-conditioned (not singular,
# but basically singular)
matrixInverse = np.linalg.inv(genMatrix)

# Just choosing a random event n to look at
n = np.random.randint(num_evnt,size=1)[0]


# Getting the weights of that event, making it a column vector
T = all_weight[n]
T = np.reshape(T, (len(T), 1))

# printing the event, the inverse of the matrix, and the column
# vector of known weights
print('Wilson Coefficients: \n', total, '\n\n\n')
print('\nEvent # ', n, '\n\n\n')
print("\nA^-1 = \n", matrixInverse, '\n\n\n')
print("\nT = \n", T, '\n\n\n')

"""From here down, we get the cross section for the standard model,
we get the cross section in for each different case of Wilson Coefficients 
(allCross).  These are then combined into the cross section coefficient
present on rhat, crossCoef.  From there, we get the standard model weight,
tsm.  T is the total weight of an event, and is already stored in all_weight.
So we have T / tsm defined as Tdivtsm.  We also go ahead and solve for 
t, denoting it tVector.  This is done for all events at once, taking advantage
of numpy."""
smCross = np.mean(all_weight[:,0])

allCross = []
for i in range(len(all_weight[0])):
    allCross.append(np.mean(all_weight[:,i]))


crossCoef = smCross/allCross
tsm = all_weight[:,0]


Tdivtsm = all_weight / tsm

tVector = [np.matmul(matrixInverse,item) for item in all_weight]
print('t = ',np.reshape(tVector[n],(len(tVector[n]),1)))

rhat = [crossCoef * item for item in Tdivtsm]
print('\n\n\n rhat = ',rhat[n])