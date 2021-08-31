"""The whole purpose of this script is to automate the writing of a 
reweight card so that once we have a ton of Wilson Coefficients, we
aren't just copy-pasting the reweight_card a million times."""

import math
import numpy as np
import random

# This is the number of Wilson Coefficients
# params is the only thing that should ever be changed, for the HEL_UFO
# model.  Numbers correspond to couplings, seen in model's parameters.py
params = [7,11]
n = int(len(params))


# Defining a combinatorics function for later use,
# stolen from stackoverflow
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)



# This is the head of the reweight.dat file, stupid long line, 
# hesitant to actually break up though, python is dumb
header = "#*************************************************************************#                          Reweight Module                               *#                Matrix-Element reweighting at LO/NLO                    *#  Mattelaer Olivier                                    arxiv:1607.00763 *  #*************************************************************************#          # Note: #   1) the value of alpha_s will be used from the event so the value in#      the param_card is not taken into account.#   2) It is (in general) dangerous/wrong to change parameters by a large#      amount, if this changes the shape of the matrix elements a lot.#      (For example, changing a particle's mass by much more than its#      width leads to very inaccurate result). In such a case, separate #      event generation runs are needed.##************************************************************************# ENTER YOUR COMMANDS BELOW. #************************************************************************"
header = header.split('#')
header = ['#' + item for item in header]
header = [item + '\n' for item in header]
footer = "#  SPECIFY A PATH OR USE THE SET COMMAND LIKE THIS:# set sminputs 1 130 # modify 1/alpha_EW#************************************************************************# Manual:      https://cp3.irmp.ucl.ac.be/projects/madgraph/wiki/Reweight#************************************************************************## Example of (standard) code for the computation of two weights:## launch                    ! tag to start the computation of the first weight#  set BLOCKNAME ID VALUE   ! rule to modify the current param_card#  set BLOCKNAME ID VALUE   ! rule to modify the current param_card# launch                    ! start to compute a second weight#  /home/Cards/param_card_2.dat ! you can also enter a path to a valid card## Note: The command to specify the parameter are AFTER the associated \"launch\" ## Possible options: #   You can enter one of the following lines to customize the reweighting #   procedure. These need to be given before the \'launch\' command.## change model NAME : use another model for the matrix-elements to reweight #        with. In this case you need to provide the path to a correct #        param_card for the new model; you cannot modify the original one #        with the \'set\' command.# change process DEF [--add]: change the process by which you reweight. #        The initial and final state particles of the new process should #        be exactly identical to the ones in the original process.# change helicity False: perform the reweighting by helicity summed #        matrix-elements even if the events have been written with a #        single helicity state.       # change mode XXX: change the type of reweighting performed.#        allowed values: LO, NLO, LO+NLO#        - This command has no effect for reweighting an .lhe event file with LO accuracy. #          In that case LO mode is always used (whatever entry is set).#        - When the .lhe file reweighted is at NLO accuracy, then all modes are allowed.#          * \"LO\" is an approximate leading order method#          * \"NLO\" is the NLO accurate method#          * \"LO+NLO\" runs both#       - \"NLO\" and \"LO+NLO\" modes requires \'store_rwgt_info\' equals True (run_card.dat)#          If the reweighting is done at generation level this parameter will#          automatically be set on True.#************************************************************************"
footer = footer .split('#')
footer = ['#' + item for item in footer ]
footer = [item + '\n' for item in footer ]


mode = "\nchange mode LO\n\n"

model = "change model HEL_UFO\n\n"

launch = "launch --rwgt_name=case"

terms = 2*n + 1 + nCr(n,2)
terms = int(terms)

i = range(1,terms+1)

launches = [launch + str(j) +'\n' for j in i]

text = header
text.append(mode)
text.append(model)

i = range(1, 40)
setString = '   set NEWCOUP '
setStrings = [setString + str(j) + '\n' for j in i]


with open('wilsonParams.dat', 'w') as f:
    for x in params:
        f.write(str(x)+',')

# These are the wilson coefficients, all 39 in the HEL_UFO model
# for MadGraph5
wilsonCo = [[] for _ in range(39)]
for x in params:
    wilsonCo[x-1] = [random.uniform(0.0, 0.3) for _ in range(terms-1)]
    wilsonCo[x-1].insert(0,0.0) 

setStrings = []


for j in range(0,terms):
    for i in range(1,40):
        if i in params:
            setStrings.append(setString + str(i) + ' ' + str(wilsonCo[i-1][j]) + '\n')
        else:
            setStrings.append(setString + str(i) + ' ' + '0.' + '\n')

with open('reweight_card.dat', 'w') as file:
    for line in text:
        file.write(line)
    for j in range(0,len(launches)):
        file.write(launches[j])
        for i in range(0,39):
            file.write(setStrings[i+j*39])
        file.write('\n')
    for line in footer:
        file.write(line)
file.close()
