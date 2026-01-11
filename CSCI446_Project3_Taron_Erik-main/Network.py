import math
import numpy as np
import os
import random 
import sys
import re
import copy as cp


class Network:
    # ---------------- INSTANTIATION ------------------
    def __init__(self, model, toReport, evidence):
        self.model = model # model imported from file
        self.toReport = toReport
        self.evidence = evidence
        self.probs = {} # dictionary of dictionaries for absolute probabilities
        self.eprobs = {} # dictionary of dictionaries for absolute probabilities based upon children
        self.tprobs = {} # dictionary of true probabilities

        self.pprobs = {}

        self.eprobsUnnorm = {}
        
        self.stateCounts = {} # cxount the number of occurances of each state for gibbs sampling

        self.probsSolved = {} # tracks whether state probabilities have been solved for
        self.eprobsSolved = {} # tracks whether state probabilities have been solved for
        self.tprobsSolved = {} # tracks whether state probabilities have been solved for

        
        self.vars = [] # array of all variables in model
        self.varsStates = {} # dictionary providing all states names for each variable
        self.varsNumStates = {} # records the number of states for each variable
        self.varParents = {} # records parents associated with each variable
        self.varNumParents = {} # records number of parents associated with each variable
        
        self.varChildren = {} # records children associated with each variable
        self.varNumChildren = {}
        
        self.varCDPs = {} # TabularCDP datastructure for each variable
        self.varCDPsVals = {} # For each variable, stores the probabilities of each state 
        # for each combination of states for all parents. Parent order should be identical 
        # to that in self.varParents. 
        self.parentsStates = {} # redundant with varsStates, but stores ordered states for all parents of each variable
        self.parentsNumStates = {} # redundant with varsNumStates, but stores number of states for all parents of each variable

        self.childrenStates = {} # redundant with varsStates, but stores ordered states for all children of each variable
        self.childrenNumStates = {} # redundant with varsNumStates, but stores number of states for all children of each variable

        self.varsWithEvidence = [] # list of all variables with evidence
        self.evidenceDict = {} # dictionary of evidence variables and associated confirmed states
        self.isEvidenceDict = {} # dictionary containing True/False values for whether evidence was provided

        self.probsStateDistributions = {} # for some child, for some state, the probability that this state being confirmed is associated with
        # this probability distribution (as provided in the CDP ordering) is this.
        self.numProbsStateDistributions = {}

        for piece in self.evidence:
            self.varsWithEvidence.append(piece[0])
            #print("evidenceVar: " + str(piece[0]))
            #print("evidenceState: " + str(piece[1]))
            evidenceVar = piece[0]
            evidenceStateIndexF = 1
            evidenceState = piece[evidenceStateIndexF]
            evidenceStateName = evidenceState
            if (evidenceState[0] == "“"):
                evidenceStateName = evidenceState[1:]
                while (evidenceState[-1] != "”"):
                    evidenceStateIndexF += 1
                    evidenceState = piece[evidenceStateIndexF]
                    evidenceStateName += "=" + evidenceState
                evidenceStateName = evidenceStateName[:-1]
                
            self.evidenceDict.update({evidenceVar : evidenceStateName})
        
        # for each state for each variable
        for var in model.nodes(): # for every variable in tree
            varcdp = model.get_cpds(var) # get given cdps
            varStates = varcdp.state_names[var] # to get possible states
            varProbDict = {} # initialize absolute probability dictionary
            varPProbDict = {} # initialize absolute probability dictionary
            varEProbDict = {} # initialize absolute probability dictionary
            varEProbUnnormDict = {} # initialize absolute probability dictionary
            varTProbDict = {} # initialize absolute probability dictionary
            varStateCounts = {} # initialize storage for counting states on vars
            newVarStates = []
            for state in varStates: # for all possible states
                varProbDict.update({state : -1}) # initialize as unknown, only update once
                varPProbDict.update({state : -1}) # initialize as unknown, only update once
                varEProbDict.update({state : -1}) # initialize as unknown, only update once
                varEProbUnnormDict.update({state : -1}) # initialize as unknown, only update once
                varTProbDict.update({state : -1}) # initialize as unknown, only update once
                varStateCounts.update({state : 0}) # initialize as 0
                newVarStates.append(state)
                
            self.probs.update({var : varProbDict})
            self.eprobs.update({var : varEProbDict})
            self.tprobs.update({var : varTProbDict})
            self.eprobsUnnorm.update({var : varEProbUnnormDict})
            self.pprobs.update({var : varPProbDict})
            
            self.stateCounts.update({var : varStateCounts})

            self.probsSolved.update({var: False})
            self.eprobsSolved.update({var: False})
            self.tprobsSolved.update({var: False})
                
            self.vars.append(var)
            self.varsStates.update({var : newVarStates})
            self.varsNumStates.update({var : len(newVarStates)})

            self.varParents.update({var : model.get_parents(var)})
            self.varNumParents.update({var: len(model.get_parents(var))})
            
            self.varChildren.update({var : model.get_children(var)})
            self.varNumChildren.update({var: len(model.get_children(var))})
            
            self.varCDPs.update({var: varcdp})
            self.varCDPsVals.update({var : varcdp.get_values()})

            self.probsStateDistributions.update({var: {}})

        for var in model.nodes(): # for every variable in tree (again, but separate due to previous datastructure dependencies)
            parStatesDict = {}
            parNumStatesDict = {}
            childStatesDict = {}
            childNumStatesDict = {}
            for parent in self.varParents[var]:
                parStatesDict.update({parent : self.varsStates[parent]})
                parNumStatesDict.update({parent : self.varsNumStates[parent]}) 

            for child in self.varChildren[var]:
                childStatesDict.update({child : self.varsStates[child]})
                childNumStatesDict.update({child : self.varsNumStates[child]})

            self.parentsStates.update({var: parStatesDict})
            self.parentsNumStates.update({var: parNumStatesDict})
            
            self.childrenStates.update({var: childStatesDict})
            self.childrenNumStates.update({var: childNumStatesDict})

            self.isEvidenceDict.update({var: False})

        for var in self.varsWithEvidence:
            self.isEvidenceDict[var] = True
        

    # ---------------- END INSTANTIATION ------------------

    # ---------------- GET METHODS ------------------
    def getVars(self):
        return self.vars # return list of all variables in network
    
    def getProbs(self):
        return self.probs # returns all absolute probabilities
    
    def getVarProbs(self, var): # returns probabilities for all states of a variable
        return self.tprobs[var]
    
    def getStateProb(self, var, state): # returns probability of a specific state on a specific variable
        return (self.getVarProbs(var)[state])
    
    def getVarsStates(self): # returns all possible states for all possible variables
        return self.varsStates
    
    def getVarStates(self, var): # returns all possible states for a specific variable
        return self.varsStates[var]
    
    # ---------------- END GET METHODS ------------------

    # ---------------- SET METHODS ------------------

    
    # ---------------- END SET METHODS ------------------

    # ************************** SHARED METHODS *******************************


    # ************************** END SHARED METHODS *******************************

    # ************************** VARIABLE ELIMINATION METHODS *******************************

    # ------------------------ DO VARIABLE ELIMINATION ---------------------------------
    def doVariableElim(self):
        for report in self.toReport:
            # Solve the variable with VE
            self.variableElim(report)
            print("Variable = " + str(report))
            print("Found Probabilities: " + str(self.getVarProbs(report)))
        
        return
    # ------------------------ END DO VARIABLE ELIMINATION ---------------------------------
    
    # ------------------------ VARIABLE ELIMINATION ---------------------------------
    def variableElim(self, report): # returns probability distribution of variable selected
        
        for parent in self.varParents[report]: # start at top of tree, work back down to variable in question
            if (not self.probsSolved[parent]): # if parent probability has not been solved for
                self.variableElim(parent)

        localDistribution = np.zeros(self.varsNumStates[report], dtype=float)

        
        if (not self.probsSolved[report]): # if we have not solved for this forward probability
            stateNumber = 0
            for state in self.varsStates[report]: # get forward probability (based purely on parent evidence)           
                # print("state: " + str(state))
                self.probs[report][state] = self.computeStateProbabilityFromParents(report, state, stateNumber)
                stateNumber += 1
            self.probsSolved[report] = True

        # print(self.probs[report])

        for child in self.varChildren[report]: # following all parents, solve down for children. Then, solve backwards
            if (not self.eprobsSolved[child]): # if child backwards probability has not been found
                self.variableElim(child)

        if (not self.eprobsSolved[report]): # if we have not solved for this backward probability
            stateNumber = 0 
            eUnnormalizedSum = 0
            for state in self.varsStates[report]: # get backwards probability (extracted from child evidence, based on that parent evidence)           
                # print("state: " + str(state))
                self.eprobs[report][state] = self.computeStateProbabilityFromChildren(report, state, stateNumber)
                self.eprobsUnnorm[report][state] = cp.copy(self.eprobs[report][state])
                eUnnormalizedSum += self.eprobs[report][state]
                stateNumber += 1

            for state in self.varsStates[report]:
                if (eUnnormalizedSum > 0):
                    self.eprobs[report][state] = self.eprobs[report][state] / eUnnormalizedSum
                else:
                    self.eprobs[report][state] = 1 / self.varsNumStates[report]
                    
            self.eprobsSolved[report] = True


        for parent in self.varParents[report]: # start at top of tree, work back down to variable in question
            if (not self.tprobsSolved[parent]): # if parent probability has not been solved for
                self.variableElim(parent)
        
        stateNumber = 0
        for state in self.varsStates[report]:
            if (self.varNumParents[report] == 0):
                self.tprobs[report][state] = self.probs[report][state] * self.eprobsUnnorm[report][state]
            else:
                self.tprobs[report][state] = self.computeTrueState(report, state, stateNumber) * self.eprobsUnnorm[report][state] 
                stateNumber += 1

        
        tUnnormalizedSum = 0.0
        for state in self.varsStates[report]:
            tUnnormalizedSum += self.tprobs[report][state]
        
        if tUnnormalizedSum > 0:
            for state in self.varsStates[report]:
                self.tprobs[report][state] = self.tprobs[report][state] / tUnnormalizedSum
        # print("report: " + str(report))
        # print("eprobs: " + str(self.eprobs[report]))
        # print("tprobs: " + str(self.tprobs[report]))
        
        self.tprobsSolved[report] = True
        
        return

        
    # ------------------------ END VARIABLE ELIMINATION ---------------------------------

    # ------------------------ COMPUTE STATE PROBABILITY ---------------------------------
    def computeStateProbabilityFromParents(self, var, state, stateNumber): # CURRENTLY: returns probability of state on variable with the assumption that 
        # all parents are solved for
        # used to update self.Eprobs
        
        parentsNumsStates = np.zeros(self.varNumParents[var], dtype=int) # number of possible states for each parent
        currentParStateNums = np.zeros(self.varNumParents[var], dtype=int) # current indexes of states for each parent
        numProbsToSum = 1
        parentIndex = 1
        for parent in self.varParents[var]:
            parentsNumsStates[self.varNumParents[var] - parentIndex] = self.varsNumStates[parent] # order of parent's states must be reversed to match 
            # indexing of pgmpy's cdp
            numProbsToSum = self.varsNumStates[parent] * numProbsToSum
            parentIndex += 1
            
        probsToSum = np.zeros(numProbsToSum, dtype=float)

        overflow = 0
        #print()
        # print("Var: " + var)
        # print("parents: " + str(self.varParents[var]))
        for parentsComboIndex in range(numProbsToSum):
            thisProbability = 1
            theseParentStatesProbability = 1
            
            for parentIndexI in range(self.varNumParents[var]): # get each parent state corresponding to the current index
                currentParStateNums[parentIndexI] += overflow
                if(currentParStateNums[parentIndexI] == parentsNumsStates[parentIndexI]):
                    currentParStateNums[parentIndexI] = 0
                    overflow = 1
                else:
                    overflow = 0
                    
            overflow = 1
                
            parentIndexI = 0
            #print(parentsComboIndex)
            for parent in reversed(self.varParents[var]): # multiply the probability of each parent state at the current state index
                parentStateName = self.varsStates[parent][currentParStateNums[parentIndexI]] # get state name of current parent state's index
                theseParentStatesProbability = theseParentStatesProbability * self.probs[parent][parentStateName]
                parentIndexI += 1 # -= 1
            
            thisProbability = self.varCDPsVals[var][stateNumber][parentsComboIndex] * theseParentStatesProbability
            probsToSum[parentsComboIndex] = thisProbability # probability given this set of parents

        probability = 0
        for i in range(numProbsToSum):
            probability += probsToSum[i]

        uniformProbsToSum = np.zeros(numProbsToSum, dtype=float)
        if (probability != 0):     
            for i in range(numProbsToSum):
                uniformProbsToSum[i] = (probsToSum[i]/probability)
        else:
            for i in range(numProbsToSum):
                uniformProbsToSum[i] = (1/numProbsToSum)
        #print("Uniform to sum: " + str(uniformProbsToSum))
        self.probsStateDistributions[var].update({state: uniformProbsToSum})
        self.numProbsStateDistributions.update({var: len(probsToSum)})

        self.pprobs[var][state] = probability
        if (self.isEvidenceDict[var]): # if probability is evidence, return a certainty
            if (state == self.evidenceDict[var]):
                probability = 1.00
                return probability
            else:
                probability = 0.00
                return probability
        
        return probability


    def computeStateProbabilityFromChildren(self, var, state, stateNumber):
        if self.isEvidenceDict[var]:
            if state == self.evidenceDict[var]:
                return 1.00 
            else:
                return 0.00 

        numberOfChildren = self.varNumChildren[var]
        if numberOfChildren == 0:
            # return self.probs[var][state]
            return 1.00
    
        probsOfStateOnEachChild = np.zeros(numberOfChildren, dtype=float)
        childIndex = 0

        for child in self.varChildren[var]:
            childNumDists = self.numProbsStateDistributions[child]
            childNumStates = self.varsNumStates[child]
            childStates = self.varsStates[child]
    
            childProbsOfDists = np.zeros(childNumDists, dtype=float)
            for parentIndex in range(childNumDists):
                for stateIndex in range(childNumStates):
                    stateName = childStates[stateIndex]
                    # probability of state given distribution times probability of state
                    childProbsOfDists[parentIndex] += self.varCDPsVals[child][stateIndex][parentIndex] * self.eprobs[child][stateName]
    
            # Decode parentIndex back to parent states 
            childParentsNumbersStates = list(self.parentsNumStates[child].values()) # number of states on each parent on child
            childNumParents = self.varNumParents[child]
            parentPosition = 0
            childParentsOrder = self.varParents[child] # order of parents on child
            while var != childParentsOrder[parentPosition]:
                parentPosition += 1
            parentPosition = childNumParents - parentPosition - 1  # Reversed pos
    
            parentsStatesNums = np.array(list(reversed(childParentsNumbersStates)), dtype=int)
    
            stateProbSum = 0.0
            for parentIndex in range(childNumDists):
                remainder = parentIndex
                parentStates = np.zeros(childNumParents, dtype=int)
                for parentIndexAssorted in range(childNumParents):
                    parentStates[parentIndexAssorted] = remainder % parentsStatesNums[parentIndexAssorted]
                    remainder //= parentsStatesNums[parentIndexAssorted]
                if parentStates[parentPosition] == stateNumber:
                    stateProbSum += childProbsOfDists[parentIndex]
    
            probsOfStateOnEachChild[childIndex] = stateProbSum
            childIndex += 1
    
        # Product over children
        totalLikelihood = 1.0
        for prob in probsOfStateOnEachChild:
            totalLikelihood *= prob
    
        return totalLikelihood

        
    def computeTrueState(self, var, state, stateNumber): # Gets true probability using top-down recursion, from eprobs
        
        parentsNumsStates = np.zeros(self.varNumParents[var], dtype=int) # number of possible states for each parent
        currentParStateNums = np.zeros(self.varNumParents[var], dtype=int) # current indexes of states for each parent
        numProbsToSum = 1
        parentIndex = 1
        for parent in self.varParents[var]:
            parentsNumsStates[self.varNumParents[var] - parentIndex] = self.varsNumStates[parent] # order of parent's states must be reversed to match 
            # indexing of pgmpy's cdp
            numProbsToSum = self.varsNumStates[parent] * numProbsToSum
            parentIndex += 1
            
        probsToSum = np.zeros(numProbsToSum, dtype=float)

        overflow = 0
        #print()
        # print("Var: " + var)
        # print("parents: " + str(self.varParents[var]))
        for parentsComboIndex in range(numProbsToSum):
            thisProbability = 1
            theseParentStatesProbability = 1
            
            for parentIndexI in range(self.varNumParents[var]): # get each parent state corresponding to the current index
                currentParStateNums[parentIndexI] += overflow
                if(currentParStateNums[parentIndexI] == parentsNumsStates[parentIndexI]):
                    currentParStateNums[parentIndexI] = 0
                    overflow = 1
                else:
                    overflow = 0
                    
            overflow = 1
                
            parentIndexI = 0
            #print(parentsComboIndex)
            for parent in reversed(self.varParents[var]): # multiply the probability of each parent state at the current state index
                parentStateName = self.varsStates[parent][currentParStateNums[parentIndexI]] # get state name of current parent state's index
                theseParentStatesProbability = theseParentStatesProbability * self.tprobs[parent][parentStateName]
                parentIndexI += 1 # -= 1
            
            thisProbability = self.varCDPsVals[var][stateNumber][parentsComboIndex] * theseParentStatesProbability
            probsToSum[parentsComboIndex] = thisProbability # probability given this set of parents

        probability = 0
        for i in range(numProbsToSum):
            probability += probsToSum[i]

            
        uniformProbsToSum = np.zeros(numProbsToSum, dtype=float)
        if (probability != 0):     
            uniformProbsToSum = np.zeros(numProbsToSum, dtype=float)
            for i in range(numProbsToSum):
                uniformProbsToSum[i] = (probsToSum[i]/probability)
        else:
            for i in range(numProbsToSum):
                uniformProbsToSum[i] = (1/numProbsToSum)
        
        self.probsStateDistributions[var][state] = probsToSum
        #self.numProbsStateDistributions.update({var: len(probsToSum)})
        if (self.isEvidenceDict[var]): # if probability is evidence, return a certainty
            if (state == self.evidenceDict[var]):
                probability = 1.00
                return probability
            else:
                probability = 0.00
                return probability
        
        return probability
    # ------------------------ END COMPUTE STATE PROBABILITY ---------------------------------   
    


    # ************************** END VARIABLE ELIMINATION METHODS *******************************

    # ************************** GIBBS SAMPLING METHODS *******************************

    # ------------------------ DO GIBBS SAMPLING ---------------------------------

    def doGibbsSample(self):
        numberSamples = 50000
        numberToIgnore = numberSamples/10
        self.gibbsSampling(numberSamples, numberToIgnore) # Solve the variable with GS
        
        for var in self.toReport:
            # Perform Division across samples on reported variable, save to tprobs

            totalSamplesOnReported = 0
            for state in self.varsStates[var]:
                totalSamplesOnReported += self.stateCounts[var][state]
            
            for state in self.varsStates[var]:
                self.stateCounts[var][state] = self.stateCounts[var][state] / totalSamplesOnReported
                self.tprobs[var][state] = self.stateCounts[var][state]

            
            
            print("Variable = " + str(var))
            print("Found Probabilities: " + str(self.getVarProbs(var)))
        
        return

    def gibbsSampling(self, numberOfSamples, numberToIgnore):
        for var in self.varsWithEvidence: # initialize evidence
            for state in self.varsStates[var]:
                self.probs[var][state] = 0
                self.eprobs[var][state] = 0
                self.tprobs[var][state] = 0
                
            state = self.evidenceDict[var]
            self.probs[var][state] = 1
            self.eprobs[var][state] = 1
            self.tprobs[var][state] = 1
            self.stateCounts[var][state] = 1

        for var in self.vars:
            if (not self.isEvidenceDict[var]):
                stateInit = random.randint(1, self.varsNumStates[var]) - 1 # generate random starting variables for non-evidence
                stateInit = self.varsStates[var][stateInit]
                for state in self.varsStates[var]:
                    if (state == stateInit):
                        self.probs[var][state] = 1
                        self.eprobs[var][state] = 1
                        self.tprobs[var][state] = 1
                        self.pprobs[var][state] = 1
                    else:
                        self.probs[var][state] = 0
                        self.eprobs[var][state] = 0
                        self.tprobs[var][state] = 0
                        self.pprobs[var][state] = 0

        networkSize = len(self.vars) # number of variables in network

        for i in range(numberOfSamples): # big number, consider scaling to network size
            varNumberChange = random.randint(1, networkSize) - 1
            varToChange = self.vars[varNumberChange]
            
            if (not self.isEvidenceDict[varToChange]): # if it not evidence, we can update it

                stateNumber = 0
                for state in self.varsStates[varToChange]:  
                    self.probs[varToChange][state] = self.computeStateProbabilityFromParents(varToChange, state, stateNumber)
                    stateNumber += 1
                
                for child in self.varChildren[varToChange]:
                    childStateNumber = 0
                    for state in self.varsStates[child]:
                        self.probs[child][state] = self.computeStateProbabilityFromParents(child, state, childStateNumber)
                        childStateNumber += 1
                        for grandchild in self.varChildren[child]:
                            if (not grandchild in self.numProbsStateDistributions): 
                                numGrandChildDists = 1
                                for parentOfChild in self.varParents[grandchild]:
                                    numGrandChildDists *= self.parentsNumStates[grandchild][parentOfChild]
                                self.numProbsStateDistributions[grandchild] = numGrandChildDists
                            if (self.isEvidenceDict[grandchild]):
                                grandStateNumber = 0
                                for grandState in self.varsStates[grandchild]:
                                    self.computeStateProbabilityFromParents(grandchild, grandState, grandStateNumber)
                                    grandStateNumber += 1
                
                for child in self.varChildren[varToChange]:
                    childStateNumber = 0
                    for state in self.varsStates[child]:
                        self.eprobsUnnorm[child][state] = self.computeStateProbabilityFromChildrenGibbs(child, state, childStateNumber)
                        childStateNumber += 1
                
                stateNumber = 0
                eUnnormalizedSum = 0.0
                for state in self.varsStates[varToChange]:  
                    self.eprobsUnnorm[varToChange][state] = self.computeStateProbabilityFromChildrenGibbs(varToChange, state, stateNumber)
                    eUnnormalizedSum += self.eprobsUnnorm[varToChange][state]
                    stateNumber += 1

                if eUnnormalizedSum == 0.0:
                    numStates = float(self.varsNumStates[varToChange])
                    for state in self.varsStates[varToChange]:
                        self.eprobs[varToChange][state] = 1.0 / numStates
                else:
                    for state in self.varsStates[varToChange]:
                        self.eprobs[varToChange][state] = self.eprobsUnnorm[varToChange][state] / eUnnormalizedSum

                tUnnormalizedSum = 0.0
                for state in self.varsStates[varToChange]:
                    finalProb = self.eprobs[varToChange][state] * self.probs[varToChange][state]
                    self.tprobs[varToChange][state] = finalProb
                    tUnnormalizedSum += finalProb

                if tUnnormalizedSum == 0.0:
                    # Everything collapsed to zero: use uniform distribution
                    numStates = float(self.varsNumStates[varToChange])
                    for state in self.varsStates[varToChange]:
                        self.tprobs[varToChange][state] = 1.0 / numStates
                else:
                    # Normalize by sum
                    for state in self.varsStates[varToChange]:
                        self.tprobs[varToChange][state] = self.tprobs[varToChange][state] / tUnnormalizedSum

                newState = self.getRandomFromDist(varToChange, self.tprobs[varToChange])

                if (numberToIgnore == 0):
                    self.stateCounts[varToChange][newState] += 1
                else: 
                    numberToIgnore -= 1

                for state in self.varsStates[varToChange]: 
                    if (state == newState):
                        self.probs[varToChange][state] = 1 # varToChange
                        self.eprobs[varToChange][state] = 1
                        self.tprobs[varToChange][state] = 1
                        self.pprobs[varToChange][state] = 1
                    else:
                        self.probs[varToChange][state] = 0
                        self.eprobs[varToChange][state] = 0
                        self.tprobs[varToChange][state] = 0
                        self.pprobs[varToChange][state] = 0
                        
                for child in self.varChildren[varToChange]: 
                    for state in self.varsStates[child]:
                        if (self.tprobs[child][state] == 0):
                            self.probs[child][state] = 0 # varToChange
                            self.eprobs[child][state] = 0
                            self.tprobs[child][state] = 0
                            self.pprobs[child][state] = 0
                        else:
                            self.probs[child][state] = 1
                            self.eprobs[child][state] = 1
                            self.tprobs[child][state] = 1
                            self.pprobs[child][state] = 1
                
    
    # ------------------------ END DO GIBBS SAMPLING ---------------------------------
    def getRandomFromDist(self, var, dist):
        probNumber = 0
        selections = np.zeros(1000, dtype=int)
        selectionsPlaced = 0
        for state in self.varsStates[var]:

            prob = dist[state]
            numberSelectionSpaces = int(prob * 1000)
            for i in range(numberSelectionSpaces):
                selections[selectionsPlaced] = probNumber
                selectionsPlaced += 1
            probNumber += 1

        selectedIndex = random.randint(0, selectionsPlaced - 1)

        selectedStateNum = selections[selectedIndex]

        selectedState = self.varsStates[var][selectedStateNum]

        return selectedState

    def computeStateProbabilityFromChildrenGibbs(self, var, state, stateNumber):
        numberOfChildren = self.varNumChildren[var]
        if numberOfChildren == 0:
            return 1.00
   
        probsOfStateOnEachChild = np.zeros(numberOfChildren, dtype=float)
        childIndex = 0
        probVarStateFromParents = self.probs[var][state]
        for child in self.varChildren[var]:
            childNumDists = self.numProbsStateDistributions[child]
            childNumStates = self.varsNumStates[child]
            childStates = self.varsStates[child]
            isEvidence = self.isEvidenceDict[child]
            isLeaf = (self.varNumChildren[child] == 0)
            stateProbSum = 0.0
           
            if (not isLeaf):
   
                effectivePriorLower = 0.0
                probVarStateGivenLower = 0.0
               
                for stateIndex in range(childNumStates):
                    stateName = childStates[stateIndex]
                    unnormChildSignal = self.eprobsUnnorm[child][stateName]
                    parentsComputedChildProb = self.probs[child][stateName]
   
                    effectivePriorLower += parentsComputedChildProb * unnormChildSignal
                   
                childParentsNumbersStates = list(self.parentsNumStates[child].values()) # number of states on each parent on child
                childNumParents = self.varNumParents[child]
                parentPosition = 0
                childParentsOrder = self.varParents[child] # order of parents on child
                while var != childParentsOrder[parentPosition]:
                    parentPosition += 1
                parentPosition = childNumParents - parentPosition - 1 # Reversed pos
       
                parentsStatesNums = np.array(list(reversed(childParentsNumbersStates)), dtype=int)
   
                if (effectivePriorLower > 0 and probVarStateFromParents > 0):
                    for parentComboIndex in range(childNumDists):
                        remainder = parentComboIndex
                        parentStates = np.zeros(childNumParents, dtype=int)
                        for parentIndexAssorted in range(childNumParents):
                            parentStates[parentIndexAssorted] = remainder % parentsStatesNums[parentIndexAssorted]
                            remainder //= parentsStatesNums[parentIndexAssorted]
                           
                        assigned = (parentStates[parentPosition] == stateNumber)
                        distributionContribution = 0.0
                        for stateIndex in range(childNumStates):
                            stateName = childStates[stateIndex]
                            childProbDistGivenState = self.probsStateDistributions[child][stateName][parentComboIndex] # probability of distribution given state
                            unnormChildSignal = self.eprobsUnnorm[child][stateName]
                            parentsComputedChildProb = self.probs[child][stateName]
                            distributionContribution += childProbDistGivenState * unnormChildSignal * parentsComputedChildProb / effectivePriorLower
                        if assigned:
                            probVarStateGivenLower += distributionContribution
                    stateProbSum = probVarStateGivenLower * effectivePriorLower / probVarStateFromParents
                    if (isEvidence):
                        ChildEvidenceState = self.evidenceDict[child]
                        stateProbSum *= self.pprobs[child][ChildEvidenceState]
                else:
                    stateProbSum = 0.00
               
            else:
                if (isEvidence):
                    probParentGivenEvidence = 0.00
                    ChildEvidenceState = self.evidenceDict[child]
                   
                    childParentsNumbersStates = list(self.parentsNumStates[child].values()) # number of states on each parent on child
                    childNumParents = self.varNumParents[child]
                    parentPosition = 0
                    childParentsOrder = self.varParents[child] # order of parents on child
                    while var != childParentsOrder[parentPosition]:
                        parentPosition += 1
                    parentPosition = childNumParents - parentPosition - 1 # Reversed pos
           
                    parentsStatesNums = np.array(list(reversed(childParentsNumbersStates)), dtype=int)
                    for parentComboIndex in range(childNumDists):
                        remainder = parentComboIndex
                        parentStates = np.zeros(childNumParents, dtype=int)
                        for parentIndexAssorted in range(childNumParents):
                            parentStates[parentIndexAssorted] = remainder % parentsStatesNums[parentIndexAssorted]
                            remainder //= parentsStatesNums[parentIndexAssorted]
                           
                        assigned = (parentStates[parentPosition] == stateNumber)
                        if assigned:
                            probParentGivenEvidence += self.probsStateDistributions[child][ChildEvidenceState][parentComboIndex]
                    if (probVarStateFromParents > 0):
                        evidenceProbFromParents = self.pprobs[child][ChildEvidenceState]
                        stateProbSum = probParentGivenEvidence * evidenceProbFromParents / probVarStateFromParents # 1, in this case
                    else:
                        stateProbSum = 0.0
                else:
                    stateProbSum = 1.00
               
                   
            probsOfStateOnEachChild[childIndex] = stateProbSum
            childIndex += 1
   
        # Product over children
        totalLikelihood = 1.0
        for prob in probsOfStateOnEachChild:
            totalLikelihood *= prob

        # doesn't actually matter
        if self.isEvidenceDict[var]:
            if state == self.evidenceDict[var]:
                return 1.00
            else:
                return 0.00
   
        return totalLikelihood
        


    # ************************** END GIBBS SAMPLING METHODS *******************************
    
    
    