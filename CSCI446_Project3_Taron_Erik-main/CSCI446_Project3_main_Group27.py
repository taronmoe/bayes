import math
import numpy as np
import os
import random 
import sys
import re
import copy as cp
from Network import Network
from pgmpy.readwrite import BIFReader

def fileImport(NETWORK_NAME, toReport, EVIDENCE): # imports file as pgmpy.models.BayesianNetwork.BayesianNetwork datatype type
    
    reader = BIFReader(NETWORK_NAME) # https://pgmpy.org/readwrite/bif.html
    model = reader.get_model()

    evidence = getEvidence(EVIDENCE)
    
    newNetwork = Network(model, toReport, evidence)
        
    return newNetwork

def getVarsToReport(REPORT):
    toReport = REPORT.split(' ; ')
    
    return toReport

def getEvidence(EVIDENCE):
    evidenceList = EVIDENCE.split('; ')
    # print("Evidence List: " + evidenceList)
    evidence = []
    for part in evidenceList:
        evidence.append(part.split('='))
    if EVIDENCE == '':
        evidence = []
    # print(evidence)
    return evidence

def getReports(toReport, network):
    
    reportString = ""
    
    for var in toReport: # for each variable we want to report
        reportString += var 
        varStates = network.getVarStates(var) # gets possible states for our variable
        varProbs = []
        for state in varStates:
            stateProb = network.getStateProb(var, state)
            varProbs.append(stateProb)
            reportString += ", "
            reportString += state
        reportString += "\n"
        reportString += "("
        for prob in varProbs:
            reportString += str(round(prob, 2)) + ", "
        reportString = reportString[:-2]
        reportString += ")"
        reportString += "\n"
        print(reportString)
        
    return reportString
        
    
def saveOutput(GROUP_ID, ALGORITHM, NETWORK_NAME, EVIDENCE_LEVEL, toReport, network): 
    # saves solved network reported variables to output file
    # Formatted name as: [GROUP_ID]_[ALGORITHM]_[NETWORK_NAME]_[EVIDENCE_LEVEL].csv
    netPathArr = NETWORK_NAME.split('/')
    netName = netPathArr[-1]
    fileName = GROUP_ID + "_" + ALGORITHM + "_" + netName + "_" + EVIDENCE_LEVEL + ".csv"
    writeString = getReports(toReport, network)
        
    with open(fileName, "w") as f:
        f.write(writeString)
        
    return

def main(GROUP_ID, ALGORITHM, NETWORK_NAME, REPORT, EVIDENCE_LEVEL, EVIDENCE): 

    toReport = getVarsToReport(REPORT)
    network = fileImport(NETWORK_NAME, toReport, EVIDENCE)
    
    if (ALGORITHM == 've'):
        # code to run variable elimination
        # will be method on Network class
        print("run VE")
        network.doVariableElim()
        
    elif (ALGORITHM == 'gibbs'):
        # code to run gibbs sampling
        # will be method on Network class
        print("run gibbs")
        network.doGibbsSample()
        
    else:
        print("Not a valid algorithm. Terminating...")
        sys.exit() # exit program



    saveOutput(GROUP_ID, ALGORITHM, NETWORK_NAME, EVIDENCE_LEVEL, toReport, network)
    
    
    
    
    