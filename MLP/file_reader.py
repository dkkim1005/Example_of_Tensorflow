#!/usr/bin/python2.7
import numpy as np

def convertData(nameFileRed,nameFileBlue,dtype='float32'):
    def returnNumRow(nameFile):
        with open(nameFile) as f:
            temp = f.readlines()
            num = len(temp)
        return num 

    def readData(nameFile,numRow):
        data = np.zeros([numRow,2],dtype)
        with open(nameFile) as f:
            for n in xrange(numRow):
                x,y = f.readline().split()
                data[n,0] = float(x); data[n,1] = float(y);
        return data

    numRowRed = returnNumRow(nameFileRed); numRowBlue = returnNumRow(nameFileBlue);
    dataRed = readData(nameFileRed,numRowRed); dataBlue = readData(nameFileBlue,numRowBlue)

    dataTot = np.zeros([numRowRed + numRowBlue,2],dtype)
    dataTot[:numRowRed] = dataRed[:]; dataTot[numRowRed:] = dataBlue[:];

    dataTag = np.zeros([numRowRed + numRowBlue,2],dtype)
    dataTag[:numRowRed,0] = 0; 
    dataTag[:numRowRed,1] = 1;  

    dataTag[numRowRed:,0] = 1;
    dataTag[numRowRed:,1] = 0;

    return dataTot, dataTag
