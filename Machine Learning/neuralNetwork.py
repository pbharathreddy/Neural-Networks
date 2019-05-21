import pandas
import numpy as np
import random

class neural:

    dataset = []
    input = []
    output = []
    networkOutput = []
    #add layer weights here
    layer1Weights = []
    layer2Weights = []
    layer3Weights = []
    error = []

    def __init__(self,dataset):
        self.dataset = dataset
        self.dataset= np.array(dataset)
        self.dataset = pandas.DataFrame(self.dataset, columns=['r', 'g', 'b', 'grey'])
        self.input, self.output = self.formatData(self.dataset)
        self.layer1Weights = self.createLayer(10,3)#change the number of nodes here
        self.layer3Weights = self.createLayer(5,10)
        self.layer2Weights = self.createLayer(3,5)


    def errorResult(self,input):
        errorMatrix = input-self.output
        red,green,blue = 0,0,0

        for i in errorMatrix:
            red += i[0]*i[0]
            green += i[1]*i[1]
            blue += i[2]*i[2]

        meanSqrError = np.array([red**(1/2),green**(1/2),blue**(1/2)])
        return (meanSqrError)

    def layerOutputResult(self,input):
        layerResult = []
        for i in range(len(input)):
            temp = []
            for j in range(len(self.layer2Weights)):
                temp.append(self.layer2Weights[j].dot(input[i][j]))
            #relu function
            layerResult.append(sum(temp))
        layerResult = np.array(layerResult)
        for i in range(len(layerResult)):
            for j in range(len(layerResult[i])):
                if layerResult[i][j]<0:
                    layerResult[i][j]=0
        return np.array(layerResult)

    def layerInputResult(self,input):
        layerResult = []
        for i in range(len(input)):
            temp = []
            temp1 = self.layer1Weights[0].dot(input[i][0])
            temp2 = self.layer1Weights[1].dot(input[i][1])
            temp3 = self.layer1Weights[2].dot(input[i][2])
            temp = temp1+temp2+temp3
            layerResult.append(temp)
        layerResult = np.array(layerResult)

        for i in range(len(layerResult)):
            for j in range(len(layerResult[i])):
                if layerResult[i][j]<0:
                    layerResult[i][j]=0
        return layerResult

    def layerMiddleResult(self, input):
        layerResult = []
        for i in range(len(input)):
            temp = []
            for j in range(len(self.layer3Weights)):
                temp.append(self.layer3Weights[j].dot(input[i][j]))
                # relu function
            layerResult.append(sum(temp))
        layerResult = np.array(layerResult)
        for i in range(len(layerResult)):
            for j in range(len(layerResult[i])):
                if layerResult[i][j] < 0:
                    layerResult[i][j] = 0
        return layerResult


    def createLayer(self,nodes,inputToLayer):
        layerWeights = []
        for i in range(inputToLayer):
            nodeWeights = []
            for j in range(nodes):
                nodeWeights.append(round(random.uniform(0.2,0.5),2))
            layerWeights.append(nodeWeights)
        return np.array(layerWeights)



    def formatData(self,dataset):
        input = dataset['grey']
        input = np.array(input)
        finalInput = []
        tempInput = []
        count = 1
        for i in range(len(input)):
            tempInput.append(input[i])
            if count%3 == 0 and count != 0:
                finalInput.append(tempInput)
                tempInput = []
            count+=1
        input = np.array(finalInput)
        output = dataset.drop(['grey'], axis=1)
        output = np.array(output)
        temp = []
        for i in range(0, len(output), 3):
            temp.append((output[i] + output[i + 1] + output[i + 2]) / 3)
        output = np.array(temp)
        return input, output

dataSet1 = pandas.read_csv('D:\Masters\Projects\\520\Machine Learning\DataSet\image0.csv',index_col=[0])
dataSet2 = pandas.read_csv('D:\Masters\Projects\\520\Machine Learning\DataSet\image1.csv',index_col=[0])
dataSet3 = pandas.read_csv('D:\Masters\Projects\\520\Machine Learning\DataSet\image2.csv',index_col=[0])

network = neural(dataSet1)

for xyz in range(200):

    data = network.layerOutputResult(network.layerMiddleResult(network.layerInputResult(network.input)))
    meanSqrErr = network.errorResult(data)
    lr = .0000001
    network.layer2Weights = network.layer2Weights - lr * (meanSqrErr)
    #network.layer2Weights = network.layer2Weights - lr*(np.transpose(meanSqrErr.dot(np.transpose(sum(network.input))/3000)))
    for i in range(len(network.layer1Weights)):
        for j in range(len(network.layer1Weights[i])):
            affect = network.layer1Weights[i][j]/sum(network.layer1Weights[i])
            network.layer1Weights[i][j] = network.layer1Weights[i][j] - lr*affect*sum(meanSqrErr)
    for i in range(len(network.layer3Weights)):
        for j in range(len(network.layer3Weights[i])):
            affect = network.layer3Weights[i][j]/sum(network.layer3Weights[i])
            network.layer3Weights[i][j] = network.layer3Weights[i][j] - lr*affect*sum(meanSqrErr)
    data = pandas.DataFrame(data)
    print(meanSqrErr)



