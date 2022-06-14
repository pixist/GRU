# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:59:42 2022

@author: novar
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

class GRUlayer:
    def forward(self, x_t, h_prev, params):
        # update and reset gates
        self.z = sigmoid(np.dot(params["Wz"],x_t) + np.dot(params["Uz"], h_prev) + params["bz"])
        self.r = sigmoid(np.dot(params["Wr"],x_t) + np.dot(params["Ur"], h_prev) + params["br"])
        
        # hidden units
        self.h_ = np.tanh(np.dot(params["Wh"],x_t) + np.dot(params["Uh"], np.multiply(self.r, h_prev)) + params["bh"])
        self.h = np.multiply(self.z, h_prev) + np.multiply((1-self.z), self.h_)
        
        #hid to v
        self.v = np.dot(params["Wv"],self.h) + params["bv"]
        self.y_t = softmax(self.v)
        self.h_prev = h_prev
        self.x_t = x_t
        return self.h, self.y_t
        
    def backward(self, params, y, dh_next):
        #run forward first for creating self parameters
        #daXXX; a denotes activaiton
        grads_step = {}
        
        dv = self.y_t.copy() - y
        grads_step["dWv"] = np.dot(dv, self.h.T)
        grads_step["dbv"] = dv
        
        dh = np.dot(params["Wv"].T, dv) + dh_next
        
        dh_ = np.multiply(dh, (1 - self.z))
        dh_l = dh_ * (1-np.square((self.h_))) # try tanh squared
        
        grads_step["dWh"] = np.dot(dh_l, self.x_t.T)
        grads_step["dUh"] = np.dot(dh_l, np.multiply(self.r, self.h_prev).T)
        grads_step["dbh"] = dh_l

        drh = np.dot(params["Uh"].T, dh_l)
        dr = np.multiply(drh, self.h_prev)
        dr_l = dr * self.r*(1-(self.r)) #check again

        grads_step["dWr"] = np.dot(dr_l, self.x_t.T)
        grads_step["dUr"] = np.dot(dr_l, self.h_prev.T)
        grads_step["dbr"] = dr_l
        
        dz = np.multiply(dh, self.h_prev - self.h_)
        dz_l = dz*self.z*(1-(self.z)) #check
        
        grads_step["dWz"] = np.dot(dz_l, self.x_t.T)
        grads_step["dUz"] = np.dot(dz_l, self.h_prev.T)
        grads_step["dbz"] = dz_l
        dh_prev = (np.dot(params["Uz"].T, dz_l) + np.dot(params["Ur"].T, dr_l)
                   + np.multiply(drh, self.r) + np.multiply(dh, self.z))
    
        return dh_prev, grads_step


class GRU:
    def __init__(self, Lfeature, Lxdim, Lhid, Lclass, bptt):
        self.Lhid = Lhid
        self.Lfeature = Lfeature
        self.Lxdim = Lxdim
        self.Lclass = Lclass
        self.bptt = bptt
        
    def initializeWeights(self):
        rH = np.sqrt(1/self.Lhid)
        rX = np.sqrt(1/self.Lxdim)
        self.params = {}
        # z
        self.params["Wz"] = np.random.uniform(-rX,rX,(self.Lhid,self.Lxdim))
        self.params["Uz"] = np.random.uniform(-rH,rH,(self.Lhid, self.Lhid))
        self.params["bz"] = np.ones((self.Lhid,1))
        # r
        self.params["Wr"] = np.random.uniform(-rX,rX,(self.Lhid,self.Lxdim))
        self.params["Ur"] = np.random.uniform(-rH,rH,(self.Lhid, self.Lhid))
        self.params["br"] = np.ones((self.Lhid,1))
        # h
        self.params["Wh"] = np.random.uniform(-rX,rX,(self.Lhid,self.Lxdim))
        self.params["Uh"] = np.random.uniform(-rH,rH,(self.Lhid, self.Lhid))
        self.params["bh"] = np.ones((self.Lhid,1))
        # hid to out
        self.params["Wv"] = np.random.uniform(-rH,rH,(self.Lclass,self.Lhid))
        self.params["bv"] = np.ones((self.Lclass,1))
        # grads
        self.grads = {}
        self.gradsNew = {}
        for key in self.params:
            self.grads["d"+key] = np.zeros(self.params[key].shape)
            self.gradsNew["d"+key] = np.zeros(self.params[key].shape)
    
    def forward(self, data):
        # data is (T,) size timeseries, parellelize after implementing single steps
        T = len(data)
        hidden = np.zeros((self.Lhid,1))
        foldedLayers = []
        for t in range(T):
            layer = GRUlayer()
            hidden, self.outProb = layer.forward(data[t][:,None], hidden, self.params)
            foldedLayers.append(layer)
        self.outProb = self.outProb.T # transpose for convention
        self.Layers = foldedLayers
        return foldedLayers
    
    def forwardOut(self):
        # call forward before to update self.rnnLayers
        # out is OneHot encoded
        self.out = np.zeros(self.Lclass)
        self.out[np.argmax(self.outProb)] = 1
        return self.out, self.outProb
    
    def crossEntropy(self, ground):
        # call forward before to update self.outProb
        assert ground.shape == self.outProb.shape
        return -np.sum(ground*np.log(self.outProb))
    
    def calcGrad(self, data, ground):
        #run after forward
        lyr = self.Layers
        t = self.Lfeature - 1
        #
        dh_next, grads_step = lyr[t].backward(self.params, ground.T, np.zeros((self.Lhid,1)))
        for key in self.gradsNew:
            self.gradsNew[key] = self.gradsNew[key] + grads_step[key]
        for i in range(t-1, max(-1, t-self.bptt-1), -1): # change t with (self.Lfeature - 1) if you want
            y_t = lyr[i].y_t
            dh_next, grads_step = lyr[i].backward(self.params, y_t, dh_next) # input y_t so that dWv is zero
            for key in self.gradsNew:
                self.gradsNew[key] = self.gradsNew[key] + grads_step[key]
        
    def updateWeights(self, learningRate, momentum):
        #run after grad calculation is done for enough samples depending on the batch size
        for key in self.params:
            self.grads['d'+key] = momentum*self.grads['d'+key] + self.gradsNew['d'+key]
            self.params[key] = self.params[key] - learningRate*self.grads['d'+key]
            self.gradsNew['d'+key] = np.zeros(self.params[key].shape)

    def trainStep(self, sample, target):
        # calls required methods sequentally except updateWeights()
        self.forward(sample)
        guess, _ = self.forwardOut()
        loss = self.crossEntropy(target)
        self.calcGrad(sample, target)
        return loss, guess

def trainMiniBatch(nnModel, data, ground, valX, valD, testX, testD, epoch, learningRate, momentum, batchSize = 32):
    countSamples = 0
    lossListT, lossListV, accuracyListT, accTest= [], [], [], []
    totalSamples = len(ground)
    batchCount = totalSamples//batchSize
    remainder = totalSamples % batchSize
    remLimit = totalSamples - remainder
    for e in range(epoch):
        permutation = list(np.random.permutation(totalSamples))
        shuffled_samples = data[permutation]
        shuffled_grounds = ground[permutation]
        samples = np.array_split(shuffled_samples[:remLimit], batchCount)
        grounds = np.array_split(shuffled_grounds[:remLimit], batchCount)
        samples.append(shuffled_samples[remLimit:])
        grounds.append(shuffled_grounds[remLimit:])
        
        estimatesT = []
        loss = 0
        for j in range(len(grounds)):
            bSize = grounds[j].shape[0]
            for i in range(bSize):
                countSamples += 1
                l, g = nnModel.trainStep(samples[j][i], grounds[j][i][None,:])
                estimatesT.append(g)
                loss += l
            nnModel.updateWeights(learningRate, momentum)
        loss = loss/totalSamples
        lossListT.append(loss)
        
        gndidx = np.array([np.where(r==1)[0][0] for r in shuffled_grounds]) + 1
        estidx = np.array([np.where(r==1)[0][0] for r in estimatesT]) + 1
        
        falses = np.count_nonzero(gndidx-estidx)
        accuracy = 1-falses/totalSamples
        accuracyListT.append(accuracy)
        
        loss = 0
        for i in range(valD.shape[0]):
            nnModel.forward(valX[i])
            guess, _ = nnModel.forwardOut()
            loss += nnModel.crossEntropy(valD[i][None,:])
        loss = loss/valD.shape[0]
        lossListV.append(loss)
        
        estTest = []
        for i in range(testD.shape[0]):
            nnModel.forward(testX[i])
            guess, _ = nnModel.forwardOut()
            estTest.append(guess)
        
        Tgndidx = np.array([np.where(r==1)[0][0] for r in testD]) + 1
        estTestidx = np.array([np.where(r==1)[0][0] for r in estTest]) + 1
        
        falses = np.count_nonzero(Tgndidx-estTestidx)
        accuracy = 1-falses/testD.shape[0]
        accTest.append(accuracy)
        
        print(f"Validation Loss in epoch {e+1}: {loss}, Test Accuracy: {accuracy}")
        if loss > 1.2*lossListV[0]: 
            print("Termnated due to increased loss")
            return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)
        elif (e > 1) & (lossListT[e-1] - lossListT[e] < 0.0001):
            print("Terminated due to convergence")
            return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)
    return lossListT, lossListV, accuracyListT, accTest, comp_confmat(gndidx,estidx), comp_confmat(Tgndidx,estTestidx)

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def comp_confmat(actual, predicted):
    np.seterr(divide='ignore')
    classes = np.unique(actual)
    confmat = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))
    return confmat 

def plotTwinParameter(metric, labels):
    xlabel = [i for i in range(len(metric[0]))]
    plt.plot(xlabel, metric[0], marker='o', markersize=6, linewidth=2, label=labels[0])
    plt.legend()
    plt.ylabel(labels[0])
    plt.ylim((0,1.1))
    ax2 = plt.twinx()
    ax2.plot(xlabel, metric[1], marker='o', color = 'red', markersize=6, linewidth=2, label=labels[1])
    plt.ylabel(labels[1])
    plt.title('Parameter vs Metrics Plot')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def plotParameter(metric, labels, metricName):
    plt.figure(figsize = (12,6))
    xlabel = [i for i in range(len(metric[0]))]
    for i in range(len(labels)):
        plt.plot(xlabel, metric[i], marker='o', markersize=6, linewidth=2, label=labels[i])
    plt.ylabel(metricName[0])
    plt.title(f'{metricName[1]} with Learning Rate: {metricName[2]}, Momentum: {metricName[3]}, BPTT: {metricName[4]}')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def plotConf(mat_con, Title):
    fig, px = plt.subplots(figsize=(7.5, 7.5))
    px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
    for m in range(mat_con.shape[0]):
        for n in range(mat_con.shape[1]):
            px.text(x=m,y=n,s=int(mat_con[m, n]), va='center', ha='center', size='xx-large')
    
    # Sets the labels
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix for '+Title, fontsize=15)
    plt.show()
# In[Read the data]
filename = "data3.h5"

with h5py.File(filename, "r") as f:
    groupKeys = list(f.keys())
    sets = []
    for key in groupKeys:
        sets.append(list(f[key]))
del key
# In[]
idx = np.random.permutation(3000)
trainX = np.array(sets[0])[idx]
trainD = np.array(sets[1])[idx]
testX = np.array(sets[2])
testD = np.array(sets[3])
valX = trainX[:300]
valD = trainD[:300]
trainX = trainX[300:]
trainD = trainD[300:]
# In[]
bptt = 10
model = GRU(150, 3, 128, 6, bptt)
model.initializeWeights()
lossT, lossV, accT, accTest = [], [], [], []
# In[]
lr = 0.01
mm = 0.85
epoch = 10
print(f"Started Training with learning rate = {lr}, momentum = {mm}, bptt = {bptt}")
l1, l2, a1, a2, confT, confTest = trainMiniBatch(model, trainX, trainD, valX, valD, testX, testD, epoch, lr, mm)
lossT.extend(l1)
lossV.extend(l2) 
accT.extend(a1)
accTest.extend(a2)
# In[]
plotConf(confT, "Training Set, GRU")
plotConf(confTest, "Test Set, GRU")
# In[plot]
plotParameter([lossT, lossV], ["Training","Validation"], ["Loss","GRU",lr,mm,bptt])
#%%
plotParameter([accT, accTest], ["Training","Validation"], ["Accuracy","GRU",lr,mm,bptt])