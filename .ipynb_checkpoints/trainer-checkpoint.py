
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import entropy
from getData import * 

# Seed for reproducibility
SEED = 25
torch.manual_seed(SEED)
np.random.seed(SEED)

class blHandler():
    def __init__(self, net, criterion, optimizer, device, scheduler=None, num_epochs=50, bestPath="./models/best_0420.pth"):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.NUM_EPOCHS = num_epochs
        self.device = device
        self.bestPath = bestPath
      
        self.history = {
              "train": {"loss": [], "accuracy": []}, 
              "validation":{"loss": [], "accuracy": []},
              }
        
    def train(self, tLoader, vLoader=None):
        eStopThreshold, eStopCounter = 8, 0 
        best_loss, preValLoss = 100, 100
        
        for epoch in range(self.NUM_EPOCHS):
            totalLoss, totalAcc = 0, 0
            valLoss, valAcc = 0, 0

            self.net.train()
            for i, data in enumerate(tLoader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                output = self.net(inputs)
                loss = self.criterion(output, labels)
                totalLoss += loss.item()
                loss.backward()
    
                _, predicted = torch.max(output, 1)
                acc = accuracy_score(labels.detach().cpu().numpy(), predicted.detach().cpu().numpy())
                totalAcc += acc
                
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step() 
            totalLoss = totalLoss/len(tLoader)
            totalAcc = totalAcc/len(tLoader)
            
            self.history["train"]["loss"].append(totalLoss)
            self.history["train"]["accuracy"].append(totalAcc)
            
            print("epoch {} --> trainLoss: {:0.3f}, trainAcc: {:0.3f}"
              .format(epoch+1, totalLoss, totalAcc), end="")
            
            if vLoader:
                with torch.no_grad():
                    self.net.eval()
                    for i, data in enumerate(vLoader, 0):
                        inputs, labels = data[0].to(self.device), data[1].to(self.device)
                        output = self.net(inputs)
                        loss = self.criterion(output, labels)

                        valLoss += loss.item()

                        _, predicted = torch.max(output, 1)
                        acc = accuracy_score(labels.detach().cpu().numpy(), predicted.detach().cpu().numpy())

                        valAcc += acc 
         
                    valLoss = valLoss/len(vLoader)
                    valAcc = valAcc/len(vLoader)

                    self.history["validation"]["loss"].append(valLoss)
                    self.history["validation"]["accuracy"].append(valAcc)
            
                print(", validLoss: {:0.3f}, validAcc: {:0.3f}"
                  .format(valLoss, valAcc))
                
                if valLoss <= best_loss:
                    # Save the model with the lowest validation loss.
                    best_loss = valLoss
                    torch.save(self.net.state_dict(), self.bestPath)
                    print("Model Saved!")
        
                if valLoss >= preValLoss:
                    eStopCounter += 1
                    if eStopCounter >= eStopThreshold:
                        print("Training Stopped!")
                        break;
                else:
                    eStopCounter = 0
                preValLoss = valLoss
        
            else:
                print("")

        return self.net, self.history

    def infer(self, sLoader):
        """
            @Inference
        """
        
        acc, tLoss = 0, 0
        predicted = []
        with torch.no_grad():
            self.net.eval()
            for inputs, gTruth in sLoader:
                inputs, gTruth = inputs.to(self.device), gTruth.to(self.device)  
                outputs = self.net(inputs)

                _, preds = torch.max(outputs, 1)
                predicted.append(preds)
                loss = self.criterion(outputs, gTruth)
                tLoss += loss.item()
                acc += accuracy_score(gTruth.detach().cpu().numpy(), preds.detach().cpu().numpy())

            acc = acc/len(sLoader)
            tLoss = tLoss/len(sLoader)
        return predicted, acc
    

    def late_inference(self, sLoader, threshold=0.05, verbose=False):
        """
            @Inference: we compare the output confidence (entropy) at a branch with a certain threshold.
        """
        softmaxLayer = nn.Softmax(dim=1)
        acc = 0
        predicted = []
        recorder = {x: [] for x in range(2)}
        self.net.eval()
        with torch.no_grad():
            for inputs, gTruth in sLoader:
                inputs, gTruth = inputs.to(self.device), gTruth.to(self.device)
                x = self.net(inputs)
                for iSample in range(x.shape[0]): # a sample by sample
                    out1 = self.net(x[iSample:iSample+1])
                    y = softmaxLayer(out1)
                    e = entropy(y.detach().cpu().numpy().squeeze(), base=10)
                    if e <= threshold:
                        if verbose:
                            print(e)
                        _, label = torch.max(out1, 1)
                        predicted.append(label)
                        if label == gTruth[iSample].item():
                            recorder[0].append(1)
                            acc+=1
                        else:
                            recorder[0].append(0)
                        continue
                    out2 = self.net(x[iSample:iSample+1])
                    _, label = torch.max(out2, 1)
                    predicted.append(label)
                    if label == gTruth[iSample].item():
                        acc+=1
                        recorder[1].append(1)
                    else:
                        recorder[1].append(0) 
            
            acc = acc / sum([len(recorder[x]) for x in range(2)])

        return recorder, torch.FloatTensor(predicted), acc