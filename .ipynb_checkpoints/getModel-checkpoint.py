import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

# Seed for reproducibility
SEED = 25
torch.manual_seed(SEED)
np.random.seed(SEED)

class eeModel(nn.Module):

    def __init__(self, num_classes=2, input_dim = 256):
        super(eeModel, self).__init__()
        
        self.baseModel = None
        self.shortBranch = None
        self.longBranch = None
   
              
    def forward(self, X):
        X = self.baseModel(X)
        X1 = self.shortBranch(X)
        X2 = self.longBranch(X)
        
        return X1, X2
    
    def short_forward(self, X):
        X = self.baseModel(X)
        X = self.shortBranch(X)
        return X
    
    def long_forward(self, X):
        X = self.baseModel(X)
        X = self.longBranch(X)
        return X
    
    def forward_timeTest(self, X):
        selected_data = X[0: 50]
        remaining_data = X[50: X.shape[0]]
        _ = self.short_forward(selected_data)
        _ = self.long_forward(remaining_data)
    
    def print_summary(self):
        basemodel_total_params = sum(p.numel() for p in self.baseModel.parameters())
        basemodel_trainable_params = sum(p.numel() for p in self.baseModel.parameters() if p.requires_grad)

        shortBranch_total_params = sum(p.numel() for p in self.shortBranch.parameters())
        shortBranch_trainable_params = sum(p.numel() for p in self.shortBranch.parameters() if p.requires_grad)

        longBranch_total_params = sum(p.numel() for p in self.longBranch.parameters())
        longBranch_trainable_params = sum(p.numel() for p in self.longBranch.parameters() if p.requires_grad)

        print("Number of base parameters: {}".format(basemodel_total_params))
        print("Number of short branch parameters: {}".format(shortBranch_total_params))
        print("Number of long branch parameters: {}".format(longBranch_total_params))
        print("Difference = {}".format(longBranch_total_params-shortBranch_total_params))
    
    def compute_size(self):
        param_size = 0
        for param in self.baseModel.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.shortBranch.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.longBranch.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.baseModel.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.shortBranch.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.longBranch.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def compute_short_branch_size(self):
        param_size = 0
        for param in self.baseModel.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.shortBranch.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.baseModel.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.shortBranch.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def compute_long_branch_size(self):
        param_size = 0
        for param in self.baseModel.parameters():
            param_size += param.nelement() * param.element_size()
        for param in self.longBranch.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.baseModel.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        for buffer in self.longBranch.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb


class blModel(eeModel):

    def __init__(self, num_classes=2, input_dim = 3):
        super(blModel, self).__init__()
        
        self.baseModel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride = 1,padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
            
        )

        self.shortBranch = nn.Sequential()
        
        self.longBranch = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2, padding = 0),
            
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
 

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Sigmoid()

        )



    
                
    def forward(self, X):
        X = self.baseModel(X)
        X = self.longBranch(X)     
        return X
    

