import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

# Seed for reproducibility
SEED = 120
torch.manual_seed(SEED)
np.random.seed(SEED)

class eeModel(nn.Module):

    def __init__(self, num_classes=1, input_dim = 3):
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
        _ = self.shortBranch(X)
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


class blModel(nn.Module):

    def __init__(self, num_classes=1, input_dim = 3):
        super(blModel, self).__init__()
        
        self.baseModel = nn.Sequential(

            #Layer 1
            nn.Conv2d(in_channels=input_dim, out_channels=4, kernel_size=3,stride = 1,padding='same'),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            ##Layer 2
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(4),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            #Layer 3
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            ##Layer 4
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            #Layer 5
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            

            #Layer 6
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            ##Layer 7
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)

            
        
        
        )

        self.shortBranch = nn.Sequential()
        
        self.longBranch = nn.Sequential(

            #FC
            nn.Flatten(),
            nn.Linear(31*31*16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 2),
            nn.Softmax(dim=1)
            

        )


                
    def forward(self, X):
        X = self.baseModel(X)
        X = self.longBranch(X)
        return X
    

class eeModel_V0(eeModel):

    """
        A class for the model in which the short branch is the closest to the baseModel compared to the other models.
    """
    def __init__(self, num_classes=1, input_dim = 3):
        super(eeModel_V0, self).__init__()

        self.baseModel = nn.Sequential(

            #Layer 1
            nn.Conv2d(in_channels=input_dim, out_channels=4, kernel_size=3,stride = 1,padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            ##Layer 2
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2)



            
        )
        
        self.shortBranch = nn.Sequential(

            #FC
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Flatten(),
            nn.Linear(15876, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        
        )
        
        self.longBranch = nn.Sequential(


            #Layer 3
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            ##Layer 4
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            
            #Layer 5
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            #Layer 6
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            ##Layer 7
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            #FC
            nn.Flatten(),
            nn.Linear(31*31*16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        
        )
   
        self.print_summary()

    def forward(self, X):    
        X = self.baseModel(X)
        X1 = self.shortBranch(X)
        X2 = self.longBranch(X)
        
        return X1, X2



