import torch
import torch.nn as nn

class EmotiClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
       
        self.l1 = nn.Sequential(
            
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32,64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            
            
            nn.Conv2d(64,128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(128,256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        
       
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        )
        
        
        self.loss = nn.CrossEntropyLoss();

        
    
    def forward(self, x):
        out = self.l1(x);
        out = out.view(-1, 256);
        out = self.fc(out);
        
        return out
    
    def predict(self, x):
        self.eval();
        with torch.no_grad():
            out = self.forward(x);
        
        return out;

    
        

        

