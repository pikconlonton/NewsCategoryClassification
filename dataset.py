import torch
from torch.utils.data import DataLoader,Dataset

class MyDataset(Dataset):
    def __init__(self,X,Y):
        
        self.X = torch.tensor(X,dtype=torch.float32)
        self.Y = torch.tensor(Y,dtype=torch.long)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index],self.Y[index]
        

