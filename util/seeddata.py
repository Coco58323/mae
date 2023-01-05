import torch
import numpy as np
import pandas as pd
import scipy.io as sio

from typing import Optional
from forecaster.dataset import DATASETS
from torch.utils.data import Dataset

class SEEDDataset(Dataset):
    def __init__(
        self,
        prefix: str="/home/v-yike/teamdrive/msrashaiteamdrive/data/SEED",
        name: str = "DE",
        window: int = 5,
        train_ratio: float = 0.8,
        test_ratio: float = 0.2,
        addtime: bool = False,
        individual_index: Optional[int] = 0,
        dataset_name: Optional[str] = None,
    ):
        super().__init__()
        self.window=window
        self.addtime=addtime
        self.dataset_name = dataset_name
        self.new_index = 0
        #prepare DE data 
        file = prefix+"/"+name+"/"
        data_file_path=file+"DE_{}.mat"
        d_labels_path=file+"DE_labels.mat"
        
        candidate_list = [individual_index] if individual_index!=-1 else [i for i in range(45)]
        self.data = np.array([sio.loadmat(data_file_path.format(i+1))['DE_feature'] for i in candidate_list])
        self.data = self.data.transpose([0,2,1,3])
        self.label = np.array([sio.loadmat(d_labels_path)['de_labels'] for _ in candidate_list])
        self._normalize()
        if addtime:
            self._addtimewindow(window)
        self._split(train_ratio, test_ratio, self.dataset_name)

    
    def _normalize(self):
        # self.data -= np.mean(np.mean(self.data,axis=2,keepdims=True),axis=1,keepdims=True)
        for i in range(self.data.shape[0]):
            for j in range(5):
                val = np.mean(self.data[i,:2010,:,j])
                self.data[i,:,:,j] = 2*(self.data[i,:,:,j]-val)/val
            
    def _addtimewindow(self,window):
        shape = self.data.shape
        DE_wtime = np.zeros(shape=(shape[0],shape[1],window,shape[2],shape[3]))
        label_wtime=np.zeros(shape=(shape[0],shape[1]))
        for i in range(shape[0]):
            sample_index=0
            for j in range(shape[1]-window):
                if j==2010:
                    self.new_index=sample_index
                if self.label[i,j] == self.label[i,j+window-1]:
                    DE_wtime[i,sample_index,:,:,:] = self.data[i,sample_index:sample_index+window,:,:]
                    label_wtime[i,sample_index]=self.label[i,j]
                    sample_index += 1

        self.data = DE_wtime[:,:sample_index,:,:,:]
        self.label = label_wtime[:,:sample_index]

    def _split(self,train_ratio,test_ratio,dataset_name):
        if not self.addtime:
            total_size = 3394
            train_size = 2010
            test_size = total_size - train_size
        else:
            total_size = self.label.shape[1]
            train_size = self.new_index
            test_size = total_size - train_size

        if dataset_name == "train":
            self.length = train_size
            self.start_idx = 0
        elif dataset_name == "test":
            self.length = test_size
            self.start_idx = train_size
        else:
            raise ValueError

    def get_index(self):
        return self.label.index
    
    def __len__(self):
        return self.label.shape[0]*self.length
    
    def __getitem__(self,idx):
        shape=self.label.shape
        return torch.tensor(self.data[idx//self.length,self.start_idx + (idx % self.length)],dtype=torch.float), torch.tensor(self.label[idx//self.length,self.start_idx+(idx % self.length)],dtype=torch.long).squeeze()+1

    def freeup(self):
        pass

    def load(self):
        pass


if __name__=="__main__":
    dataset = SEEDDataset()