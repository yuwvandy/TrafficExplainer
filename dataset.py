import torch
import numpy as np


class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, flag, PAD_TRUNC_DIGIT):
        if flag == 'train':
            self.data = np.load(f'./dataset/{dataset}/train_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_train_pyg.npz')
        elif flag == 'val':
            self.data = np.load(f'./dataset/{dataset}/val_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_val_pyg.npz')
        else:
            self.data = np.load(f'./dataset/{dataset}/test_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_test_pyg.npz')

        seq, label = self.data['data'], self.data['label']
        header_seq = self.header['data']
        
        self.traffic_seq = torch.tensor(seq)
        self.header_seq = torch.tensor(header_seq)
        self.label = torch.tensor(label)

        self.PAD_TRUNC_DIGIT = PAD_TRUNC_DIGIT

    def __getitem__(self, idx):
        return self.traffic_seq[idx], self.label[idx], (self.traffic_seq[idx] == self.PAD_TRUNC_DIGIT).long(), self.header_seq[idx], (self.header_seq[idx] == self.PAD_TRUNC_DIGIT).long()


    def __len__(self):
        return len(self.traffic_seq)



class TrafficDataset_class(torch.utils.data.Dataset):
    def __init__(self, dataset, flag, class_idx, pred_label = None):
        if flag == 'train':
            self.data = np.load(f'./dataset/{dataset}/train_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_train_pyg.npz')
        elif flag == 'val':
            self.data = np.load(f'./dataset/{dataset}/val_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_val_pyg.npz')
        else:
            self.data = np.load(f'./dataset/{dataset}/test_pyg.npz')
            self.header = np.load(f'./dataset/{dataset}/header_test_pyg.npz')
        
        seq, label = self.data['data'], self.data['label']
        header_seq = self.header['data']

        #select the one with label == class_idx
        seq = seq[label == class_idx]
        header_seq = header_seq[label == class_idx]
        label = label[label == class_idx]

        
        self.traffic_seq = torch.tensor(seq)
        self.header_seq = torch.tensor(header_seq)
        self.label = torch.tensor(label)
        self.pred_label = pred_label

        if pred_label:
            self.pred_labels= torch.tensor(pred_label)

    def __getitem__(self, idx):
        if self.pred_label:
            return self.traffic_seq[idx], self.label[idx], (self.traffic_seq[idx] == 256).long(), self.header_seq[idx], (self.header_seq[idx] == 256).long(), self.pred_labels[idx]
        else:
            return self.traffic_seq[idx], self.label[idx], (self.traffic_seq[idx] == 256).long(), self.header_seq[idx], (self.header_seq[idx] == 256).long()


    def __len__(self):
        return len(self.traffic_seq)