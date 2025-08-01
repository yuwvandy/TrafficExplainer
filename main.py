import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from config import *


from model import build_transformer
from dataset import TrafficDataset
from utils import seed_everything, compute_emb
from torch.utils.data import WeightedRandomSampler

torch.autograd.set_detect_anomaly(True)

def run(train_dataloader, val_dataloader, optimizer, loss_fn, model, header_model, pack_model, header_pack_model, config):
    best_val_loss = float('inf')

    for epoch in tqdm(range(config.n_epochs)):
        torch.cuda.empty_cache()
        model.train()
        header_model.train()

        pack_model.train()
        header_pack_model.train()

        train_loss = []
        for batch in train_dataloader:
            seq, label, mask, seq_header, seq_header_mask = batch
            batch_size = seq.shape[0]

            seq, label, mask, seq_header, seq_header_mask = seq.to(config.device), label.to(config.device), mask.to(config.device), seq_header.to(config.device), seq_header_mask.to(config.device)
            
            seq_emb = compute_emb(model, pack_model, seq, mask, config.BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)

            header_emb = compute_emb(header_model, header_pack_model, seq_header, seq_header_mask, config.HEADER_BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)

            emb = torch.cat([seq_emb, header_emb], dim = 1) # (batch, 2*d_model)
            
            
            pred = model.project(emb) # (batch, class_size)
            

            loss = loss_fn(pred, label)
            loss.backward()

            train_loss.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        train_loss = np.mean(train_loss)
        
        with torch.no_grad():
            model.eval()
            val_loss, preds, labels = [], [], []
            for batch in val_dataloader:
                seq, label, mask, seq_header, seq_header_mask = batch
                batch_size = seq.shape[0]

                seq, label, mask, seq_header, seq_header_mask = seq.to(config.device), label.to(config.device), mask.to(config.device), seq_header.to(config.device), seq_header_mask.to(config.device)
                
                seq_emb = compute_emb(model, pack_model, seq, mask, config.BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)

                header_emb = compute_emb(header_model, header_pack_model, seq_header, seq_header_mask, config.HEADER_BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)

                emb = torch.cat([seq_emb, header_emb], dim = 1) # (batch, 2*d_model)
                
                
                pred = model.project(emb) # (batch, class_size)
                loss = loss_fn(pred, label)

                val_loss.append(loss.item())
                preds.append(pred)
                labels.append(label)
        
        val_loss = np.mean(val_loss)
        #calculate acc
        preds = torch.cat(preds, dim = 0)
        labels = torch.cat(labels, dim = 0)
        acc = (preds.argmax(dim = 1) == labels).float().mean()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./model/{config.baseline}/{config.dataset}/model.pth')
            torch.save(header_model.state_dict(), f'./model/{config.baseline}/{config.dataset}/model_header.pth')
            torch.save(pack_model.state_dict(), f'./model/{config.baseline}/{config.dataset}/model_pack.pth')
            torch.save(header_pack_model.state_dict(), f'./model/{config.baseline}/{config.dataset}/model_header_pack.pth')

            print(f'Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {acc}')

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--baseline", type=str, help="baseline", required=True)
    opt = parser.parse_args()

    if opt.dataset == 'iscx-vpn':
        config = ISCXVPNConfig()
    elif opt.dataset == 'iscx-nonvpn':
        config = ISCXNonVPNConfig()
    elif opt.dataset == 'iscx-tor':
        config = ISCXTorConfig()
    elif opt.dataset == 'iscx-nontor':
        config = ISCXNonTorConfig()
    else:
        raise Exception('Dataset Error')
    
    config.baseline = opt.baseline
    seed_everything(config.SEED)

    header_model, header_pack_model = build_transformer(config.seq_vocab_size, config.seq_seq_len, config.seq_pack_len, config.class_size, header = True)
    model, pack_model = build_transformer(config.seq_vocab_size, config.seq_seq_len, config.seq_pack_len, config.class_size)

    model.to(config.device)
    header_model.to(config.device)
    pack_model.to(config.device)
    header_pack_model.to(config.device)
    
    train_dataset, val_dataset, test_dataset = TrafficDataset(config.dataset, flag = 'train', PAD_TRUNC_DIGIT = config.PAD_TRUNC_DIGIT), TrafficDataset(config.dataset, flag = 'val', PAD_TRUNC_DIGIT = config.PAD_TRUNC_DIGIT), TrafficDataset(config.dataset, flag = 'test', PAD_TRUNC_DIGIT = config.PAD_TRUNC_DIGIT)

    num_labels = torch.tensor([(train_dataset.label == _).sum().item() for _ in range(config.class_size)])
    weight = num_labels.sum() / num_labels
    weight = weight / weight.sum()
    # sampler = WeightedRandomSampler(weight, len(train_dataset.label), replacement=True)

    train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = config.batch_size, shuffle=False)
    

    optimizer = optim.Adam(list(model.parameters()) + list(header_model.parameters()) + list(pack_model.parameters()) + list(header_pack_model.parameters()), lr = config.lr)
    # loss_fn = nn.CrossEntropyLoss(weight = weight).to(config.device)
    loss_fn = nn.CrossEntropyLoss().to(config.device)

    run(train_dataloader, val_dataloader, optimizer, loss_fn, model, header_model, pack_model, header_pack_model, config)
