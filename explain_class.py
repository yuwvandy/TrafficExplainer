import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from config import *
from torch.nn.functional import softmax


from model import build_transformer, explainer_model
from dataset import TrafficDataset, TrafficDataset_class
from utils import seed_everything, cal_class_entropy, budget_constrain
from collections import defaultdict





def run(test_dataloaders, unique_label, model, header_model, config):
    model.eval()

    explain_header_idxs = [defaultdict(list) for _ in range(len(unique_label))]
    explain_payload_idxs = [defaultdict(list) for _ in range(len(unique_label))]

    count = 0
    for test_dataloader, class_label in zip(test_dataloaders, unique_label):
        print(f'===========Class_label:{class_label}===========')

        explainer = explainer_model().to(config.device)
        explainer.train()
        optimizer = optim.Adam(explainer.parameters(), lr = 0.002)

        for epoch in tqdm(range(400)):
            avg_loss = []
            for batch in test_dataloader:
                seq, label, mask, seq_header, seq_header_mask, pred_label = batch

                seq, label, mask, seq_header, seq_header_mask, pred_label = seq.to(config.device, non_blocking=True), label.to(config.device, non_blocking=True), mask.to(config.device, non_blocking=True), seq_header.to(config.device, non_blocking=True), seq_header_mask.to(config.device, non_blocking=True), pred_label.to(config.device, non_blocking=True)

                pred = explainer(model, seq, mask, seq_header, seq_header_mask)
                loss1 = cal_class_entropy(pred, pred_label)
                loss2 = budget_constrain(explainer.mask_header, explainer.mask_payload, 0.1, 0.1)

                loss = loss1 + loss2
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                avg_loss.append(loss.item())
            
            print(f'Epoch:{epoch}, Loss:{np.mean(avg_loss)}')
            
        keep_header = torch.sigmoid(explainer.mask_header)
        keep_payload = torch.sigmoid(explainer.mask_payload)

        for budget in [0.05, 0.1, 0.2]:
            _, top_keep_header = torch.topk(keep_header, int(keep_header.shape[0] * budget))
            _, top_keep_payload = torch.topk(keep_payload, int(keep_payload.shape[0] * budget))

            explain_header_idxs[count][budget] = top_keep_header.tolist()
            explain_payload_idxs[count][budget] = top_keep_payload.tolist()

        count += 1

    torch.save(explain_header_idxs, f'./res/{config.dataset}/explain_header_idxs.pt')
    torch.save(explain_payload_idxs, f'./res/{config.dataset}/explain_payload_idxs.pt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
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
    
    seed_everything(config.SEED)

    header_model = build_transformer(config.seq_vocab_size, config.seq_seq_len, config.class_size, header = True).to(config.device)
    model = build_transformer(config.seq_vocab_size, config.seq_seq_len, config.class_size).to(config.device)

    model.load_state_dict(torch.load(f'./model/{config.dataset}/model.pth'))
    header_model.load_state_dict(torch.load(f'./model/{config.dataset}/model_header.pth'))
    
    unique_label = np.unique(np.load(f'./dataset/{config.dataset}/train_pyg.npz')['label'])

    test_datasets = [TrafficDataset_class(config.dataset, flag = 'test', class_idx = i) for i in unique_label]
    test_dataloaders = [DataLoader(test_dataset, batch_size = 16, shuffle=False) for test_dataset in test_datasets]

    print('==========Start Model Prediction==========')
    pred_labels = defaultdict(list)
    for i, test_dataloader in enumerate(test_dataloaders):
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                seq, label, mask, seq_header, seq_header_mask = batch
                batch_size = seq.shape[0]

                seq, label, mask, seq_header, seq_header_mask = seq.to(config.device, non_blocking=True), label.to(config.device, non_blocking=True), mask.to(config.device, non_blocking=True), seq_header.to(config.device, non_blocking=True), seq_header_mask.to(config.device, non_blocking=True)
                
                seq_emb = model.embed(seq.reshape(-1, 150)) # (batch, seq_len, d_model)
                seq_emb = model.encode(seq_emb, mask = mask.reshape(-1, 150)) # (batch, seq_len, d_model)

                seq_emb = seq_emb * (1 - mask).reshape(-1, 150).unsqueeze(-1)
                seq_emb = seq_emb.mean(dim = 1).reshape(batch_size, -1, seq_emb.shape[-1]).mean(dim = 1) # (batch, d_model)

                # print(seq_header.shape, seq_header_mask.shape)
                header_emb = model.embed(seq_header.reshape(-1, 40)) # (batch, seq_len, d_model)
                header_emb = model.encode(header_emb, mask = seq_header_mask.reshape(-1, 40)) # (batch, seq_len, d_model)

                header_emb = header_emb * (1 - seq_header_mask).reshape(-1, 40).unsqueeze(-1)
                header_emb = header_emb.mean(dim = 1).reshape(batch_size, -1, header_emb.shape[-1]).mean(dim = 1)

                emb = torch.cat([seq_emb, header_emb], dim = 1) # (batch, 2*d_model)
                
                pred = model.project(emb) # (batch, class_size)

                pred_labels[i].extend(pred.argmax(dim = 1).detach().cpu().numpy())
    
    print('==========Finish Model Prediction==========')
    
    test_datasets = [TrafficDataset_class(config.dataset, flag = 'test', class_idx = i, pred_label = pred_labels[i]) for i in unique_label]
    test_dataloaders = [DataLoader(test_dataset, batch_size = 24, shuffle=True) for test_dataset in test_datasets]

    run(test_dataloaders, unique_label, model, header_model, config)
    
