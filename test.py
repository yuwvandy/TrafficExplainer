import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from config import *
from sklearn.metrics import classification_report


from model import build_transformer
from dataset import TrafficDataset
from utils import seed_everything, compute_emb




def run(test_dataloader, loss_fn, model, header_model, pack_model, header_pack_model, config):
    
    with torch.no_grad():
        model.eval()
        header_model.eval()
        pack_model.eval()
        header_pack_model.eval()

        test_loss, preds, labels = [], [], []
        for batch in test_dataloader:
            seq, label, mask, seq_header, seq_header_mask = batch
            batch_size = seq.shape[0]

            seq, label, mask, seq_header, seq_header_mask = seq.to(config.device), label.to(config.device), mask.to(config.device), seq_header.to(config.device), seq_header_mask.to(config.device)
            
            seq_emb = compute_emb(model, pack_model, seq, mask, config.BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)

            header_emb = compute_emb(header_model, header_pack_model, seq_header, seq_header_mask, config.HEADER_BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)


            emb = torch.cat([seq_emb, header_emb], dim = 1) # (batch, 2*d_model)
            
            
            pred = model.project(emb) # (batch, class_size)
            loss = loss_fn(pred, label)

            test_loss.append(loss.item())
            preds.append(pred)
            labels.append(label)
    
    test_loss = np.mean(test_loss)
    #calculate acc
    preds = torch.cat(preds, dim = 0)
    labels = torch.cat(labels, dim = 0)
    
    print(classification_report(labels.cpu().numpy(), preds.argmax(dim = 1).cpu().numpy(), digits=4))





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

    model.load_state_dict(torch.load(f'./model/{config.baseline}/{config.dataset}/model.pth'))
    header_model.load_state_dict(torch.load(f'./model/{config.baseline}/{config.dataset}/model_header.pth'))
    pack_model.load_state_dict(torch.load(f'./model/{config.baseline}/{config.dataset}/model_pack.pth'))
    header_pack_model.load_state_dict(torch.load(f'./model/{config.baseline}/{config.dataset}/model_header_pack.pth'))
    
    test_dataset = TrafficDataset(config.dataset, flag = 'test', PAD_TRUNC_DIGIT = config.PAD_TRUNC_DIGIT)
    test_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss().to(config.device)

    run(test_dataloader, loss_fn, model, header_model, pack_model, header_pack_model, config)
