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
from dataset import TrafficDataset
from utils import seed_everything, cal_class_entropy_instance, budget_constrain, compute_emb
from collections import defaultdict



def run(test_dataloader, model, header_model, pack_model, header_pack_model, config):
    model.eval()
    header_model.eval()
    pack_model.eval()
    header_pack_model.eval()

    ground_truths, pred_labels, pred_rm_labels, pred_add_labels = [], [], defaultdict(list), defaultdict(list)
    explain_header_idx = defaultdict(list)
    explain_payload_idx = defaultdict(list)

    count = 0
    for batch in tqdm(test_dataloader):
        seq, label, mask, seq_header, seq_header_mask = batch
        batch_size = seq.shape[0]

        seq, label, mask, seq_header, seq_header_mask = seq.to(config.device, non_blocking=True), label.to(config.device, non_blocking=True), mask.to(config.device, non_blocking=True), seq_header.to(config.device, non_blocking=True), seq_header_mask.to(config.device, non_blocking=True)
        
        seq_emb = compute_emb(model, pack_model, seq, mask, config.BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)

        header_emb = compute_emb(header_model, header_pack_model, seq_header, seq_header_mask, config.HEADER_BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)

        emb = torch.cat([seq_emb, header_emb], dim = 1) # (batch, 2*d_model)
        
        pred = model.project(emb) # (batch, class_size)
        
        logits = softmax(pred, dim = 1)
        pred_label = logits.argmax(dim = 1).detach().cpu().numpy()

        ground_truths.extend(label.detach().cpu().numpy())
        pred_labels.extend(pred_label)

        if config.explain_strategy == 'byte-level':
            explainer = explainer_model(config).to(config.device)
            explainer.train()
            optimizer = optim.Adam(explainer.parameters(), lr = 0.01)

            for epoch in range(200):
                pred = explainer(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask)
                loss1 = cal_class_entropy_instance(pred, pred_label)
                loss2 = budget_constrain(explainer.mask_header, explainer.mask_payload, 0.1, 0.1)

                loss = loss1 + loss2
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                # print(f'Epoch: {epoch}, Loss: {loss.item()}, pred: {pred.detach().cpu().numpy()}, pred_label: {pred_label}, label: {label.detach().cpu().numpy()}')

            keep_header = torch.sigmoid(explainer.mask_header)
            keep_payload = torch.sigmoid(explainer.mask_payload)
        
        elif config.explain_strategy == 'saliency map':
            logits.squeeze(0)[pred_label[0]].backward()
            grad = torch.norm(model.seq_embed.embedding.weight.grad, dim = 1)
            grad_header = torch.norm(header_model.seq_embed.embedding.weight.grad, dim = 1)

            grad = 0.5*(grad + grad_header)

        # print(keep_header)

        for budget in [0.01]:
            if config.explain_strategy == 'byte-level':
                _, top_keep_header = torch.topk(keep_header, int(keep_header.shape[0] * budget))
                _, top_keep_payload = torch.topk(keep_payload, int(keep_payload.shape[0] * budget))
            
            elif config.explain_strategy == 'random':
                top_keep_header = torch.randint(0, 257, (int(257 * budget),))
                top_keep_payload = torch.randint(0, 257, (int(257 * budget),))

            elif config.explain_strategy == 'saliency map':
                _, top_keep_header = torch.topk(grad, int(grad.shape[0] * budget))
                _, top_keep_payload = torch.topk(grad, int(grad.shape[0] * budget))

            # print(label)
            explain_header_idx[label[0].item()].append(keep_header.tolist())
            explain_payload_idx[label[0].item()].append(keep_payload.tolist())

            #check whether remove those byte would change the prediction
            mask_header = torch.ones(257, dtype = torch.long)
            mask_header[top_keep_header] = 0
            mask_header = mask_header.to(config.device)

            mask_payload = torch.ones(257, dtype = torch.long)
            mask_payload[top_keep_payload] = 0
            mask_payload = mask_payload.to(config.device)


            seq_emb = compute_emb(model, pack_model, seq, mask, config.BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = mask_payload, baseline = config.baseline)
            header_emb = compute_emb(header_model, header_pack_model, seq_header, seq_header_mask, config.HEADER_BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = mask_header, baseline = config.baseline)

            emb = torch.cat([seq_emb, header_emb], dim = 1)
            
            pred = model.project(emb)
            pred_rm_labels[budget].append(pred.argmax(dim = 1).detach().cpu().numpy().item())



            # check whether only keep those byte would keep the prediction
            mask_header = 1 - mask_header
            mask_payload = 1 - mask_payload


            seq_emb = compute_emb(model, pack_model, seq, mask, config.BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = mask_payload, baseline = config.baseline)
            header_emb = compute_emb(header_model, header_pack_model, seq_header, seq_header_mask, config.HEADER_BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = mask_header, baseline = config.baseline)

            emb = torch.cat([seq_emb, header_emb], dim = 1)
            
            pred = model.project(emb)
            pred_add_labels[budget].append(pred.argmax(dim = 1).detach().cpu().numpy().item())

            # print('model prediction:', pred.argmax(dim = 1).detach().cpu().numpy().item(), \
            #       'ground-truth:', label.detach().cpu().numpy().item(), \
            #       'remove prediction:', pred_rm_labels[budget][-1],\
            #       'add prediction:', pred_add_labels[budget][-1])

        # break
        count += 1

    torch.save(ground_truths, f'./res/{config.dataset}/{config.explain_strategy}/ground_truths.pt')
    torch.save(pred_labels, f'./res/{config.dataset}/{config.explain_strategy}/pred_labels.pt')
    torch.save(pred_rm_labels, f'./res/{config.dataset}/{config.explain_strategy}/pred_rm_labels.pt')
    torch.save(pred_add_labels, f'./res/{config.dataset}/{config.explain_strategy}/pred_add_labels.pt')


    # for key in explain_header_idx.keys():
    #     explain_header_idx[key] = np.stack(explain_header_idx[key]).mean(axis = 0)
    #     explain_payload_idx[key] = np.stack(explain_payload_idx[key]).mean(axis = 0)
        
    torch.save(explain_header_idx, f'./res/{config.dataset}/explain_header_idx.pt')
    torch.save(explain_payload_idx, f'./res/{config.dataset}/explain_payload_idx.pt')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--baseline", type=str, help="baseline", required=True)
    parser.add_argument("--explanation", type=str, help="explain_strategy", required=True)
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
    config.explain_strategy = opt.explanation
    seed_everything(config.SEED)

    header_model, header_pack_model = build_transformer(config.seq_vocab_size, config.seq_seq_len, config.seq_pack_len, config.class_size, header = True)
    model, pack_model = build_transformer(config.seq_vocab_size, config.seq_seq_len, config.seq_pack_len, config.class_size)

    model.load_state_dict(torch.load(f'./model/{config.baseline}/{config.dataset}/model.pth'))
    pack_model.load_state_dict(torch.load(f'./model/{config.baseline}/{config.dataset}/model_pack.pth'))
    header_model.load_state_dict(torch.load(f'./model/{config.baseline}/{config.dataset}/model_header.pth'))
    header_pack_model.load_state_dict(torch.load(f'./model/{config.baseline}/{config.dataset}/model_header_pack.pth'))

    model = model.to(config.device)
    pack_model = pack_model.to(config.device)
    header_model = header_model.to(config.device)
    header_pack_model = header_pack_model.to(config.device)
    
    test_dataset = TrafficDataset(config.dataset, flag = 'test', PAD_TRUNC_DIGIT = config.PAD_TRUNC_DIGIT)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    run(test_dataloader, model, header_model, pack_model, header_pack_model, config)
    
