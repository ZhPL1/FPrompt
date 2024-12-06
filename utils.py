import numpy as np
import torch
import math
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.sparse import coo_matrix


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)

    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()
    

def evaluate(logits, label, sens):
    predictions = (logits > 0).type_as(label)
    acc = accuracy_score(y_true=label.cpu().detach().numpy(), y_pred=predictions.cpu().detach().numpy())
    
    y_score = torch.sigmoid(logits).cpu().detach().numpy() 
    auc = roc_auc_score(y_true=label.cpu().detach().numpy(), y_score=y_score)

    parity, equality = fair_metric(predictions.cpu().detach().numpy(), label.cpu().detach().numpy(),sens.cpu().detach().numpy())
    return acc, auc, parity, equality


def fixed_prompt(data):
    emb = data.x
    train_emb = emb[data.idx_train_list]
    train_sens = data.sens[data.idx_train_list]

    group1 = (train_sens == 1).long()
    group0 = 1 - group1
    mean_group1 = torch.sum(train_emb * group1.unsqueeze(1), 0) / group1.count_nonzero().item()
    mean_group0 = torch.sum(train_emb * group0.unsqueeze(1), 0) / group0.count_nonzero().item()
    
    sim_pos = torch.nn.functional.cosine_similarity(emb, mean_group1.unsqueeze(0), dim=1)
    sim_neg = torch.nn.functional.cosine_similarity(emb, mean_group0.unsqueeze(0), dim=1)

    coeff_pos = torch.exp(-torch.tanh(sim_pos)) / (torch.exp(-torch.tanh(sim_pos)) + torch.exp(-torch.tanh(sim_neg)))
    coeff_neg = torch.exp(-torch.tanh(sim_neg)) / (torch.exp(-torch.tanh(sim_pos)) + torch.exp(-torch.tanh(sim_neg)))

    coeff_pos = coeff_pos.unsqueeze(1)  
    coeff_neg = coeff_neg.unsqueeze(1) 
    aug_emb = emb + coeff_pos * mean_group1 + coeff_neg * mean_group0
    return aug_emb


def sensitive_group_assignment(args, data):
    emb = data.x.to(args.device)
    sens = data.sens.to(args.device)
    train_list = data.idx_train_list.to(args.device)[:500]

    group0_indices = train_list[sens[train_list] == 0]
    group1_indices = train_list[sens[train_list] == 1]

    X_group0 = emb[group0_indices] 
    X_group1 = emb[group1_indices] 

    XtX_group0 = X_group0 @ X_group0.T
    XtX_group1 = X_group1 @ X_group1.T
    PROJ_group0 = args.gamma * X_group0.T @ torch.inverse(torch.eye(XtX_group0.size(0)).to(args.device) + XtX_group0) @ X_group0 
    PROJ_group1 = args.gamma * X_group1.T @ torch.inverse(torch.eye(XtX_group1.size(0)).to(args.device) + XtX_group1) @ X_group1
    PROJ_group0 = emb @ PROJ_group0.T
    PROJ_group1 = emb @ PROJ_group1.T

    dist_group0 = torch.norm(emb - PROJ_group0, dim=1)
    dist_group1 = torch.norm(emb - PROJ_group1, dim=1)


    logits = torch.stack([-dist_group0, -dist_group1], dim=1) 
    probabilities = torch.softmax(logits, dim=1)
    sens_pred = torch.argmax(probabilities, dim=1)
    return sens_pred


def fair_edge_mask(args, data):
    adj = data.adj.tocoo()
    sens_pred = sensitive_group_assignment(args, data).cpu()
    row, col, data = adj.row, adj.col, adj.data

    sens_row = sens_pred[row]  
    sens_col = sens_pred[col]  

    probs = np.where(sens_row != sens_col,args.epsilon, 1 - args.epsilon)
    mask = np.random.rand(len(data)) < probs 

    row_masked = row[mask]
    col_masked = col[mask]

    edge_index = torch.tensor([row_masked, col_masked], dtype=torch.long, device=args.device)
    return edge_index