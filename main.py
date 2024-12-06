import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.cuda as cuda
from torch import optim
import random
from utils import fixed_prompt, fair_metric, evaluate, fair_edge_mask
from data_loader import load_data
from models import PreTrainGNN, TuneAdapter, TuneClassifier, TunePrompt

warnings.filterwarnings('ignore')

def train_process(data, gnn, adapter, classifier, prompt, optimizer):
    # aug_x = fixed_prompt(data)
    aug_x = data.x
    prompt_x = prompt(data.x)
    prompt_aug_x = prompt(aug_x)

    gnn_output = gnn(prompt_x, data.edge_index).clone().detach()
    gnn_aug_output = gnn(prompt_aug_x, data.edge_index).clone().detach()

    logits = classifier(adapter(gnn_output[data.idx_train_list]))
    logits_aug = classifier(adapter(gnn_aug_output[data.idx_train_list]))
    
    loss1 = F.binary_cross_entropy_with_logits(logits, data.y[data.idx_train_list].unsqueeze(1).float())
    loss2 = F.binary_cross_entropy_with_logits(logits_aug, data.y[data.idx_train_list].unsqueeze(1).float())
    loss3 = torch.norm(logits - logits_aug)
    total_loss = loss1 + args.lambda_1 * loss2 + args.lambda_2 * loss3

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    adapter.eval()
    classifier.eval()
    logits = classifier(adapter(gnn_output[data.idx_train_list]))
    acc, auc, parity, equality = evaluate(logits, data.y[data.idx_train_list], data.sens[data.idx_train_list])
    return acc, auc, parity, equality 


def eval_process(data, gnn, adapter, classifier, prompt, mode):
    prompt_x = prompt(data.x)
    gnn_output = gnn(prompt_x, data.edge_index).clone().detach()

    if mode == 'valid':
        logits = classifier(adapter(gnn_output[data.idx_valid_list]))
        acc, auc, parity, equality = evaluate(logits, data.y[data.idx_valid_list], data.sens[data.idx_valid_list])
    elif mode == 'test':
        logits = classifier(adapter(gnn_output[data.idx_test_list]))
        acc, auc, parity, equality = evaluate(logits, data.y[data.idx_test_list], data.sens[data.idx_test_list])       
    return acc, auc, parity, equality 


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    pgm_file = f'saved_models/pretrain/infomax/pokec_n_7_leakyrelu_hidden-dim(24)_num-layer(2)_epochs(2000)_lr(0.001)_weight_decay(0.0)_weights.pt'
    data = load_data(args)
    data.edge_index = fair_edge_mask(args, data)
    
    GNN = PreTrainGNN(data.input_dim, args.hidden_dim, args.num_layer).to(args.device)
    GNN.load_state_dict(torch.load(pgm_file, map_location=torch.device(args.device)))

    Adapter = TuneAdapter(args.hidden_dim).to(args.device)
    Classifier = TuneClassifier(args.hidden_dim).to(args.device)
    Prompt = TunePrompt(data.x.shape[1]).to(args.device)

    parameters = list(Classifier.parameters()) + list(Adapter.parameters()) + list(Prompt.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate, weight_decay=0)

    best_res = -10000
    test_acc, test_auc, test_equality, test_parity = 0,0,0,0
    GNN.eval()
    for epoch in range(1, args.epochs+1):
        Adapter.train()
        Classifier.train()
        Prompt.train()
        acc, auc, parity, equality = train_process(data, GNN, Adapter, Classifier, Prompt, optimizer)

        Adapter.eval()
        Classifier.eval()
        Prompt.eval()
        valid_acc, valid_auc, valid_equality, valid_parity = eval_process(data, GNN, Adapter, Classifier, Prompt, 'valid')

        valid_res = valid_auc
        if valid_res > best_res and epoch > 200:
            best_res = valid_res
            test_acc, test_auc, test_equality, test_parity = eval_process(data, GNN, Adapter, Classifier, Prompt, 'test')
            print(f"Epoch: {epoch}, Valid Acc: {valid_acc:.4f}, Valid AUC: {valid_auc:.4f}, Valid DP: {valid_parity:.4f}, Valid EO: {valid_equality:.4f}")
        
    print(f"Final Results: Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test DP: {test_parity:.4f}, Test EO: {test_equality:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='pokec_n')
    parser.add_argument('--pre_train', type=str, default='infomax')
    parser.add_argument('--gamma', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epsilon', type=int, default=0.5)
    parser.add_argument('--lambda_1', type=int, default=0.2)
    parser.add_argument('--lambda_2', type=int, default=1e-5)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--hidden_dim', type=int, default=24)

    args = parser.parse_args()
    main(args)