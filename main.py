
from torch.nn import CrossEntropyLoss
from utils import set_seed, get_dataloader, Lookahead, get_kfold_data, show_result
from dataset import load_dataset
from train import train, valid, test
from torch.optim import Optimizer, SGD
from torch.utils.data import random_split
import os
from hyperparameter import hyperparameter
from model.model import DTI

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
hp = hyperparameter()
type = hp.type
DATASET = hp.DATASET
set_seed(hp.seed_cross)
device = hp.device
config = {
            'layer_output': hp.mlp_layer_output,
            'l_c': hp.l_c,
            'l_d': hp.l_d,
            'num_layers': hp.num_layers,
            'num_heads': hp.num_heads,
            'dropout': hp.dropout,
            'drug_in_channels': 1024,
            'drug_out_channels': 1024,
            'protein_in_channels': 1280,
            'protein_out_channels': 1280
        }

dataset, molecule_graph, protein_graph, molecule_embedding, protein_embedding = load_dataset(dataname=DATASET)
Epoch_List_test, Accuracy_List_test, Precision_List_test, Recall_List_test, F1_List_test, AUC_List_test, AUPR_List_test = [], [], [], [], [], [], []
all_test_labels, all_test_scores = [], []
for i_fold in range(hp.K_Fold):
    train_dataset, test_dataset = get_kfold_data(i_fold, dataset, hp.K_Fold)
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    valid_size = int(train_size * 0.2)
    train_size = train_size - valid_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])
    print(f"Train dataset size: {train_size}")
    print(f"Valid dataset size: {valid_size}")
    print(f"Test dataset size: {test_size}")
    train_drug_loader, train_protein_loader, valid_drug_loader, valid_protein_loader, test_drug_loader, test_protein_loader = get_dataloader(
        train_dataset, valid_dataset, test_dataset, molecule_graph, protein_graph, molecule_embedding,
        protein_embedding, batch=hp.batch)
    model = DTI(**config).to(device)
    optimizer_inner = SGD(model.parameters(), lr=hp.Learning_rate, weight_decay=hp.weight_decay)
    optimizer = Lookahead(optimizer_inner, k=0, alpha=0.5)
    criterion = CrossEntropyLoss().to(device)
    best_epoch, es, max_valid_auc = 0, 0, 0
    for epoch in range(hp.epochs):
        train_loss, train_acc, train_auc, train_aupr, train_precision, train_recall, train_f1, train_MCC = train(
            i_fold, epoch, model, train_drug_loader, train_protein_loader, optimizer, criterion, device)
        valid_loss, valid_acc, valid_auc, valid_aupr, valid_precision, valid_recall, valid_f1, valid_MCC = valid(
            i_fold, epoch, model, valid_drug_loader, valid_protein_loader, criterion, device)
        if max_valid_auc < valid_auc:
            es = 0
            max_valid_auc = valid_auc
        else:
            es += 1
        if es == 5:
            Epoch_List_test.append(epoch)
            break
        print('*' * 25 + ' End Metrics ' + '*' * 25)
        print(f"Epoch： {epoch} Train: Loss={train_loss:.8f}, Acc={train_acc:.4f}, AUC={train_auc:.4f}, AUPR={train_aupr:.4f}, Precision={train_precision:.4f}, Recall={train_recall:.4f}, F1={train_f1:.4f}, MCC={train_MCC:.4f}")
        print(f"Epoch： {epoch} Valid: Loss={valid_loss:.8f}, Acc={valid_acc:.4f}, AUC={valid_auc:.4f}, AUPR={valid_aupr:.4f}, Precision={valid_precision:.4f}, Recall={valid_recall:.4f}, F1={valid_f1:.4f}, MCC={valid_MCC:.4f}")
    predict_result, fpr, tpr, test_acc, test_auc, test_aupr, test_precision, test_recall, test_f1, test_MCC = test(
        i_fold, model, test_drug_loader, test_protein_loader, device)
    print(f"Test:  Acc={test_acc:.4f}, AUC={test_auc:.4f}, AUPR={test_aupr:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}, MCC={test_MCC:.4f}")
    # 打印验证集结果
    Accuracy_List_test.append(test_acc)
    Precision_List_test.append(test_precision)
    Recall_List_test.append(test_recall)
    F1_List_test.append(test_f1)
    AUC_List_test.append(test_auc)
    AUPR_List_test.append(test_aupr)
show_result("Test", type, DATASET,  Accuracy_List_test, Precision_List_test, Recall_List_test, F1_List_test, AUC_List_test, AUPR_List_test)




