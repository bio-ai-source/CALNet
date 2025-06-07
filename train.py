import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score,precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.functional import softmax
import numpy as np

def train(fold, epoch, model, train_drug_loader, train_protein_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    train_labels = []
    train_preds = []
    train_scores = []
    pbar = tqdm(
        zip(train_drug_loader, train_protein_loader),
        total=len(train_drug_loader),
        desc=f"Fold {fold}, Epoch {epoch} training",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for batch_idx, (drug_batch, protein_batch) in enumerate(pbar):
        drug_batch.to(device)
        protein_batch.to(device)
        label = drug_batch.label.to(device)

        optimizer.zero_grad()
        combined_embedding, predicted = model(drug_batch, protein_batch)

        loss = criterion(predicted, label)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        loss.backward()
        clip_grad_norm_(parameters=model.parameters(), max_norm=5)
        optimizer.step()
        interaction = label.cpu().detach().numpy()
        ys = softmax(predicted, 1).cpu().detach().numpy()
        train_labels.extend(interaction)

    pbar.clear()
    pbar.refresh()

    # 计算指标
    fpr, tpr, thresholds = roc_curve(train_labels, train_scores)
    train_acc = accuracy_score(train_labels, train_preds)
    train_auc = roc_auc_score(train_labels, train_scores)
    train_aupr = average_precision_score(train_labels, train_scores)
    precision = precision_score(train_labels, train_preds, zero_division=1)
    recall = recall_score(train_labels, train_preds)
    f1 = f1_score(train_labels, train_preds)
    MCC = matthews_corrcoef(train_labels, train_preds)

    return total_loss / len(train_drug_loader), train_acc, train_auc, train_aupr, precision, recall, f1, MCC


def valid(fold, epoch, model, valid_drug_loader, valid_protein_loader, criterion, device):
    model.eval()
    valid_labels = []
    valid_preds = []
    valid_scores = []
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(
            zip(valid_drug_loader, valid_protein_loader),
            total=len(valid_drug_loader),
            desc=f"Fold {fold}, Epoch {epoch} validating",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        for batch_idx, (drug_batch, protein_batch) in enumerate(pbar):
            drug_batch.to(device)
            protein_batch.to(device)
            label = drug_batch.label.to(device)

            combined_embedding, predicted = model(drug_batch, protein_batch)

            loss = criterion(predicted, label)
            total_loss += loss.item()
            interaction = label.cpu().detach().numpy()
            ys = softmax(predicted, 1).cpu().detach().numpy()

            valid_labels.extend(interaction)
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            valid_preds.extend(predicted_labels)
            valid_scores.extend(predicted_scores)

        pbar.clear()
        pbar.refresh()

        # 计算指标
        valid_acc = accuracy_score(valid_labels, valid_preds)
        valid_auc = roc_auc_score(valid_labels, valid_scores)
        valid_aupr = average_precision_score(valid_labels, valid_scores)
        precision = precision_score(valid_labels, valid_preds, zero_division=1)
        recall = recall_score(valid_labels, valid_preds)
        f1 = f1_score(valid_labels, valid_preds)
        MCC = matthews_corrcoef(valid_labels, valid_preds)

    return total_loss / len(valid_drug_loader), valid_acc, valid_auc, valid_aupr, precision, recall, f1, MCC



def test(fold, model, test_drug_loader, test_protein_loader, device):
    model.eval()
    test_labels = []
    test_preds = []
    test_scores = []

    results = []
    with torch.no_grad():
        pbar = tqdm(
            zip(test_drug_loader, test_protein_loader),
            total=len(test_drug_loader),
            desc=f"Fold {fold}, Testing",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        for batch_idx, (drug_batch, protein_batch) in enumerate(pbar):
            drug_batch.to(device)
            protein_batch.to(device)
            label = drug_batch.label.to(device)

            combined_embedding, predicted = model(drug_batch, protein_batch)


            interaction = label.cpu().detach().numpy()
            ys = softmax(predicted, 1).cpu().detach().numpy()

            test_labels.extend(interaction)
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))

            drug_names = []
            protein_names = []
            interaction_labels = []
            drug_names.extend(drug_batch.drug_name)
            protein_names.extend(protein_batch.protein_name)
            interaction_labels.extend(drug_batch.label)
            for drug_name, protein_name, label, score in zip(drug_names, protein_names, interaction, predicted_scores):
                results.append([drug_name, protein_name, label, score])
            
            test_preds.extend(predicted_labels)
            test_scores.extend(predicted_scores)



        pbar.clear()
        pbar.refresh()


        fpr, tpr, thresholds = roc_curve(test_labels, test_scores)
        test_acc = accuracy_score(test_labels, test_preds)
        test_auc = roc_auc_score(test_labels, test_scores)
        test_aupr = average_precision_score(test_labels, test_scores)
        precision = precision_score(test_labels, test_preds, zero_division=1)
        recall = recall_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        MCC = matthews_corrcoef(test_labels, test_preds)

    return results, fpr, tpr, test_acc, test_auc, test_aupr, precision, recall, f1, MCC



