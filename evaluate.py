from typing import List
import torch
torch.cuda.empty_cache()
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from scipy import optimize, interpolate
from sklearn.metrics import confusion_matrix
from data_processing import heldout_split, dataset_split
from model import CustomResNet18

import argparse
parser = argparse.ArgumentParser(description='Evaluate script for Liveness Detection')
parser.add_argument('--stride', type=int, default=125, help="Stride value (default: 125)")
parser.add_argument('--window_size', type=int, default=1500, help="Window size (default: 1500)")
parser.add_argument('--sas', type=int, default=1, help="SAS 1 or 2?")


args = parser.parse_args()

live_path = './ETPAD.v2/LIVE_EYE_MOVEMENTS/LIVE_'
if args.sas == 1:
    sas_path = './ETPAD.v2/SASI_EYE_MOVEMENTS/SAS_I_'
else:
    sas_path = './ETPAD.v2/SASII_EYE_MOVEMENTS/SAS_II_'


train_subjects, _, heldout_subjects = dataset_split()

X_test, y_test = heldout_split(args.stride, args.window_size, train_subjects, heldout_subjects, live_path, sas_path)
print(f'Testing the model with Stride:{args.stride}, Window size:{args.window_size}, Attack-Scenario:{args.sas}')


test_dataset = TensorDataset(X_test, y_test)

torch.manual_seed(0)
batch_size = 64
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)

@torch.no_grad()
def evaluate_model(dataloader, model):
    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    test_loss = 0
    correct_predictions = 0
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    all_y_true, all_y_pred = [], []

    for X_batch, y_batch in tqdm(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        
        test_loss += loss_fn(y_pred, y_batch)
        y_pred = torch.sigmoid(y_pred).round().long()
        y_batch = y_batch.round().long()
        
        y_true_np = y_batch.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        
        all_y_true.append(y_true_np)
        all_y_pred.append(torch.sigmoid(y_pred).detach().cpu().numpy())

        confusion_mat = confusion_matrix(y_true_np, y_pred_np, labels=[True, False])
        tn, fp, fn, tp = confusion_mat.ravel()
        total_tn += tn
        total_fp += fp
        total_fn += fn
        total_tp += tp
        
        correct_predictions += (abs(y_pred - y_batch) < 0.5).type(torch.float).sum().item()

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(all_y_true, all_y_pred, pos_label=1)
    eer = optimize.brentq(lambda x: 1.0 - x - interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0)
    
    test_loss /= num_batches
    
    apcer = total_fp / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    npcer = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0
    acer = (apcer + npcer) / 2
    acr = 1 - acer

    print(f"Performance Metrics:\nACR: {(100 * acr):>0.2f}%, ACER: {(100 * acer):>0.2f}%")
    print(f"APCER: {100 * apcer:>0.2f}%, NPCER: {100 * npcer:>0.2f}%")
    print(f"EER: {(eer * 100):0.2f}%")

# Loss function
loss_fn = nn.BCEWithLogitsLoss()

# Load and evaluate model
model_save_path = f"./pretrained_model/SAS{args.sas}_W{args.window_size}_S{args.stride}.pt"
model = CustomResNet18()

if torch.cuda.is_available():
    model.cuda()

model.load_state_dict(torch.load(model_save_path))
model.eval()

evaluate_model(test_dataloader, model)
