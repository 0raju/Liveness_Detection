import torch
torch.cuda.empty_cache()
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from data_processing import train_test_split, dataset_split
from model import CustomResNet18

import argparse
parser = argparse.ArgumentParser(description='Train script for Liveness Detection')
parser.add_argument('--stride', type=int, default=125, help="Stride value (default: 125)")
parser.add_argument('--window_size', type=int, default=1500, help="Window size (default: 1500)")
parser.add_argument('--sas', type=int, default=1, help="SAS 1 or 2?")


args = parser.parse_args()

live_path = './ETPAD.v2/LIVE_EYE_MOVEMENTS/LIVE_'
if args.sas == 1:
    sasi_path = './ETPAD.v2/SASI_EYE_MOVEMENTS/SAS_I_'
else:
    sasi_path = './ETPAD.v2/SASII_EYE_MOVEMENTS/SAS_II_'


train_subjects, test_subjects, _ = dataset_split()
X_train, y_train, X_test, y_test = train_test_split(args.stride, args.window_size, train_subjects, test_subjects, live_path, sasi_path)

print(f'Training the model with Stride:{args.stride}, Window size:{args.window_size}, Attack-Scenario:{args.sas}')

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

torch.manual_seed(0)
batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize weights for the model
def initialize_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)

# Training function
def train_model(dataloader, model, loss_fn, optimizer, epoch):
    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_loss, correct_predictions = 0, 0

    for batch_idx, (X_batch, y_batch) in enumerate(tqdm(dataloader)):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        y_pred = model(X_batch)
        y_batch = y_batch.round()
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        y_pred = torch.sigmoid(y_pred).round().long()
        correct_predictions += (y_pred == y_batch).type(torch.float).sum().item()

    total_loss /= num_batches
    accuracy = correct_predictions / dataset_size * 100

    print(f"Training: Accuracy: {accuracy:.2f}%, Loss: {total_loss:.4f}")

# Testing/Validation function
@torch.no_grad()
def test_model(dataloader, model, loss_fn):
    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    total_loss, correct_predictions = 0, 0
    sum_tp, sum_tn, sum_fp, sum_fn = 0, 0, 0, 0

    for X_batch, y_batch in tqdm(dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        y_pred = model(X_batch)
        total_loss += loss_fn(y_pred, y_batch).item()
        
        # Calculate predictions and confusion matrix
        y_pred = torch.sigmoid(y_pred).round().long()
        y_batch = y_batch.round().long()

        y_true_np = y_batch.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        cm = confusion_matrix(y_true_np, y_pred_np, labels=[True, False])
        tn, fp, fn, tp = cm.ravel()
        
        sum_tn += tn
        sum_fp += fp
        sum_fn += fn
        sum_tp += tp
        correct_predictions += (y_pred == y_batch).type(torch.float).sum().item()

    total_loss /= num_batches
    accuracy = correct_predictions / dataset_size * 100

    # Calculate metrics
    apcer = sum_fp / (sum_tn + sum_fp) if (sum_tn + sum_fp) > 0 else 0
    npcer = sum_fn / (sum_fn + sum_tp) if (sum_fn + sum_tp) > 0 else 0
    acer = (apcer + npcer) / 2
    acr = 1 - acer

    print(f"Test: Accuracy: {accuracy:.2f}%, Loss: {total_loss:.4f}")


    return total_loss

# Model initialization
model = CustomResNet18()
initialize_weights(model)
model = model.to(device)

# Optimizer and loss function
learning_rate = 3e-4
weight_decay = 1e-5
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop with early stopping
n_epochs_stop = 6
epochs_no_improve = 0
early_stop = False
best_loss = np.inf
num_epochs = 100

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train_model(train_dataloader, model, loss_fn, optimizer, epoch)
    
    test_loss = test_model(test_dataloader, model, loss_fn)
    if test_loss < best_loss:
        best_loss = test_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch > 10 and epochs_no_improve == n_epochs_stop:
        print("Early stopping!")
        early_stop = True
        break

# Save the model
model_save_path = f"./pretrained_model/SAS{args.sas}_W{args.window_size}_S{args.stride}.pt"
torch.save(model.state_dict(), model_save_path)
