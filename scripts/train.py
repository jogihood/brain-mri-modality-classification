print("Importing modules...")
import os
import random
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import modules.dataset as ds
import modules.model as mdl
import csv
import time
import datetime as dt
from torch.utils.data import DataLoader, Dataset
print("Importing complete.")

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class MRIDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        np_image = sitk.GetArrayFromImage(sitk.ReadImage(self.files[idx]))
        X = torch.from_numpy(np_image.astype(np.float32)).unsqueeze(0)
        Y = torch.from_numpy(self.labels[idx])

        return X, Y
    
def train(model, data_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        torch.cuda.empty_cache()

    return train_loss / len(data_loader.dataset)

def evaluate(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_samples = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            _, preds = torch.max(outputs, 1) # predict with current weights

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == torch.argmax(labels, dim=1))
            num_samples += inputs.size(0)

    epoch_loss = running_loss / num_samples
    epoch_acc = float(running_corrects.double() / num_samples)
    return epoch_loss, epoch_acc

classes = ["T1", "T2", "FLAIR"]
num_classes = 3
num_epochs = 50
batch_size = 64
learning_rate = 1e-5
patience = 5
seed = 200
image_size = (224,224)

random.seed(seed)
paths = ds.images_2d # preprocessed 224*224 2d MRI slices
random.shuffle(paths)

labels = []

# one-hot encoding
for p in paths:
    if "T1" in p:   labels.append([1., 0., 0.])
    elif "T2" in p: labels.append([0., 1., 0.])
    else:           labels.append([0., 0., 1.])

# train 60%, val 20%, test 20%
i = int(len(paths)*.6)
j = int(len(paths)*.8)

train_paths     = np.array(paths[:i])
val_paths       = np.array(paths[i:j])
test_paths      = np.array(paths[j:])

train_labels    = np.array(labels[:i])
val_labels      = np.array(labels[i:j])
test_labels     = np.array(labels[j:])

train_dataset   = MRIDataset(train_paths, train_labels)
train_loader    = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset     = MRIDataset(val_paths, val_labels)
val_loader      = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset    = MRIDataset(test_paths, test_labels)
test_loader     = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

t = dt.datetime.today().strftime("%y%m%d_%H%M")

model = mdl.MRI2DResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# scheduler will reduce learning rate if val loss will not reduce for [patience] epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)

best_val_loss = float('inf')
early_stop_counter = 0

loss = []
save_name = f"../models/model_{t}.pt"

memo = f"ResNet Transfer Learning, 50 epoch, with improved model, 5 patience"

start = time.time()

for epoch in range(num_epochs):
    v = True
    if v: print(f"Epoch [{epoch+1}/{num_epochs}]\tTraining...", end="")
    train_loss = train(model, train_loader, criterion, optimizer)
    if v: print(f"\rEpoch [{epoch+1}/{num_epochs}]\tTrain Loss: {train_loss:.4f}\tEvaluating...", end="")
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    if v: print(f"\rEpoch [{epoch+1}/{num_epochs}]\tTrain Loss: {train_loss:.4f}\tVal Loss: {val_loss:.4f}\tVal Acc: {val_acc*100:.2f}%")
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), save_name)
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping.")
            break
    loss.append([train_loss, val_loss, val_acc])

end = time.time()
sec = end - start
print(f"Elapsed time - {str(dt.timedelta(seconds=sec)).split('.')[0]}")
print()

with open(f"../losses/loss_{t}.csv",'w') as file :
    write = csv.writer(file)
    write.writerow([memo])
    write.writerow(['batch_size', batch_size])
    write.writerow(['epoch', num_epochs])
    write.writerow(['lr', learning_rate])
    write.writerow(['model', '2d_Modified_ResNet'])
    write.writerow(['model_params', None])
    write.writerows(loss)

# Model loading / Evaluation
print("Testing...")

model.load_state_dict(torch.load(save_name))
model.eval()

with torch.no_grad():
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")