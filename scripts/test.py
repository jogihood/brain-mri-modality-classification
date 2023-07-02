import sys
# args
if len(sys.argv) != 3:
    print("test.py - Script used for testing models")
    print()
    print("<Usage>")
    print("\tpython test.py [model_no] [train_dataset]")
    print()
    print("\t\t[model_no] - Model number. Model files in /cgh_study/models")
    print("\t\t\tIf 'default', Modified ResNet50 model is used.")
    print()
    print("\t\t[train_dataset] - Set a dataset for testing a model")
    print("\t\t\tOnly 'oasis3' dataset is available at the moment.")
    sys.exit()

print("Importing modules...")
import os
import gc
import numpy as np
import modules.model as mdl
import modules.dataset as ds
import modules.preprocessing as pre
import torch
from tqdm import tqdm
print("Importing complete.")

if sys.argv[1] == "default":
    model_no = "default"
else:
    model_no = sys.argv[1]

if sys.argv[2] == "oasis3":
    train_dataset = ds.oasis3
elif sys.argv[2] == "something else":
    pass

# GPU Setup
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["T1", "T2", "FLAIR"]
loss = []

model = mdl.MRI2DResNet().to(device) # using Modified ResNet50
model.load_state_dict(torch.load(f"../models/model_{model_no}.pt"))
model.eval()

total = 0
correct = 0

# full path of correct or wrong files
correct_files = []
wrong_files = []

print("Testing...")
pbar = tqdm(train_dataset.files)

for i, f in enumerate(pbar):
    # preprocessing
    try: test_image = pre.preprocess_from_file(f, (224, 224))
    except Exception as e: continue # some files cannot be read, so I added exception
    X = torch.from_numpy(test_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
        if classes[torch.argmax(outputs).item()] in f: # correct inference
            correct += 1
            correct_files.append(f)
        else: # wrong inference
            wrong_files.append(f)

        total += 1
        running_acc = correct / total * 100
        pbar.set_description(f"Running acc: {running_acc: .1f}% ")

print()
print(f"Total test images: \t{total}")
print(f"Correct classification: \t{correct}")
print(f"Test accuracy: \t{correct/total*100:.1f}%\n")

counts = [0,0,0]
for w in wrong_files:
    for i, m in enumerate(classes):
        if m in w: counts[i] += 1
for i, m in enumerate(train_dataset.modalities):
    print(f"{m.method} Accuracy: \t{(100-counts[i]/len(m.files)):.4f}%\tWrong:\t{counts[i]}/{len(m.files)}")

# Garbage collection
gc.collect()
torch.cuda.empty_cache()