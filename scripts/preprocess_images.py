import os
import SimpleITK as sitk
from tqdm import tqdm
import modules.preprocessing as pre
import modules.dataset as ds

# Modality counts
c = [0, 0, 0]
error_cnt = 0
pbar = tqdm(ds.files)

for f in pbar:
    pbar.set_description(f"T1: {c[0]:4d} / T2: {c[1]:4d} / FLAIR: {c[2]:4d} / Error: {error_cnt:4d}")
    try:
        p = pre.preprocess_from_file(f, (224, 224))
    except: error_cnt += 1; continue
    
    if "T1" in f:
        m = "T1"; c[0] += 1; c_ = c[0];
    elif "T2" in f:
        m = "T2"; c[1] += 1; c_ = c[1];
    elif "FLAIR" in f:
        m = "FLAIR"; c[2] += 1; c_ = c[2];
    else: continue

    s = sitk.GetImageFromArray(p)
    n = os.path.join("/nasdata4/csgradproj/2d_data/224", f"{m}_224_{c_:04d}.nii")
    sitk.WriteImage(s, n)

print(f"Resampling Finished. Error occured {error_cnt} times")
