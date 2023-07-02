import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
from datetime import datetime

# Load image file as numpy ndarray
# original size
def load_image_as_numpy(file_path: str) -> np.ndarray :
    sitk_image = sitk.ReadImage(file_path)
    np_image = sitk.GetArrayFromImage(sitk_image)
    return np_image

# resample to new size
# new size
def resample_image(cropped_image: np.ndarray, new_size: tuple, interpolator=sitk.sitkLinear) -> np.ndarray:
    sitk_image = sitk.GetImageFromArray(cropped_image)
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()

    # Adjust the size for SimpleITK (z, y, x)
    new_size_sitk = new_size[::-1]

    new_spacing = [old_spacing * old_size / new_dim for old_spacing, old_size, new_dim in zip(original_spacing, original_size, new_size_sitk)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size_sitk)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())

    resampled_image = sitk.GetArrayFromImage(resampler.Execute(sitk_image))
    return resampled_image

# Execute z-score normalization
def z_normalize(image: np.ndarray) -> np.ndarray:
    mean = np.mean(image)
    std = np.std(image)

    return (image-mean)/std

# Get head area
def crop(np_image: np.ndarray, visualization=False, slice=None) -> np.ndarray:
    time = datetime.today().strftime("%y%m%d_%H%M")

    # Thresholding
    mean = np.mean(np_image)
    mask = np.where((np_image > 1.5*mean), 1, 0)
    d, h, w = mask.shape

    # Getting head area z values
    for zbottom, s in enumerate(mask):
        if 1 in s : break
    for ztop, s in enumerate(mask[::-1]):
        if 1 in s : break
    ztop = mask.shape[0]-ztop-1

    # About 68% point of head
    if slice == None: s = int((ztop-zbottom)*.68)
    else: s = slice
    if s >= d: s = d-1
    if s < 0: s = 0
    
    for i, m in enumerate(mask[zbottom:ztop]):
        m_ = np.transpose(m)
        if i == s:
            # from top
            for t, h in enumerate(m):
                if 1 in h : break

            # from bottom
            for b, h in enumerate(m[::-1]):
                if 1 in h : break

            # from left
            for l, w in enumerate(m_):
                if 1 in w : break

            # from right
            for r, w in enumerate(m_[::-1]):
                if 1 in w : break
    
    if visualization:
        i1 = np_image[:,:,np_image.shape[2]//2][::-1]
        i2 = mask[:,:,np_image.shape[2]//2][::-1]
        i3 = np_image[zbottom:ztop,:,np_image.shape[2]//2][::-1]
        f, a = plt.subplots(ncols=3, nrows=2)
        for _ in a.flat: _.axis("off")
        a[0,0].imshow(i1, cmap='gray')
        a[0,1].imshow(i2, cmap='gray')
        a[0,1].plot([0,np_image.shape[1]-1],[np_image.shape[0]-ztop]*2,color='red',linewidth=2)
        a[0,1].plot([0,np_image.shape[1]-1],[np_image.shape[0]]*2,color='red',linewidth=2)
        a[0,1].plot([0,np_image.shape[1]-1],[np_image.shape[0]-s]*2,color='blue',linewidth=1)
        a[0,2].imshow(i3, cmap='gray')
        a[0,0].set_title("Original")
        a[0,1].set_title("Mask")
        a[0,2].set_title("Z-axis Cropped")
        a[1,0].imshow(np_image[s+zbottom], cmap='gray')
        a[1,0].set_title("Selected Slice")
        a[1,1].imshow(mask[s+zbottom], cmap='gray')
        a[1,1].set_title("Mask")
        a[1,1].plot([0,np_image.shape[2]-1],[np_image.shape[1]-t]*2,color='red',linewidth=2)
        a[1,1].plot([0,np_image.shape[2]-1],[b]*2,color='red',linewidth=2)
        a[1,1].plot([l]*2,[0,np_image.shape[1]-1],color='red',linewidth=2)
        a[1,1].plot([np_image.shape[2]-r+1]*2,[0,np_image.shape[1]-1],color='red',linewidth=2)
        a[1,2].imshow(np_image[s+zbottom, t:(len(mask[s+zbottom])-b), l:(len(mask[s+zbottom][0])-r)], cmap='gray')
        a[1,2].set_title("Cropped Slice")
        f.savefig(f"../../preprocessing_visualizations/visualization_{time}.png")

    return np_image[s+zbottom, t:(len(mask[s+zbottom])-b), l:(len(mask[s+zbottom][0])-r)]

def check_orientation(file_path, np_image) -> bool:
    # some MRI files are oriented wrong
    # we have to re-orient some images manually
    # this part is particularly hard-coded, simplification and modification is needed
    oasis3check = np.array([[-1.,0.,0.,127.5],[0.,1.,0.,-127.5],[0.,0.,1.25,-79.375],[0.,0.,0.,1.]])
    if ("IXI" in file_path or
        "sub-CC" in file_path or
        "OASIS-1" in file_path or
        ("OASIS-3" in file_path and "TSE" in file_path) or
        ("OASIS-3" in file_path and not (False in (nib.load(file_path).affine == oasis3check)))):
        np_image = np.rot90(np_image, k=1, axes=(1,0))
        np_image = np.rot90(np_image, k=1, axes=(1,2))
    return np_image

def preprocess_from_file(file_path:str, new_size:tuple, visualization=False) -> np.ndarray:
    np_image = load_image_as_numpy(file_path)
    np_image = check_orientation(file_path, np_image)

    cropped_image       = crop(np_image, visualization=visualization)
    resampled_image     = resample_image(cropped_image, new_size)
    normalized_image    = z_normalize(resampled_image)

    return normalized_image
    
def get_zslice(file_path:str, new_size:tuple, slice:int) -> np.ndarray :
    np_image = load_image_as_numpy(file_path)
    np_image = check_orientation(file_path, np_image)

    cropped_image       = crop(np_image, visualization=False, slice=slice)
    resampled_image     = resample_image(cropped_image, new_size)

    return resampled_image