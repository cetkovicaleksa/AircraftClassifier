# %%
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

# %matplotlib inline

device = 'cuda' if torch.cuda.is_available() else 'cpu'
display(f"Using device: `{device}`")

np.random.seed(14_02_2003)

torch.random.manual_seed(14_02_2003)
if torch.cuda.is_available():
    torch.cuda.random.manual_seed_all(14_02_2003)

# %% [markdown]
# # Aircraft Classifier
# %%
variants = {
    "F/A-18",
    "F-16A/B",
    "Eurofighter Typhoon",
    "Hawk T1",
    "Tornado",
    "C-130",
    "An-12",
    "Il-76",
    "737-800",
    "747-200"
}

variants_arg = " ".join(map('"{}"'.format, variants))

import os

FGVCAIRCRAFT_ROOT = os.getenv('FGVCAIRCRAFT_ROOT', "./data")

if any(os.path.exists(path) for path in ["data/train", "data/test", "data/val", "data/trainval"]): 
    # %run download_fgvcaircraft.py -r $FGVCAIRCRAFT_ROOT -o "./data" --splits train test val --classes $variants_arg
    pass

# %%
class RemoveCopyright:
    def __call__(self, img):
        match type(img):
            case torch.Tensor:
                _, h, w = img.shape
            case _:
                w, h = img.size

        return F.crop(img, top=0, left=0, height=(h - 20), width=w) # 20px copyright bottom banner
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    
normalize = T.Compose([
    T.ToTensor(),
    RemoveCopyright(),
    T.Resize((244, 244)),
])

augment = T.Compose([
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2)
])

# %%
train_dataset = torchvision.datasets.ImageFolder(
    root="./data/train",
    transform=T.Compose([*normalize.transforms, *augment.transforms]) # unpack just to display it nicely (without nested compose)
)

val_dataset = torchvision.datasets.ImageFolder(
    root="./data/val",
    transform=normalize
)

test_dataset = torchvision.datasets.ImageFolder(
    root="./data/test",
    transform=normalize
)

display(train_dataset, val_dataset, test_dataset)

# %%
sample_dataset = train_dataset

labels, indices = np.unique([label for _, label in sample_dataset.samples], return_index=True)
images = [sample_dataset[i][0] for i in indices]

_, axes = plt.subplots(nrows := 2, len(images) // nrows, figsize=(16, 9), frameon=False, tight_layout=True)

for i, [ax, image, label] in enumerate(zip(axes.ravel(), images, labels)):
    image = torch.clamp(image, 0, 1).numpy().transpose([1, 2, 0])
    ax.imshow(image)
    ax.axis(i == 0)

    ax.text(
        0.03, 0.05, 
        sample_dataset.classes[label],
        fontsize=12, color="green", backgroundcolor="black",
        ha="left", va="top",
        transform=ax.transAxes,
    )   

for ax in axes[len(images):]:
    ax.remove()

plt.show()

# %%
BATCH_SIZE = 2**5

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=2
)

# %% [markdown]
# - https://colab.research.google.com/github/pytorch/vision/blob/gh-pages/main/_generated_ipynb_notebooks/plot_transforms_illustrations.ipynb
# - 