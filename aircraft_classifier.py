# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
# ---

# %%capture
# %pip install -r requirements.txt

# %%
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt

from math import ceil, sqrt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from train import train_model, TrainingResult, visualize_training_results

# %matplotlib inline

device = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if torch.xpu.is_available() else 'cpu')
display(f"Using device: {device}")

seed = 14_02_2003
np.random.seed(seed)

torch.random.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.random.manual_seed_all(seed)

# %% [markdown]
"""
# Класификација модела авиона са фотографија коришћењем неуронских мрежа

У овом пројекту развијамо систем за класификацију различитих модела авиона на основу фотографија. Примјењене су савремене технике рачунарског 
вида и дубоког учења како би се развила три модела, са циљем поређења њихових перформанси и проналажења најбољег приступа. Прво је имплементирана 
конволутивна неуронска мрежа од нуле, а затим примјењено трансфер учење са Resnet18 и EfficientNetV2s моделима као основе.  

## Скуп података

Користимо _FGVC-Aircraft_ скуп података. Од 120 варијанти авиона класификоваћемо следеће:
"""

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
};

# %% [markdown]
"""
Преузимамо скуп података уз помоћ `download_fgvcaircraft.py` скрипте. Она ће скинути комплетан 
скуп података у привремени фолдер, копирати фотографије жељених варијанти у `data_dir`, а затим 
обрисати оригинални скуп података. При копирању фотографије се организују у под-директоријуме 
за тренинг, валидацију и тест, а затим и по класама.  

_Препорука је да се команди испод дода аргумент путање гдје је FGVC-Aircraft већ или гдје да буде сачуван, 
како би избјегли поновно преузимање. (`-r "./data"`)_
"""

# %%
_variants_arg = ' '.join(map('"{}"'.format, variants))

data_dir = Path('./data')
# %run download_fgvcaircraft.py -o $data_dir -a variant -s train val test --classes $_variants_arg

# %%
showcase_dataset = datasets.ImageFolder(data_dir/"train", transform=T.Compose([T.Resize([224, 224]), T.ToTensor()]))

classes = showcase_dataset.classes
num_classes = len(classes)

labels, indices = np.unique([label for _, label in showcase_dataset.samples], return_index=True)
images = torch.stack([showcase_dataset[i][0] for i in indices])

nrow = ceil(sqrt(len(images) * 2))
grid = torchvision.utils.make_grid(images, nrow=nrow).numpy().transpose(1, 2, 0)

display(
    np.reshape(classes, [-1, nrow], copy=True).tolist(),
    T.ToPILImage(mode='RGB')(grid),
    showcase_dataset
)

# %% [markdown]
"""
### Претпроцесирање

Претпроцесирање у _PyTorch-у_ се обавља коришћењем трансформација. Трансформација прима податке на улазу, обрађује их, и даје друге податке на излазу. 
За визуелну класификацију постоје предефинисане трансформације (у модулу `torchvision.transforms`), које су корисне за обраду фотографија. 
Ове трансформације се могу комбиновати тако да излаз једне буде улаз у другу, формирајући пајплајн (pipeline).  

У _Python-у_ фотографије у меморији се углавном представљају `Image` објектом из _Python Imaging Library (PIL)_ библиотеке, 
која омогућава рад са разним форматима растерских фотографија (.jpg, .png, ...). Он пружа ефикасну манипулацију над фотографијама 
— ротација, исецање, промјена контраста и слично. Међутим за тренирање неуронских мрежа оваква репрезентација није погодна, зато _PyTorch_ библиотека користи 
`torch.Tensor` (тензор - генерализација скалара, вектора и матрица). Фотогравија је представљена тродимензијалним тензором димензија _C_ x _W_ x _H_, 
гдје је _C_ број канала (нпр. црвени, плави и зелени), _W_ ширина, а _H_ висина фотографије. Свака скаларна вриједност коју садржи представља интензитет на 
датој висини, ширини и по датом каналу фотографије.  

Да би се максимизовала ефикасност фотографије обрађујемо користећи `Image` објекте, које непосредно прије прослеђивања неуронској мрежи конвертујемо у тензор.  

Како све фотографије у скупу података садрже банер са ауторским правима (на дну, висине 20 пиксела), прије употребе морамо уклонити овај банер. 
Затим, јер неуронске мреже тако захтјевају, све скалирати на исте димензије. Па на крају, како би побољшали тренинг перформансе и стабилност, стандардизујемо 
вриједности интензитета. Тим смо завршили процес нормализације фотографија.  
"""

# %%
IMG_SHAPE = 240, 240

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

remove_copyright = RemoveCopyright()
resize = T.Resize(IMG_SHAPE)
to_tensor = T.ToTensor()
to_pil = T.ToPILImage(mode='RGB') # inverse from to_tensor (used for displaying images)
# TODO: Perform standardization with mean and std

# %% [markdown]
"""
_Због јаснијег приказа пајплајна трансформација не користимо уграђену `torchvision.transforms.Lambda` трансформацију него дефинишемо нову._  

При обучавању конволутивне неуронске мреже од нуле, параметре нормализације — као што је величина слике, средња вриједност (mean) 
и стандардна девијација (std) — можемо слободно прилагодити како би дали најбоље резултате за наш модел и податке.  

Међутим, када користимо **трансфер учење** (transfer learning) и ослањамо се на унапред обучене моделе (у нањем случају _ResNet18_ и _EfficientNetV2s_), 
онда морамо поштовати исте нормализационе параметре који су коришћени током њихове иницијалне обуке — у супротном мрежа неће исправно „разумјети” нове улазе.  

Поред трансформација намјењених нормализацији улазних података, пошто не располажемо великим скупом података, користићемо и 
насумичне трансформације **тренинг** података — поступак познат као аугментација (augmentation). Циљ ових трансформација јесте да „вјештачки” повећамо скуп података, 
увођењем нпр. насумичних ротација или промјена контраста. Пошто моделу показујемо исту фотографију (са диска) више пута, овим трансформацијама 
добијамо варијације исте слике, што помаже бољој генерализацији модела.    

#### КНМ од нуле

У блоку испод дефинишемо комплетан пајплајне претпроцесирања за конволутивну неуронску мрежу од нуле.  
"""

# %%
transforms_train = T.Compose([
    remove_copyright,
    resize,
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=7),
    T.ColorJitter(brightness=2e-1, contrast=2e-1),
    to_tensor
])

transforms_eval = T.Compose([
    remove_copyright,
    resize,
    to_tensor
]);

# %% [markdown]
"""
Како се само тренинг подаци аугментују, дефинишемо посебне трансформације за тренинг, а посебне 
за валидацију и тестирање.  

#### Трансфер учење - ResNet18

Библиотека `torchvision` нуди разне унапред трениране моделе познатих архитектура. 
Један од њих је _ResNet18_, обучен над ImageNet v1 1k (1000 класа) скупом података. 
Дефинишемо пајплајнe претпроцесирања за трансфер учење са _ResNet18_ моделом.  
"""

# %%
from torchvision.models import ResNet18_Weights, resnet18

resnet_weights = ResNet18_Weights.IMAGENET1K_V1
display(resnet_weights.transforms())

# %% [markdown]
"""
Енумерацијама формата `<architecture name>_Weights`, бирамо скуп података над којим је архитектура тренирана. 
Такође од конкретне вриједности енумерације можемо сазнати коришћене параметре нормализације. 
"""

# %%
resnet_transforms_train = T.Compose([
    remove_copyright,  
    T.Resize(256, interpolation=InterpolationMode.BILINEAR),
    T.CenterCrop(224), 
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=7),
    T.ColorJitter(brightness=2e-1, contrast=2e-1),
    to_tensor, 
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

resnet_transforms_eval = T.Compose([
    remove_copyright,  
    T.Resize(256, interpolation=InterpolationMode.BILINEAR),
    T.CenterCrop(224), 
    to_tensor, 
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
]);
  
# %% [markdown]
"""
#### Трансфер учење - EfficientNetV2s

Иако `torchvision` нуди EfficientNetV2s архитектуру, не нуди опцију унапред трениране над 
ImageNet v1 21k (21,000 класа) — зато користимо библиотеку `timm`. Инспекцијом конфигурације нормализационих 
параметара за модел, креирамо претпроцесирајућe пајплајнe.  
"""

# %%
import timm
from timm.data.config import resolve_data_config

enet_model_name = "tf_efficientnetv2_s.in21k"
enet_config = resolve_data_config({}, model=enet_model_name)
display(enet_config)

# %%
enet_transforms_train = T.Compose([
    remove_copyright,
    T.Resize(int(224 // 0.875), interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=7),
    T.ColorJitter(brightness=2e-1, contrast=2e-1),
    to_tensor,
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

enet_transforms_eval = T.Compose([
    remove_copyright,
    T.Resize(int(224 // 0.875), interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    to_tensor,
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
]);

# %% [markdown]
"""
## Тренирање модела

Процес тренирања извршава се у више **епоха** (`EPOCHS`) — једна епоха обухвата један пролазак кроз комплетан тренинг скуп података. 
Током тренирања пратимо перформансе модела на **валидационом скупу**. Ако се метрика модела не побољша током фише узастопних епоха (`PATIENCE`), 
прекидамо тренинг да би спријечили претренирање (overfitting). 

_Имплементација тренинг петље са визуелизацијом се налази у `train.py` модулу._
"""

# %%
EPOCHS = 50
BATCH_SIZE = 2**5 # 32
PATIENCE = 8

# %% [markdown]
"""
### КНМ од нуле

Имплементирана конволутивна неуронска мрежа се састоји из више слојева:  
- **4 конволутивна блока** — намјена препознавање карактеристика (feature)
- **Global Average Pooling** — сманјује димензије
- **Потпуно повезан (fully connected) класификатор** са `Dropоut`-ом

Ова структура омогућава мрежи да постепено учи све сложеније карактеристике, уз контролу _overfitting-a_.  
"""

# %%
class AircraftCNNClassifier(nn.Module):

    def __init__(self, num_classes: int):
        super(AircraftCNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# %%
dataset = {
    'train': datasets.ImageFolder(data_dir/"train", transform=transforms_train), 
    'val': datasets.ImageFolder(data_dir/"val", transform=transforms_eval),
    'test': datasets.ImageFolder(data_dir/"test", transform=transforms_eval)
}

model = AircraftCNNClassifier(num_classes).to(device)

# %%
results = train_model(
    model, 
    train_loader=DataLoader(dataset['train'], BATCH_SIZE, shuffle=True), 
    val_loader=DataLoader(dataset['val'], BATCH_SIZE), 
    patience=PATIENCE, 
    num_epochs=EPOCHS,
    num_classes=num_classes,
    optimizer=optim.SGD,
    device=device, 
    lr=1e-4, 
    out_file="cnn_best_model.pth",
)

# %% [markdown]
"""
#### Анализа резултата
"""

# %%
visualize_training_results(results, labels=classes)

# %% [markdown]
"""
### Трансфер учење - Resnet18
"""

# %%
resnet_dataset = {
    'train': datasets.ImageFolder(data_dir/"train", transform=resnet_transforms_train), 
    'val': datasets.ImageFolder(data_dir/"val", transform=resnet_transforms_eval),
    'test': datasets.ImageFolder(data_dir/"test", transform=resnet_transforms_eval)
}

resnet_model = resnet18(weights=resnet_weights)

for param in resnet_model.parameters():
    param.requires_grad = False

resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

for param in resnet_model.fc.parameters():
    param.requires_grad = True

resnet_model = resnet_model.to(device);

# %%
resnet_results = train_model(
    resnet_model, 
    train_loader=DataLoader(resnet_dataset['train'], BATCH_SIZE, shuffle=True),
    val_loader=DataLoader(resnet_dataset['val'], BATCH_SIZE),
    patience=PATIENCE,
    num_epochs=EPOCHS,
    num_classes=num_classes,
    optimizer=optim.Adam,
    device=device,
    lr=1e-3,
    out_file="resnet_best_model.pth",
)

# %% [markdown]
"""
#### Анализа резултата
"""

# %%
visualize_training_results(resnet_results, labels=classes)

# %% [markdown]
"""
### Трансфер учење - EfficientNetV2s
"""

# %%
enet_dataset = {
    'train': datasets.ImageFolder(data_dir/"train", transform=enet_transforms_train), 
    'val': datasets.ImageFolder(data_dir/"val", transform=enet_transforms_eval),
    'test': datasets.ImageFolder(data_dir/"test", transform=enet_transforms_eval)
}

enet_model = timm.create_model(enet_model_name, pretrained=True)

for param in enet_model.parameters():
    param.requires_grad = False

enet_model.reset_classifier(num_classes=num_classes) # type: ignore # pyright: ignore
#enet_model.classifier = nn.Linear(enet_model.classifier.in_features, len(classes)) # type: ignore # pyright: ignore

for param in enet_model.get_classifier().parameters(): # type: ignore # pyright: ignore
    param.requires_grad = True

enet_model = enet_model.to(device);

# %%
enet_results = train_model(
    enet_model,
    train_loader=DataLoader(enet_dataset['train'], BATCH_SIZE, shuffle=True),
    val_loader=DataLoader(enet_dataset['val'], BATCH_SIZE),
    patience=PATIENCE,
    num_epochs=EPOCHS,
    num_classes=num_classes,
    optimizer=optim.Adam,
    lr=1e-3,
    out_file='enet_best_model.pth',
)

# %% [markdown]
"""
#### Анализа резултата
"""

# %%
visualize_training_results(resnet_results, labels=classes)

# %% [markdown]
"""
### Евалуација модела

Као што је и очекивано, најгоре резултате остварује КНМ од нуле. Њена прецизност је значајно мања од 
трансфер учења, док је њено тренирање највише временски захтјевно (не рачунајући иницијални тренинг модела Resnet18 и EfficientNetV2s).  

Најбољи резултати добијени сз примјеном трансфер учења са модерним **EfficientNetV2s** моделом, као што се може видјети из евалуације испод.  
"""

# %%

def eval_model(model, test_loader):
    from torchmetrics.classification import Accuracy
    model =  model.to(device)
    model.eval()

    accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            preds = model(images)
            preds = torch.argmax(preds, dim=1)
            accuracy.update(preds, labels)
    
    return accuracy.compute().item()

# %%
model = AircraftCNNClassifier(num_classes)
model.load_state_dict(torch.load('cnn_best_model.pth'))

resnet_model = resnet18()
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
resnet_model.load_state_dict(torch.load('resnet_best_model.pth'))

enet_model = timm.create_model(enet_model_name, pretrained=False, num_classes=num_classes)
enet_model.load_state_dict(torch.load('enet_best_model.pth'))


accuracy = eval_model(model, DataLoader(dataset['test'], BATCH_SIZE))
resnet_accuracy = eval_model(resnet_model, DataLoader(resnet_dataset['test'], BATCH_SIZE))
enet_accuracy = eval_model(enet_model, DataLoader(enet_dataset['test'], BATCH_SIZE))

display(
    f"CNN from scratch accuracy:  {accuracy:.3f}",
    f"Resnet18 accuracy: {resnet_accuracy:.3f}",
    f"EfficientNetV2s accuracy: {enet_accuracy:.3f}"
)
