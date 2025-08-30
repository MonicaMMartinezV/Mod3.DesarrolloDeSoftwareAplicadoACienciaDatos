# Mod3.DesarrolloDeSoftwareAplicadoACienciaDatos

Okay, my git now is gitting lol, so I will make a second commit just to be sure


# Overview:
This notebook provides a comprehensive guide to preparing the CIFAR-10 dataset for deep learning tasks using PyTorch. It covers the essential steps of data loading, exploration, normalization, and augmentation that are crucial for effective model training.

# Data set information:
CIFAR-10 is a well-known computer vision dataset consisting of 60,000 images total (50,000 training, 10,000 test), 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), 32×32 pixel resolution with 3 color channels (RGB) and Balanced distribution across all classes, which is something extremely useful as we do not have to make any extra adjustments for uneaven rows/columns in the data set.

# Steps taken in Data Preparation:
## Environment setup
First, we set up reproducibility with 42 fixed random seeds for reproducibility, check GPU availability (specially important if the notebook is being run on a local device) and import the necessary libraries to work with the data base which will be downloaded (PyTorch, TorchVision, NumPy, Matplotlib)

## Data loading:
Then, the CIFAR-10 data set is downloaded using TorchVision's datasets module (base_transform = transforms.ToTensor()), and create a train and test splits..

```python
train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=base_transform)
test_set  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=base_transform)
```

## Data exploration:
First, it's important to see what we're working with before preparing our data for analysis, and for that, we first examine the dataset's dimensons and structure (len(train_set), len(test_set), train_set.data.shape), visualize some random samples from the dataset (def show_images(dataset, n=5):) and display the class names (classes = train_set.classes).

## Normalization:
Firstly it is important to note that a batch size was set in order to load and compute the training.

It is extremely important to normalize the data we're working with, as this enables us to scale the data range to a standard scale, ensuring that all data have a comparable scale to one another, specially if we have different lenghths, which is fortunately not the case this time. This way we ensure that larger values don't dominate the training process. The normalization equation used was:

```math

X norm = X - min(X) / max(X) - min(X)

```
```python
normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

train_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

train_set_norm = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=train_transform)
test_set_norm  = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_transform)

len(train_set_norm), len(test_set_norm)
```

This equation computes per-channel mean and standard deviation from the training set using calculated statistics. After normalizing, we geta size of:

(50000, 10000)


Lastly, it is very important to verify the normalization by using (mean ≈ 0, std ≈ 1) by using DataLoader from the previusly installed library torch.utils.data:
```python
train_loader_norm = DataLoader(train_set_norm, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
mean_norm, std_norm
```

After post-normalization means and stds, the printed result should be close to [0,0,0] and [1,1,1]. (If small deviations are showed, there shouldn't be a problem.

## Data Augmentation:
A random horizontal flapping was implemented in order to apply augmentation only to the training set (not the test set).
```python
aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    #anything else you may want to add is welcomed
    normalize,
])

train_set_aug = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=aug_transform)

len(train_set_aug)
```

Augmentation helps by increasing the dataset diversity without collecting new data, presenting different parts of the data sert to train in a randomly manner in order to simulate new data presented in a real-world scenario. This way, overfitting is reduced, as we expose the model to varied examples each time as if it was the first time they were ever seen, and thus, improving model generalization.

It is important to note, that for small images like CIFAR-10 (32×32), heavy augmentations should be used cautiously as they may remove too much semantic content or introduce unrealistic artifacts because of the pixel array dimentions.

## Class Distribution Analysis:
Fortunately, CIFAR-10 is perfectly distributed among classes, athough this will not always be the case, plots help and numbers help, so in order to see the distribution we:
```python
show class distribution
show plot 
```


