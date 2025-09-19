# Mod3.DesarrolloDeSoftwareAplicadoACienciaDatos

## Team Members: 
- A01027715 - Ulises Orlando Carrizalez Lerin
- A01028008 - Tomás Pérez Vera
- A01799609 - Bárbara Paola Alcántara Vega
- A01710965 - Mónica Monserrat Martínez Vásquez
- A01705840 - María José Soto Castro

## Overview:
This repository contains multiple notebooks that explore the application of PyTorch to real-world data challenges.

1. CIFAR-10 Preprocessing
- Provides a comprehensive guide to preparing the CIFAR-10 dataset for deep learning tasks.

- Covers data loading, exploration, normalization, and augmentation.

- Ensures balanced training and testing sets for effective model training.

2. COVID-19 CT Images Segmentation Challenge
- Implements a baseline semantic segmentation pipeline for COVID-19 CT scans.

- Focuses on detecting ground-glass opacities and consolidations from lung CT slices.

- Uses datasets from MedSeg and Radiopaedia, with images standardized and preprocessed for consistency.

- Employs U-Net with EfficientNet-B2 encoder trained using PyTorch, with evaluation based on pixel accuracy and mean Intersection over Union (mIoU).

- Produces predictions and a submission file in the format required by the challenge.

## Data set information:

- CIFAR-10 is a well-known computer vision dataset consisting of 60,000 images total (50,000 training, 10,000 test), 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), 32×32 pixel resolution with 3 color channels (RGB) and Balanced distribution across all classes, which is something extremely useful as we do not have to make any extra adjustments for uneaven rows/columns in the data set.

- The dataset used for COVID-19 CT segmentation combines two main sources. MedSeg provides 100 CT slices with annotated masks from over 40 patients, while Radiopaedia contributes 829 slices extracted from full CT volumes. The masks are annotated with four classes:

    - Class 0: Ground-glass opacity
    - Class 1: Consolidation
    - Class 2: Other lung areas (ignored for analysis)
    - Class 3: Background (ignored for analysis)

## Report
- `report.pdf` is a detailed report in which we describe in a precise way all the data transfomrations and decissions taken in order to prepare a given data set to be loaded into a machine learning model. 

- `Image segmentation for COVID-19 CT scans` is a complete technical report of the COVID-19 segmentation baseline, covering preprocessing, dataset preparation, model training, evaluation, and submission.