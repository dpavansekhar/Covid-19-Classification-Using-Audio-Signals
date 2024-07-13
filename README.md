# Covid-19 Classification from Audio Signals

This repository contains the code and data used for classifying Covid-19 from audio signals. The dataset consists of 1207 negative samples and 150 positive samples. The project includes data visualization, feature extraction, data preprocessing, and model selection.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Visualization](#data-visualization)
4. [Feature Extraction and Data Preprocessing](#feature-extraction-and-data-preprocessing)
5. [Model Selection](#model-selection)
    - [Random Forest Classifier](#random-forest-classifier)
    - [Support Vector Machine](#support-vector-machine)
    - [Gradient Boosting Classifier](#gradient-boosting-classifier)
    - [Convolutional Neural Networks](#convolutional-neural-networks)
6. [Results](#results)
7. [Contributing](#contributing)

## Introduction

The purpose of this project is to classify audio signals as Covid-19 positive or negative using various machine learning models. The project includes steps for data visualization, feature extraction, data preprocessing, model building, and evaluation.

## Dataset

The dataset consists of audio samples labeled as either Covid-19 positive or negative. The dataset is divided into:

- 1207 negative samples
- 150 positive samples

## Data Visualization

Data visualization helps in understanding the audio signals better. The following visualizations are provided for both Covid positive and negative audio samples:

- **Waveform**: Shows the amplitude of the audio signal over time.
- **Zero Crossing Rate**: Indicates the rate at which the signal changes sign.
- **Spectral Centroid**: Represents the center of mass of the spectrum.
- **Spectral Bandwidth**: Measures the width of the spectral band.
- **Mel Frequency Cepstral Coefficients (MFCCs)**: Represents the short-term power spectrum of a sound.

## Feature Extraction and Data Preprocessing

Feature extraction involves extracting useful information from the audio signals which can be used for classification. The following features are extracted:

- **MFCC coefficients**: Mel Frequency Cepstral Coefficients represent the short-term power spectrum of a sound.
- **Spectral Centroid**: Indicates where the "center of mass" of the spectrum is located.
- **Spectral Bandwidth**: Describes the width of the frequency range.
- **Zero Crossing Rate**: The rate of sign changes along the audio signal.
- **Spectral Roll-off**: The frequency below which a specified percentage of the total spectral energy lies.
- **RMS Energy**: The root-mean-square value of the signal, providing a measure of its intensity.

These features are extracted using the librosa library in Python. After extraction, they are saved in a structured dataset suitable for model training and evaluation.

# Audio Signal Classification Models

In this project, we explored various machine learning models for classifying audio signals. Below are brief descriptions of the models used:

## Random Forest Classifier
- **Description**: Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.
- **Strengths**: It's robust against overfitting, handles high-dimensional data well, and provides good accuracy in classification tasks.
- **Applications**: Used in various fields such as finance, healthcare, and remote sensing for tasks like fraud detection, disease prediction, and image classification.

## Support Vector Machine (SVM)
- **Description**: SVM is a supervised machine learning algorithm that finds an optimal hyperplane which best separates classes in a high-dimensional feature space.
- **Strengths**: Effective in high-dimensional spaces, memory-efficient due to its use of a subset of training points (support vectors), and capable of handling non-linear data with the kernel trick.
- **Applications**: Widely used in text classification, image recognition, and bioinformatics for tasks like spam detection, facial expression recognition, and protein structure prediction.

## Gradient Boosting Classifier
- **Description**: Gradient Boosting is an ensemble technique where weak learners (often decision trees) are sequentially trained to correct the errors of the previous models.
- **Strengths**: Produces strong predictive performance, handles heterogeneous features well, and can capture complex relationships in data.
- **Applications**: Used in web search ranking, anomaly detection, and ecological modeling for tasks like click-through rate prediction, fraud detection, and species distribution modeling.

## Convolutional Neural Networks (CNN)
- **Description**: CNNs are deep learning models designed to process structured grid-like data, such as images and audio spectrograms, by using convolutional layers to automatically learn hierarchical representations.
- **Strengths**: Highly effective for tasks involving spatial relationships (like image or audio data), capable of learning feature hierarchies, and robust to variations in input data.
- **Applications**: Dominant in computer vision tasks such as image classification, object detection, and facial recognition. Also increasingly used in audio processing for tasks like speech recognition and sound classification.

These models were evaluated based on their performance metrics such as accuracy, precision, recall, and F1-score. The CNN model showed the best performance in classifying Covid-19 positive and negative samples.

## Results

The models were evaluated using accuracy, precision, recall, and F1-score. The CNN model showed the best performance, achieving high accuracy in classifying Covid-19 positive and negative samples.


# Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. **Open an Issue**: If you have a feature request, bug report, or general feedback, please open an issue to discuss it.

2. **Fork the Repository**: If you want to contribute code, fork the repository on GitHub.

3. **Create a Branch**: Create a new branch off of the `main` branch for your feature or bug fix.

4. **Make Changes**: Make your changes in the new branch. Ensure that your code follows the project's coding conventions and style.

5. **Test Your Changes**: Test your changes thoroughly to ensure they work as expected.

6. **Submit a Pull Request**: Once your changes are ready, submit a pull request. Provide a clear description of your changes and why they are beneficial.

7. **Review and Discuss**: Participate in the code review process and address any feedback or concerns.

8. **Merge**: After approval, your pull request will be merged into the main branch.



Thank you for considering contributing to our project!

