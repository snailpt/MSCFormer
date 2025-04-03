## MSCFormer
# Multi-Scale Convolutional Transformer Network for Motor Imagery Brain-Computer Interface

# Abstract
Brain-computer interface (BCI) systems allow users to communicate with external devices by translating neural signals into real-time commands. Convolutional neural networks (CNNs) have been effectively utilized for decoding motor imagery electroencephalography (MI-EEG) signals in BCIs. However, traditional CNN-based methods face challenges such as individual variability in EEG signals and the limited receptive fields of CNNs. This study presents the Multi-Scale Convolutional Transformer (MSCFormer) model that integrates multiple CNN branches for multi-scale feature extraction and a Transformer module to capture global dependencies, followed by a fully connected layer for classification. The multi-branch multi-scale CNN structure effectively addresses individual variability in EEG signals, enhancing the model's generalization capabilities, while the Transformer encoder strengthens global feature integration and improves decoding performance. Extensive experiments on the BCI IV-2a and IV-2b datasets show that MSCFormer achieves average accuracies of 82.95% (BCI IV-2a) and 88.00% (BCI IV-2b), with kappa values of 0.7726 and 0.7599 in five-fold cross-validation, surpassing several state-of-the-art methods. These results highlight MSCFormer’s robustness and accuracy, underscoring its potential in EEG-based BCI applications. The code has been released in https://github.com/snailpt/MSCFormer.


Scientific Reports
DOI: 10.1038/s41598-025-96611-5

### Communication
QQ discussion group (Motor imagery and Seizure Detection): 837800443

Email: zhaowei701@163.com

Thank you for your interest!

