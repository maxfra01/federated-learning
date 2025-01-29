# Federated Learning

## 1. Project overview
### CIFAR-100
 - **Model:** LeNet-5
 - **Training Modes:**
    - Centralized Training
    - Federated Learning
- **Learning Approaches:**
    - IID (Indipendent and Identically Distributed)
    - Non-IID (Partitioned by class)
- **Schedulers:**
    - Step LR
    - Exponential LR
    - CosineAnnealing LR
- **Optimizer:** SGD (Stochastic Gradient Descent)

### Shakespeare
 - **Model:** Shakespeare LSTM
 - **Training Modes:**
    - Centralized Training
    - Federated Learning
- **Learning Approaches:**
    - IID (Indipendent and Identically Distributed)
    - Non-IID (Partitioned by class)
- **Schedulers:**
    - Step LR
    - Exponential LR
    - CosineAnnealing LR
- **Optimizer:** SGD (Stochastic Gradient Descent)**

## 2. Installation & Dependencies
Before running the experiments, ensure that all required libraries are installed and that your environment is correctly set up.
- ### 2.1 Install Required Libraries
    Run the first cell in the notebook. It installs all required libraries in one step.
    The libraries are essential for: 
    - **Torch & Torchvision:** Core frameworks for building and training deep learning models.
    - **Numpy & Pandas:** Data manipulation and numerical operations.
    - **Matplotlib:** Visualization and progress tracking.
    - **Scikit-learn:** Data splitting and preprocessing utilities.

## 3. Dataset Preparation
 Run the third cell in the notebook to prepare the CIFAR-100 and/or Shakespeare dataset.
- ### 3.1 CIFAR-100
   
    This step will download the CIFAR-100 dataset (if not already present) directly from the official repository and apply data augmentations and normalizations for the training and testing datasets
    Why this step is necessary?
    - **CIFAR-100 Overview:** It contains 100 classes, each with 600 images (500 for training, 100 for testing).
    Images are 32x32 pixels with RGB color channels.
    - **Data Augmentations:**
        - **Random Horizontal Flip:** Helps improve model generalization by flipping images randomly.
        - **Random Cropping:**: Prevents overfitting by cropping random parts of the image.
        - **Normalization:** Ensures that input features have zero mean and unit variance.

- ### 3.2 Shakespeare
    This step will preprocess the Shakespeare dataset (if not already prepared) by tokenizing text, padding sequences, and partitioning the data for training and testing.
    Why this step is necessary?
    - **Shakespeare Overview:**
       The Shakespeare dataset is derived from The Complete Works of William Shakespeare, split into lines of text. Each client in the dataset represents a character, and their data consists of lines spoken by that character.
    - **Text Pre-processing Steps:**
        - **Character-to-Index Mapping:** Coverts each character in the text into a unique numerical index using a predefined vocabulary of all possible characters.
        - **Padding and Truncation:** Ensures that all sequences are of a fixed length.
        - **Partitioning:**
            - **Train/Test Split:** Each client's data is divided into training and testing sets.
            - **Sharding Options:**
            IID (randomly partitions data across clients) & Non-IID (partitions data such that each client only has access to a subject of unique text)

## 4. Model Initialization & Training
Running the next cell in the notebook, it initializes the deep learning models for CIFAR-100 and Shakespeare datasets.
 - ### 4.1 CIFAR-100 - LeNet-5
    The CIFAR-100 dataset is trained using the LeNet-5 architecture, a convolutional neural network (CNN) designed for image classification.
- ### 4.2 Shakespeare - Shakespeare LSTM
    The Shakespeare dataset is trained using an LSTM-based recurrent neural network (RNN), which processes sequences of characters for text generation tasks.

## 5. Centralized Training
 - ### 5.1 CIFAR-100
 - ### 5.2 Shakespeare 
## 6. Federated Learning Implementation
 - ### 6.1 CIFAR-100
 - ### 6.2 Shakespeare 