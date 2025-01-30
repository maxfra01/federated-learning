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
- **Optimizer:** SGD (Stochastic Gradient Descent)

## 2. Installation & Dependencies
Before running the experiments, ensure that all required libraries are installed and that your environment is correctly set up.
- ### 2.1 Install Required Libraries
    Run the first cell in the notebooks `cifarFL.ipynb` and `shakesprFL.ipynb`. It installs all required libraries in one step.
    The libraries are essential for: 
    - **Torch & Torchvision:** Core frameworks for building and training deep learning models.
    - **Numpy & Pandas:** Data manipulation and numerical operations.
    - **Matplotlib:** Visualization and progress tracking.
    - **Scikit-learn:** Data splitting and preprocessing utilities.

## 3. Dataset Preparation
 Run the third cell in the notebooks `cifarFL.ipynb` and `shakesprFL.ipynb` to prepare the CIFAR-100 and/or Shakespeare dataset.
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

## 4. Model Initialization 
 - ### 4.1 CIFAR-100 - LeNet-5
    Running the next cell in the notebook, it initializes the deep learning models for CIFAR-100 `cifarFL.ipynb`. The CIFAR-100 dataset is trained using the LeNet-5 architecture, a convolutional neural network (CNN) designed for image classification.
    Model Architecture:
    - **Convolutional Layer 1:** 64 filters, kernel size 5x5, ReLU activation.
    - **Max Pooling Layer 1:** Reduces dimensionality.
    - **Convolutional Layer 2:** 64 filters, kernel size 5x5, ReLU activation.
    - **Max Pooling Layer 2:** Further reduces dimensionality
    - **Fully Connected Layer 1:** 384 neurons, ReLU activation.
    - **Fully Connected Layer 2:** 192 neurons, ReLU activation.
    - **Output Layer:** 100 neurons (corresponding to the 100 classes in CIFAR-100), Softmax activation.
- ### 4.2 Shakespeare - Shakespeare LSTM
    Running the next cell in the notebook, it initializes the deep learning models for Shakespeare datasets `shakesprFL.ipynb`.The Shakespeare dataset is trained using an LSTM-based recurrent neural network (RNN), which processes sequences of characters for text generation tasks.
    Model Architecture:
    - **Embedding Layer:** Convert character indices into dense vectors of size 8.
    - **LSTM Layer 1:** 256 hidden units.
    - **LSTM Layer 2:** 256 hidden units.
    - **Fully Connected Layer:** Outputs a probability distribution over the vocabulary size.

## 5. Centralized Training
 - ### 5.1 CIFAR-100
     Run the fifth and sixth cells in the notebook `cifarFL.ipynb` to train the CIFAR-100 dataset using centralized training.
     Hyperparameters such as the number of epochs, learning rate, and optimizer can be adjusted to reproduce experiments.
 - ### 5.2 Shakespeare 
    Run the fifth and sixth and seventh cells in the notebook `shakesprFL.ipynb` to train the Shakespeare dataset using centralized training.
    Hyperparameters such as the number of epochs, learning rate, and optimizer can be adjusted to reproduce experiments.

## 6. Federated Learning Implementation
 - ### 6.1 CIFAR-100
    Run the last three cells in the notebook `cifarFL.ipynb` to train the CIFAR-100 dataset using federated learning. In particular in the last cell, you can all hyperparameters such as the number of rounds, number of clients, learning rate. To run uniform selection experiments it is sufficient to run FedAVG on the `server_uniform` object.
    For skewed participation, you can adjust the `SKEWNESS` parameter and run the FedAVG on the `server_skewed` object.

    To reproduce non-IID experiments, set the `sharding='niid'` and `Nc=5,1,10...` arguments in the `CIFAR100Dataset` class constructor.
 - ### 6.2 Shakespeare   
    Run the last three cells in the `shakesprFL.ipynb` notebook to train the Shakespeare dataset using federated learning. In particular in the last cell, you can all hyperparameters such as the number of rounds, number of clients, learning rate. To run uniform selection experiments it is sufficient to run FedAVG on the `server_uniform` object.
    To run non-IID experiments, modify in the function _load data this 'client_id': str(user), because we have strings in this case. We also change this parameter for having 100 classes
    preprocess_params = {
        'sharding': 'iid',
        'sf': 0.01,
        'iu': 0.089,
        't': 'sample',
        'tf': 0.8,
    }


## 7. Personal contribution

The `contribution_cifarFL.ipynb` contains the code for the personal contribution to the project.  The structure is the same as the `cifarFL.ipynb` notebook, with the addition of the personal contribution at the end of the notebook. To run the experiments run all cells in the notebook. In the last cell of the notebook, you can adjust the hyperparameters such as the number of rounds, number of clients, learning rate, and the `ALPHA` parameter for the personal contribution. To run the experiments, run the last cell in the notebook.