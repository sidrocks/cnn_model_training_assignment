# Model Architecture Details

This document outlines key details of the neural network model defined in `Assignment.ipynb`.

## Assignment.ipynb Code Description

The `Assignment.ipynb` notebook implements a Convolutional Neural Network (CNN) for image classification, on the MNIST dataset given the input size of (1, 28, 28).

The network architecture (`Net` class) consists of:
- **Convolutional Layers:** Three `nn.Conv2d` layers with 3x3 kernels, and padding in the first two layers. The third convolutional layer uses a stride of 2.
- **Batch Normalization:** Applied after each convolutional layer (`nn.BatchNorm2d`) and after the first fully connected layer (`nn.BatchNorm1d`).
- **Activation Function:** ReLU activation is used after each batch normalization.
- **Pooling Layers:** Two `F.max_pool2d` layers with a 2x2 kernel are used for downsampling.
- **Flattening:** `torch.flatten` is used to convert the 2D feature maps into a 1D vector before the fully connected layers.
- **Fully Connected Layers:** Two `nn.Linear` layers: `fc1` (180 -> 64) and `fc2` (64 -> 10).
- **Dropout:** A `nn.Dropout` layer with a probability of 0.3 is applied after the first fully connected layer to prevent overfitting.
- **Output Layer:** The final output is passed through `F.log_softmax` for multi-class classification.

The notebook also includes:
- **`torchsummary`**: Used to display a summary of the model, including layer types, output shapes, and parameter counts.
- **Data Transformations:** `torchvision.transforms` are used for both training and testing data, including `RandomApply`, `Resize`, `RandomRotation`, `ToTensor`, and `Normalize`.
- **Data Loading:** The MNIST dataset is downloaded and loaded using `DataLoader` with specified batch sizes and `kwargs` for `num_workers` and `pin_memory`. The test set is split into validation and actual test sets.
- **Training and Testing Functions:** `train` and `test` functions are defined to handle the training and evaluation loops, including loss calculation (`F.nll_loss`), optimizer steps, and accuracy tracking.
- **Optimizer:** `optim.SGD` with a learning rate of 0.1 and momentum of 0.9 is used for training.
- **Training Loop:** The model is trained for 20 epochs, with validation performed after each training epoch.

## Test and Validation Accuracy Logs

Here are the training and validation logs from running the `Assignment.ipynb` notebook in a tabular format:

| Epoch | Val. Loss | Val. Accuracy | Test Loss | Test Accuracy |
|-------|-----------|---------------|-----------|---------------|
| 1     | 0.0860    | 94.00%        | 0.0525    | 98.33%        |
| 2     | 0.0258    | 100.00%       | 0.0323    | 98.98%        |
| 3     | 0.1014    | 96.00%        | 0.0364    | 98.79%        |
| 4     | 0.0158    | 100.00%       | 0.0307    | 99.07%        |
| 5     | 0.0643    | 96.00%        | 0.0326    | 98.92%        |
| 6     | 0.0134    | 100.00%       | 0.0254    | 99.17%        |
| 7     | 0.0140    | 100.00%       | 0.0227    | 99.31%        |
| 8     | 0.0332    | 98.00%        | 0.0240    | 99.20%        |
| 9     | 0.0474    | 98.00%        | 0.0222    | 99.31%        |
| 10    | 0.0519    | 98.00%        | 0.0200    | 99.29%        |
| 11    | 0.0420    | 96.00%        | 0.0230    | 99.21%        |
| 12    | 0.0381    | 98.00%        | 0.0234    | 99.30%        |
| 13    | 0.0119    | 100.00%       | 0.0213    | 99.26%        |
| 14    | 0.0374    | 98.00%        | 0.0225    | 99.28%        |
| 15    | 0.0081    | 100.00%       | 0.0201    | 99.34%        |
| 16    | 0.0049    | 100.00%       | 0.0205    | 99.38%        |
| 17    | 0.0031    | 100.00%       | 0.0224    | 99.27%        |
| 18    | 0.0172    | 98.00%        | 0.0226    | 99.26%        |
| 19    | 0.0070    | 100.00%       | 0.0179    | 99.43%        |

## Total Parameter Count

- **Total Parameters:** 18,002

## Regularization and Layer Usage

- **Batch Normalization:** Batch Normalization is used after convolutional layers (`nn.BatchNorm2d`) and after the first fully connected layer (`nn.BatchNorm1d`).
- **Dropout:** Dropout is applied with a probability of 0.3 after the first fully connected layer.
- **Fully Connected Layer:** Yes, Fully Connected Layers (`nn.Linear`) are used. Specifically, `fc1` (180 -> 64) and `fc2` (64 -> 10). Global Average Pooling (GAP) is not used; instead, `torch.flatten` is used to reshape the tensor before the fully connected layers.
