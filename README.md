# AI-ML-Assignment-3-Simple-NN
# MNIST Digit Classification with a Feedforward Neural Network

**Student:** Victoria Salomon  
**Course:** AI / Machine Learning Assignment  
**Framework:** TensorFlow / Keras (TensorFlow 2.12.0)

## Project Overview
This project trains a simple Feedforward Neural Network (FNN) to classify handwritten digits (0–9) using the MNIST dataset.  
The model learns to recognize patterns in grayscale images of handwritten numbers and predicts which digit (0 through 9) is shown.

The work in this repository includes:
- A Jupyter Notebook that loads and preprocesses the MNIST dataset
- A neural network model built with TensorFlow/Keras
- Model training and evaluation
- A demo prediction on a random test digit
- Final accuracy results

## Dataset
**MNIST handwritten digits dataset**
- 60,000 training images
- 10,000 test images  
- Each image is 28x28 pixels, grayscale  
- Labels are digits from 0 to 9

## Preprocessing Steps
1. **Normalization:**  
   Pixel values were scaled from the range [0, 255] down to [0, 1] to help the model train more efficiently and stabilize gradients.

2. **Flattening:**  
   Each 28x28 image was flattened into a 784-dimensional vector so it could be passed into a dense (fully connected) layer.

3. **One-hot encoding of labels:**  
   The digit labels (0–9) were converted to one-hot vectors of length 10.  
   Example: the label `7` becomes `[0,0,0,0,0,0,0,1,0,0]`.

## Model Architecture
The model is a simple Feedforward Neural Network (fully connected / dense network) with the following layers:

1. **Input Layer:**  
   - Shape: 784 (flattened 28x28 image)

2. **Hidden Layer:**  
   - Dense layer with 128 neurons  
   - Activation function: ReLU  
   - ReLU (Rectified Linear Unit) helps the model learn nonlinear patterns efficiently by zeroing out negative values.

3. **Output Layer:**  
   - Dense layer with 10 neurons  
   - Activation function: Softmax  
   - Softmax converts the output into a probability distribution across the 10 digit classes (0–9).

In Keras code, the model looks like this:

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])
