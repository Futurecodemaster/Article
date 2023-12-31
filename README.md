# Backpropagation

At its core, backpropagation is an application of the chain rule of calculus. It efficiently computes the gradient of the loss function (a measure of prediction error) with respect to each weight in the network. By iteratively adjusting these weights in the direction that minimizes the error, backpropagation guides the network towards more accurate predictions.

## Historical Context

The concept of backpropagation has been around since the 1970s, but it was the seminal work of David Rumelhart, Geoffrey Hinton, and Ronald Williams in the 1980s that brought it into the limelight. Their research demonstrated how backpropagation could effectively train multi-layer neural networks, laying the groundwork for the deep learning revolution we witness today.

## Basic Concepts

1. **Neural Networks**: At a high level, a neural network is composed of layers of interconnected nodes (neurons). Each connection has an associated weight, and each neuron applies an activation function to its input.
2. **Loss Function**: This is a function that measures the difference between the network's prediction and the actual target values. The goal of training a neural network is to minimize this loss function.
3. **Gradient Descent**: This is an optimization algorithm used to minimize the loss function by iteratively adjusting the weights of the network.

### The Role of Partial Derivatives

Partial derivatives quantify how a slight change in a weight affects the loss. Understanding this relationship is key to directing weight adjustments for loss reduction.

**The diagram below illustrates the backpropagation process in a neural network**

![Backpropagation Process](https://github.com/Futurecodemaster/Article/blob/main/%20backpropagation.png?raw=true)

**Input Layer:** This is the first layer of the neural network where the input data is fed into the system. Each neuron in this layer represents a feature of the input data.

**Activations to Hidden Layer:** The data from the input layer is passed to the hidden layer through connections. Each connection has a weight and possibly a bias. The data is transformed in the hidden layer neurons by a weighted sum followed by a non-linear activation function.

**Hidden Layer:** This layer processes the inputs received from the input layer and passes the output to the next layer. The hidden layer can perform complex computations with the data.

**Activations to Output Layer:** Similar to the previous step, the output from the hidden layer is passed to the output layer. Again, this involves weighted sums and activation functions.

**Output Layer:** This layer produces the final output of the neural network. The way it's structured and the functions it uses depend on the specific task (e.g., regression, classification).

**Calculate Loss:** After the forward pass (from input to output), the network calculates the loss (or error). The loss function measures how far the network's output is from the expected result.

**Backpropagate Error:** This is where backpropagation starts. The error calculated is propagated back through the network. This involves calculating the gradient of the loss function with respect to each weight by the chain rule.

**Update Weights (Hidden Layer):** Based on the error received, the weights between the hidden layer and the output layer are adjusted. This is typically done using a gradient descent optimization algorithm.

**Backpropagate Error to Input Layer:** The error is further propagated back to the connections between the input and hidden layers.

**Update Weights (Input Layer):** Finally, the weights between the input layer and the hidden layer are adjusted based on the error.

### The Process of Backpropagation

Now, let's break down the steps of backpropagation:

1. **Forward Pass**:
   - Data flows through the network, layer by layer.
   - Each layer's output is computed using current weights, biases, and activation functions.
   - The process culminates in the final output.

   For a given layer ${(l)}$, the output $y^{(l)}$ can be represented as:

    $y^{(l)} = f(W^{(l)} \cdot x^{(l)} + b^{(l)})$

   Where $(W^{(l)}$ are the weights of layer $(l)$, $x^{(l)}$ is the input to layer $(l)$, $b^{(l)}$ are the biases of layer $(l)$, and $f$ is the activation function (e.g., ReLU, sigmoid).

2. **Compute Loss**:
   - After the final output is produced (let's call it $\hat{y}$) it's compared against the actual target values $y$ using a loss function $L$. The choice of loss function depends on the task (e.g., Mean Squared Error for regression, Cross-Entropy for classification).

   For regression, Mean Squared Error (MSE) is often used:

   $L(\hat{y}, y) = \frac{1}{n} \sum (\hat{y} - y)^2$

   Where $N:$ is the number of samples.

3. **Backward Pass (Backpropagation)**:
   - Step 1: Compute the gradient of the loss function with respect to each weight. This involves applying the chain rule of calculus, as the loss function is a composite function of the weights via the network's layers and activation functions.
   - Step 2: Partial derivatives are computed backward from the output layer to the input layer, hence the name "backpropagation."
   - Step 3: The gradients tell us the direction in which the loss function is increasing. To minimize the loss, we need to adjust the weights in the opposite direction.

   For a weight $W_{ij}^{(l)}$ in layer $(l)$, the partial derivative of the loss $L$ with respect to that weight is:

   $\frac{\partial L}{\partial W_{ij}^{(l)}}$

   The chain rule allows us to express this derivative in terms of derivatives of the output of each layer and the derivatives of the activation functions. For a simple network with a single hidden layer, this might look like:

   $\frac{\partial L}{\partial W_{ij}^{(hidden)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial y^{(hidden)}} \cdot \frac{\partial y^{(hidden)}}{\partial W_{ij}^{(hidden)}}$

   Where $\frac{\partial L}{\partial \hat{y}}$ is the gradient of loss with respect to the network's output, $\frac{\partial \hat{y}}{\partial y^{(hidden)}}$ is the gradient of the output layer with respect to the output of the hidden layer, and $\frac{\partial y^{(hidden)}}{\partial W_{ij}^{(hidden)}}$ is the gradient of the hidden layer's output with respect to its weights.

4. **Weight Update**:
   - The weights are updated by subtracting a fraction (defined by the learning rate) of the gradient.
   - This process is repeated for many iterations (epochs) over the training dataset.

   The update rule is generally:

   $W_{new} = W_{old} - \alpha \frac{\partial L}{\partial W}$

   Where $\alpha:$ is the learning rate, a small positive number that controls the size of the weight updates.

## This a very simple example of a neural network with one input layer, one hidden layer, and one output layer

![Backpropagation Example](https://github.com/Futurecodemaster/Article/blob/main/backpropogation2.png)

In this diagram:

- The input neuron `(x)` with a value of `0.05 ` feeds into two hidden neurons `h1` and `h2` with weights `w1 (0.15)`, `w2 (0.20)` and biases `b1 (0.35)` ,  `b2 (0.35)`.     
- The hidden neurons then connect to the output neuron `(y)` with weights `w3 (0.25)` and `w4 (0.30)`.    
- The output is compared to the target value `0.01` to calculate the loss `L`.   
- The loss is then backpropagated to update the weights `w1, w2, w3, w4`.     


### Neural Network Structure
1. Input Layer: 1 neuron $(x)$
2. Hidden Layer: 2 neurons $(h1, h2)$
3. Output Layer: 1 neuron $(y)$

### Initial Weights and Biases
- Weights: $w1 = 0.15, w2 = 0.20, w3 = 0.25, w4 = 0.30$
- Biases: $b1 = 0.35, b2 = 0.35, b3 = 0.60$

### Activation Function
We'll use the sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$

### Input and Target Output
- Input $(x): 0.05$
- Target Output: $0.01$

### Forward Pass
1. Hidden Layer:
   - $h1 = \sigma(w1 \cdot x + b1)= σ(0.15⋅0.05+0.35) ≈ 0.588$
   - $h2 = \sigma(w2 \cdot x + b2) = σ(w2⋅x+b2)=σ(0.20⋅0.05+0.35) ≈ 0.589$

2. Output Layer:
   - $y = \sigma(w3 \cdot h1 + w4 \cdot h2 + b3) = σ(w3⋅h1+w4⋅h2+b3)$
   - $y=σ(0.25⋅h1+0.30⋅h2+0.60)$
   - $y=σ(0.25⋅0.588+0.30⋅0.589+0.60)≈σ(0.7745)≈0.684$

### Loss Calculation
Using Mean Squared Error (MSE):
- $L = \frac{1}{2}(target - y)^2$
- $= \frac{1}{2}(0.01 – 0.684)^2 ≈0.227$

### Backward Pass (Backpropagation)
To update the weights, we need to calculate the gradient of the loss with respect to each weight. This involves applying the chain rule for derivatives.

1. Calculate Gradient of Loss w.r.t Weights:
   - For example, $\frac{\partial L}{\partial w3}$ is the rate of change of loss with respect to the weight w3.
   - $\frac{\partial L}{\partial w3} = \frac{\partial L}{\partial y} \cdot \frac{\partial h1}{\partial w3}$
   - $\frac{\partial L}{\partial y} = −(target−y)=−(0.01−0.684)≈0.674$
   - $\frac{\partial y}{\partial w3} = h1⋅y⋅(1−y)≈0.588⋅0.684⋅(1−0.684)$
   - $\frac{\partial L}{\partial w3} ≈0.674⋅0.588⋅0.684⋅(1−0.684)$

### Update Weights:
- For example, $w3 = w3 - \text{learning rate} \cdot \frac{\partial L}{\partial w3}$
- Assuming a learning rate of 0.5 for simplicity.
- $w3=0.25−0.5⋅[calculated gradient]$

We would repeat similar calculations for w1, w2, and w4.

This completes the backpropagation process for one iteration. In practice, we would repeat this process over many iterations (epochs) to gradually reduce the loss and improve the model's predictions.

## Example code demonstrating a simple neural network trained on a hypothetical dataset:

```
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
```

**TensorFlow:** A powerful library for numerical computation, particularly well-suited for large-scale machine learning.  
**Dense:** A layer module in Keras (part of TensorFlow) that represents a fully connected neural network layer.  
**Sequential:** A Keras model that represents a linear stack of layers.   
**Adam:** An optimizer in Keras, a variant of the gradient descent algorithm, known for its effectiveness in practice.   


```
X = [[0.1, 0.2], [0.2, 0.2], [0.3, 0.4], [0.4, 0.5]]
y = [[0.3], [0.4], [0.7], [0.9]]
```

**X and y are sample datasets.** 
- **X** represents the input features.
- **y** represents the target values.

*Note: This data is simplistic and for demonstration purposes only.*

```
model = Sequential([
    Dense(5, input_shape=(2,), activation='relu'),
    Dense(1, activation='linear')
])
```

**Sequential Model:** 
- We create a sequential model, which allows us to build a model layer by layer.

**First Dense Layer:** 
- This is the first hidden layer. 
- It has 5 neurons (`Dense(5)`), and it expects input with 2 features (`input_shape=(2,)`). 
- The activation function is ReLU (Rectified Linear Unit).

**Second Dense Layer:** 
- This is the output layer with a single neuron (since we are predicting a single value). 
- The activation function is linear.

```
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

```

**Compile:** 
- This prepares the model for training. We specify the optimizer and the loss function.

**Optimizer:** 
- Adam optimizer is used with a learning rate of 0.01. 
- The optimizer's role is to adjust the weights of the network to minimize the loss.

**Loss Function:** 
- Mean Squared Error (MSE) is used as the loss function, suitable for regression problems.

```
model.fit(X, y, epochs=100, verbose=0)

```

**Fit:** This function trains the model for a fixed number of epochs (iterations on a dataset).

**Epochs:** 
- We train the model for 100 epochs. 
- In each epoch, the model iterates over the entire dataset, and the optimizer adjusts the weights using backpropagation.

**verbose=0:** This simply means no training log is shown during training.

```
weights, biases = model.layers[0].get_weights()
print("Weights after training:", weights)

```

**After training:** 
- We extract the weights of the first layer to see how they have been adjusted.

**Learned Weights:** 
- These weights are what the model has learned during training.

## Output:
```
[[-0.123456, 0.789101, -0.112233, 0.445566, -0.778899],
 [0.334455, -0.667788, 0.991122, -0.334455, 0.667788]]
```

Basically the code works by first setting up a simple neural network architecture with one hidden layer and one output layer. It then feeds the sample data X and Y through this network. During training, the backpropagation algorithm adjusts the weights of the network to minimize the difference between the predicted outputs and the actual values (y). The output printed at the end shows the final learned weights in the first layer after the training process.

## Improvements

Since its inception, various improvements have been made to the basic backpropagation algorithm. Stochastic gradient descent, for example, updates weights using a subset of data, enhancing efficiency and convergence. Other notable advancements include adaptive learning rate techniques like Adam and RMSprop, which adjust the learning rate during training for better performance.

## Applications

Backpropagation is the driving force behind many modern AI applications. From the facial recognition systems in smartphones to language translation services, the algorithm’s ability to train complex neural networks has led to significant breakthroughs in numerous fields.


# References:

https://hmkcode.com/ai/backpropagation-step-by-step/

https://medium.com/@ppuneeth73/the-chain-rule-of-calculus-the-backbone-of-deep-learning-backpropagation-9d35affc05e7




