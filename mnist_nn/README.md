# MNIST MODEL
## A convolutional neural network used to predict digits based on handwritten digit images

The handwritten dataset looks like this:
![Dataset exemple](https://upload.wikimedia.org/wikipedia/commons/f/f7/MnistExamplesModified.png)

Input images are 28 per 28 pixels in grayscale.

### Neural network architecture
The convolutional neural network architecture consists of two convolutional layers, each followed by a max pooling 2D layer and a dropout layer.

After these two blocks, a flatten layer is applied, followed by two dense layers with a ReLU activation function.

The final layer is a softmax layer, which is used to calculate the probability for each class, ranging from 0 to 9.

### Usage
```sh
python3 main.py --verbose=1
```

