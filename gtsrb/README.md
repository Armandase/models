# Model for GTSRB dataset
## A convolutional neural network used to classify traffic signs among 43 different types

The dataset looks like this:
![Dataset exemple](https://www.researchgate.net/publication/343758119/figure/fig1/AS:963538231234571@1606736824148/Example-images-from-the-GTSRB-dataset-Identifying-the-class-of-traffic-signs-in-the.jpg)

The shape of images could vary, so they are reshaped into a square, either 24x24 or 48x48.

There is also a comparison between multiple pixel transformations, such as grayscale, RGB + histogram equalization, and other types of pixel transformations.


### Neural network architecture
__Input Layer:__
This layer specifies the input shape of the image data (24 or 48).

__Convolutional Layers:__
Two convolutional layers are utilized.

The first layer consists of 96 filters of size 3x3, followed by a ReLU activation. Max pooling with a 2x2 window and a dropout layer with a rate of 0.25 are applied after the first convolutional layer. 

The second convolutional layer consists of 192 filters of size 3x3, followed by ReLU activation. Again, max pooling with a 2x2 window and a dropout layer with a rate of 0.25 are applied after this layer.

__Flatten Layer:__
This layer flattens the output from the convolutional layers into a one-dimensional tensor, preparing it for input into the fully connected layers.

__Dense Layers:__
Two dense (fully connected) layers follow the flatten layer.

The first dense layer has 1500 units with ReLU activation and a dropout rate of 0.5.

The second dense layer has 43 units, corresponding to the number of classes in the classification task, with softmax activation.

Softmax activation is commonly used in the output layer of a classification model to produce probabilities for each class, allowing for easy interpretation of the model's confidence in its predictions.

__Summary__:
Overall, this model incorporates convolutional layers for feature extraction, dropout layers for regularization to prevent overfitting, and dense layers for classification.
The output is a probability for each class produced by the softmax function.

### Usage
_Take a look at the constants file to set up your own environment_

__First step__
```sh
python3 preprocessing.py
```
This will create few improved datasets
![new datasets](https://raw.githubusercontent.com/Armandase/models/main/gtsrb/tools/differents_dataset.png)

__Second step__
```sh
python3 create_model.py --no-multiple
```
This script creates one (default dataset) or few models (every improved datasets).
It saves the best generated model as "best_model.tf". Additionally, you can utilize TensorBoard in the "models/logs" directory to access all training metrics.

__Last step__
```sh
python3 main.py
```
The main function will print the metrics for every dataset in the terminal.
Also, a matrix of confusion is plotted using the best model.


