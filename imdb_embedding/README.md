# Model for IMDB dataset
## Word embedding network used to classify movie reviews (positive or negative)  

The dataset is composed of 50000 review, splitted between train and test. The length of each review is reshaped to 10 000 words and reformat to one hot encoding matrix.

There is also a dictionary which is used to assign a value to a word. 
</br>
In this case, it's composed of the most 10 000 commons words of every review (keras already provide the dictionary for this dataset).


### Neural network architecture
__Input Layer:__

This layer takes the input shape as parameter (in our case, it's 10000).


__Embedding Layers:__

An embedding layer is used to convert the input (n reviews of 10,000 words) into n matrices of size 256 by 16.

These matrices represent the input reviews in a latent space, which is very useful for extracting specific features.

__GlobalAvgPool1D:__

The global average pooling 1D layer converts a matrix of size (x, y) into a vector of y elements.

The output vector is the mean of each column.

This technique is used to reduce the spatial dimensions of the data, summarizing the feature map and prevent overfitting.

__Output Layers:__

A Dense layer with relu activation function follows the global average pooling to learn how much each feature is important.

A Sigmoid layer with 1 unit is used as output function.

Sigmoid activation is commonly used in the output layer of binary classifications models to produce probabilities for between two class.


__Summary:__

Overall, this model incorporates an embedding layer for feature representation, a global average pooling layer to reduce spatial dimensions and prevent overfitting, and fully connected layers for feature importance learning.

The final Sigmoid layer provides the probability of a review being positive or negative.



### Usage

__Trainning:__
```sh
python3 main.py
```
This will create the model and plot training metrics.

![metrics](https://raw.githubusercontent.com/Armandase/models/main/imdb_one_hot/assets/accuracy.png)


__Evaluation:__
```sh
python3 evaluation.py -a= 1 or 0
```

Evaluation file write, in the terminal, loss and accuracy with train and test dataset.

If you set 'a=1', the evaluation will be on a review that you will need to write 
