# Model for IMDB dataset
## A dense network used to classify movie reviews (positive or negative)  

The dataset is composed of 50000 review, splitted between train and test. The lenght of each review is reshaped to 10 000 words and reformat to one hot encoding matrix.

There is also a dictionnary which is used to assign a value to a word. 
</br>
In this case, it's composed of the most 10 000 commons words of every review (keras already provide the dictionnary for this dataset).


### Neural network architecture
__Input Layer:__

This layer takes the input shape as parameter (in our case, it's 10000).


__Dense Layers:__

Two dense (fully connected) layers follow the input layer.

The two denses layers have 32 units with ReLU activation.

Between them, a dropout layers is used to try to reduce overfitting with a rate of 0.5%.


__Output Layer:__

A Sigmoid layer with 1 unit is used as ouput function.

Sigmoid activation is commonly used in the output layer of binary classifications models to produce probabilities for between two class.


__Summary:__

Overall, this model incorporates fully connected layers for feature extraction, dropout layers for regularization to prevent overfitting, and a Sigmoid layers for classification.

The output is a probability between positive and negative review.



### Usage

__Trainning:__
```sh
python3 main.py
```
This will create the model and plot trainning metrics. 


__Evaluation:__
```sh
python3 evaluation.py
```

Evaluation file write, in the terminal, loss and accurancy with train and test dataset.
