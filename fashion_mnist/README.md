# Model for fashion mnist dataset
## A dense neural network used to predict clothe type based between 10 differents category

The dataset looks like this:
![Dataset exemple](https://mlops.systems/posts/2022-05-11-fashion-mnist-pixel-similarity_files/figure-html/cell-33-output-1.png)

Input images are 28 per 28 pixels in grayscale.

### Neural network architecture
The neural network architecture consists just of a flatten layer followed by a dense layer (ReLu as activation).

The last layer is a softmax layer, which is used to calculate the probability for each class, ranging from 0 to 9.

### Usage

__Multiples arguments can be provided:__

_batch_size_: Number of images processed in one training step.

_learning_rate_: Rate of adjustment for model parameters during training.

_epochs_: Number of times the entire dataset is passed through the model during training.

_path_: Location to save the trained model.

```sh
python3 main.py -p='/home/models'
```

If a model already exists in the provided path, it is loaded and used for prediction.


