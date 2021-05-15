# DetectNet
We developed a deep learning model for modeling the transfer function of galaxy detection in wide-field surveys.

Each location in the sky is surveyed several times by telescopes producing a stack of images of the same location. Different statistics are developed to characterize these observations as a function of position in the sky. We have sky maps of these statistics, including the fluctuation of sky brightness and the blurriness of point-like sources at each point for different passbands (filters). The passbands we consider are g, r, i, and z, from bluer to redder. For each passband and location in the sky, we use two of these observing conditions from the sky maps and the true galaxy magnitude, a measure of brightness, from deep-field surveys as the input to our network. Our model predicts the probability that a given galaxy would be detected by the Dark Energy Survey (DES).

## Dataset
* Our training dataset consists of ∼20 million galaxies.
* We use 10,000 galaxies for validation.
* We use 10,000 galaxies for testing.

## Setup
```bash
git clone https://github.com/georgehalal/DetectNet.git
cd DetectNet/
pip install -r requirements.txt
```

## Code Layout
* `preprocess.py` - prepare the raw data into a format useful for the model.
* `model/detect_net.py` and `model/detect_net_withz.py` - define the model architecture (without and with a random standard normal variable input, respectively), the loss function, and the evaluation metrics.
* `model/dataloader.py` - specify how the data should be fed into the model.
* `tests/detectz1/params.json` - an example directory with a JSON file containing the model and training hyperparameters. Similar directories can be created there containing different hyperparameters. These directories also store the logging info and plots corresponding to the given set of hyperparameters.
* `train_and_evaluate.py` and `train_and_evaluate_withz.py` - train the model on the training dataset, evaluating it along the way on the validation dataset (without and with a random standard normal variable input, respectively).
* `test.py` and `test_withz.py` - run the trained model on the testing dataset, calculate the accuracy of the result, and plot a ROC curve (for the model without and with a random standard normal variable input, respectively).
* `utils.py` - contains functions for handling hyperparameters, logging information, and storing model states.

## Performance
![roc](https://github.com/georgehalal/DetectNet/blob/main/img/roc_curve.png)
