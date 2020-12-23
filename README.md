# DetectNet
We develop a deep learning model for modeling the transfer function of galaxy detection in wide-field surveys.

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
* `model/detect_net.py` and `model/detect_net_withz.py` - define the model architecture, the loss function, and the evaluation metrics (without and with a random standard normal variable input, respectively).
* `model/dataloader.py` - specify how the data should be fed into the model.
* `tests/detectz1/params.json` - an example directory with a JSON file containing the model and training hyperparameters. Similar directories can be created there containing different hyperparameters. These directories also store the logging info and plots corresponding to the given set of hyperparameters.
* `train_and_evaluate.py` and `train_and_evaluate_withz.py` - train the model on the training dataset, evaluating it along the way on the validation dataset (without and with a random standard normal variable input, respectively).
* `test.py` and `test_withz.py` - run the trained model on the testing dataset and calculate the accuracy of the result by plotting a ROC curve (without and with a random standard normal variable input, respectively).
* `utils.py` - contains functions for handling hyperparameters, logging information, and storing model states
