# Task 1: Image classification + OOP
In this task, you need to use a publicly available simple MNIST dataset and build 3 classification
models around it. It should be the following models:
1) Random Forest;
2) Feed-Forward Neural Network;
3) Convolutional Neural Network;

Each model should be a separate class that implements MnistClassifierInterface with 2
abstract methods - train and predict. Finally, each of your three models should be hidden under
another MnistClassifier class. MnistClassifer takes an algorithm as an input parameter.
Possible values for the algorithm are: cnn, rf, and nn for the three models described above.

The solution should contain:
- Interface for models called MnistClassifierInterface.
- 3 classes (1 for each model) that implement MnistClassifierInterface.
- MnistClassifier, which takes as an input parameter the name of the algorithm and
provides predictions with exactly the same structure (inputs and outputs) not depending on the selected algorithm.

## Solution details
Overall, the task is simple and the results are known beforehand: CNN is the best, Random Forest is the worst.  
I didn't put much effort into accelerating this task on the GPU because it's a huge overkill.
The solution requires Python 3.10+ (and I'm slightly sorry for that!).  

### Key Points:
- The methods of the `MnistClassifier` class were not specified in the assignment, so instead of just the `predict` method, I added `train`, which can be useful as it allows easy dataset switching.  
- The training loop is implemented as a callback within the `MnistClassifier` class. This helps avoid code duplication and makes it easy to add new models.  
- Simple inference call thanks to `np.ndarray` support: you don't need to cast a loaded image to `torch.Tensor` or even use a PyTorch dataloader.  

## Setup
You can use any virtual environment you prefer.  

First, install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```
Then, simply run all in [demo.ipynb](demo.ipynb).