# A3: Eyes on the Road

CIFAR-10 image classification (32×32 colour images, 10 classes) for a low-cost roadside camera:
a neural network written from scratch in NumPy versus a convolutional network in PyTorch.
The first 45,000 training images are used for training, the last 5,000 for validation, and the
10,000-image test set is evaluated once.

## Assignment overview

The assignment moves from hand-derived backpropagation to a NumPy network and finally to a CNN,
measuring how much the convolutional structure gains.

**Part I: Neurons by hand**
- Two-layer network architecture with bias units and its parameter count for CIFAR-10 (308,310)
- Forward pass, backpropagation and one gradient-descent step on a 2–2–1 toy network; why zero initialization fails
- Squared, cross-entropy and multiclass hinge losses, and the link to the SVM hinge loss
- MAP estimation as L2-regularized least squares (λ = σ²/τ²); why stacked linear layers add no expressiveness

**Part II: A neural network from scratch (NumPy only)**
- Vectorized affine, ReLU, sigmoid, softmax and hinge layers in `nn_numpy.py`, verified with numerical gradients
- Overfit 50 images, reproduce the TA baseline, tune hidden size, learning rate and L2; compare softmax and hinge loss
- Five-hidden-layer design study: zero, small, large and He initialization; sigmoid vs. ReLU; L2 strength

**Part III: Convolutional networks (PyTorch)**
- Output size and parameter count of a convolutional layer vs. a fully connected layer; why weight sharing helps
- A VGG-style CNN with batch normalization, data augmentation and a one-cycle schedule
- Sensitivity study of the dropout probability, then a single final evaluation on the test set

## Contents

| File | Description |
|---|---|
| `A3_315511079.pdf` | Report (Part I derivations, results, analysis) |
| `A3_315511079.ipynb` | Part II and Part III: data, gradient check, training, design study, CNN, final evaluation and Kaggle submissions |
| `nn_numpy.py` | NumPy-only layers, losses, numerical gradient, MLP and mini-batch SGD |
| `kaggle_submission_part2.csv`, `kaggle_submission_part3.csv` | Validation-set predictions submitted to the Kaggle practice leaderboard (5,000 rows each) |
| `ML-part2-figure/`, `ML-part3-figure/` | Exported figures and result tables |

## Key results

- Gradient check: largest relative error 1.1 × 10⁻⁷ over every layer and both losses (requirement < 10⁻⁶)
- NumPy MLP (500 hidden units, lr 0.05, L2 10⁻³): validation **55.46%** vs. TA baseline 21.66%; test 55.10%
- He initialization keeps gradients constant across five ReLU layers; small and large initializations vanish or explode,
  and sigmoid gradients shrink by three orders of magnitude towards the input
- CNN (1.15M parameters): validation **91.92%** vs. TA baseline CNN 72.42%; test **91.68%** (+36.6 points over the MLP)
- Kaggle practice leaderboard: 0.5546 (Part II) and 0.9192 (Part III)

## Run

Open the notebook in Google Colab
([link](https://colab.research.google.com/drive/19xXAplgJy12qeEqRIxEpTqfZfyfXiYiI?usp=sharing)),
select a GPU runtime, upload `sample_submission.csv` and `validation_ids.csv` from the course Kaggle
competition, and run all cells. CIFAR-10 is downloaded with torchvision. Random seeds are fixed to 0.
