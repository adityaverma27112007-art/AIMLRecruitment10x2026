# AIML-Recruitment-2026-Aditya

## Candidate Details

- **Name:** Aditya
- **Role:** Student
- **Submission:** AI/ML Recruitment 2026
- **Repository:** `AIMLRecruitment10x2026`

> Add your college, branch, year/semester, and any other required candidate details before submission.



# Task 2 — MNIST Handwritten Digit Classification

## Problem Statement

The objective is to build and train a simple neural network that classifies handwritten grayscale images into one of ten digit classes, from **0 to 9**, using the MNIST dataset.

The project also studies how changing one part of the neural network affects its performance.

## Dataset Understanding

MNIST is a standard benchmark dataset for handwritten digit recognition.

- **Number of classes:** 10
- **Classes:** digits `0, 1, 2, ..., 9`
- **Image size:** `28 × 28` pixels
- **Channels:** 1 grayscale channel
- **Training images:** 60,000
- **Test images:** 10,000
- **Pixel range before normalization:** 0–255
- **Labels:** the digit represented by each image

## Approach

The implementation uses PyTorch.

### Preprocessing

1. Load MNIST using `torchvision.datasets.MNIST`.
2. Convert images to tensors.
3. Normalize using the standard MNIST mean `0.1307` and standard deviation `0.3081`.
4. Split the original training set into training and validation subsets.
5. Flatten each `28 × 28` image into `784` values for the fully connected network.

### Baseline Architecture

```text
28 × 28 image
     ↓
Flatten: 784
     ↓
Linear: 784 → 128
     ↓
ReLU
     ↓
Linear: 128 → 10 logits
     ↓
Softmax probabilities
```

During training, the model uses raw logits with `CrossEntropyLoss`. Softmax is used when probabilities are required for interpretation.

## Activation Functions

### ReLU

`ReLU(x) = max(0, x)`.

It introduces non-linearity into the hidden layer. Without a non-linear activation, a stack of linear layers would remain equivalent to one linear transformation.

### Softmax

Softmax converts the ten output scores into a probability distribution whose values sum to 1. It is suitable for MNIST because each image belongs to exactly one of ten classes.

## Training Configuration

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 8 |
| Loss | Cross-Entropy Loss |
| Baseline hidden neurons | 128 |
| Modified hidden neurons | 256 |

The script records training/validation loss and accuracy and evaluates the final models on the test set.

## Evaluation

The project reports:

- Accuracy
- Confusion matrix
- Weighted precision
- Weighted recall
- Weighted F1-score

The confusion matrix shows actual classes against predicted classes. Diagonal entries are correct predictions; off-diagonal entries show which digits were confused.

## Experiment

The baseline hidden layer contains **128 neurons**. The modified model contains **256 neurons**.

Only this architectural variable is changed; preprocessing, data split, optimizer, learning rate, batch size, and number of epochs remain the same.

Increasing hidden neurons increases model capacity and can allow the network to represent more complex patterns. The actual effect should be judged from the measured validation/test metrics rather than assumed in advance.

## Results

Run the training script to generate the actual metrics and plots. Numerical results are intentionally not fabricated.

Generated files include:

- `results/metrics.json`
- `results/comparison.csv`
- `results/baseline_training_loss.png`
- `results/baseline_training_accuracy.png`
- `results/modified_training_loss.png`
- `results/modified_training_accuracy.png`
- `results/baseline_confusion_matrix.png`
- `results/modified_confusion_matrix.png`
- `results/model_comparison.png`
- classification reports for both models

## Key Learnings

1. Normalization improves the numerical scale of image inputs for neural-network training.
2. ReLU introduces non-linearity and enables hidden layers to learn non-linear patterns.
3. Softmax is appropriate for single-label, multi-class classification.
4. Increasing hidden-layer width increases representational capacity and parameter count.
5. Confusion matrices reveal class-specific errors that overall accuracy does not show.

## Challenges and Solutions

### Challenge 1 — Image representation

A fully connected layer expects a vector rather than a 2D image.

**Solution:** flatten each `28 × 28` image into a `784`-element vector.

### Challenge 2 — Fair model comparison

Changing several hyperparameters at once would make the experiment difficult to interpret.

**Solution:** change only the hidden-layer width from 128 to 256 neurons.

### Challenge 3 — Reproducible metrics

Results should come from an actual training run rather than hand-written numbers.

**Solution:** the script automatically trains, evaluates, and saves all metrics and plots.

## Technologies Used

- Python 3
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab
- GitHub

## Project Structure

```text
AIMLRecruitment10x2026/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── train_mnist.py
├── notebooks/
│   └── MNIST_Neural_Network.ipynb
└── results/
    └── README.md
```

## How to Run

### Google Colab

Open `notebooks/MNIST_Neural_Network.ipynb` in Google Colab and run the cells.

### Local

```bash
pip install -r requirements.txt
python src/train_mnist.py
```

The first run downloads MNIST automatically.

