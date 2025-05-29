以下是更新后的项目文档，已将 **BABD-13 数据集网址** 添加至合适位置：

---

# Blockchain Fraud Detection System

This project implements a state-of-the-art fraud detection system for blockchain transactions using a novel ensemble approach with deep feature pyramids. The system is designed to detect illegal transactions in large-scale blockchain networks with high accuracy and efficiency.

## Key Features

* 🚀 **High-Performance Detection**: Detects fraudulent transactions with high precision
* 🧠 **Ensemble Learning**: Combines 15+ base models with a deep feature pyramid
* 📈 **Feature Pyramid Architecture**: Extracts multi-granularity features (transaction, address, network levels)
* 🔄 **Incremental Learning**: Adapts to new transaction patterns without full retraining
* 🌐 **Federated Learning Support**: Enables collaborative training across multiple nodes
* ⚡ **Dynamic Routing**: Intelligently selects fast or deep analysis paths based on confidence

## Model Architecture

The system uses a hybrid architecture combining ensemble learning with deep neural networks:

```
1. Base Model Ensemble
   ├── Random Subspace Sampling
   ├── LightGBM Classifiers
   └── Dynamic Feature Weighting

2. Feature Pyramid Network
   ├── Transaction-level Attention
   ├── Address-level Processing
   └── Network-level GRU

3. Dual-Path Classifier
   ├── Fast Inference Path (for high-confidence cases)
   └── Deep Analysis Path (for ambiguous cases)
```

## Dataset

We use the [BABD-13: Blockchain Anomaly Behavior Dataset](https://www.kaggle.com/datasets/lemonx/babd13) from Kaggle, which includes labeled blockchain transactions across multiple behavior types. You can download and place the dataset in the `data/` directory for training and evaluation.

## Installation

### Prerequisites

* Python 3.8+
* PyTorch 1.10+
* CUDA 11.3 (for GPU acceleration)
* LightGBM

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/GYMN-C/Enhancing-Illicit-Transaction-Detection-on-Blockchain-via-High-Order-Semantic-Subspace-Learning.git
cd blockchain-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Place your blockchain transaction data (e.g., BABD-13) in CSV or Excel format in the `data/` directory. The dataset should include transaction features and a 'label' column indicating legitimate (0) or fraudulent (1) transactions.

### 2. Training the Model

```python
from detector import BlockchainFraudDetector
from utils import load_large_dataset

# Load dataset
data = load_large_dataset('data/transactions.xlsx')

# Initialize detector
detector = BlockchainFraudDetector(
    n_base_models=15,
    subspace_ratio=0.6,
    max_features=60,
    embed_dim=128,
    chunk_size=50000
)

# Preprocess data
X, y, feature_names = detector.preprocess_data(data)

# Train model
train_losses, val_losses = detector.train_full_model(
    X, y,
    epochs=20,
    batch_size=4096,
    lr=0.001,
    top_k=60
)
```

### 3. Making Predictions

```python
# Load test data
test_data = load_large_dataset('data/test_transactions.xlsx')
X_test, y_test, _ = detector.preprocess_data(test_data)

# Make predictions
predictions, confidences, paths = detector.predict(X_test)

# Evaluate performance
metrics = detector.evaluate_model(X_test, y_test)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test F1 Score: {metrics['f1']:.4f}")
```

### 4. Incremental Learning

```python
# Load new data
new_data = load_large_dataset('data/new_transactions.xlsx')

# Perform incremental learning
detector.incremental_learning(new_data)
```

## Results

Performance on blockchain transaction dataset (500,000+ transactions):

| Metric          | Value |
| --------------- | ----- |
| Accuracy        | 98.2% |
| Precision       | 97.8% |
| Recall          | 95.6% |
| F1 Score        | 96.7% |
| AUC             | 99.1% |
| Fast Path Usage | 83.4% |

## File Structure

```
blockchain-fraud-detection/
├── data/                   # Dataset storage
├── models/                 # Trained model files
├── src/
│   ├── detector.py         # Main fraud detector implementation
│   ├── model.py            # PyTorch model definitions
│   ├── utils.py            # Utility functions
│   └── config.py           # Configuration settings
├── requirements.txt        # Python dependencies
├── train.py                # Training script
├── predict.py              # Prediction script
└── README.md               # This document
```

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## Contact

For questions or support, please contact:

* [Yannan Guo](mailto:gyn13944041446@outlook.com)

---

如需进一步美化或国际会议投稿准备，我也可以帮你调整格式和语气。需要我继续吗？
