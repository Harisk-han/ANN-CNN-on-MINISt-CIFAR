# Deep Learning Image Classification Project

A comprehensive deep learning project implementing various neural network architectures for image classification tasks on popular datasets including MNIST and CIFAR-10.

## 📋 Project Overview

This project contains implementations of both Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN) for image classification. The codebase includes training scripts, model architectures, and pre-trained models for MNIST and CIFAR-10 datasets.

## 🗂️ Project Structure

```
├── README.md                    # Project documentation
├── cifar_10.ipynb              # CIFAR-10 CNN implementation
├── min.ipynb                   # MNIST ANN implementation
├── man.ipynb                   # Additional model experiments
├── mnist_ANN_model.h5          # Pre-trained ANN model for MNIST
├── mnist_cnn_model.h5          # Pre-trained CNN model for MNIST
└── data.py                     # Data preprocessing utilities
```

## 🎯 Datasets

### MNIST Dataset
- **Description**: Handwritten digit recognition (0-9)
- **Image Size**: 28x28 grayscale images
- **Classes**: 10 (digits 0-9)
- **Samples**: 60,000 training + 10,000 test images

### CIFAR-10 Dataset
- **Description**: Object recognition across 10 categories
- **Image Size**: 32x32x3 color images
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Samples**: 50,000 training + 10,000 test images

## 🏗️ Model Architectures

### CNN for CIFAR-10
- **Architecture**: Deep Convolutional Neural Network
- **Layers**:
  - 5 Convolutional layers (32-512 filters)
  - MaxPooling layers for downsampling
  - Flatten layer
  - Dense output layer with softmax activation
- **Parameters**: ~2.5M trainable parameters
- **Performance**: Achieves ~75% accuracy on CIFAR-10 test set

### ANN for MNIST
- **Architecture**: Multi-layer Perceptron
- **Layers**:
  - Input layer: 784 neurons (28×28 pixels)
  - Hidden layers: 256 → 128 → 64 → 32 neurons
  - Output layer: 10 neurons (softmax activation)
- **Parameters**: ~270K trainable parameters
- **Performance**: Achieves ~98% accuracy on MNIST test set

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install dependencies
pip install tensorflow matplotlib numpy
```

### Running the Models

#### For MNIST ANN:
```bash
jupyter notebook min.ipynb
```

#### For CIFAR-10 CNN:
```bash
jupyter notebook cifar_10.ipynb
```

## 📊 Results

### MNIST Performance
| Model | Architecture | Test Accuracy | Training Time |
|-------|--------------|---------------|---------------|
| ANN   | 5-layer MLP  | ~98%          | ~2 minutes    |
| CNN   | ConvNet      | ~99%          | ~5 minutes    |

### CIFAR-10 Performance
| Model | Architecture | Test Accuracy | Training Time |
|-------|--------------|---------------|---------------|
| CNN   | Deep CNN     | ~75%          | ~15 minutes   |

## 🔧 Usage

### Loading Pre-trained Models
```python
from tensorflow.keras.models import load_model

# Load MNIST models
ann_model = load_model('mnist_ANN_model.h5')
cnn_model = load_model('mnist_cnn_model.h5')

# Make predictions
import numpy as np
# For MNIST
predictions = ann_model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
```

### Custom Training
Each notebook contains complete training pipelines with:
- Data preprocessing and normalization
- Model architecture definition
- Training with validation monitoring
- Performance visualization
- Model saving functionality

## 📈 Visualization

The notebooks include comprehensive visualization of:
- Training/validation accuracy curves
- Loss progression over epochs
- Sample predictions with confidence scores
- Confusion matrices for detailed performance analysis

## 🛠️ Technical Details

### Data Preprocessing
- **Normalization**: Pixel values scaled to [0,1] range
- **Flattening**: 2D images converted to 1D vectors for ANN
- **One-hot encoding**: Labels converted to categorical format

### Training Configuration
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 64 (CIFAR-10), varies for MNIST
- **Epochs**: 5 (configurable)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- MNIST and CIFAR-10 dataset creators
- The open-source community for continuous inspiration and support

## 📞 Contact

For questions or suggestions, please open an issue or reach out through the repository discussions.
