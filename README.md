# 🧠 MNIST Digit Classifier using PyTorch

This project builds and trains a **Neural Network** using **PyTorch** to classify handwritten digits from the MNIST dataset. The network is composed of fully connected layers with ReLU activations and is trained using the **Adam optimizer**.

## 📦 Dataset

We use the **MNIST** dataset, which consists of 28x28 grayscale images of handwritten digits (0 through 9).  
- Training samples: 60,000  
- Test samples: 10,000  
The dataset is automatically downloaded via `torchvision.datasets`.

## 🧠 Model Architecture

The model is a simple **feedforward neural network** with the following layers:

- Input layer: 784 (28x28 flattened image)
- Hidden layer 1: 64 nodes
- Hidden layer 2: 64 nodes
- Output layer: 10 nodes (one for each digit class)

```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(784, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, data):
        data = F.relu(self.input_layer(data))
        data = F.relu(self.hidden1(data))
        data = F.relu(self.hidden2(data))
        data = self.output_layer(data)
        return F.log_softmax(data, dim=1)
```

## ⚙️ Training Details

- **Loss Function**: Negative Log Likelihood (`F.nll_loss`)
- **Optimizer**: Adam (`lr = 0.009`)
- **Epochs**: 10
- **Batch size**: 10

## 📊 Results

- The model prints training loss per epoch.
- After training, the model is evaluated on the test set.
- It outputs the final **accuracy**.

## 🧪 Dependencies

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `tabulate`

See [`requirements.txt`](./requirements.txt) for full list.

## 🚀 Usage

1. Clone the repository:
```bash
git clone https://github.com/RimAmarat/MNIST_DNN.git
cd mnist-dnn
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the training script:
```bash
python train_mnist.py
```

---

## 🧾 License

This project is for educational and experimental use.

## 🙌 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
