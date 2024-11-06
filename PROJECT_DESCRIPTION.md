
# Neural Network for Image Classification using MNIST Dataset

## Objective
This project aims to build a neural network that can classify handwritten digit images from the MNIST dataset, providing foundational experience with deep learning concepts and demonstrating a basic image classification pipeline.

## Technologies Used
- **Python**: For data preprocessing and model implementation.
- **Keras (TensorFlow)**: For building and training the neural network model.

## Project Structure
- `neural_network.py`: Contains the neural network code to load, preprocess, and classify images from the uploaded MNIST `.idx` files.
- `requirements.txt`: Lists all the required libraries for the project.
- `data/`: Contains the uploaded MNIST `.idx` files.

## Installation and Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/Image-Classification-Project.git
    cd Image-Classification-Project
    ```

2. **Install Required Packages**:
    Install the necessary Python packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Model**:
    Execute `neural_network.py` to train the model on the MNIST dataset.
    ```bash
    python neural_network.py
    ```

## Usage

### Training the Model
The `neural_network.py` script loads the MNIST dataset from `.idx` files, preprocesses it, and trains a neural network model. After training, it evaluates the model on the test set, printing out the test accuracy.

## Final Outcomes
After training, the model achieves over 90% accuracy on the MNIST test set, demonstrating effective image classification capabilities for handwritten digits.

## License
This project is licensed under the MIT License.
