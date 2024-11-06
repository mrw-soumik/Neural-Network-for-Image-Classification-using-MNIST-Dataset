# Neural Network for Image Classification using MNIST Dataset

## Objective
This project builds a neural network to classify handwritten digit images from the MNIST dataset, providing foundational experience with deep learning concepts and image classification.

## Technologies Used
- **Python**: For data preprocessing and model implementation.
- **Keras (TensorFlow)**: For building and training the neural network model.

## Project Structure
- `neural_network.py`: Contains the code to load, preprocess, and classify images from the uploaded MNIST `.idx` files.
- `requirements.txt`: Lists the required libraries for the project.
- `data/`: Contains the MNIST `.idx` files used for training and testing, along with a sample image for testing.

## Installation and Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/Image-Classification-Project.git
    cd Image-Classification-Project
    ```

2. **Place Dataset in the `data/` Directory**:
   Extract the `MNIST Dataset.zip` file into the `data/` directory so it includes the following files:
   - `train-images.idx3-ubyte`
   - `train-labels.idx1-ubyte`
   - `t10k-images.idx3-ubyte`
   - `t10k-labels.idx1-ubyte`

3. **Install Required Packages**:
    Use `requirements.txt` to install the necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Model**:
    Execute the neural network script to train and evaluate the model on the MNIST dataset:
    ```bash
    python neural_network.py
    ```

## Usage

### Training the Model
The `neural_network.py` script loads the MNIST dataset from `.idx` files, preprocesses it, and trains a neural network model. After training, it evaluates the model on the test set, outputting the test accuracy.

### Testing on the Sample Image
The script also includes a function to classify a sample image located in `data/sample_image.png`. This image will be displayed, and the model's prediction for the digit will be printed.

## Final Outcomes
After training, the model achieves over 90% accuracy on the MNIST test set, demonstrating effective image classification capabilities for handwritten digits.

## License
This project is licensed under the MIT License.
