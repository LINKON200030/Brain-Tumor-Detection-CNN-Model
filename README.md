Brain Tumor Detection using CNN (TensorFlow/Keras)

A deep learning project for detecting brain tumors from MRI images using a Convolutional Neural Network (CNN).
This model classifies MRI scans into two categories:

Yes → Tumor present

No → No tumor present

The model is trained, validated, and tested using a structured dataset of MRI scans and implemented entirely in Python, TensorFlow/Keras, and Google Colab with GPU acceleration.

📂 Project Structure
Brain-Tumor-Detection-CNN-Model/
│
├── Brain Tumor Detection Model.ipynb      # Main training & testing notebook
├── Brain_Tumor_Detection/                 # Dataset (yes/no/pred folders)
│   ├── yes/                               # MRI scans with tumor
│   ├── no/                                # MRI scans without tumor
│   └── pred/                              # Extra images for prediction demo
│
└── README.md                              # Project documentation

📑 Dataset Information

The dataset contains MRI brain scan images divided into labeled folders:

Folder	Description
yes	Images showing brain tumors
no	Images without tumors
pred	Additional images used for model prediction demos

The dataset was split into:

70% Training data

15% Validation data

15% Test data

This ensures unbiased evaluation and prevents overfitting.

🧪 Model Architecture (CNN)

A custom Convolutional Neural Network was designed with the following structure:

Conv2D + ReLU (32 filters)

MaxPooling2D

Conv2D + ReLU (64 filters)

MaxPooling2D

Conv2D + ReLU (128 filters)

MaxPooling2D

Flatten

Dense (256 neurons)

Dropout (0.5)

Output Layer (Softmax activation)

This architecture balances simplicity and efficiency while achieving reliable classification performance.

⚙️ Training Details

Framework: TensorFlow / Keras

Optimizer: Adam

Loss Function: Categorical Crossentropy

Batch Size: 32

Epochs: 15

Input Size: 150 × 150 × 3

The model was trained in Google Colab with GPU support for faster computation.

📈 Evaluation Metrics

The model is evaluated on:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

These metrics assess how well the model detects tumors and distinguishes between tumor/no-tumor classes.

🖼️ Predicting on New MRI Images

The notebook includes a function that allows you to load and classify any new MRI image:

predict_single_image("/path/to/image.jpg")


This function:

Loads the image

Resizes it to 150×150

Normalizes pixel values

Predicts tumor vs. no tumor

Displays the probability distribution

▶️ How to Run This Project
1. Open the Notebook in Google Colab

Upload the notebook:

Brain Tumor Detection Model.ipynb

2. Mount Google Drive

The notebook includes:

from google.colab import drive
drive.mount('/content/drive')

3. Extract and Load Dataset

Dataset folders must be placed in Colab or Google Drive (as shown in the notebook).

4. Train the Model

Run all cells to train the model and view training/validation curves.

5. Test the Model

Evaluate accuracy, precision, recall, and F1-score.

6. Predict New Images

Use images from the pred/ folder or upload your own MRI scans.

💾 Saving the Model

The model is saved as:

brain_tumor_cnn_model.h5


This file can be reused for deployment, further training, or real-time predictions.

🎯 Project Goals

Understand and implement Convolutional Neural Networks

Work with real medical imaging data

Build a complete deep-learning pipeline

Evaluate model performance using industry-standard metrics

Demonstrate prediction on unseen MRI scans

👤 Author

Shahadat Hossain (LINKON200030)
Computer Science Student
Ravensbourne University London

📬 Contact

If you need any clarification or improvements in the model, feel free to open an Issue or contact me directly.
