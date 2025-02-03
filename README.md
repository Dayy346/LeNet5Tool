# LeNet-5 Based CNN Model Generator

## Overview
This project is a comprehensive implementation and extension of the LeNet-5 Convolutional Neural Network, originally designed for handwritten digit recognition on the MNIST dataset. This project evolved from a machine learning assignment into a versatile tool capable of adapting to various datasets and classification tasks, Showcasing the versatility of deep learning in real world applications in a full stack web application. The objective was to build and enhance the LeNet 5 model, initially trained on the MNIST dataset, and then expand its capabilities to process RGB data, incorporate advanced regularization techniques, and provide a user friendly interface for training and testing custom datasets. Then producing a downloadable model for deployment.

---

## Features

1. **Dynamic Model Adaptation**
   - Initially trained on grayscale images using LeNet5, the model was modified to handle RGB datasets effectively.
   - Incorporates a toggle to switch between grayscale and RGB inputs seamlessly, allowing adaptability for diverse datasets.

2. **Regularization and Overfitting Prevention**
   - To combat overfitting, the following enhancements were implemented:
     - Batch Normalization.
     - Data Augmentation (random rotations, flips, and color adjustments).
     - Dropout layers for regularization.
     - Optimized the learning rate and added L2 regularization for further stability.

3. **End-to-End Machine Learning Workflow**
   - **Backend**:
     - Flask-based server for handling data uploads, model training, and predictions.
     - Dynamic handling of number of epochs, number of classes, and input format (grayscale/RGB).
   - **Frontend**:
     - React-based interface for dataset upload, model training, and single-image testing.
     - Displays training progress through a dynamically simulated loading bar.

4. **Performance Metrics**
   - Final train and test accuracies are displayed post-training.
   - Test accuracy consistently averages over 70% across multiple complex datasets.

5. **Single Image Testing**
   - Upload a single image (JPEG or JPG), and the model predicts the class with the corresponding folder name for intuitive results.

6. **Downloadable Trained Models**
   - The trained model can be downloaded for external use or future reference.

---

## Implementation Details

### 1. Original Assignment
The project began as an academic assignment to implement LeNet-5 from scratch, using only the original research paper as a reference. Libraries like PyTorch and NumPy were provided, and the primary goal was to achieve digit classification on the MNIST dataset.

### 2. Model Enhancements
After achieving initial success with MNIST, the model was significantly improved:
- **Transition to RGB**: Adapted the CNN to handle RGB input by modifying convolutional layers to accommodate three input channels.
- **Regularization Techniques**: Implemented batch normalization, data augmentation, and dropout to counter overfitting, ensuring robust performance on challenging datasets.
- **Optimization**: Switched to the Adam optimizer and tuned hyperparameters for faster convergence.

### 3. Backend Development
- Built a Flask server to manage dataset uploads, model training, and predictions.
- Supported dynamic configurations such as epochs, grayscale/RGB selection, and dataset-specific classes.
- Added endpoints for downloading trained models and testing single images.

### 4. Frontend Development
- Created a React-based interface for seamless user interaction.
- Integrated a progress bar for training visualization, calculated by timing a single epoch and estimating the total training duration.
- Allowed dynamic testing of single images after model training, ensuring the usability of the tool.

### 5. Challenges and Solutions
- **Overfitting**: Resolved through regularization and data augmentation techniques.
- **Loading Bar**: Innovatively calculated progress using the duration of a single epoch, ensuring a smooth user experience even with unpredictable training times.

---


2. **Install Dependencies**
- **Backend**:
  ```
  pip install -r requirements.txt
  ```
- **Frontend**:
  ```
  npm install
  ```

3. **Run the Application**
- Start the Flask server:
  ```
  python app.py
  ```
- Start the React frontend:
  ```
  npm start
  ```

4. **Train a Model**
- Upload a dataset (ZIP file containing `train` and `test` folders with images classified by folders(classes)).
- Configure settings like the number of epochs and input format (grayscale/RGB).
- Click "Upload and Train" to start training.

5. **Test a Single Image**
- Upload a JPEG or JPG image.
- View the predicted class name (derived from the folder structure of the dataset).

6. **Download Trained Model**
- Once training is complete, download the model using the provided button.

---

## Results

The final model demonstrated exceptional flexibility and adaptability:
- Achieved consistent test accuracy exceeding 70% across various datasets.
- Successfully integrated RGB input without compromising model performance.
- Provided real-time feedback and usability through the frontend interface.

---

## Key Learning Outcomes

- **Advanced CNN Architectures**: Gained in-depth knowledge of convolutional layers, batch normalization, and dropout techniques.
- **Backend Development**: Strengthened Flask skills by building robust APIs for model training and data handling.
- **Frontend Integration**: Improved React skills to create a seamless user experience, bridging machine learning and application development.
- **Problem-Solving**: Tackled unique challenges like loading bar implementation and model overfitting with innovative solutions.

---

## Future Enhancements

- Extend support for additional image formats like PNG.
- Implement a feature to visualize intermediate feature maps of the CNN.
- Add options for hyperparameter tuning directly from the frontend.
- Dockerize the application for easy deployment.

---

## Credits

- **Model Architecture**: Based on Yann LeCun's LeNet-5, enhanced for modern use cases.
- **Libraries Used**: PyTorch, Flask, React, NumPy, Matplotlib.

Feel free to contribute or reach out for collaboration opportunities!
