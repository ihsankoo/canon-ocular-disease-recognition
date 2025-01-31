## Fundus Disease Detection App

### Overview
The **Fundus Disease Detection App** is a machine learning-powered tool for analyzing fundus (eye) images to detect potential ocular diseases. Users can upload left and right fundus images, and the app will predict diseases, if any, and display the results interactively.

### How It Works
1. Upload your left and right fundus images in `.jpg`, `.png`, or `.jpeg` format.
2. The app processes the images and predicts the presence of diseases.
3. Results are displayed:
   - Diseases detected are shown in bold **red** text with an exclamation mark.
   - If no diseases are detected, a green "No Disease Detected" message is shown.

### Setup and Installation

#### Prerequisites
- Python 3.8 or higher
- GPU support (optional but recommended for training)

#### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ihsankoo/canon-ocular-disease-recognition.git
   cd canon-ocular-disease-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

### File Structure
- `notebook.ipynb`: Script for training the model on fundus image data.
- `app.py`: Streamlit app for uploading images and viewing predictions.
- `fundus_disease_model.h5`: Trained deep learning model.
- `README.md`: Documentation for the project.

### Model Training
To train the model from scratch, use the `notebook.ipynb` script. Ensure your dataset is structured correctly and update the `image_dir` path.

### Model Deployment
The trained model is currently deployed on AWS, making it accessible for testing. You can visit the demo at:
http://44.200.73.145:8501/
This live deployment allows users to upload their fundus images and receive disease predictions instantly.

### Technologies
- **TensorFlow/Keras**: For training and deploying the CNN model.
- **Streamlit**: For building the interactive UI.
- **Pillow**: For image preprocessing.
- **Numpy**: For numerical operations.

### Contributing
Contributions are welcome! Please fork the repository and create a pull request for any changes.

### License
This project is licensed under the MIT License. See the LICENSE file for details.
