 Marine Eco-AI: Step-by-Step AI for Monitoring Oil Spills using Satellite Imagery
 Tool: Python, TensorFlow/Keras, OpenCV, Streamlit, GitHub
 Application: Detecting oil spills in ocean satellite images

 =======================================
 Step 1: Setup the Project Structure
 =======================================
 Create a GitHub repository called "marine-eco-ai"
 Folder structure:
 marine-eco-ai/
 ├── data/                 For storing images
 ├── models/               Saved model files
 ├── notebooks/            Jupyter notebooks
 ├── app/                  Web app interface
 ├── utils/                Helper functions
 ├── README.md             Project description
 └── requirements.txt      List of libraries to install

 =======================================
 Step 2: Install Required Libraries
 =======================================
 Run this command in your terminal or save in requirements.txt:
 pip install tensorflow opencv-python numpy pandas matplotlib seaborn streamlit scikit-learn

 =======================================
 Step 3: Data Collection and Preparation
 =======================================
 Use Sentinel-2 or Landsat satellite images of ocean areas with oil spills.
 Save sample images into the data/ directory under two folders: `oil_spill` and `clean_ocean`

 =======================================
 Step 4: Image Preprocessing
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
def load_images(data_path):
    images = []
    labels = []
    for label, category in enumerate(["clean_ocean", "oil_spill"]):
        folder_path = os.path.join(data_path, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

images, labels = load_images("data")
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

 Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

 =======================================
 Step 5: Create the AI Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

 =======================================
 Step 6: Train the Model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save("models/oil_spill_detector.h5")

 =======================================
 Step 7: Create Streamlit Web App (app/app.py)
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

st.title("Marine Eco-AI: Oil Spill Detector")
model = load_model("../models/oil_spill_detector.h5")

uploaded_file = st.file_uploader("Upload an ocean satellite image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        st.error("⚠️ Oil Spill Detected")
    else:
        st.success("✅ Clean Ocean Surface")

 =======================================
 Step 8: Run the App
 In terminal, navigate to the app folder and run:
 streamlit run app.py

 =======================================
 Step 9: Upload to GitHub
 1. Initialize git: `git init`
 2. Add files: `git add .`
 3. Commit: `git commit -m "Initial commit"`
 4. Link repo: `git remote add origin <repo_url>`
 5. Push: `git push -u origin main`

 =======================================
 Step 10: Share Your Project
 - Add a README.md describing the project, its goals, usage, and results
 - Share your GitHub link with others (classmates, instructors, organizations)
 - Add screenshots and sample predictions in your README or a demo video
import streamlit as st
from PIL import Image
import numpy as np

st.title("🌊 Marine Eco-AI: Oil Spill Detector")

 Dummy Prediction Placeholder (since model isn't uploaded yet)
def predict(image):
    return "⚠️ Oil Spill Detected" if np.random.rand() > 0.5 else "✅ Clean Ocean Surface"

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    result = predict(image)
    st.subheader("Prediction:")
    st.write(result)
