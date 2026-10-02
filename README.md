# 🛡️ MaskDetect AI

### AI-Powered Face Mask Detection Web Application

MaskDetect AI is a web-based face mask detection system built using **Python, Flask, and Machine Learning**. The application allows users to detect whether a person is wearing a face mask through uploaded images and live camera detection.

The project combines an AI-based detection model with a modern web interface for an easy and user-friendly experience.

---

## ✨ Features

- 🛡️ **Face Mask Detection**
  - Detect whether a person is wearing a mask or not.

- 📷 **Image Detection**
  - Upload an image and perform face mask detection.

- 🎥 **Live Detection**
  - Perform real-time face mask detection using a camera.

- 📊 **Detection Results**
  - View the prediction results after detection.

- 🕒 **Detection History**
  - Keep track of previous detection results.

- 👤 **User Profile**
  - User profile section with detection information.

- 🔐 **User Authentication**
  - Login and signup functionality.
  - Forgot password and password reset pages.

- 🎨 **Modern User Interface**
  - Responsive and visually designed web interface.
  - Dedicated pages for About, Features, and How It Works.

- 🤖 **AI-Based Prediction**
  - Uses a trained machine learning/deep learning model for face mask classification.

---

## 🖥️ Application Pages

The application includes:

- 🏠 Home
- 🔐 Login
- 📝 Signup
- 🔑 Forgot Password
- 🔄 Reset Password
- 📷 Detect
- 🎥 Live Detection
- 📊 Results
- 🕒 Detection History
- 👤 Profile
- ℹ️ About
- ⚡ Features
- 🔧 How It Works

---

## 🛠️ Technologies Used

### Frontend

- HTML5
- CSS3
- JavaScript

### Backend

- Python
- Flask

### Machine Learning

- TensorFlow / Keras
- OpenCV
- NumPy
- Machine Learning / Deep Learning

### Development Tools

- Git
- GitHub
- VS Code

---

## 📁 Project Structure

```text
MaskDetect-AI/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── models/
│   └── model.h5
│
├── static/
│   ├── uploads/
│   ├── logo.png
│   ├── ai-character.png
│   ├── ai-character-auth.png
│   ├── ai-character-right.png
│   ├── ai-character-about.png
│   ├── ai-character-features.png
│   └── ai-character-how-it-works.png
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── forgot_password.html
│   ├── reset_password.html
│   ├── detect.html
│   ├── live.html
│   ├── results.html
│   ├── history.html
│   ├── profile.html
│   ├── about.html
│   ├── features.html
│   └── how_it_works.html
│
├── app.py
├── detect_live.py
├── train_model.py
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/varshini7995/MaskDetect-AI.git
```

### 2. Open the project directory

```bash
cd MaskDetect-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 5. Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

Then open the following address in your web browser:

```text
http://127.0.0.1:5000
```

---

## 📷 How It Works

The general workflow of MaskDetect AI is:

```text
User
  ↓
Upload Image / Start Camera
  ↓
Image Processing
  ↓
AI Model
  ↓
Face Mask Prediction
  ↓
Detection Result
  ↓
History / Profile
```

---

## 🧠 Machine Learning

The project uses a trained machine learning/deep learning model stored in:

```text
models/model.h5
```

The model is used to classify face mask detection input and generate the corresponding prediction.

The project also contains:

```text
train_model.py
```

which contains the model training workflow.

---

## 🎥 Live Detection

The live detection functionality uses the system camera to process video frames and perform face mask detection in real time.

The live detection implementation is contained in:

```text
detect_live.py
```

---

## 📊 Project Highlights

### Image-Based Detection

Users can upload an image through the web interface and receive a face mask detection result.

### Real-Time Detection

The application provides a live camera-based detection mode.

### User Dashboard

Users can access their profile and detection history.

### Modern Web Interface

The application contains a dedicated interface for authentication, detection, results, history, and informational pages.

---

## 🎯 Project Objectives

The main objectives of MaskDetect AI are:

1. Build an AI-powered face mask detection system.
2. Provide an easy-to-use web interface.
3. Support image-based detection.
4. Support real-time camera detection.
5. Maintain user detection history.
6. Provide a complete user authentication experience.
7. Integrate machine learning with a Flask web application.

---

## 🚀 Future Enhancements

Possible future improvements include:

- Improved detection accuracy
- Multi-face detection optimization
- Deployment to a cloud platform
- Mobile-friendly application
- Improved real-time performance
- Analytics dashboard
- Additional AI-based safety detection features

---

## 👩‍💻 Author

**Varshini Reddy**

GitHub:

https://github.com/varshini7995

---

## 📜 License

This project is created for educational and project demonstration purposes.
