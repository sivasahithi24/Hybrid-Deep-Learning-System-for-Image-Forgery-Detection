
# Hybrid Deep Learning System for Image Forgery Detection

##  Problem Statement

With the rapid advancement of AI-powered image editing tools, digitally manipulated images can appear highly realistic, making it difficult to distinguish between authentic and forged content. This creates serious concerns in areas such as media integrity, security, and digital forensics. Therefore, there is a need for an intelligent system that can accurately detect image forgery and highlight tampered regions.

---

##  Overview

This project focuses on detecting whether a given image is **real or fake** and further **highlighting the tampered regions** using heatmap visualization.

It uses a **hybrid deep learning approach (CNN-based models)** to analyze image features and identify manipulation patterns. The system provides both **classification** and **visual explanation** of forgery.

---

##  Features

*  Classifies images as **Real** or **Fake**
*  Highlights tampered regions using **heatmaps**
*  Uses hybrid CNN-based deep learning models
*  Interactive UI built with Streamlit
*  Real-time image analysis

---

##  Tech Stack

* **Programming Language:** Python
* **Libraries:** TensorFlow, Keras, OpenCV, NumPy
* **Visualization:** Heatmaps for tampering detection
* **Frontend:** Streamlit

---

##  Project Structure

```
project/
│── app.py
│── config.json
│── utils.py
│── metrics.xls
│── requirements.txt
│── README.md
```

---

##  Model Files

Due to GitHub size limitations, trained model files are not included.

 Download models from:
 [https://drive.google.com/drive/folders/1F56uhGQPYnxzerd-6ZHQvea_k8uAn4sg?usp=sharing]

---

##  How to Run

### 1. Clone Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Requirements

```
pip install -r requirements.txt
```

### 3. Add Model Files

* Download models from the link
* Place them in the project folder

### 4. Run App

```
streamlit run app.py
```

---

##  Usage

1. Launch the app
2. Upload an image
3. System will:

   * Predict **Real / Fake**
   * Display **heatmap highlighting tampered regions**

---

##  Output

*  Classification: Real or Fake
*  Heatmap showing manipulated areas

---

##  Future Improvements

* Improve localization accuracy
* Add support for video forgery detection
* Deploy as a web application

