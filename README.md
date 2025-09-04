# 🧠 Cancer-Checker-AI

A machine learning web application designed to **predict cancer presence** based on patient data and gene expression patterns. Built with Python, Flask, and multiple ML models (KNN, Naive Bayes), this app offers fast, accessible, and intelligent early cancer detection.

---

## 🚀 Features

- 🧬 Predict cancer risk from real patient datasets
- 🧪 Uses trained ML models (KNN, Naive Bayes)
- 🧮 Scaled input features for model precision
- 🧑‍⚕️ Simple web interface for prediction and result interpretation
- 💾 Pretrained models loaded from `.pkl` files
- 🎨 Clean and responsive UI with HTML/CSS (Bootstrap-like)

---

## 🧠 Machine Learning

- **Algorithms:**  
  - K-Nearest Neighbors (KNN)  
  - Naive Bayes Classifier  

- **Preprocessing:**  
  - Feature scaling (StandardScaler)  
  - Gene expression and patient data  

---

## 🛠️ Tech Stack

| Layer       | Technology       |
|-------------|------------------|
| Language    | Python 3.12       |
| Web Backend | Flask             |
| ML Models   | Scikit-learn      |
| Frontend    | HTML5, CSS3       |
| Tools       | Git + GitHub      |

---

## 📁 Project Structure

```text
.
├── CAN/
│   ├── static/
│   │   └── css/
│   │       ├── style.css
│   │       └── style_index.css
│   ├── templates/
│   │   ├── index.html
│   │   ├── predict.html
│   │   ├── prediction_result.html
│   │   └── results.html
│   ├── models/
│   │   ├── knn_model.pkl
│   │   ├── nb_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_names.pkl
│   └── main.py           ← Flask entry point
|
├── *.csv                 ← Cancer-related datasets
├── requirements.txt      ← Python dependencies
└── README.md             ← This file
```
## 🖼️ Screenshots

<details>
<summary><strong>🧪 Home Page</strong></summary>

<br>

![image](https://github.com/user-attachments/assets/cf0029ab-c2d2-4a59-9341-3445b08ff1a3)
</details>

<details>
<summary><strong>📊 Prediction Result</strong></summary>

<br>

![image](https://github.com/user-attachments/assets/a405efe0-7b6c-43af-8602-4f898048e15e)

</details>

<details>
<summary><strong>📈 Prediction</strong></summary>

<br>

![image](https://github.com/user-attachments/assets/40590b1a-8f3e-4116-a427-dac51eda9bb4)

</details>
