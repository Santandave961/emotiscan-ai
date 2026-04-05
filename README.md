# 😊 EmotiScan AI
### Facial Emotion Recognition for Fintech KYC & Beyond

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.99-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Demo
👉 [EmotiScan AI on Streamlit Cloud](https://emotiscan-ai.streamlit.app)

---

## 📌 Overview
EmotiScan AI is a real-time facial emotion recognition web application built with DeepFace and Streamlit. It detects faces in uploaded images and classifies emotions with confidence scores, providing actionable insights tailored for Nigerian fintech use cases like KYC onboarding, EdTech engagement monitoring, and healthcare patient analysis.

---

## 🎯 Features
- 🧠 **Deep Learning** — Powered by DeepFace multi-model ensemble
- 😊 **7 Emotions Detected** — Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- 👥 **Multi-Face Support** — Detects and analyses multiple faces simultaneously
- 📊 **Confidence Scores** — Interactive Plotly bar charts per face
- 🗺️ **Face Detection Map** — Annotated image with bounding boxes and emotion labels
- 💼 **Fintech KYC Insights** — Real-time alerts for customer emotional state
- 📋 **Session Summary** — Dominant emotion and average confidence metrics

---

## 💼 Use Cases

| Industry | Application |
|----------|-------------|
| 💼 Fintech KYC | Detect stress or confusion during video onboarding |
| 🎓 EdTech | Monitor student engagement during online learning |
| 🏥 Healthcare | Detect emotional distress in patient monitoring |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Streamlit | Web app framework |
| DeepFace | Facial emotion recognition |
| OpenCV | Face detection & annotation |
| Plotly | Interactive charts |
| Pandas / NumPy | Data processing |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Santandave961/emotiscan-ai.git
cd emotiscan-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure
```
emotiscan-ai/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 📸 How It Works
1. 📤 Upload a photo with one or more faces
2. 🔍 Haar Cascade detects face regions
3. 🧠 DeepFace CNN classifies emotion per face
4. 📊 Confidence scores displayed as interactive charts
5. 💼 Fintech KYC insight generated per detected emotion

---

## 🇳🇬 Built For Nigerian Fintech
This project is designed with Nigerian fintech companies in mind — Kuda, Moniepoint, Flutterwave, and Interswitch — where video KYC and customer emotion intelligence are becoming critical to onboarding and fraud prevention.

---

## 👨‍💻 Author
**Okparaji Wisdom**
- GitHub: [@Santandave961](https://github.com/Santandave961)
- LinkedIn: [Okparaji Wisdom](https://linkedin.com/in/okparaji-wisdom)

---

## 📄 License
MIT License — free to use and modify.

