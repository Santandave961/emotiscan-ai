import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
from deepface import DeepFace
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="EmotiScan AI", page_icon="😊", layout="wide")

st.markdown("""
<style>
    .header-title { font-size: 3rem; font-weight: 900; color: #00d4aa; text-align: center; }
    .sub-title { text-align: center; color: #aaa; font-size: 1.1rem; margin-bottom: 2rem; }
    .emotion-card { border-radius: 20px; padding: 30px; text-align: center; color: white; }
    .info-card { background: #1e2130; border-radius: 12px; padding: 20px; margin: 8px 0; }
    .metric-card { background: #1e2130; border-radius: 12px; padding: 15px; text-align: center; }
</style>
""", unsafe_allow_html=True)

EMOTION_CONFIG = {
    "happy":    {"emoji": "😊", "color": "#00b894", "gradient": "linear-gradient(135deg, #00b894, #00d4aa)", "label": "Happy"},
    "sad":      {"emoji": "😢", "color": "#0984e3", "gradient": "linear-gradient(135deg, #0984e3, #74b9ff)", "label": "Sad"},
    "angry":    {"emoji": "😠", "color": "#d63031", "gradient": "linear-gradient(135deg, #d63031, #ff4757)", "label": "Angry"},
    "fear":     {"emoji": "😨", "color": "#6c5ce7", "gradient": "linear-gradient(135deg, #6c5ce7, #a29bfe)", "label": "Fear"},
    "surprise": {"emoji": "😲", "color": "#e17055", "gradient": "linear-gradient(135deg, #e17055, #fdcb6e)", "label": "Surprise"},
    "disgust":  {"emoji": "🤢", "color": "#00b894", "gradient": "linear-gradient(135deg, #00b894, #55efc4)", "label": "Disgust"},
    "neutral":  {"emoji": "😐", "color": "#636e72", "gradient": "linear-gradient(135deg, #636e72, #b2bec3)", "label": "Neutral"},
}

FINTECH_INSIGHTS = {
    "happy":    "✅ Customer appears satisfied. Ideal moment to upsell products.",
    "sad":      "⚠️ Customer may be distressed. Consider offering support.",
    "angry":    "🚨 High frustration detected. Escalate to senior support.",
    "fear":     "⚠️ Customer appears anxious. Reassurance needed during onboarding.",
    "surprise": "💡 Customer reacting to something unexpected. Clarify product terms.",
    "disgust":  "🚨 Negative reaction detected. Check for UX friction.",
    "neutral":  "✅ Customer is calm. Standard onboarding can proceed.",
}


def detect_and_annotate_faces(img_array, emotion_results):
    annotated = img_array.copy()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    color_map = {
        "happy": (0, 212, 170), "sad": (9, 132, 227), "angry": (214, 48, 49),
        "fear": (108, 92, 231), "surprise": (225, 112, 85),
        "disgust": (0, 184, 148), "neutral": (99, 110, 114)
    }
    for i, (x, y, w, h) in enumerate(faces):
        if i < len(emotion_results):
            emotion = emotion_results[i].get("dominant_emotion", "neutral")
            label = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])["label"]
            color = color_map.get(emotion, (0, 212, 170))
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated, (x, y-th-15), (x+tw+10, y), color, -1)
            cv2.putText(annotated, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return annotated, len(faces)


# ── HEADER ──
st.markdown('<p class="header-title">😊 EmotiScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Facial Emotion Recognition powered by Deep Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ── SIDEBAR ──
st.sidebar.markdown("## 😊 EmotiScan AI")
st.sidebar.markdown("---")
st.sidebar.markdown("### 7 Emotions Detected")
for emotion, config in EMOTION_CONFIG.items():
    st.sidebar.markdown(f"{config['emoji']} **{config['label']}**")
st.sidebar.markdown("---")
st.sidebar.markdown("### How it works")
st.sidebar.markdown("1. 📤 Upload a photo\n2. 🔍 DeepFace detects faces\n3. 🧠 CNN classifies emotion\n4. 📊 Confidence scores shown\n5. 💼 Fintech insight provided")
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by:** Okparaji Wisdom")

# ── UPLOAD ──
st.markdown("### 📤 Upload Image")
st.info("Upload a clear photo with one or more faces. Supported formats: JPG, PNG, JPEG")
uploaded_file = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(pil_image, use_container_width=True)
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h4 style="color:#00d4aa">Image Info</h4>
            <p>📐 Size: {pil_image.width} x {pil_image.height}px</p>
            <p>🎨 Mode: {pil_image.mode}</p>
            <p>📁 Format: {uploaded_file.type}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <h4 style="color:#00d4aa">Analysis Engine</h4>
            <p>🧠 DeepFace — Multi-model ensemble</p>
            <p>👤 Haar Cascade — Face Detection</p>
            <p>📊 7 emotion classes</p>
            <p>💼 Fintech KYC insights</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("😊 Analyse Emotions", use_container_width=True):
        results = []
        annotated_img = img_array.copy()
        face_count = 0
        analysis_success = False

        with st.spinner("🧠 Detecting faces and analysing emotions..."):
            try:
                results = DeepFace.analyze(
                    img_path=img_array,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(results, dict):
                    results = [results]
                annotated_img, face_count = detect_and_annotate_faces(img_array, results)
                analysis_success = True
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

        if analysis_success and len(results) > 0:
            st.markdown("## 🎯 Emotion Analysis Results")
            st.success(f"✅ Detected **{len(results)} face(s)** in the image")

            for i, result in enumerate(results):
                dominant_emotion = result.get("dominant_emotion", "neutral").lower()
                emotion_scores = result.get("emotion", {})
                config = EMOTION_CONFIG.get(dominant_emotion, EMOTION_CONFIG["neutral"])

                st.markdown(f"### 👤 Face {i+1}")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="emotion-card" style="background: {config['gradient']}">
                        <h1 style="font-size:4rem; margin:0">{config['emoji']}</h1>
                        <h2>{config['label'].upper()}</h2>
                        <h3>{emotion_scores.get(dominant_emotion, 0):.1f}% confidence</h3>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    emotions_df = pd.DataFrame({
                        "Emotion": [EMOTION_CONFIG[e]["emoji"] + " " + e.capitalize() for e in emotion_scores.keys()],
                        "Score": list(emotion_scores.values())
                    }).sort_values("Score", ascending=True)
                    fig = px.bar(emotions_df, x="Score", y="Emotion", orientation="h",
                                 title="Emotion Confidence Scores", color="Score",
                                 color_continuous_scale=["#636e72", "#00d4aa"])
                    fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                                      font_color="white", height=300, showlegend=False,
                                      coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color:#00d4aa">Face Attributes</h4>
                        <p>😊 Emotion: <b>{config['label']}</b></p>
                        <p>🎯 Confidence: <b>{emotion_scores.get(dominant_emotion, 0):.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    insight = FINTECH_INSIGHTS.get(dominant_emotion, "")
                    st.markdown(f"""
                    <div class="info-card" style="margin-top:10px">
                        <h4 style="color:#00d4aa">💼 Fintech KYC Insight</h4>
                        <p>{insight}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ── ANNOTATED IMAGE ──
            st.markdown("---")
            st.markdown("### 🗺️ Face Detection Map")
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_img, caption="Detected Faces with Emotion Labels", use_container_width=True)

            # ── MULTI FACE ──
            if len(results) > 1:
                st.markdown("---")
                st.markdown("### 👥 Multi-Face Emotion Comparison")
                comparison_data = []
                for i, result in enumerate(results):
                    scores = result.get("emotion", {})
                    for emotion, score in scores.items():
                        comparison_data.append({"Face": f"Face {i+1}", "Emotion": emotion.capitalize(), "Score": score})
                comp_df = pd.DataFrame(comparison_data)
                fig = px.bar(comp_df, x="Emotion", y="Score", color="Face", barmode="group",
                             title="Emotion Scores by Face",
                             color_discrete_sequence=["#00d4aa", "#ff4757", "#fdcb6e", "#a29bfe"])
                fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", font_color="white", height=350)
                st.plotly_chart(fig, use_container_width=True)

            # ── SUMMARY ──
            st.markdown("---")
            st.markdown("### 📋 Session Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Faces Detected", len(results))
            dominant_emotions = [r.get("dominant_emotion", "neutral") for r in results]
            if dominant_emotions:
                most_common = max(set(dominant_emotions), key=dominant_emotions.count)
                cfg = EMOTION_CONFIG.get(most_common, EMOTION_CONFIG["neutral"])
                col2.metric("Dominant Emotion", f"{cfg['emoji']} {cfg['label']}")
                avg_confidence = np.mean([
                    r.get("emotion", {}).get(r.get("dominant_emotion", "neutral"), 0)
                    for r in results
                ])
                col3.metric("Avg Confidence", f"{avg_confidence:.1f}%")

        elif analysis_success:
            st.warning("No faces detected. Try a clearer photo with a visible face.")

else:
    st.markdown("### 💡 Use Cases")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-card"><h4>💼 Fintech KYC</h4><p>Detect customer stress or confusion during video onboarding and identity verification</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card"><h4>🎓 EdTech</h4><p>Monitor student engagement and emotional state during online learning sessions</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="info-card"><h4>🏥 Healthcare</h4><p>Assist in detecting emotional distress or pain indicators in patient monitoring</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#aaa'>Upload an image above to begin emotion analysis</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa'>EmotiScan AI | Built by Okparaji Wisdom 🇳🇬 | Computer Vision Portfolio Project</p>", unsafe_allow_html=True)