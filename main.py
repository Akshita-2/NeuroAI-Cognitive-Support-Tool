import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import base64
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import google.generativeai as genai
import json
from deepface import DeepFace
import requests


# Configure page
st.set_page_config(
    page_title="NeuroAI - Cognitive Support Tool",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #6200EE;
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #BB86FC;
    }
    
    .feature-header {
        font-size: 24px;
        font-weight: 600;
        color: #6200EE;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    .success-box {
        padding: 20px;
        background-color: #E8F5E9;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    
    .info-box {
        padding: 20px;
        background-color: #E3F2FD;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin-bottom: 20px;
    }
    
    .emotion-card {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        background-color: white;
        transition: transform 0.3s ease;
    }
    
    .emotion-card:hover {
        transform: translateY(-5px);
    }
    
    .emotion-result {
        background-color: #F3E5F5;
        border-radius: 15px;
        padding: 25px;
        border-left: 5px solid #9C27B0;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .camera-feed {
        border: 3px solid #6200EE;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .stButton>button {
        background-color: #6200EE;
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3700B3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .upload-section {
        border: 2px dashed #BB86FC;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #6200EE;
        background-color: rgba(187, 134, 252, 0.05);
    }
    
    .emotion-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin-right: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .joy-badge { background-color: #FFC107; }
    .sadness-badge { background-color: #2196F3; }
    .anger-badge { background-color: #F44336; }
    .fear-badge { background-color: #9C27B0; }
    .surprise-badge { background-color: #FF9800; }
    .neutral-badge { background-color: #78909C; }
    .disgust-badge { background-color: #8BC34A; }
    .happy-badge { background-color: #FFC107; }
    
    .history-item {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    .history-item:hover {
        background-color: #F5F5F5;
        transform: translateX(5px);
    }
    
    div.stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 5px;
    }
    
    div.stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        border: none;
        font-weight: 500;
    }
    
    div.stTabs [aria-selected="true"] {
        background-color: #F3E5F5 !important;
        color: #6200EE !important;
        border-bottom: 3px solid #6200EE;
    }
    
    .stats-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stats-value {
        font-size: 36px;
        font-weight: bold;
        color: #6200EE;
    }
    
    .stats-label {
        font-size: 14px;
        color: #757575;
        margin-top: 5px;
    }
    
    .chart-container {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .emoji-large {
        font-size: 60px;
        text-align: center;
        margin: 10px 0;
    }
    
    .emotion-name {
        font-size: 24px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 15px;
        color: #6200EE;
    }
    
    .emotion-description {
        text-align: center;
        color: #555;
        font-size: 16px;
    }
    
    .color-bar {
        height: 10px;
        background-image: linear-gradient(to right, #FFC107, #FF9800, #F44336, #9C27B0, #2196F3, #8BC34A, #78909C);
        border-radius: 5px;
        margin: 20px 0;
    }
    
    .footer {
        text-align: center;
        padding: 20px 0;
        font-size: 14px;
        color: #757575;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'emotion_results' not in st.session_state:
    st.session_state.emotion_results = None
if 'image_emotion_results' not in st.session_state:
    st.session_state.image_emotion_results = None
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'prev_time' not in st.session_state:
    st.session_state.prev_time = time.time()


# Function to get emoji for emotion
def get_emotion_emoji(emotion):
    emotion_map = {
        'happy': '😄',
        'joy': '😄',
        'sad': '😢',
        'sadness': '😢',
        'angry': '😠',
        'anger': '😠',
        'neutral': '😐',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'contempt': '😒'
    }
    return emotion_map.get(emotion.lower(), '❓')


# Function to get color for emotion
def get_emotion_color(emotion):
    emotion_map = {
        'happy': '#FFC107',
        'joy': '#FFC107',
        'sad': '#2196F3',
        'sadness': '#2196F3',
        'angry': '#F44336',
        'anger': '#F44336',
        'neutral': '#78909C',
        'fear': '#9C27B0',
        'surprise': '#FF9800',
        'disgust': '#8BC34A',
        'contempt': '#795548'
    }
    return emotion_map.get(emotion.lower(), '#9E9E9E')


# Function to detect faces and emotions
def detect_emotion(image, is_path=False):
    try:
        if is_path:
            # For file path input
            result = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                enforce_detection=False
            )
        else:
            # For direct image array input
            result = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                enforce_detection=False
            )
        
        if isinstance(result, list):
            result = result[0]  # Get the first face if multiple faces
        
        # Extract emotion data
        emotion_data = {
            "dominant_emotion": result['dominant_emotion'],
            "emotion_scores": result['emotion'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return emotion_data
    except Exception as e:
        st.error(f"Error in emotion detection: {str(e)}")
        return None


# Function for real-time facial emotion detection
def process_webcam():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    status_text = st.empty()
    emotion_display = st.empty()
    
    # Counters for emotion detection stats
    emotion_counts = {
        'happy': 0, 'sad': 0, 'angry': 0, 'fear': 0, 
        'neutral': 0, 'surprise': 0, 'disgust': 0
    }
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Mirror the frame for a more natural appearance
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for display in Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process every 3rd frame to improve performance
        current_time = time.time()
        if current_time - st.session_state.prev_time > 0.3:  # Process every 0.3 seconds
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            # If faces detected, analyze emotions
            if len(faces) > 0:
                try:
                    emotion_result = detect_emotion(rgb_frame)
                    
                    if emotion_result:
                        # Update stats
                        dominant_emotion = emotion_result['dominant_emotion']
                        if dominant_emotion in emotion_counts:
                            emotion_counts[dominant_emotion] += 1
                        
                        # Display emotion on frame
                        for (x, y, w, h) in faces:
                            # Draw rectangle around face
                            color = get_emotion_color(dominant_emotion)
                            hex_color = color.lstrip('#')
                            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            rgb_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Convert to BGR
                            
                            cv2.rectangle(frame, (x, y), (x+w, y+h), rgb_color, 2)
                            
                            # Put text above rectangle
                            emoji = get_emotion_emoji(dominant_emotion)
                            text = f"{dominant_emotion.upper()} {emoji}"
                            cv2.putText(frame, text, (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rgb_color, 2)
                            
                        # Display detailed emotion data
                        with emotion_display.container():
                            emotion_data = emotion_result['emotion_scores']
                            
                            # Convert to displayable format and sort by value
                            emotion_pairs = [(k, v) for k, v in emotion_data.items()]
                            emotion_pairs.sort(key=lambda x: x[1], reverse=True)
                            
                            cols = st.columns(len(emotion_pairs))
                            for i, (emotion, score) in enumerate(emotion_pairs):
                                with cols[i]:
                                    emoji = get_emotion_emoji(emotion)
                                    st.markdown(f"<div style='text-align: center;'>{emoji}<br>{emotion.capitalize()}</div>", unsafe_allow_html=True)
                                    st.progress(score/100)
                                    st.markdown(f"<div style='text-align: center;'>{score:.1f}%</div>", unsafe_allow_html=True)
                            
                            # Add to history if significant change
                            if len(st.session_state.emotion_history) == 0 or dominant_emotion != st.session_state.emotion_history[-1]['dominant_emotion']:
                                history_item = {
                                    'dominant_emotion': dominant_emotion,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'type': 'realtime'
                                }
                                st.session_state.emotion_history.append(history_item)
                    else:
                        status_text.warning("Could not determine emotion. Please adjust lighting or position.")
                        
                except Exception as e:
                    status_text.error(f"Processing error: {str(e)}")
            else:
                status_text.info("No face detected. Please position yourself in front of the camera.")
                
            st.session_state.prev_time = current_time
            
        # Convert processed frame back to RGB for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_column_width=True)
        
    cap.release()


# Sidebar for navigation
with st.sidebar: 
    st.image("images.png", width=100)
    st.markdown("<h2 style='text-align: center;'>NeuroAI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Advanced Facial Expression Analysis</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation
    nav_option = st.radio(
        "Navigation",
        ["Dashboard", "Expression Analysis","AAC","storify","Quiz"]
    )
    
    st.markdown("---")
    st.markdown("### About NeuroAI")
    st.markdown("""
    NeuroAI is an AI-powered platform that supports neurodiverse children through:

    - AAC Tools(visual-to-speech cards)
    - Emotion Recognition (real-time facial emotion detection)
    - Storify(attention, memory boosters)
    - Parent Dashboard (track communication, emotion, performance)
    """)
    
    st.markdown("---")
    st.markdown(
        "<div class='success-box'>All systems operational</div>", 
        unsafe_allow_html=True
    )


# Main content area based on navigation
if nav_option == "Dashboard":
    st.markdown("<div class='main-header'>Facial Expression Dashboard</div>", unsafe_allow_html=True)
    
    # Dashboard intro
    st.markdown("""
    <div class='info-box'><span style ='color : green;'>
        Welcome to NeuroAI! This dashboard gives you an overview of your emotional analysis activities
        and provides insights into expression patterns.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='stats-card'>
            <div class='stats-value'>{}</div>
            <div class='stats-label'>TOTAL ANALYSES</div>
        </div>
        """.format(len(st.session_state.emotion_history)), unsafe_allow_html=True)
    
    with col2:
        # Calculate most common emotion
        if st.session_state.emotion_history:
            emotions = [item['dominant_emotion'] for item in st.session_state.emotion_history]
            most_common = max(set(emotions), key=emotions.count)
            emoji = get_emotion_emoji(most_common)
        else:
            most_common = "N/A"
            emoji = "❓"
            
        st.markdown("""
        <div class='stats-card'>
            <div class='stats-value'>{} {}</div>
            <div class='stats-label'>TOP EMOTION</div>
        </div>
        """.format(most_common.upper(), emoji), unsafe_allow_html=True)
    
    with col3:
        # Calculate realtime sessions
        realtime_count = len([item for item in st.session_state.emotion_history if item.get('type') == 'realtime'])
        st.markdown("""
        <div class='stats-card'>
            <div class='stats-value'>{}</div>
            <div class='stats-label'>REALTIME SESSIONS</div>
        </div>
        """.format(realtime_count), unsafe_allow_html=True)
    
    with col4:
        # Calculate image uploads
        image_count = len([item for item in st.session_state.emotion_history if item.get('type') == 'image'])
        st.markdown("""
        <div class='stats-card'>
            <div class='stats-value'>{}</div>
            <div class='stats-label'>IMAGES ANALYZED</div>
        </div>
        """.format(image_count), unsafe_allow_html=True)
    
    # Decorative color bar
    st.markdown("<div class='color-bar'></div>", unsafe_allow_html=True)
    
    # Emotion distribution chart
    st.markdown("<div class='feature-header'>Emotion Distribution</div>", unsafe_allow_html=True)
    
    if st.session_state.emotion_history:
        # Count emotions
        emotions = [item['dominant_emotion'] for item in st.session_state.emotion_history]
        emotion_counts = {}
        for emotion in emotions:
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
        
        # Create chart data
        chart_data = pd.DataFrame({
            'Emotion': list(emotion_counts.keys()),
            'Count': list(emotion_counts.values())
        })
        
        # Sort by count
        chart_data = chart_data.sort_values('Count', ascending=False)
        
        # Display chart
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Get colors for each emotion
        colors = [get_emotion_color(emotion) for emotion in chart_data['Emotion']]
        
        bars = ax.bar(chart_data['Emotion'], chart_data['Count'], color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Detected Emotions', fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No emotion data available yet. Start analyzing expressions to see distribution.")
    
    # Recent activity
    st.markdown("<div class='feature-header'>Recent Expression Analysis</div>", unsafe_allow_html=True)
    
    if st.session_state.emotion_history:
        # Display the 5 most recent analyses in cards
        col1, col2 = st.columns(2)
        
        recent_items = st.session_state.emotion_history[-5:] if len(st.session_state.emotion_history) >= 5 else st.session_state.emotion_history
        
        for i, item in enumerate(reversed(recent_items)):
            with col1 if i % 2 == 0 else col2:
                emotion = item.get('dominant_emotion', 'unknown')
                timestamp = item.get('timestamp', '')
                activity_type = item.get('type', 'analysis')
                emoji = get_emotion_emoji(emotion)
                
                st.markdown(f"""
                <div class='emotion-card' style='border-left: 5px solid {get_emotion_color(emotion)};'>
                    <div style='display: flex; align-items: center;'>
                        <div style='font-size: 40px; margin-right: 15px;'>{emoji}</div>
                        <div>
                            <div style='font-size: 22px; font-weight: 600;'>{emotion.upper()}</div>
                            <div style='color: #757575; font-size: 14px;'>{timestamp}</div>
                            <div style='margin-top: 5px;'>
                                <span class='emotion-badge' style='background-color: #7E57C2; font-size: 12px;'>
                                    {activity_type.upper()}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recent activity to display. Start analyzing expressions to build history.")
    
    # Emotion insights
    st.markdown("<div class='feature-header'>Understanding Expressions</div>", unsafe_allow_html=True)
    
    emotion_insight_cols = st.columns(3)
    
    emotions_info = [
        {
            "name": "Happiness",
            "emoji": "😄",
            "description": "Characterized by a genuine smile, raised cheeks, and crinkled eyes. Indicates positive emotions and contentment."
        },
        {
            "name": "Sadness",
            "emoji": "😢",
            "description": "Features drooping eyelids, downturned mouth, and sometimes furrowed brows. Suggests unhappiness or disappointment."
        },
        {
            "name": "Surprise",
            "emoji": "😲",
            "description": "Shown through raised eyebrows, widened eyes, and often an open mouth. Indicates unexpected information or events."
        }
    ]
    
    for i, col in enumerate(emotion_insight_cols):
        with col:
            info = emotions_info[i]
            st.markdown(f"""
            <div class='emotion-card'>
                <div class='emoji-large'>{info['emoji']}</div>
                <div class='emotion-name'>{info['name']}</div>
                <div class='emotion-description'>{info['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        NeuroAI • Advanced Cognitive Support • © 2025
    </div>
    """, unsafe_allow_html=True)


elif nav_option == "AAC":
    import pyttsx3
    import logging
    from typing import List, Dict, Tuple

    # --- Page Setup ---
    st.markdown("""
    <style>
    .aac-button button {
        min-height: 100px;
        font-size: 24px !important;
    }
    .constructed-sentence {
        font-size: 32px;
        padding: 20px;
        border-radius: 10px;
        background-color: #004080 !important;  /* Dark blue background for contrast */
        color: #ffffff !important;             /* White, bold text */
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown("<h1 class='main-header'>AAC Communication Board</h1>", unsafe_allow_html=True)
    st.markdown("### Tap symbols to build your message")

    # --- Session State Management ---
    class AACState:
        @staticmethod
        def init():
            defaults = {
                'constructed_sentence': [],
                'aac_category': "Core",
            }
            for key, value in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = value

    AACState.init()

    # --- Vocabulary Setup ---
    class AACVocabulary:
        CORE: Dict[str, List[Tuple[str, str]]] = {
            "Core": [
                ("I", "🧑"), ("want", "🙏"), ("like", "👍"), ("don't", "🙅‍♂️"),
                ("go", "🏃"), ("stop", "✋"), ("more", "➕"), ("finished", "✅"),
                ("you", "🫵"), ("it", "📦"), ("help", "🆘"), ("please", "🙏"),
                ("yes", "✔️"), ("no", "❌"), ("my", "🙋‍♂️"), ("your", "🙋‍♀️"),
            ],
            "Needs": [
                ("eat", "🍽️"), ("drink", "🥤"), ("water", "💧"), ("toilet", "🚽"),
                ("sleep", "🛌"), ("pain", "🤕"), ("medicine", "💊"), ("wash", "🧼"),
            ],
            "Feelings": [
                ("happy", "😊"), ("sad", "😢"), ("angry", "😠"), ("scared", "😱"),
                ("tired", "😴"), ("sick", "🤒"), ("excited", "🤩"), ("love", "❤️"),
            ]
        }

        @classmethod
        def get_categories(cls) -> List[str]:
            return list(cls.CORE.keys())

    # --- TTS Engine Management ---
    import threading

    class TTSEngine:
        @staticmethod
        def speak(text: str):
            try:
                st.session_state['is_speaking'] = True
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
            except RuntimeError as e:
                logging.error(f"TTS Error: {str(e)}")
                st.error("Speech error. Please wait until current speech finishes.")
            except Exception as e:
                logging.exception("TTS Failure")
                st.error(f"Speech error: {str(e)}")
            finally:
                st.session_state['is_speaking'] = False





    # --- UI Components ---
    class AACInterface:
        @staticmethod
        def category_selector():
            current_category = st.session_state.aac_category
            new_category = st.selectbox(
                "Category",
                options=AACVocabulary.get_categories(),
                index=AACVocabulary.get_categories().index(current_category),
                key="aac_category_selector"
            )
            st.session_state.aac_category = new_category

        @staticmethod
        def symbol_grid():
            cols = st.columns(4)
            category = st.session_state.aac_category
            for idx, (word, emoji) in enumerate(AACVocabulary.CORE[category]):
                with cols[idx % 4]:
                    if st.button(
                        f"{emoji} {word}",
                        key=f"btn_{category}_{word}",
                        use_container_width=True,
                        type="primary"
                    ):
                        st.session_state.constructed_sentence.append(word)

        @staticmethod
        def sentence_builder():
            if st.session_state.constructed_sentence:
                sentence = " ".join(st.session_state.constructed_sentence)
                st.markdown(
                    f"<div class='constructed-sentence'>🗨️ {sentence}</div>",
                    unsafe_allow_html=True
                )


        @staticmethod
        def control_buttons():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if st.button("🔊 Speak Message", use_container_width=True, disabled=st.session_state.get('is_speaking', False)):
                    TTSEngine.speak(" ".join(st.session_state.constructed_sentence))

            with col2:
                if st.button("↩️ Undo", use_container_width=True):
                    if st.session_state.constructed_sentence:
                        st.session_state.constructed_sentence.pop()
            with col3:
                if st.button("🧹 Clear All", use_container_width=True, type="secondary"):
                    st.session_state.constructed_sentence = []

    # --- Main UI Flow ---
    AACInterface.category_selector()
    AACInterface.symbol_grid()
    AACInterface.sentence_builder()
    AACInterface.control_buttons()

    # --- Accessibility Notes ---
    st.markdown("""
    <div style='margin-top: 50px; font-size: 14px; color: #666;'>
    Accessibility Features:
    <ul>
        <li>Large touch-friendly buttons</li>
        <li>Clear visual feedback</li>
        <li>Consistent category organization</li>
        <li>Undo/Clear functionality</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
elif nav_option == "storify":
    import streamlit as st
    import requests
    import re

    # --- Custom CSS for Aesthetics ---
    st.markdown("""
    <style>
    .storify-hero {
        text-align: center;
        margin-bottom: 2rem;
    }
    .storify-title {
        font-size: 2.7rem;
        font-weight: 800;
        color: #1e3a8a;
        letter-spacing: 1px;
        margin-bottom: 0.2em;
    }
    .storify-subtitle {
        font-size: 1.15rem;
        color: #2563eb;
        margin-bottom: 1.5rem;
    }
    .storify-form {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 4px 24px rgba(30,58,138,0.08);
        padding: 2rem 2rem 1rem 2rem;
        margin-bottom: 2rem;
        max-width: 650px;
        margin-left: auto;
        margin-right: auto;
    }
    .storify-card {
        background: linear-gradient(100deg, #f1f5ff 70%, #e0e7ff 100%);
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(30,58,138,0.07);
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .storify-image {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(30,58,138,0.08);
        margin-bottom: 0.8rem;
    }
    .storify-section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .storify-badge {
        display: inline-block;
        background: #2563eb;
        color: #fff;
        border-radius: 8px;
        padding: 0.3rem 0.8rem;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }
    .stButton>button {
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        background: linear-gradient(90deg, #2563eb 80%, #38bdf8 100%) !important;
        color: #fff !important;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(30,58,138,0.05);
        transition: background 0.2s, transform 0.1s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1e40af 80%, #0ea5e9 100%) !important;
        transform: scale(1.04);
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Hero Section ---
    st.markdown("""
    <div class="storify-hero">
        <img src="https://cdn-icons-png.flaticon.com/512/3064/3064197.png" width="90" style="margin-bottom: 0.7rem;" />
        <div class="storify-title">✨ Storify: Turn Prompts into Illustrated Stories</div>
        <div class="storify-subtitle">Generate a captivating story with matching images using AI</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Story Form ---
    with st.form("story_generator"):
        st.markdown('<div class="storify-form">', unsafe_allow_html=True)
        prompt = st.text_input("Enter a story prompt:", placeholder="A robot discovering emotions in a post-apocalyptic world")
        num_images = st.slider("Number of images to generate:", min_value=1, max_value=5, value=3)
        generate_button = st.form_submit_button("🚀 Generate Story")
        st.markdown('</div>', unsafe_allow_html=True)

    if generate_button and prompt:
        st.markdown('<div class="storify-card">', unsafe_allow_html=True)
        st.markdown('<span class="storify-badge">Your AI-Generated Story</span>', unsafe_allow_html=True)

        with st.spinner("Generating your story..."):
            try:
                import google.generativeai as genai
                api_key = "add your key here for gemini api"
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                story_prompt = f"""
                Create an engaging short story based on this prompt: {prompt}
                The story should:
                - Be approximately 400-600 words
                - Have a clear beginning, middle, and end
                - Include vivid descriptions that would work well for generating {num_images} distinct images
                - Divide the story into {num_images} sections with clear scene breaks
                - Include a short, descriptive image prompt at the end of each section enclosed in [IMAGE: prompt here]
                Make the story captivating and creative!
                """
                response = model.generate_content(story_prompt)
                story = response.text
                if story:
                    pattern = r'\[IMAGE:\s*(.*?)\]'
                    image_prompts = re.findall(pattern, story)
                    story_sections = story.split('[IMAGE:')
                    if len(story_sections) <= 1:
                        st.write(story)
                    else:
                        st.write(story_sections[0])
                        for i, section in enumerate(story_sections[1:], 1):
                            parts = section.split(']', 1)
                            if len(parts) < 2:
                                continue
                            image_prompt, section_text = parts
                            col1, col2 = st.columns([1.3, 2.7])
                            with col1:
                                HF_API_TOKEN = Add your key here
                                API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
                                headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
                                payload = {
                                    "inputs": image_prompt,
                                    "options": {"wait_for_model": True}
                                }
                                with st.spinner(f"Generating image for: '{image_prompt}'"):
                                    try:
                                        response = requests.post(API_URL, headers=headers, json=payload)
                                        if response.status_code == 200:
                                            st.image(response.content, caption=f"Scene {i}", use_column_width=True)
                                        else:
                                            st.error(f"Failed to generate image {i}: {response.status_code}")
                                    except Exception as e:
                                        st.error(f"Error during image generation: {str(e)}")
                            with col2:
                                st.markdown(f'<div class="storify-section-title">Scene {i}</div>', unsafe_allow_html=True)
                                st.write(section_text)
                else:
                    st.error("Generated story is empty. Please try again with a different prompt.")
            except Exception as e:
                st.error(f"Error generating content: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- User Tips & Footer ---
    st.markdown("""
    <div style='margin-top: 2.5rem; color: #64748b; font-size: 1.05rem; text-align: center;'>
        <span>🧑‍💻 <strong>Tips:</strong> Try creative prompts and adjust the image count for different experiences.<br>
        All images are AI-generated for each scene. Enjoy your story adventure! 📚✨</span>
    </div>
    """, unsafe_allow_html=True)

        
        


elif nav_option == "Expression Analysis":
    st.markdown("<div class='main-header'>Expression Analysis</div>", unsafe_allow_html=True)
    st.markdown("""
    Analyze facial expressions from your webcam in real-time or upload images for detailed emotion analysis.
    """)
    
    # Create tabs for different analysis methods
    tabs = st.tabs(["📷 Real-time Camera", "🖼️ Image Upload"])
    
    with tabs[0]:
        st.markdown("<div class='section-tab'>", unsafe_allow_html=True)
        st.markdown("### Real-time Expression Analysis")
        st.markdown("""
        Use your webcam to analyze facial expressions in real-time. 
        Make sure you have good lighting for the best results.
        """)
        
        # Camera controls
        camera_col1, camera_col2 = st.columns([1, 1])
        
        with camera_col1:
            start_camera = st.button("Start Camera" if not st.session_state.camera_active else "Stop Camera")
            
            if start_camera:
                st.session_state.camera_active = not st.session_state.camera_active
        
        # Camera feed and processing
        if st.session_state.camera_active:
            st.markdown("<div class='camera-feed'>", unsafe_allow_html=True)
            process_webcam()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='border: 2px dashed #BB86FC; border-radius: 15px; padding: 40px; text-align: center;'>
                <img src="https://www.svgrepo.com/show/513324/camera.svg" width="100">
                <p style="margin-top: 20px; font-size: 18px;">Click "Start Camera" to begin real-time expression analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("<div class='section-tab'>", unsafe_allow_html=True)
        st.markdown("### Image Upload Analysis")
        st.markdown("""
        Upload an image to analyze facial expressions. For best results, 
        choose images with clear, well-lit faces looking toward the camera.
        """)
        
        # Create a nice upload section
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image for expression analysis", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if st.button("Analyze Expression"):
                    with st.spinner("Analyzing facial expression..."):
                        # Convert PIL Image to numpy array for DeepFace
                        img_array = np.array(image)
                        
                        # Analyze the image
                        emotion_result = detect_emotion(img_array)
                        
                        if emotion_result:
                            st.session_state.image_emotion_results = emotion_result
                            
                            # Add to history
                            history_item = {
                                'dominant_emotion': emotion_result['dominant_emotion'],
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'type': 'image'
                            }
                            st.session_state.emotion_history.append(history_item)
                        else:
                            st.error("No faces detected or could not analyze emotions. Please try another image.")
            
            # Display results if available
            if st.session_state.image_emotion_results:
                st.markdown("<div class='emotion-result'>", unsafe_allow_html=True)
                results = st.session_state.image_emotion_results
                
                dominant_emotion = results['dominant_emotion']
                emotion_scores = results['emotion_scores']
                
                # Display dominant emotion with emoji
                emoji = get_emotion_emoji(dominant_emotion)
                st.markdown(f"""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <span style='font-size: 60px;'>{emoji}</span>
                    <h2>Primary Expression: {dominant_emotion.upper()}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Display all emotion scores as a horizontal bar chart
                st.markdown("### Expression Confidence Scores")
                
                # Sort emotions by score
                sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
                
                for emotion, score in sorted_emotions:
                    # Create a color-coded progress bar for each emotion
                    emotion_color = get_emotion_color(emotion)
                    emotion_emoji = get_emotion_emoji(emotion)
                    
                    st.markdown(f"""
                    <div style='margin-bottom: 15px;'>
                        <div style='display: flex; align-items: center; margin-bottom: 5px;'>
                            <span style='font-size: 24px; margin-right: 10px;'>{emotion_emoji}</span>
# ... [Previous code remains the same until the last line you provided] ...

                            <span style='font-weight: 500; font-size: 16px;'>{emotion.capitalize()}</span>
                            <span style='margin-left: auto; font-weight: 600;'>{score:.1f}%</span>
                        </div>
                        <div style='height: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                            <div style='height: 100%; width: {score}%; background-color: {emotion_color}; border-radius: 5px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add emotional context from Gemini
                try:
                    with st.spinner("Getting emotional context from Gemini..."):
                        prompt = f"""Provide a brief psychological analysis of someone showing {dominant_emotion} expression.
                        Include:
                        1. Possible reasons for this emotion
                        2. Typical body language cues
                        3. Suggestions for responding appropriately
                        
                        Keep it concise (3-4 sentences max)."""
                        
                        response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
                        
                        st.markdown("### Emotional Context")
                        st.markdown(f"""
                        <div style='background-color: #f5f5f5; padding: 15px; border-radius: 10px;'>
                            {response.text}
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning("Couldn't get additional context from Gemini API")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
elif nav_option == "Quiz":
    import random
    import time

    # --- Session State Initialization ---
    if 'quiz_level' not in st.session_state:
        st.session_state.quiz_level = 1
    if 'quiz_cards' not in st.session_state:
        st.session_state.quiz_cards = []
    if 'quiz_flipped' not in st.session_state:
        st.session_state.quiz_flipped = []
    if 'quiz_matched' not in st.session_state:
        st.session_state.quiz_matched = []
    if 'quiz_attempts' not in st.session_state:
        st.session_state.quiz_attempts = 0
    if 'quiz_start_time' not in st.session_state:
        st.session_state.quiz_start_time = time.time()

    # --- Emoji Pool ---
    EMOJI_POOL = [
        "🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼",
        "🦁", "🐮", "🐷", "🐸", "🐵", "🐤", "🦄", "🐙"
    ]

    def reset_quiz_game(level):
        num_pairs = min(2 + level, len(EMOJI_POOL))  # Increase pairs with level
        emojis = random.sample(EMOJI_POOL, num_pairs)
        cards = emojis * 2
        random.shuffle(cards)
        st.session_state.quiz_cards = cards
        st.session_state.quiz_flipped = []
        st.session_state.quiz_matched = []
        st.session_state.quiz_attempts = 0
        st.session_state.quiz_start_time = time.time()

    # --- Start/Restart Button ---
    st.markdown("""
    <style>
    .memory-title {font-size:2.2rem;font-weight:800;color:#2563eb;}
    .memory-sub {font-size:1.1rem;color:#64748b;}
    .stButton>button {font-size:1.2rem;font-weight:600;border-radius:10px;background:linear-gradient(90deg,#2563eb 80%,#38bdf8 100%)!important;color:#fff!important;}
    .stButton>button:hover {background:linear-gradient(90deg,#1e40af 80%,#0ea5e9 100%)!important;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="memory-title">🧠 Memory Match Game</div>', unsafe_allow_html=True)
    st.markdown('<div class="memory-sub">Flip the cards and find all the matching pairs! The game adapts to your progress.</div>', unsafe_allow_html=True)

    if st.button("🔄 Start New Game"):
        reset_quiz_game(st.session_state.quiz_level)

    if not st.session_state.quiz_cards:
        reset_quiz_game(st.session_state.quiz_level)

    cards = st.session_state.quiz_cards
    flipped = st.session_state.quiz_flipped
    matched = st.session_state.quiz_matched

    # --- Game Board ---
    n = len(cards)
    cols = 4 if n > 6 else 2
    rows = (n + cols - 1) // cols

    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx >= n:
                continue
            card_id = f"quiz_card_{idx}_{cards[idx]}"
            if idx in matched or idx in flipped:
                columns[col].button(cards[idx], key=card_id, disabled=True, help="Matched!" if idx in matched else "Flipped!")
            else:
                if columns[col].button("❓", key=card_id):
                    flipped.append(idx)
                    if len(flipped) == 2:
                        st.session_state.quiz_attempts += 1
                        i1, i2 = flipped
                        if cards[i1] == cards[i2]:
                            matched.extend([i1, i2])
                        time.sleep(0.5)
                        st.session_state.quiz_flipped = []

    # --- Progress and Feedback ---
    st.markdown("---")
    matched_pairs = len(matched) // 2
    total_pairs = len(cards) // 2
    st.success(f"Matches found: {matched_pairs} / {total_pairs}")
    st.info(f"Attempts: {st.session_state.quiz_attempts}")

    # --- Win Condition ---
    if matched_pairs == total_pairs:
        elapsed = int(time.time() - st.session_state.quiz_start_time)
        st.balloons()
        st.markdown(f"## 🎉 Well Done! You completed Level {st.session_state.quiz_level} in {st.session_state.quiz_attempts} attempts and {elapsed} seconds.")
        if st.button("Next Level 🚀"):
            st.session_state.quiz_level += 1
            reset_quiz_game(st.session_state.quiz_level)
        if st.button("Restart at Level 1"):
            st.session_state.quiz_level = 1
            reset_quiz_game(1)

    # --- Caregiver Tracking ---
    with st.expander("📊 Caregiver Progress Tracking"):
        st.write(f"**Current Level:** {st.session_state.quiz_level}")
        st.write(f"**Attempts this round:** {st.session_state.quiz_attempts}")
        st.write(f"**Time taken:** {int(time.time() - st.session_state.quiz_start_time)} seconds")
        st.write(f"**Total pairs:** {total_pairs}")

    st.markdown("""
    ---
    *This game is designed to improve memory, attention, and matching skills. Caregivers can use the progress data to track cognitive improvements over time.*
    """)


# Add footer
st.markdown("""
<div class='footer'>
    NeuroAI• Advanced Cognitive Support • © 2025
</div>
""", unsafe_allow_html=True)

# Initialize Gemini (if not already done)
if 'genai_configured' not in st.session_state:
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        st.session_state.genai_configured = True
    except Exception as e:
        st.error(f"Failed to configure Gemini: {str(e)}")