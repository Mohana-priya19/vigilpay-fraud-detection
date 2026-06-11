import streamlit as st
import cv2
import numpy as np
from mediapipe import solutions as mp
import pyttsx3
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RehabAI", layout="wide")

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #4facfe, #00f2fe);
}
h1, h2, h3 {
    color: white;
}
.stButton>button {
    background-color: #ff4b2b;
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- VOICE ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- MEDIAPIPE ----------------
mp_drawing = mp.drawing_utils
mp_pose = mp.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "login"

# ---------------- LOGIN PAGE ----------------
if st.session_state.page == "login":

    st.title("🏥 RehabAI Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        st.session_state.page = "dashboard"

# ---------------- DASHBOARD ----------------
elif st.session_state.page == "dashboard":

    st.title("📊 RehabAI Dashboard")

    st.markdown("### Welcome back! 👋")

    # Progress Section
    st.subheader("Today's Progress")

    progress = 0.6  # example
    st.progress(progress)

    st.write(f"Completed: {int(progress*100)}%")

    st.subheader("Today's Goal")
    st.write("Complete 20 reps of Arm Exercise")

    if st.button("▶ Start Exercise"):
        st.session_state.page = "exercise"

# ---------------- EXERCISE PAGE ----------------
elif st.session_state.page == "exercise":

    st.title("💪 Arm Exercise Session")

    st.info("Get ready... Starting camera")

    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None
    last_feedback = ""
    last_spoken_time = 0
    movement_started = False

    frame_placeholder = st.empty()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]

                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle < 60:
                    feedback = "Increase range"
                    voice = "Raise your arm higher"
                elif angle > 120:
                    feedback = "Reduce range"
                    voice = "Do not lift too high"
                else:
                    feedback = "Good movement"
                    voice = "Good job"

                if angle > 20:
                    movement_started = True

                current_time = time.time()

                if movement_started:
                    if feedback != last_feedback or current_time - last_spoken_time > 5:
                        speak(voice)
                        last_feedback = feedback
                        last_spoken_time = current_time

                if angle < 60:
                    stage = "down"

                if angle > 120 and stage == "down":
                    stage = "up"
                    counter += 1

                cv2.putText(image, f'Reps: {counter}',
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0), 2)

            except:
                pass

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            frame_placeholder.image(image, channels="BGR")

    cap.release()

    st.success(f"Session Completed! Reps: {counter}")

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"