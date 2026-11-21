import json
import os
import urllib.request

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageOps

st.set_page_config(page_title="Hand Pose Detection", page_icon="🖐️", layout="wide")

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "/tmp/hand_landmarker.task"
COLORS = [(225, 29, 72), (245, 158, 11), (16, 185, 129), (59, 130, 246), (168, 85, 247)]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
JOINT_NAMES = ["wrist"] + [f"{f}_{j}" for f in FINGERS for j in ("mcp", "pip", "dip", "tip")]

st.title("🖐️ Hand Pose Detection: Upload a Photo")
st.caption(
    "Project context: FreiHAND dataset and toolbox (github.com/lmb-freiburg/freihand). "
    "Detector: MediaPipe Hand Landmarker, which is not trained on FreiHAND. "
    "Joint order matches FreiHAND: wrist, then thumb, index, middle, ring, pinky with four joints each."
)


@st.cache_resource(show_spinner="Loading the detection model...")
def get_detector(min_conf):
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(opts)


def draw(img, hands):
    out = img.copy()
    d = ImageDraw.Draw(out)
    W, H = out.size
    t = max(2, W // 250)
    for hand in hands:
        pts = [(p.x * W, p.y * H) for p in hand]
        for fi in range(5):
            ids = [0] + [1 + fi * 4 + k for k in range(4)]
            d.line([pts[i] for i in ids], fill=COLORS[fi], width=t)
        for i, (x, y) in enumerate(pts):
            col = (255, 255, 255) if i == 0 else COLORS[(i - 1) // 4]
            r = t * 2
            d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(0, 0, 0))
    return out


src = st.radio("Input", ["Upload a photo", "Use camera"], horizontal=True)
f = (st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
     if src == "Upload a photo" else st.camera_input("Take a photo"))
min_conf = st.slider("Minimum detection confidence", 0.1, 0.9, 0.4, 0.05)

if f is None:
    st.info("Upload a hand photo to begin.")
else:
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    try:
        det = get_detector(min_conf)
    except Exception as e:
        st.error(f"Could not load the detection model: {e}")
        st.stop()
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img)))
    if not res.hand_landmarks:
        st.warning("No hand found. Try a clearer, closer photo with the whole hand visible.")
        st.image(img, use_container_width=True)
    else:
        n = len(res.hand_landmarks)
        st.success(f"Detected {n} hand{'s' if n > 1 else ''}.")
        c1, c2 = st.columns([1.3, 1])
        c1.image(draw(img, res.hand_landmarks), caption="2D keypoints", use_container_width=True)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        for hand in res.hand_world_landmarks:
            P = np.array([[p.x, p.z, -p.y] for p in hand])
            for fi in range(5):
                ids = [0] + [1 + fi * 4 + k for k in range(4)]
                ax.plot(P[ids, 0], P[ids, 1], P[ids, 2], color=np.array(COLORS[fi]) / 255, lw=2)
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="k", s=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("y (m)")
        c2.pyplot(fig)
        c2.caption("3D keypoints (estimated, metres, centered on the hand).")

        W, H = img.size
        data = {"image_size": [W, H], "joint_names": JOINT_NAMES, "hands": []}
        for i, (lm, wl) in enumerate(zip(res.hand_landmarks, res.hand_world_landmarks)):
            cat = res.handedness[i][0]
            data["hands"].append({
                "handedness": cat.category_name, "score": cat.score,
                "keypoints_2d_px": [[p.x * W, p.y * H] for p in lm],
                "keypoints_3d_m": [[p.x, p.y, p.z] for p in wl],
            })
            with st.expander(f"Hand {i+1}: {cat.category_name} ({cat.score:.0%}) joint table"):
                st.dataframe({"joint": JOINT_NAMES,
                              "x_px": [round(p.x * W, 1) for p in lm],
                              "y_px": [round(p.y * H, 1) for p in lm],
                              "x_m": [round(p.x, 4) for p in wl],
                              "y_m": [round(p.y, 4) for p in wl],
                              "z_m": [round(p.z, 4) for p in wl]})
        st.download_button("Download keypoints (JSON)", json.dumps(data, indent=1),
                           "keypoints.json", "application/json")

st.markdown("---")
st.caption("The 3D values are model estimates and less reliable than the 2D positions. "
           "This app predicts keypoints only, not the MANO hand mesh described in the FreiHAND paper.")
