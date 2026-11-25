import os
import streamlit as st

st.set_page_config(page_title="Project Report: Hand Pose Detection", page_icon="📄", layout="wide")

# ---------- EDIT THESE ----------
TEAM = "[Team member names / roll numbers]"
INSTITUTE = "[Institute / Internship name]"
PPT_LINK = "https://github.com/your-username/your-repo/blob/main/presentation.pptx"
VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"        # project explanation video (placeholder)
SCREEN_REC_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"   # PPT screen recording (placeholder)
GITHUB_URL = "https://github.com/lmb-freiburg/freihand"
# --------------------------------

st.markdown("""
<style>
.hero{background:linear-gradient(135deg,#0f3d33,#1f6f5c 60%,#2f9c80);color:#fff;padding:34px 30px;border-radius:16px;margin-bottom:14px}
.hero h1{color:#fff;margin:0 0 6px;font-size:2.1rem;line-height:1.2}
.hero p{color:#d8efe7;margin:4px 0}
.pill{display:inline-block;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.35);border-radius:999px;padding:3px 14px;margin:8px 8px 0 0;font-size:.9rem}
.card{border:1px solid rgba(128,128,128,.3);border-radius:12px;padding:14px 16px;height:100%}
.card b{color:#1f9c7c}
</style>
<div class="hero">
<h1>3D Hand Pose Detection from a Single Photo</h1>
<p>Project report: an interactive Streamlit application inspired by FreiHAND (Zimmermann et al., ICCV 2019)</p>
<span class="pill">21 keypoints</span><span class="pill">2D + 3D</span><span class="pill">MediaPipe Hand Landmarker</span><span class="pill">Streamlit</span>
<p style="margin-top:14px;font-size:.95rem">Team: """ + TEAM + """ &nbsp;·&nbsp; """ + INSTITUTE + """</p>
</div>
""", unsafe_allow_html=True)


def show_img(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Add a screenshot at `{path}` in your repository to show: {caption}")


tabs = st.tabs(["1 Introduction", "2 Literature", "3 Methodology", "4 Experimental setup",
                "5 Results", "6 Conclusion", "7 References", "8 PPT", "9 Video"])

# ---------------- 1. INTRODUCTION ----------------
with tabs[0]:
    st.header("1. Introduction")
    st.subheader("Motivation")
    st.write("Knowing where every finger joint of a hand is in 3D, from just one ordinary photo, lets "
             "computers understand gestures and hand actions. Learning this needs large amounts of hand images "
             "with accurate joint labels. Labels are hard to get because fingers hide each other, and the FreiHAND "
             "paper reports that marking all 21 joints in all camera views takes about 15 minutes per sample.")
    st.subheader("Applications")
    c = st.columns(4)
    for col, t in zip(c, ["Gesture recognition", "Robotics and teleoperation", "AR / VR hand tracking", "Sign language and HCI"]):
        col.markdown(f'<div class="card"><b>{t}</b></div>', unsafe_allow_html=True)
    st.subheader("Problem statement")
    st.write("Given a single RGB image of a hand, estimate the positions of 21 hand keypoints (wrist plus four joints on each "
             "of five fingers) in 2D on the image and in 3D, and present them in a simple web application.")
    st.subheader("Contributions")
    st.markdown("""
- **Studied the FreiHAND paper and toolbox:** a real-image dataset with 3D pose and hand shape labels, built with 8 cameras and a human-in-the-loop labeling loop.
- **Clarified what the repository does:** it is a dataset and evaluation toolbox and does not include a trained detector.
- **Built and deployed a working web app:** upload or capture a photo, see colored 2D keypoints, a 3D keypoint plot, a per-joint table and a JSON download.
- **Used the FreiHAND joint order,** so outputs follow the same convention as the dataset.
- **Documented deployment problems and fixes** on Streamlit Cloud (Section 4).
""")

# ---------------- 2. LITERATURE ----------------
with tabs[1]:
    st.header("2. Brief Literature Review")
    st.write("**Synthetic datasets** (RHD, SynthHands, GANerated) are cheap to label but their images and poses differ from real "
             "ones. **Marker or sensor datasets** (for example FPA with magnetic sensors) give exact labels but change how hands look. "
             "**Manually labeled datasets** (STB) cover only visible joints and show limited variety. **Multi-view bootstrapping** "
             "(Simon et al.) triangulates 2D detections from many cameras to produce 3D labels.")
    st.write("FreiHAND extends this idea: it records 8 synchronized cameras, fits the deformable **MANO** hand model to sparse labels, and "
             "uses a multi-view network (MVNet) with confidence scores to speed up labeling. The authors show that networks trained "
             "on a single existing dataset generalize poorly to other datasets, while training on FreiHAND generalizes better on average.")
    st.write("Our application uses a separate, pretrained, lightweight detector (MediaPipe Hand Landmarker) that predicts 21 "
             "keypoints from one image. It is not trained on FreiHAND.")
    st.caption("Full references are in Section 7.")

# ---------------- 3. METHODOLOGY ----------------
with tabs[2]:
    st.header("3. Methodology")
    st.subheader("3.1 Our application pipeline")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8efec", fontname="Helvetica"];
      A [label="Photo upload\\nor camera"]; B [label="Preprocess\\n(fix rotation, RGB)"];
      C [label="MediaPipe\\nHand Landmarker"]; D [label="21 keypoints\\n(2D + 3D)"];
      E [label="2D overlay\\n(Pillow)"]; F [label="3D plot\\n(Matplotlib)"]; G [label="Table + JSON\\ndownload"];
      A -> B -> C -> D; D -> E; D -> F; D -> G; }
    """)
    st.write("The detector returns normalized 2D landmarks (x, y in 0 to 1 of image width and height) and 3D world landmarks in metres, "
             "relative to the hand's center. To draw on the photo, normalized coordinates are converted to pixels:")
    st.latex(r"u_k = x_k \cdot W, \qquad v_k = y_k \cdot H")
    st.write("Joint order (same as FreiHAND): wrist = 0, thumb = 1 to 4, index = 5 to 8, middle = 9 to 12, ring = 13 to 16, pinky = 17 to 20.")

    st.subheader("3.2 How the FreiHAND dataset was created (from the paper)")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8efec", fontname="Helvetica"];
      A [label="8-camera capture\\n(green screen)"]; B [label="Sparse labels:\\n6 keypoints + masks"];
      C [label="MANO fitting"]; D [label="Heuristic or\\nmanual check"]; E [label="Dataset D_i"];
      F [label="Train MVNet +\\nHandSegNet"]; A -> B -> C -> D -> E -> F; F -> C [label="repeat (4 times)", fontsize=10]; }
    """)
    st.write("MANO is a statistical hand model with 61 parameters: 10 for shape, 45 for finger articulation and 6 for global position and rotation. "
             "The fitting minimizes a sum of five terms:")
    st.latex(r"\mathcal{L} = \mathcal{L}_{kp}^{2D} + \mathcal{L}_{kp}^{3D} + \mathcal{L}_{seg} + \mathcal{L}_{shape} + \mathcal{L}_{pose}")
    st.latex(r"\mathcal{L}_{kp}^{2D} = w_{2D}\sum_i \sum_{k\in V_i} \left\| q_k^i - \Pi^i(\tilde{p}_k) \right\|_2 ,\qquad "
             r"\mathcal{L}_{kp}^{3D} = w_{3D}\sum_i \sum_{k\in V_i} \left\| p_k - \tilde{p}_k \right\|_2")
    st.write("Here q are the manually marked 2D points, Π projects a 3D point into camera view i, and p are 3D keypoints predicted by MVNet. "
             "The segmentation term compares the model's silhouette with the hand mask, and the shape and pose terms keep the hand close to likely shapes and poses.")
    st.subheader("3.3 MVNet and the confidence score")
    st.write("MVNet unprojects the features of all 8 views into a common 3D volume and predicts one voxel heatmap per keypoint:")
    st.latex(r"\mathcal{L}_{MVNet} = \frac{1}{K}\sum_k \left\| S_k - \tilde{S}_k \right\|_2 ,\qquad "
             r"c = \frac{1}{K}\sum_k \max_{i,j,l} \tilde{S}_k(i,j,l)")
    st.write("A sample is accepted automatically when the mean confidence is above 0.8, all individual keypoint confidences are at least 0.6, "
             "the mask overlap is high, and the fit agrees with the predicted keypoints to within 0.5 cm on average.")
    st.subheader("3.4 Single-image pose formulation")
    st.latex(r"p_k = s\,\hat{p}_k = s\begin{pmatrix}\hat{x}_k\\ \hat{y}_k\\ \hat{z}_k^{rel}+\hat{z}^{root}\end{pmatrix}")
    st.write("Because one photo cannot tell the true size, the pose is predicted root-relative and scale-normalized (s is the length of a reference bone). "
             "This is why the 3D values in our app describe the hand's shape but not its distance from the camera.")

# ---------------- 4. EXPERIMENTAL SETUP ----------------
with tabs[3]:
    st.header("4. Experimental Setup")
    st.subheader("Baseline algorithms")
    st.write("In the paper, the same single-view network (Iqbal et al., 2.5D heatmap regression) is trained on each dataset in turn, so training sets are compared fairly. "
             "For shape, two baselines are used: the constant mean shape, and a MANO fit to the network's predicted keypoints. "
             "In our application, MediaPipe Hand Landmarker is the detector.")
    st.subheader("Datasets")
    st.table({
        "Dataset": ["STB", "PAN", "FPA", "LSMV", "RHD", "GAN", "HO-3D", "FreiHAND"],
        "Frames (train / eval)": ["15k / 3k", "641k / 34k", "52k / 53k", "117k / 31k", "41k / 2.7k", "266k / 66k", "11k / -", "33k / 4k"],
        "Subjects": ["1", ">10", "6", "21", "20", "-", "3", "32"],
        "Real": ["yes", "yes", "yes", "yes", "no", "no", "yes", "yes"],
        "Objects": ["no", "yes", "yes", "no", "no", "yes", "yes", "yes"],
        "Shape labels": ["no", "no", "no", "no", "no", "no", "yes", "yes"],
    })
    st.caption("Table from the FreiHAND paper. The repository README lists 32,560 unique training samples and 3,960 evaluation samples.")
    st.subheader("Implementation platform")
    st.table({
        "Item": ["Language", "Web framework", "Detector", "Drawing / plots", "Deployment", "Python on server", "Pinned packages"],
        "Choice": ["Python", "Streamlit", "MediaPipe Hand Landmarker (downloaded model file)", "Pillow, Matplotlib",
                   "Streamlit Community Cloud from GitHub", "3.14", "streamlit 1.67.0, mediapipe 1.2.3, opencv-python-headless 5.1.0.95, numpy 2.5.5, matplotlib 3.11.2, pillow 13.0.0"],
    })
    st.subheader("Deployment problems we solved")
    st.markdown("""
| Problem | Cause | Fix |
|---|---|---|
| App ignored packages | Requirements file named `requirements (1).txt` | Renamed to `requirements.txt` |
| MediaPipe 0.10.14 would not install | No build for Python 3.14 | Moved to the newer MediaPipe Tasks interface |
| `libEGL.so.1` missing | Server lacks graphics libraries | Added `packages.txt` with `libgl1`, `libegl1`, `libgles2` |
| `libgthread-2.0.so.0` missing after 2 days | Rebuild picked up new package versions | Added `libglib2.0-0t64` and pinned package versions |
""")
    st.write("Lesson: unpinned packages can change between rebuilds, so pinning versions makes the app reproducible.")

# ---------------- 5. RESULTS ----------------
with tabs[4]:
    st.header("5. Results")
    st.subheader("5.1 Quantitative results from the FreiHAND paper")
    st.write("Cross-dataset generalization, measured as area under the PCK curve. Rows are training sets; columns are evaluation sets.")
    st.table({
        "Train \\ Eval": ["STB", "RHD", "GAN", "PAN", "LSMV", "FPA", "HO-3D", "FreiHAND"],
        "STB": [0.783, 0.362, 0.110, 0.459, 0.086, 0.119, 0.154, 0.473],
        "RHD": [0.179, 0.767, 0.103, 0.316, 0.209, 0.095, 0.130, 0.518],
        "GAN": [0.067, 0.184, 0.765, 0.136, 0.152, 0.084, 0.091, 0.217],
        "PAN": [0.141, 0.463, 0.092, 0.870, 0.189, 0.120, 0.111, 0.562],
        "LSMV": [0.072, 0.544, 0.206, 0.320, 0.717, 0.118, 0.149, 0.537],
        "FPA": [0.061, 0.101, 0.180, 0.184, 0.129, 0.777, 0.073, 0.128],
        "HO-3D": [0.138, 0.450, 0.087, 0.351, 0.251, 0.106, "-", 0.557],
        "FreiHAND (eval)": [0.138, 0.508, 0.183, 0.407, 0.276, 0.163, 0.169, 0.678],
        "Avg. rank": [6.0, 2.9, 5.4, 3.0, 4.1, 6.0, 6.1, 2.2],
    })
    st.caption("Same layout as Table 1 of the paper. The HO-3D evaluation set was unavailable at submission, so the paper's HO-3D column uses its training split.")
    st.subheader("Shape estimation on the FreiHAND evaluation split")
    st.table({"Method": ["Mean shape", "MANO fit to keypoints", "MANO CNN"],
              "Mesh error (lower is better)": [1.78, 1.45, 1.16],
              "F@5mm": [0.300, 0.415, 0.484], "F@15mm": [0.808, 0.884, 0.925]})
    st.subheader("Effect of iterative labeling")
    st.table({"Iteration": ["D0", "D1", "D2", "D3", "D4"], "Samples": [302, 993, 1449, 2609, 4565],
              "AUC on RHD": [0.244, 0.453, 0.493, 0.511, 0.518], "AUC on PAN": [0.347, 0.521, 0.521, 0.539, 0.562]})

    st.subheader("5.2 Our application: quantitative results")
    st.write("Fill this table from your own tests (try at least 10 photos in different conditions) and report the measured detection rate.")
    st.table({"Test condition": ["Open hand, plain background", "Fist", "Fingers pointing at camera", "Holding an object", "Dim light", "Two hands", "Blurry photo"],
              "Photos tested": ["[ ]"] * 7, "Hand detected": ["[ ]"] * 7, "Notes": ["[ ]"] * 7})

    st.subheader("5.3 Qualitative results")
    c1, c2 = st.columns(2)
    with c1:
        show_img("assets/result_2d.png", "App output: 2D keypoints drawn on the uploaded photo")
    with c2:
        show_img("assets/result_3d.png", "App output: 3D keypoint plot")
    show_img("assets/result_table.png", "App output: joint table and JSON download")

    st.subheader("5.4 Discussion and observations")
    st.markdown("""
- In the paper, each dataset scores highest on its own test split and lower on others. FreiHAND has the best average rank (2.2) but does not win every column. Training on FPA transfers worst, because the sensors and wires on the hands change their look.
- Predicting MANO parameters directly with a CNN beats fitting MANO to predicted keypoints, and the gap is larger at the strict 5 mm threshold.
- Accuracy rose with every labeling iteration, so the human-in-the-loop cycle improves data quality as it grows.
- [Add your own observations from using the app: which poses worked well, which did not.]
""")
    st.subheader("5.5 Failure cases")
    st.markdown("""
The following are likely failure modes for single-photo hand detection. Confirm each one with your own test photos and add screenshots:
- **No hand found:** small, blurry, very dark or heavily cropped hands.
- **Heavy occlusion:** fingers hidden behind each other or behind objects, where joint positions are guessed.
- **Unreliable depth:** the 3D values come from one flat photo, so they are less accurate than the 2D points.
- **Wrong left/right label** on unusual poses or mirrored images.
- **Not a full hand mesh:** the app outputs a 21-point skeleton, not the MANO surface described in the paper.
""")
    show_img("assets/failure_case.png", "Failure case from our testing")

# ---------------- 6. CONCLUSION ----------------
with tabs[5]:
    st.header("6. Conclusion")
    st.subheader("Summary")
    st.write("FreiHAND provides a large real-image dataset with 3D hand pose and shape labels, created with multi-view capture, MANO fitting and a "
             "human-in-the-loop loop, and shows better cross-dataset generalization than existing datasets. Its repository is a dataset and "
             "evaluation toolbox, not a detector. We built and deployed a Streamlit application that detects 21 hand keypoints from an uploaded photo, "
             "shows them in 2D and 3D, and exports them as JSON, using a pretrained MediaPipe model that follows FreiHAND's joint order.")
    st.subheader("Future work")
    st.markdown("""
- Train or fine-tune a network on FreiHAND and compare it with MediaPipe on the FreiHAND evaluation split.
- Predict the full MANO hand mesh and render it, as in the paper.
- Add live webcam and video processing.
- Measure detection quality on a labeled test set and report standard metrics such as PCK.
""")

# ---------------- 7. REFERENCES ----------------
with tabs[6]:
    st.header("7. References")
    st.markdown(f"""
1. C. Zimmermann, D. Ceylan, J. Yang, B. Russell, M. Argus, T. Brox. *FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images.* ICCV 2019. Code: [{GITHUB_URL}]({GITHUB_URL})
2. J. Romero, D. Tzionas, M. J. Black. *Embodied Hands: Modeling and Capturing Hands and Bodies Together.* ACM TOG, 2017.
3. T. Simon, H. Joo, I. Matthews, Y. Sheikh. *Hand Keypoint Detection in Single Images using Multiview Bootstrapping.* CVPR 2017.
4. U. Iqbal, P. Molchanov, T. Breuel, J. Gall, J. Kautz. *Hand Pose Estimation via Latent 2.5D Heatmap Regression.* ECCV 2018.
5. C. Zimmermann, T. Brox. *Learning to Estimate 3D Hand Pose from Single RGB Images.* ICCV 2017.
6. G. Garcia-Hernando et al. *First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations.* CVPR 2018.
7. MediaPipe Hand Landmarker, Google (model used by our application).
8. Streamlit documentation: https://docs.streamlit.io
""")

# ---------------- 8. PPT ----------------
with tabs[7]:
    st.header("8. Link to PPT File")
    st.markdown(f"[Open the presentation on GitHub]({PPT_LINK})")

# ---------------- 9. VIDEO ----------------
with tabs[8]:
    st.header("9. Video Demonstration and PPT Screen Recording")
    st.subheader("Project demonstration video")
    st.video(VIDEO_URL)
    st.subheader("Screen recording of the PPT")
    st.video(SCREEN_REC_URL)
    st.caption("Replace the placeholder video links at the top of this file with your own.")
