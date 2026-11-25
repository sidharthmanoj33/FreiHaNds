import os
import streamlit as st

st.set_page_config(page_title="Project Report: Hand Detection (Classical CV)", page_icon="📄", layout="wide")

# ---------- EDIT THESE ----------
TEAM = "Sidharth Manoj (264152030), Anna Pao (264152003), Sidharth Das (264152015)"
INSTITUTE = "Indian Institute of Technology Guwahati"
PPT_LINK = "https://github.com/your-username/your-repo/blob/main/presentation.pptx"
VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"        # project explanation video (placeholder)
SCREEN_REC_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"   # PPT screen recording (placeholder)
GITHUB_URL = "https://github.com/lmb-freiburg/freihand"

# Add one dict per photo you test. Put the screenshot in assets/ and set "img".
TESTS = [
    {"name": "Test 1: two fingers raised (V sign), other fingers folded, plain white background, forearm reaches the image edge",
     "expected": 2, "v1": 3, "v2": 2, "img": "assets/v_sign_new.png"},
]
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
<h1>Hand Detection with Classical Computer Vision</h1>
<p>Project report: finding the hand, palm and fingertips in a single photo without any neural network, built with OpenCV and NumPy in a Streamlit application</p>
<span class="pill">No deep learning</span><span class="pill">YCrCb skin segmentation</span><span class="pill">Mean shift</span><span class="pill">Convex hull + defects</span><span class="pill">Distance transform</span><span class="pill">Streamlit</span>
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
    st.write("Locating a hand and its fingers in an image is the first step of gesture recognition and many human-computer interaction systems. "
             "Today this is usually done with deep networks that need large labeled datasets such as FreiHAND. "
             "Before such networks, hands were found with hand-designed image processing steps. These methods need no training data, "
             "run quickly on a normal CPU and every step can be inspected and explained. We rebuilt our hand detection application using only these classical steps, "
             "to understand how far they go and where they break.")
    st.subheader("Applications")
    c = st.columns(4)
    for col, t in zip(c, ["Gesture recognition", "Finger counting", "Touch-free interfaces", "Teaching computer vision"]):
        col.markdown(f'<div class="card"><b>{t}</b></div>', unsafe_allow_html=True)
    st.subheader("Problem statement")
    st.write("Given one RGB photo that contains a hand, find the hand region, its palm centre and size, the fingertips and the valleys between fingers, "
             "using only classical computer vision (no CNN or other learned model), and present the result in a simple web application.")
    st.subheader("What changed from our first version")
    st.markdown("""
Our first application used a pretrained MediaPipe Hand Landmarker to predict 21 keypoints in 2D and 3D. The new version replaces it completely:

| | First version | This version |
|---|---|---|
| Method | Pretrained neural network | Hand-designed image processing |
| Output | 21 keypoints, 2D and 3D | Hand outline, palm circle, up to 5 fingertips, finger valleys (2D only) |
| Training data | Learned model, downloaded on first run | None needed |
| Dependencies | streamlit, mediapipe, opencv, numpy, matplotlib, pillow | streamlit, opencv-python-headless, numpy, pillow |
| Controls | Photo or camera | Photo upload only, fixed settings |
""")
    st.subheader("Contributions")
    st.markdown("""
- **Built a complete classical pipeline:** skin-colour segmentation, mean shift, contour, convex hull, convexity defects and distance transform.
- **Found and fixed two real failure modes** in our first classical version (duplicate fingertips and the forearm), described in Sections 3.6 and 5.
- **Simplified the web app:** upload a photo, see the mask and the annotated result, read the finger count and download the results as JSON.
- **Documented what the method cannot do** (Section 5.4), so the results are not over-claimed.
""")

# ---------------- 2. LITERATURE ----------------
with tabs[1]:
    st.header("2. Brief Literature Review")
    st.write("**Skin-colour segmentation.** Skin pixels cluster in a small region of chrominance space. Chai and Ngan showed that a simple box on the Cr and Cb channels of "
             "YCrCb (Cr 133 to 173, Cb 77 to 127) separates skin from most backgrounds, and Phung et al. compared many skin classifiers. "
             "Dropping the brightness channel Y makes the rule less sensitive to lighting. Its weakness is that anything skin-coloured is also selected: faces, arms and some wood or walls.")
    st.write("**Mean shift.** Comaniciu and Meer described mean shift as a way to climb to the densest point of a distribution. "
             "Applied to a skin mask, a window moved by mean shift settles on the biggest cluster of skin pixels, which is a cheap way to guess where the hand is.")
    st.write("**Contours, convex hull and convexity defects.** Border following (Suzuki and Abe) turns a binary blob into an outline. "
             "The convex hull is the smallest convex shape around it. The gaps between the outline and the hull are convexity defects. "
             "For a spread hand, hull corners far from the palm are fingertips and deep defects are the valleys between fingers.")
    st.write("**Distance transform.** Borgefors described fast distance transforms. Each pixel of the blob gets its distance to the nearest background pixel. "
             "The maximum is the centre of the largest inscribed circle, which for a hand is a good estimate of the palm centre and size.")
    st.write("**Learned alternatives.** FreiHAND (Zimmermann et al., ICCV 2019) and detectors such as MediaPipe predict 21 joints in 3D and cope with occlusion, "
             "but need training data and a trained model. Our new version is not trained on FreiHAND and does not compete with them on accuracy; it shows what rules alone can do.")
    st.caption("Full references are in Section 7.")

# ---------------- 3. METHODOLOGY ----------------
with tabs[2]:
    st.header("3. Methodology")
    st.subheader("3.1 Pipeline")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8efec", fontname="Helvetica"];
      A [label="Photo upload"]; B [label="Resize to 640 px\\n+ fix rotation"]; C [label="Skin mask\\n(YCrCb + morphology)"];
      D [label="Skin regions\\n(contours)"]; E [label="Mean shift\\npicks hand region"]; F [label="Wrist cut"];
      G [label="Distance transform\\n(palm centre, radius)"]; H [label="Convex hull +\\ndefects"]; I [label="Fingertips,\\nvalleys, JSON"];
      A -> B -> C -> D -> E -> F -> G -> H -> I; }
    """)

    st.subheader("3.2 Skin segmentation")
    st.write("The image is blurred with a 5 by 5 Gaussian filter and converted to YCrCb. A pixel is skin when its Cr and Cb values fall inside fixed ranges; Y (brightness) is ignored:")
    st.latex(r"Cr = 0.713\,(R - Y) + 128,\quad Cb = 0.564\,(B - Y) + 128,\quad Y = 0.299R + 0.587G + 0.114B")
    st.latex(r"S(x,y) = \begin{cases}1 & 133 \le Cr \le 173 \ \text{and}\ 77 \le Cb \le 127\\ 0 & \text{otherwise}\end{cases}")
    st.write("Morphological opening (erosion then dilation, elliptical 5 by 5 kernel) removes small specks, and closing (applied twice) fills small holes. "
             "The outlines of the connected skin blobs are traced, blobs smaller than 1% of the image are dropped, and the 4 largest are kept.")

    st.subheader("3.3 Choosing the hand region with mean shift")
    st.write("A window starting at the centre of the image (half the image width and height) is moved to the centroid of the mask values inside it, again and again, until it moves less than 0.5 pixels or 30 steps are done:")
    st.latex(r"x_{t+1} = \frac{\sum_{p \in W_t} p\,S(p)}{\sum_{p \in W_t} S(p)}")
    st.write("The skin region that contains (or is closest to) the centre of the final window is taken as the hand. The app no longer asks the user to pick a region.")

    st.subheader("3.4 Palm centre and radius")
    st.write("The chosen region is filled in and its Euclidean distance transform is computed:")
    st.latex(r"D(p) = \min_{q \notin \text{hand}} \lVert p - q \rVert_2,\qquad c = \arg\max_p D(p),\quad r = \max_p D(p)")
    st.write("The point c is the palm centre and r the palm radius. All later thresholds are multiples of r, so they scale with the size of the hand in the photo.")

    st.subheader("3.5 Fingertips and valleys")
    st.write("The convex hull of the contour is computed. A hull point h is a fingertip candidate if it is far from the palm centre and not on the image border:")
    st.latex(r"\lVert h - c \rVert_2 \ \ge\ \lambda\, r,\qquad \lambda = 1.6")
    st.write("Candidates are visited from farthest to nearest and at most 5 are kept. Valleys come from convexity defects: a defect point f is a valley when its depth is large enough:")
    st.latex(r"\text{depth}(f) > 0.4\,r \quad\text{and}\quad \lVert f - c \rVert_2 > 0.9\,r")

    st.subheader("3.6 Two fixes that made the method reliable")
    st.write("In our first classical version, a photo of two raised fingers (V sign) was reported as 3 fingers. Two causes were found and fixed.")
    st.markdown("""
**Fix 1: duplicate points on one fingertip.** A rounded fingertip has several hull points, and a finger is about as wide as the old spacing rule (0.35 r), so one finger could give two tips.
A new candidate j is now rejected when it lies on the same fingertip as an already accepted tip i. Two points are on the same fingertip when the contour between them never dips back towards the palm. With d the distance of each contour point to c:
""")
    st.latex(r"\max\!\big(\min_{\text{arc}_1} d,\ \min_{\text{arc}_2} d\big) \ >\ 0.8\cdot\min(d_i, d_j)\ \Rightarrow\ \text{same fingertip}")
    st.write("Between two real fingers the contour drops into the valley, which is much closer to the palm than the tips, so both fingers are kept. Around one tip the contour stays near the tip distance, so the duplicate is dropped.")
    st.markdown("""
**Fix 2: the forearm.** Photos usually include the forearm, which runs off the edge of the image. It distorted the hull and produced a false valley on the forearm.
If the hand contour touches the image border (within max(6 px, 3% of the longer side)), the direction u from the palm centre towards the border points is found, and everything beyond a cut line just below the palm is removed before the hull and the fingertips are computed:
""")
    st.latex(r"\text{remove pixels } p \text{ with } (p - c)\cdot u > 1.25\,r")

    st.subheader("3.7 Output")
    st.write("The app shows the input, the skin mask and the annotated result side by side. Yellow box: mean-shift window; green: hand contour; blue: convex hull; "
             "red circle: palm; orange: numbered fingertips; pink: valleys between fingers. It also shows the finger count, palm radius, hand area, a fingertip coordinate table and a JSON download.")

# ---------------- 4. EXPERIMENTAL SETUP ----------------
with tabs[3]:
    st.header("4. Experimental Setup")
    st.subheader("Methods compared")
    st.write("No training is involved, so there are no training datasets or baselines to fit. We compare two versions of the same pipeline: "
             "**v1** (hull points with a fixed spacing rule, no wrist handling) and **v2** (same-fingertip rule and wrist cut). "
             "The learned model from our first application (MediaPipe) is discussed only qualitatively, because we did not run both on a common labeled test set.")
    st.subheader("Fixed parameters")
    st.table({
        "Parameter": ["Cr range", "Cb range", "Gaussian blur", "Morphology kernel", "Closing iterations", "Min. region size", "Regions kept",
                      "Mean-shift stop", "Tip distance factor (λ)", "Max. fingertips", "Valley depth", "Same-fingertip ratio", "Wrist cut line", "Input size"],
        "Value": ["133 to 173", "77 to 127", "5 × 5", "5 (ellipse)", "2", "1% of image", "4", "0.5 px or 30 steps", "1.6 × r", "5", "> 0.4 r",
                  "0.8", "1.25 r beyond the palm centre", "longest side 640 px"],
    })
    st.caption("The sliders for these values were removed from the app to keep it simple; they are constants in the code.")
    st.subheader("Implementation platform")
    st.table({
        "Item": ["Language", "Web framework", "Image processing", "Arrays", "Image loading", "Deployment", "Packages (requirements.txt)"],
        "Choice": ["Python", "Streamlit", "OpenCV (opencv-python-headless)", "NumPy", "Pillow (also fixes phone-photo rotation)",
                   "Streamlit Community Cloud from GitHub", "streamlit, opencv-python-headless, numpy, pillow"],
    })
    st.subheader("Problems we ran into and how we fixed them")
    st.markdown("""
**1. Wrong finger count (3 instead of 2).** On a V-sign photo the first version reported 3 fingertips. Reading the annotated result showed that two orange points sat on the same fingertip and that the blue hull ran down the whole forearm.
*Fix:* the same-fingertip rule and the wrist cut (Section 3.6).

**2. Our fixes did not appear in the deployed app.** After correcting the code, the live app still showed the old result. The app runs on Streamlit Community Cloud, which only changes when the new files are pushed to GitHub, and Streamlit also keeps imported modules in memory.
*Fix:* we pushed both `app.py` and `hand_cv.py` to the repository and rebooted the app from **Manage app**. We also added `importlib.reload` in `app.py` so the helper module is re-read on every run.

**3. Too many controls.** The first classical app had colour sliders, a kernel slider, a fingertip factor slider, a camera input and a dropdown to choose the skin region. These made results depend on user settings.
*Fix:* we kept one upload button, fixed the settings and let mean shift choose the region.
""")
    st.info("**What we learned:** when a result looks wrong, look at the intermediate pictures (mask, hull, points) before changing thresholds. Both bugs were visible in the annotated output. "
            "Also, after every fix, confirm that the deployed app is really running the new code.")

# ---------------- 5. RESULTS ----------------
with tabs[4]:
    st.header("5. Results")
    st.subheader("5.1 Finger count on our test photo(s)")
    st.write("We checked the number of fingertips found by each version against the number of fingers actually raised. Only the photos listed here were tested, and the pass/fail "
             "judgement is by eye. No ground-truth keypoints were used.")
    st.table({
        "Test": [t["name"] for t in TESTS],
        "Raised fingers": [t["expected"] for t in TESTS],
        "v1 fingertips": [t["v1"] for t in TESTS],
        "v2 fingertips": [t["v2"] for t in TESTS],
        "v2 correct": ["yes" if t["v2"] == t["expected"] else "no" for t in TESTS],
    })
    ok = sum(t["v2"] == t["expected"] for t in TESTS)
    st.success(f"v2 gave the correct count on {ok} of {len(TESTS)} test photo(s). v1 gave the correct count on "
               f"{sum(t['v1'] == t['expected'] for t in TESTS)} of {len(TESTS)}. This is a very small sample, so it shows that the two fixes work on this case, not a general accuracy.")

    st.subheader("5.2 Before and after the fixes (V-sign photo)")
    c1, c2 = st.columns(2)
    with c1:
        show_img("assets/v_sign_old.png", "v1: input, skin mask and result. 3 fingertips (two on the same finger) and the hull runs down the forearm")
    with c2:
        show_img("assets/v_sign_new.png", "v2: input, skin mask and result. 2 fingertips and the hull stops at the wrist")
    st.table({
        "Measurement on the V-sign photo": ["Fingertips found", "Valleys reported", "Hand area used for analysis (px)", "Palm radius (px)"],
        "v1": ["3", "3 (one false, on the forearm)", "10711 (hand + forearm)", "30.0"],
        "v2": ["2", "2", "5854 (hand only)", "30.0"],
    })
    st.caption("Measured on the 183 × 275 pixel image after resizing. The two remaining valley points lie at the base of the two raised fingers, one on each side of the gap.")

    st.subheader("5.3 Sensitivity to settings")
    st.write("We re-ran v2 on the same photo with different image sizes (183, 400 and 640 px), morphology kernels (3, 5, 9), fingertip factors (1.2, 1.6, 2.0) "
             "and two sets of colour ranges.")
    st.markdown("""
- **With the default colour ranges,** v2 found exactly 2 fingertips for every combination of size, kernel and factor, except kernel size 9 on the smallest (183 px) image, where it found 1. A large kernel on a small image erodes a thin finger.
- **With a wider Cr range** (120 to 180, alone or together with Cb 70 to 135), the white background was partly counted as skin and the count dropped to 0 or 1. The colour ranges matter much more than the other settings.
""")

    st.subheader("5.4 Discussion and limitations")
    st.markdown("""
- **Skin colour is the weakest link.** Faces, necks, arms, and skin-coloured walls or wood are also selected. The mean-shift choice helps, but a hand next to a face can still be merged with it. The best results are on a plain background.
- **Lighting and skin tone.** The fixed Cr/Cb box was designed for typical skin tones and normal light. Strong coloured light, very dark or very pale skin may fall outside it. We tested only one photo, so we cannot report how often this happens.
- **Closed or curled fingers are not counted.** A fingertip is a hull point far from the palm, so folded fingers and a thumb held against the palm are missed. The V-sign result is correct because the two raised fingers are the only ones that stick out.
- **Wrist cut assumes the arm leaves through the image border.** If the whole arm is visible inside the image, nothing is cut and the arm can disturb the palm estimate.
- **One hand, 2D only.** The method analyses one region, and gives pixel positions, with no depth, no joint positions and no left/right label. The earlier MediaPipe version gives 21 keypoints in 3D.
- **Hand-tuned thresholds.** The values in Section 4 were chosen by us and checked on very few photos, so they may need changing for other cameras.
- **Fast and inspectable.** The method needs no model download and every step can be shown as a picture, which is useful for teaching and for debugging.
""")
    st.subheader("5.5 Classical method vs the learned model (qualitative)")
    st.table({
        "Aspect": ["Training data", "Output", "Curled fingers", "Cluttered or skin-coloured background", "Depth / 3D", "Setup", "Explainability"],
        "Classical (this version)": ["None", "Outline, palm, fingertips, valleys", "Not counted", "Can pick up faces and skin-coloured objects", "No", "4 small packages", "Every step visible"],
        "MediaPipe (first version)": ["Pretrained network", "21 keypoints", "Followed in our test photos", "Hand overlapping a face looked plausible (our test 6)", "Estimated, less reliable", "Larger packages and a model download", "Black box"],
    })
    st.caption("The right-hand column repeats what we observed in our first report; the two versions were not benchmarked against each other on a labeled set.")

    st.subheader("5.6 Your own test photos")
    st.write("Add more photos to the `TESTS` list at the top of this file, with the screenshot in the `assets` folder, and they will appear in the table in 5.1.")
    gallery = [t for t in TESTS if t["img"] not in ("assets/v_sign_new.png",)]
    for t in gallery:
        show_img(t["img"], t["name"])

# ---------------- 6. CONCLUSION ----------------
with tabs[5]:
    st.header("6. Conclusion")
    st.subheader("Summary")
    st.write("We replaced our neural-network hand detector with a classical computer vision pipeline: YCrCb skin segmentation, mean shift to pick the hand, "
             "a distance transform for the palm, and convex hull and convexity defects for fingertips and valleys. Two problems appeared on a V-sign photo, duplicate points on one fingertip and the forearm, "
             "and both were solved with simple geometric rules (a same-fingertip test and a wrist cut). After the fixes the app reports the correct 2 fingers on that photo and is stable across the image sizes and settings we tried, as long as the colour ranges are not widened. "
             "The method is fast, needs no training and is easy to explain, but it depends heavily on skin colour and a clean background, does not count folded fingers, and gives only 2D results.")
    st.subheader("Future work")
    st.markdown("""
- Test on a larger set of labeled photos and report the fraction of correct finger counts for different poses, lighting and skin tones.
- Use adaptive skin colour (for example, learn the Cr/Cb range from a detected face or from the hand itself) instead of a fixed box.
- Count folded fingers by looking at the contour shape of the palm, not only far hull points.
- Add video input using the previous hand position to start mean shift.
- Compare directly with the MediaPipe version and with a model trained on FreiHAND on the same test images.
""")

# ---------------- 7. REFERENCES ----------------
with tabs[6]:
    st.header("7. References")
    st.markdown(f"""
1. D. Chai, K. N. Ngan. *Face segmentation using skin-color map in videophone applications.* IEEE Transactions on Circuits and Systems for Video Technology, 1999.
2. S. L. Phung, A. Bouzerdoum, D. Chai. *Skin segmentation using color pixel classification: analysis and comparison.* IEEE TPAMI, 2005.
3. D. Comaniciu, P. Meer. *Mean shift: a robust approach toward feature space analysis.* IEEE TPAMI, 2002.
4. S. Suzuki, K. Abe. *Topological structural analysis of digitized binary images by border following.* Computer Vision, Graphics, and Image Processing, 1985.
5. G. Borgefors. *Distance transformations in digital images.* Computer Vision, Graphics, and Image Processing, 1986.
6. G. Bradski. *The OpenCV Library.* Dr. Dobb's Journal of Software Tools, 2000.
7. C. Zimmermann, D. Ceylan, J. Yang, B. Russell, M. Argus, T. Brox. *FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images.* ICCV 2019. Code: [{GITHUB_URL}]({GITHUB_URL})
8. MediaPipe Hand Landmarker, Google (used in our first version of the application).
9. Streamlit documentation: https://docs.streamlit.io
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
