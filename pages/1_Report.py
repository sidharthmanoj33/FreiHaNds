import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import hand_cv as H   # same module the main app uses

st.set_page_config(page_title="Project Report: Classical Hand Detection", page_icon="📄", layout="wide")

# ---------- EDIT THESE ----------
TEAM = "Sidharth Manoj (264152030), Anna Pao (264152003), Sidharth Das (264152015)"
INSTITUTE = "Indian Institute of Technology Guwahati"
PPT_LINK = "https://github.com/your-username/your-repo/blob/main/presentation.pptx"
VIDEO_URL = "https://youtu.be/uEc6RenVgNs"        # project explanation video (placeholder)
SCREEN_REC_URL = ""   # PPT screen recording (placeholder)
APP_URL = "https://freihands-rjzm2bqdjv9hcze5byzcpy.streamlit.app"   # CHECK: copy the exact URL from your browser
REPO_URL = "https://github.com/your-username/your-repo"             # your GitHub repository
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
<h1>Hand and Finger Detection with Classical Computer Vision</h1>
<p>Project report: an interactive Streamlit application that finds a hand and its fingertips without any deep learning</p>
<span class="pill">YCrCb skin segmentation</span><span class="pill">Mean shift</span><span class="pill">Convex hull + defects</span><span class="pill">Distance transform</span><span class="pill">OpenCV + NumPy only</span>
<p style="margin-top:14px;font-size:.95rem">Team: """ + TEAM + """ &nbsp;·&nbsp; """ + INSTITUTE + """</p>
</div>
""", unsafe_allow_html=True)


def show_img(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Add a screenshot at `{path}` in your repository to show: {caption}")


tabs = st.tabs(["1 Introduction", "2 Background", "3 Methodology", "4 Experimental setup", "5 Deployment",
                "6 Results", "7 Conclusion", "8 References", "9 PPT", "10 Video"])

# ---------------- 1. INTRODUCTION ----------------
with tabs[0]:
    st.header("1. Introduction")
    st.subheader("Motivation")
    st.write("Finding a hand in a photo and counting the raised fingers is a basic step for gesture-based interfaces. "
             "Modern detectors use neural networks, but a hand can also be found with classical image-processing ideas: "
             "skin has a characteristic colour, a hand has a recognisable outline, and fingers are the parts of that outline "
             "that stick out furthest from the palm. This project builds a complete detector from those ideas alone, "
             "so every step can be seen, explained and adjusted.")
    st.subheader("Applications")
    c = st.columns(4)
    for col, t in zip(c, ["Simple gesture commands", "Finger counting", "Teaching computer vision", "Low-power devices without a GPU"]):
        col.markdown(f'<div class="card"><b>{t}</b></div>', unsafe_allow_html=True)
    st.subheader("Problem statement")
    st.write("Given one RGB photo of a hand, locate the hand, find the palm centre and palm size, estimate the fingertip positions "
             "and the valleys between fingers, and show the result in a simple web application. No neural network or learned model is allowed.")
    st.subheader("Contributions")
    st.markdown("""
- **Built a full classical pipeline** with OpenCV and NumPy: skin-colour segmentation, morphology, mean shift, contour, convex hull, convexity defects and distance transform.
- **Added two small robustness steps:** a wrist cut that removes the forearm, and a same-finger test that stops one fingertip from being counted twice.
- **Built and deployed a web app** where the user uploads a photo and sees the skin mask, the annotated result, a metrics row, a fingertip table and a JSON download.
- **Explained every stage** with the exact thresholds used in the code (Section 3).
- **Tested the app** on three hand photos with one, two and three raised fingers (Section 6).
""")

# ---------------- 2. BACKGROUND ----------------
with tabs[1]:
    st.header("2. Background")
    st.write("**Skin-colour segmentation.** Skin pixels cluster in a small region of chrominance space. Working in the YCrCb colour space "
             "separates brightness (Y) from colour (Cr, Cb), so a fixed range on Cr and Cb is less affected by lighting than a range on raw RGB values. "
             "This idea was popularised for face and hand finding in video (Chai and Ngan, 1999).")
    st.write("**Mean shift.** Mean shift is a mode-seeking method: a window repeatedly moves to the centre of mass of the points inside it "
             "until it stops moving, so it settles on the densest cluster (Comaniciu and Meer, 2002). Here the points are skin pixels.")
    st.write("**Contour, convex hull and convexity defects.** The outline of a blob is traced as a contour. The convex hull is the smallest convex shape around it, "
             "like a rubber band. The gaps between the hull and the contour are convexity defects. For a hand, the hull touches the fingertips and the defects "
             "are the valleys between the fingers.")
    st.write("**Distance transform.** For every skin pixel, the distance transform stores the distance to the nearest background pixel. "
             "The largest value sits in the thickest part of the shape, which for a hand is the palm. Its position gives the palm centre and its value gives the palm radius.")
    st.caption("Full references are in Section 8.")

# ---------------- 3. METHODOLOGY ----------------
with tabs[2]:
    st.header("3. Methodology")
    st.subheader("3.1 Pipeline")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#e8efec", fontname="Helvetica"];
      A [label="Photo upload\\n(resize to 640 px)"]; B [label="Blur + YCrCb\\nskin mask"]; C [label="Morphology\\n(open, close)"];
      D [label="Skin blobs\\n(largest 4)"]; E [label="Mean shift\\npicks the hand"]; F [label="Wrist cut"];
      G [label="Distance transform\\npalm centre + radius"]; H [label="Convex hull\\nfingertips"]; I [label="Convexity defects\\nvalleys"];
      J [label="Overlay, table,\\nJSON"];
      A -> B -> C -> D -> E -> F -> G; G -> H; G -> I; H -> J; I -> J; }
    """)

    st.subheader("3.2 Skin segmentation and clean-up")
    st.write("The image is smoothed with a 5 x 5 Gaussian blur and converted to YCrCb. A pixel is marked as skin when its chrominance falls inside a fixed box:")
    st.latex(r"133 \le C_r \le 173, \qquad 77 \le C_b \le 127 \qquad (Y \text{ is ignored})")
    st.write("The raw mask has specks and small holes (for example nail highlights). An **opening** with a 5 x 5 elliptical kernel removes the specks, "
             "then a **closing** (2 iterations) fills the holes. The connected white blobs are traced, blobs smaller than 1% of the image area are dropped, "
             "and the four largest are kept as candidate regions.")

    st.subheader("3.3 Choosing the hand with mean shift")
    st.write("A window that starts as the central half of the image (width/2 by height/2) is shifted towards the densest area of the mask. "
             "Iteration stops after 30 steps or when the move is smaller than 0.5 pixel. The candidate region that contains, or is closest to, "
             "the centre of the final window is chosen as the hand. This is how the app decides which blob is the hand when a face or arm is also visible.")

    st.subheader("3.4 Wrist cut")
    st.write("If the chosen blob touches the image border, it is assumed to continue into the forearm. The app finds the palm centre c and radius r "
             "with the distance transform, takes the direction u from the palm centre towards the border points, and removes every pixel beyond a cut line:")
    st.latex(r"(\mathbf{x}-\mathbf{c})\cdot \mathbf{u} \;>\; 1.25\, r \;\Rightarrow\; \text{pixel removed}")
    st.write("This stops the arm from producing fake fingertips and fake valleys.")

    st.subheader("3.5 Palm centre and radius")
    st.latex(r"D(\mathbf{x}) = \min_{\mathbf{b}\in \text{background}} \|\mathbf{x}-\mathbf{b}\|_2 , \qquad "
             r"\mathbf{c} = \arg\max_{\mathbf{x}} D(\mathbf{x}), \qquad r = \max_{\mathbf{x}} D(\mathbf{x})")
    st.write("The red circle in the results is drawn at c with radius r. The radius also acts as a size scale, so the thresholds below adapt to how large the hand appears in the photo.")

    st.subheader("3.6 Fingertips from the convex hull")
    st.write("Each convex hull point p is ranked by its distance from the palm centre. Going from the furthest point inwards, a point becomes a fingertip when:")
    st.latex(r"\|\mathbf{p}-\mathbf{c}\|_2 \;\ge\; 1.6\, r")
    st.markdown("""
- the point is not within 3 pixels of the image border (points cut by the frame are ignored);
- it is **not on the same finger** as a tip already accepted. Two hull points are on the same finger when the contour between them never dips back towards the palm, tested as: the better of the two arcs between them keeps a distance above 0.8 times the smaller of the two tip distances;
- at most five fingertips are kept.
""")

    st.subheader("3.7 Valleys from convexity defects")
    st.write("A convexity defect is a dent between the contour and its hull. It is accepted as a valley between fingers when it is deep and not near the palm centre line:")
    st.latex(r"\text{depth} > 0.4\, r \qquad \text{and} \qquad \|\mathbf{f}-\mathbf{c}\|_2 > 0.9\, r")
    st.write("where f is the deepest point of the defect. The second condition ignores the straight cut edge made by the wrist cut.")

    st.subheader("3.8 Outputs")
    st.write("The app reports the number of fingers (accepted fingertips), the number of valleys, the palm radius, the hand area in pixels, "
             "the bounding box, a table of fingertip coordinates and a downloadable JSON file with all of these.")

# ---------------- 4. EXPERIMENTAL SETUP ----------------
with tabs[3]:
    st.header("4. Experimental Setup")
    st.subheader("Settings used")
    st.table({
        "Parameter": ["Input size", "Blur", "Cr range", "Cb range", "Morphology kernel", "Minimum blob size", "Candidate blobs",
                      "Mean shift stop", "Wrist cut distance", "Fingertip distance factor", "Valley depth factor", "Maximum fingertips"],
        "Value": ["longest side 640 px", "Gaussian 5 x 5", "133 to 173", "77 to 127", "ellipse 5 x 5 (open once, close twice)", "1% of image area", "4 largest",
                  "30 iterations or 0.5 px", "1.25 x palm radius", "1.6 x palm radius", "0.4 x palm radius", "5"],
    })
    st.caption("These are fixed in the app. There are no sliders, so every result in Section 6 uses exactly these values.")
    st.subheader("Implementation platform")
    st.table({
        "Item": ["Language", "Image processing", "Numerical work", "Web framework", "Deployment", "Model files"],
        "Choice": ["Python", "OpenCV", "NumPy", "Streamlit (with Pillow for image loading and rotation fix)", "Streamlit Community Cloud from GitHub",
                   "None. There is nothing to train or download."],
    })
    st.subheader("Code structure")
    st.markdown("""
- `hand_cv.py`: all image-processing functions (`skin_mask`, `skin_regions`, `mean_shift_window`, `default_region`, `cut_wrist`, `analyze`, `draw`).
- `app.py`: the Streamlit page that loads the photo, calls those functions and shows the results.
- `pages/` report file: this report.
""")
    st.subheader("Test photos")
    st.write("Three photos of one hand on a plain light background, each showing a different number of raised fingers: three, two and one. "
             "The other fingers (and thumb) are folded in front of the palm.")

# ---------------- 5. DEPLOYMENT ----------------
with tabs[4]:
    st.header("5. Deployment")
    st.write("The detector is deployed as a public web app, so anyone can upload a hand photo and see the result without installing anything.")
    c1, c2 = st.columns(2)
    c1.link_button("Open the live app", APP_URL, use_container_width=True)
    c2.link_button("Open the code on GitHub", REPO_URL, use_container_width=True)

    st.subheader("5.1 Deployment steps")
    st.markdown("""
1. **Prepare the repository on GitHub** with these files at the top level: `app.py`, `hand_cv.py`, a `pages/` folder holding this report, an `assets/` folder with the test images, and `requirements.txt`.
2. **List the Python packages** in `requirements.txt`: `streamlit`, `opencv-python-headless`, `numpy` and `pillow`. The headless OpenCV build is used because a server has no display.
3. **Add `packages.txt`** only if the server reports a missing system library when the app starts (it lists Linux packages, one per line).
4. **Create the app** on Streamlit Community Cloud: choose *New app*, select the repository and branch, set the main file to `app.py`, and press *Deploy*.
5. **Check the build log** under *Manage app* if anything fails. The log shows the real error, such as a missing package.
6. **Updates:** every push to the branch rebuilds the app automatically.
""")
    st.info("`requirements.txt` must be named exactly that. Because the app uses no trained model, there is no model file to download or upload.")

    st.subheader("5.2 Try the app here")
    st.write("This is the same pipeline as the main app (`hand_cv.py`), with the same fixed settings. Upload a photo of a hand on a plain background.")
    up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], key="report_demo_upload")
    if up is None:
        st.caption("No image uploaded yet.")
    else:
        img = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
        img.thumbnail((640, 640))
        bgr = np.array(img)[:, :, ::-1].copy()
        mask = H.skin_mask(bgr, (133, 173), (77, 127), 5)
        regions = H.skin_regions(mask)
        if not regions:
            st.warning("No skin-coloured region found. Try a clearer photo of a hand on a plain background.")
        else:
            win = H.mean_shift_window(mask)
            chosen = H.default_region(regions, win)
            info = H.analyze(regions[chosen], mask.shape, 1.6)
            out = H.draw(bgr, regions, chosen, info, win)
            d1, d2, d3 = st.columns(3)
            d1.image(img, caption="Input", use_container_width=True)
            d2.image(mask, caption="Skin mask", use_container_width=True)
            d3.image(out[:, :, ::-1], caption="Result", use_container_width=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fingers estimated", len(info["tips"]))
            m2.metric("Valleys", len(info["valleys"]))
            m3.metric("Palm radius (px)", f"{info['radius']:.1f}")
            m4.metric("Hand area (px)", int(info["area"]))

# ---------------- 6. RESULTS ----------------
with tabs[5]:
    st.header("6. Results")
    st.subheader("6.1 Output of the app on three test photos")
    st.write("For each photo we compared the number of fingers found by the app with the number of fingers we can see raised in the photo, "
             "and checked by eye whether the markers sit on the fingertips. Positions were not measured against ground-truth labels.")
    st.table({
        "Test": ["1", "2", "3"],
        "Pose": ["Three fingers raised, spread", "Two fingers raised (V sign)", "One finger raised, fist closed"],
        "Fingers visible": ["3", "2", "1"],
        "Fingers estimated": ["3", "2", "1"],
        "Valleys found": ["3", "2", "1"],
        "Palm radius (px)": ["128.4", "113.8", "120.0"],
        "Hand area (px)": ["102,704", "75,977", "64,095"],
    })
    st.success("The estimated finger count matched the raised fingers in all 3 of 3 test photos. This is a very small sample of clean photos of one hand, "
               "so it shows that the pipeline works but is not a general accuracy figure.")

    st.subheader("6.2 Qualitative results")
    show_img("assets/classical_test1_three_fingers.png",
             "Test 1: three raised fingers. Input, skin mask and result with hull, palm circle, fingertips and valleys")
    show_img("assets/classical_test2_two_fingers.png",
             "Test 2: two raised fingers (V sign)")
    show_img("assets/classical_test3_one_finger.png",
             "Test 3: one raised finger with a closed fist. The fingertip was found at pixel (328, 136)")
    st.write("In each result image, the yellow box is the final mean-shift window, the green line is the hand contour, the blue polygon is the convex hull, "
             "the red circle is the palm from the distance transform, orange dots are fingertips and pink dots are valleys. "
             "The small white number near the top left of a blob (for example the white 1 in Test 1) is the blob's index among the candidate skin regions, "
             "not a fingertip number. Fingertips are numbered by their distance from the palm centre, not by which finger they are.")

    st.subheader("6.3 Observations")
    st.markdown("""
- **Skin mask:** in all three tests the hand comes out as one solid white shape on a black background. The small holes in the fingertips and palm (nail highlights, skin creases) are small enough that they do not change the outline.
- **Fingertips:** the orange dots sit on the tips of the raised fingers. In Test 1 the spread fingers are three separate hull corners, so each gets one tip. The folded fingers and thumb do not reach 1.6 times the palm radius, so they are correctly not counted.
- **Palm circle:** the red circle sits over the palm in every test and its radius scales with the hand (113.8 to 128.4 px), which is what keeps the thresholds consistent across photos.
- **Valleys:** one valley appears between each pair of neighbouring raised fingers. The app also places one extra valley at the side of the palm where the folded thumb meets the hand, so the valley count is one higher than the number of gaps between the raised fingers (3, 2, 1 here, rather than 2, 1, 0).
- **Wrist cut:** in Test 3 the forearm leaves the bottom of the frame, and the contour is cut near the wrist, so the arm does not create extra tips.
""")

    st.subheader("6.4 Limitations seen and expected")
    st.markdown("""
- **Valley count is not a finger count:** as noted above, the extra valley at the thumb side means valleys should be read as "gaps found", not "fingers minus one". The finger count itself comes from fingertips only.
- **Closed or curled fingers:** a finger folded into the palm does not stick out of the outline, so it cannot be counted. A fist gives zero fingertips by design.
- **Skin-like colours:** because the method looks only at colour, faces, bare arms, wood or beige walls can enter the mask. Mean shift helps choose the right blob but cannot separate a hand that touches a face.
- **Lighting and skin tone:** the fixed Cr/Cb range works for the lighting and skin tone in our photos. Very dark, very bright, strongly coloured light or other skin tones may need a different range, which we did not test.
- **Two hands in one photo:** only one blob is analysed at a time.
- **2D only:** the result gives pixel positions of tips and valleys. There are no joint positions, no depth and no hand shape.
- **Limited testing:** three photos of one hand on a plain background, judged by eye, with no ground-truth labels.
""")

# ---------------- 7. CONCLUSION ----------------
with tabs[6]:
    st.header("7. Conclusion")
    st.subheader("Summary")
    st.write("We built a hand and finger detector using only classical computer vision. The app segments skin in YCrCb space, cleans the mask with morphology, "
             "uses mean shift to pick the hand, cuts off the forearm, finds the palm with the distance transform, and then reads fingertips from the convex hull "
             "and valleys from the convexity defects. On our three test photos (three, two and one raised fingers) the number of fingers found matched what is visible. "
             "The method needs no training data or model file, runs quickly and every step can be inspected. Its main weaknesses are dependence on colour, "
             "curled fingers and the lack of any 3D information.")
    st.subheader("Future work")
    st.markdown("""
- Test on many more photos, with different lighting, backgrounds and skin tones, and report a finger-count accuracy.
- Adapt the Cr/Cb range automatically from the image, for example from a sample of the detected hand.
- Remove the extra thumb-side valley so the valley count matches the gaps between raised fingers.
- Process live webcam video and track the hand with mean shift between frames.
- Handle two hands in one photo.
""")

# ---------------- 8. REFERENCES ----------------
with tabs[7]:
    st.header("8. References")
    st.markdown("""
1. D. Comaniciu, P. Meer. *Mean Shift: A Robust Approach Toward Feature Space Analysis.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 2002.
2. D. Chai, K. N. Ngan. *Face Segmentation Using Skin-Color Map in Videophone Applications.* IEEE Transactions on Circuits and Systems for Video Technology, 1999.
3. S. Suzuki, K. Abe. *Topological Structural Analysis of Digitized Binary Images by Border Following.* Computer Vision, Graphics, and Image Processing, 1985.
4. G. Borgefors. *Distance Transformations in Digital Images.* Computer Vision, Graphics, and Image Processing, 1986.
5. G. Bradski. *The OpenCV Library.* Dr. Dobb's Journal of Software Tools, 2000.
6. OpenCV documentation: contours, convex hull, convexity defects, distance transform, meanShift. https://docs.opencv.org
7. Streamlit documentation: https://docs.streamlit.io
""")

# ---------------- 9. PPT ----------------
with tabs[8]:
    st.header("9. Link to PPT File")
    st.markdown(f"[Open the presentation on GitHub]({PPT_LINK})")

# ---------------- 10. VIDEO ----------------
with tabs[9]:
    st.header("10. Video Demonstration and PPT Screen Recording")
    st.subheader("Project demonstration video")
    st.video(VIDEO_URL)
    st.subheader("")
    st.video(SCREEN_REC_URL)
    st.caption("Replace the placeholder video links at the top of this file with your own.")
