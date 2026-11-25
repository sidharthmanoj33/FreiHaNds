import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import importlib

import hand_cv as H

importlib.reload(H)   # always pick up the latest hand_cv.py (Streamlit caches imported modules)

st.set_page_config(page_title="Classical Hand Detection", page_icon="🖐️", layout="wide")
st.title("🖐️ Hand Detection with Classical Computer Vision")
st.caption("No CNN or deep learning. Pipeline: skin-colour segmentation (YCrCb) → mean shift → contour → convex hull → "
           "convexity defects → distance transform. Built with OpenCV and NumPy only.")

with st.sidebar:
    st.header("Skin colour thresholds (YCrCb)")
    cr = st.slider("Cr range", 100, 200, (133, 173))
    cb = st.slider("Cb range", 60, 160, (77, 127))
    k = st.slider("Morphology kernel size", 3, 15, 5, 2)
    tip_factor = st.slider("Fingertip distance factor (x palm radius)", 1.2, 2.5, 1.6, 0.1)
    st.caption("hand_cv version: " + getattr(H, "VERSION", "OLD - not updated"))
    st.caption("If the hand is missed, widen the ranges. If the face or background is picked up, narrow them.")

f = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if f is None:
    st.info("Upload a photo of a hand on a plain background for the best result.")
else:
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    img.thumbnail((640, 640))
    bgr = np.array(img)[:, :, ::-1].copy()

    mask = H.skin_mask(bgr, cr, cb, k)
    regions = H.skin_regions(mask)
    if not regions:
        st.warning("No skin-coloured region found. Try a clearer photo or widen the thresholds in the sidebar.")
        st.image(img, use_container_width=True)
    else:
        win = H.mean_shift_window(mask)
        auto = H.default_region(regions, win)
        chosen = auto   # hand region picked automatically by mean shift
        info = H.analyze(regions[chosen], mask.shape, tip_factor)
        out = H.draw(bgr, regions, chosen, info, win)

        c1, c2, c3 = st.columns(3)
        c1.image(img, caption="Input", use_container_width=True)
        c2.image(mask, caption="Skin mask (white = skin)", use_container_width=True)
        c3.image(out[:, :, ::-1], caption="Result", use_container_width=True)
        st.caption("Yellow box: mean-shift window · green: hand contour · blue: convex hull · red circle: palm "
                   "(distance transform) · orange: fingertips · pink: valleys between fingers")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fingers estimated", len(info["tips"]))
        m2.metric("Valleys between fingers", len(info["valleys"]))
        m3.metric("Palm radius (px)", f"{info['radius']:.1f}")
        m4.metric("Hand area (px)", int(info["area"]))
        st.dataframe({"fingertip": list(range(1, len(info["tips"]) + 1)),
                      "x_px": [t[0] for t in info["tips"]], "y_px": [t[1] for t in info["tips"]]})

        data = {"image_size": [img.width, img.height], "palm_center": info["center"], "palm_radius_px": info["radius"],
                "bounding_box": info["bbox"], "fingertips": info["tips"], "valleys": info["valleys"],
                "fingers_estimated": len(info["tips"])}
        st.download_button("Download results (JSON)", json.dumps(data, indent=1), "hand_result.json", "application/json")

with st.expander("How the algorithm works"):
    st.markdown("""
1. **Skin segmentation:** the image is converted to YCrCb. Pixels whose Cr and Cb values fall in a skin range are kept; brightness (Y) is ignored, which makes this less sensitive to lighting.
2. **Morphology:** opening removes small specks and closing fills small holes in the mask.
3. **Mean shift:** a window is moved step by step towards the densest cluster of skin pixels. The skin region it lands on is the default choice for the hand.
4. **Contour and convex hull:** the outline of the region is traced and wrapped in the smallest convex shape.
5. **Palm centre:** the distance transform gives each pixel's distance to the nearest edge. Its maximum is the palm centre and that distance is the palm radius.
6. **Fingertips:** hull points far from the palm centre (more than the chosen factor times the palm radius), spread apart from each other, up to five.
7. **Valleys:** convexity defects deep enough to be gaps between fingers.
""")
st.caption("Limits: skin-colour methods can confuse faces, arms and skin-like backgrounds. Closed or curled fingers reduce the finger count. "
           "The result is 2D only: no joint positions, depth or hand shape.")
