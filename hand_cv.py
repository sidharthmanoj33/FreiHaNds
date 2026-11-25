"""Classical (non-deep-learning) hand analysis: skin colour segmentation, mean shift,
contour, convex hull, convexity defects and distance transform. Only OpenCV + NumPy."""
import cv2
import numpy as np


def skin_mask(bgr, cr=(133, 173), cb=(77, 127), k=5):
    """Skin pixels = Cr and Cb inside a range in the YCrCb colour space, cleaned with morphology."""
    ycc = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (0, cr[0], cb[0]), (255, cr[1], cb[1]))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=2)


def skin_regions(mask, min_frac=0.01, top=4):
    """Largest connected skin blobs, sorted by area (biggest first)."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) >= min_frac * mask.size]
    return sorted(cs, key=cv2.contourArea, reverse=True)[:top]


def mean_shift_window(mask):
    """Mean shift: slide a window towards the densest cluster of skin pixels."""
    h, w = mask.shape
    prob = (mask / 255.0).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.5)
    _, win = cv2.meanShift(prob, (w // 4, h // 4, w // 2, h // 2), crit)
    return tuple(int(v) for v in win)


def default_region(regions, win):
    """Region that contains (or is closest to) the centre of the mean-shift window."""
    cx, cy = win[0] + win[2] / 2.0, win[1] + win[3] / 2.0
    scores = [cv2.pointPolygonTest(c, (float(cx), float(cy)), True) for c in regions]
    return int(np.argmax(scores))


def cut_wrist(cnt, shape):
    """If the blob runs off the image border (forearm), cut it just below the palm so the arm
    cannot create fake fingertips or valleys. Returns the (possibly new) contour."""
    h, w = shape
    blob = np.zeros(shape, np.uint8)
    cv2.drawContours(blob, [cnt], -1, 255, -1)
    pts = cnt[:, 0, :]
    m_ = max(6, int(0.03 * max(h, w)))                   # tolerance: the arm often stops a few px short of the border
    edge = (pts[:, 0] <= m_) | (pts[:, 1] <= m_) | (pts[:, 0] >= w - 1 - m_) | (pts[:, 1] >= h - 1 - m_)
    if edge.sum() < 5:
        return cnt
    _, r, _, c = cv2.minMaxLoc(cv2.distanceTransform(blob, cv2.DIST_L2, 5))
    u = pts[edge].mean(axis=0) - np.array(c)            # direction palm centre -> arm
    n = np.linalg.norm(u)
    if n < 1e-6:
        return cnt
    u /= n
    yy, xx = np.mgrid[0:h, 0:w]
    proj = (xx - c[0]) * u[0] + (yy - c[1]) * u[1]
    blob[proj > 1.25 * r] = 0                           # wrist cut line
    cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cs, key=cv2.contourArea) if cs else cnt


def _same_finger(cnt, i, j, d_cnt, ratio=0.8):
    """Two hull points are on the same fingertip if the contour between them never dips
    back towards the palm (no valley in between)."""
    n = len(cnt)
    lo, hi = sorted((i, j))
    arc1 = d_cnt[lo:hi + 1]
    arc2 = np.concatenate([d_cnt[hi:], d_cnt[:lo + 1]])
    best = max(arc1.min(), arc2.min())   # the arc without a valley has the higher minimum
    return best > ratio * min(d_cnt[i], d_cnt[j])


def analyze(cnt, shape, tip_factor=1.6):
    """Palm centre (distance transform), convex hull, fingertips (far hull points) and valleys (defects)."""
    h, w = shape
    cnt = cut_wrist(cnt, shape)
    blob = np.zeros(shape, np.uint8)
    cv2.drawContours(blob, [cnt], -1, 255, -1)
    dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    _, r, _, c = cv2.minMaxLoc(dt)               # r = palm radius, c = palm centre (x, y)
    pts = cnt[:, 0, :]
    d_cnt = np.linalg.norm(pts - np.array(c), axis=1)
    hidx = cv2.convexHull(cnt, returnPoints=False)[:, 0]
    hull = pts[hidx]
    tips, tip_idx = [], []
    for k in np.argsort(-d_cnt[hidx]):
        i, p = int(hidx[k]), pts[hidx[k]]
        if d_cnt[i] < tip_factor * r:
            break
        if p[0] <= 3 or p[1] <= 3 or p[0] >= w - 4 or p[1] >= h - 4:   # ignore points cut by the image border
            continue
        if any(_same_finger(cnt, i, j, d_cnt) for j in tip_idx):      # same fingertip, skip duplicate
            continue
        tips.append(p); tip_idx.append(i)
        if len(tips) == 5:
            break
    valleys = []
    try:
        df = cv2.convexityDefects(cnt, hidx.reshape(-1, 1))
        if df is not None:
            for s_, e, f, dep in df.reshape(-1, 4):
                if dep / 256.0 > 0.4 * r and d_cnt[f] > 0.9 * r:       # ignore the cut wrist edge
                    valleys.append(tuple(int(v) for v in cnt[f][0]))
    except cv2.error:
        pass
    return dict(center=(int(c[0]), int(c[1])), radius=float(r), area=float(cv2.contourArea(cnt)),
                bbox=tuple(int(v) for v in cv2.boundingRect(cnt)), hull=hull,
                tips=[tuple(int(v) for v in t) for t in tips], valleys=valleys)


def draw(bgr, regions, chosen, info, win):
    out = bgr.copy()
    cv2.rectangle(out, (win[0], win[1]), (win[0] + win[2], win[1] + win[3]), (0, 255, 255), 1)
    for i, c in enumerate(regions):
        cv2.drawContours(out, [c], -1, (160, 160, 160), 1)
        x, y, _, _ = cv2.boundingRect(c)
        cv2.putText(out, str(i + 1), (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.drawContours(out, [regions[chosen]], -1, (0, 200, 0), 2)
    cv2.polylines(out, [info["hull"].reshape(-1, 1, 2)], True, (255, 120, 0), 2)
    cv2.circle(out, info["center"], int(info["radius"]), (0, 0, 255), 2)
    cv2.circle(out, info["center"], 4, (0, 0, 255), -1)
    for v in info["valleys"]:
        cv2.circle(out, v, 5, (255, 0, 255), -1)
    for n, t in enumerate(info["tips"], 1):
        cv2.line(out, info["center"], t, (0, 165, 255), 1)
        cv2.circle(out, t, 7, (0, 165, 255), -1)
        cv2.putText(out, str(n), (t[0] + 8, t[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    return out
