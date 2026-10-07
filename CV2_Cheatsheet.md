# OpenCV (cv2) Exam Cheatsheet

`import cv2, numpy as np, matplotlib.pyplot as plt` — OpenCV images are **BGR, uint8, shape (rows=h, cols=w, 3)**.

## I/O & display
| Task | Code |
|---|---|
| Read | `cv2.imread(p)` · gray: `cv2.imread(p, cv2.IMREAD_GRAYSCALE)` (returns `None` on bad path) |
| Save | `cv2.imwrite("o.png", img)` |
| Show (notebook) | `plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))`; gray: `plt.imshow(g, cmap="gray")`; `plt.axis("off")` |
| Show (window) | `cv2.imshow("t", img); cv2.waitKey(0); cv2.destroyAllWindows()` |
| Size | `h, w = img.shape[:2]` · `cv2.resize(img, (w2, h2))` · `fx=0.5, fy=0.5` |
| Copy | `img.copy()` (slices are views!) |
| ROI | `img[y0:y1, x0:x1]` |
| Flip / rotate 90 | `cv2.flip(img, 1)` (0 vert, 1 horiz) · `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` |

## Colour spaces
`cv2.cvtColor(img, code)`: `COLOR_BGR2GRAY`, `BGR2RGB`, `BGR2HSV` (H 0-179, S/V 0-255), `BGR2LAB`, `GRAY2BGR`
Channels: `b, g, r = cv2.split(img)` · `cv2.merge([b, g, r])`

## Drawing (colour is BGR, thickness `-1` = filled)
```python
cv2.line(img, (x1,y1), (x2,y2), color, th)
cv2.rectangle(img, (x1,y1), (x2,y2), color, th)
cv2.circle(img, (cx,cy), r, color, th)
cv2.polylines(img, [pts_int32], True, color, th)   # closed polygon
cv2.fillPoly(img, [pts], color)
cv2.putText(img, "txt", (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th)   # y = baseline
(tw, th), base = cv2.getTextSize("txt", font, scale, th)
```

## Arithmetic, masks, blending
- Blend: `cv2.addWeighted(a, α, b, β, γ)` → `α·a + β·b + γ`
- `cv2.add / subtract / absdiff(a, b)` (saturating) · `cv2.convertScaleAbs(img, alpha=1.3, beta=0)` (contrast/brightness)
- Mask: `cv2.bitwise_and(img, img, mask=m)` · `bitwise_or / bitwise_not / bitwise_xor`
- Composite: `fg = and(a, a, mask=m)`; `bg = and(b, b, mask=not(m))`; `out = or(fg, bg)`
- Colour mask: `cv2.inRange(hsv, lower, upper)` → 255 inside range
- Blank canvas: `np.zeros((h, w, 3), np.uint8)`

## Intensity / enhancement
| Op | Code |
|---|---|
| Hist. equalisation (gray) | `cv2.equalizeHist(g)` |
| CLAHE | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(g)` |
| Colormap | `cv2.applyColorMap(g_uint8, cv2.COLORMAP_JET)` (BONE, HOT, ...) |
| Log | `c = 255/np.log(1+g.max()); s = c*np.log(1+g.astype(float))` → clip, uint8 |
| Gamma | `s = (255*(g/255.0)**γ).astype(np.uint8)` (γ<1 brightens) |
| Negative | `255 - g` |
| Histogram | `cv2.calcHist([g],[0],None,[256],[0,256])` or `plt.hist(g.ravel(),256,(0,256))` |
| Grey-world balance | scale each channel by `mean_gray / mean_channel` |

## Filtering
`cv2.blur(img,(k,k))` · `cv2.GaussianBlur(img,(k,k),0)` (k odd) · `cv2.medianBlur(img,k)` (salt & pepper) · `cv2.bilateralFilter(img,9,75,75)` (edge-preserving) · `cv2.filter2D(img,-1,kernel)` · `cv2.Sobel(g, cv2.CV_64F, 1, 0)` · `cv2.Laplacian(g, cv2.CV_64F)`

## Morphology (binary images)
`k = np.ones((5,5), np.uint8)` — `cv2.erode`, `cv2.dilate`, `cv2.morphologyEx(m, cv2.MORPH_OPEN|CLOSE|GRADIENT, k)`
OPEN = erode→dilate (removes small white specks) · CLOSE = dilate→erode (fills small black holes)

## Thresholding
```python
_, b = cv2.threshold(g, T, 255, cv2.THRESH_BINARY)          # also _INV, TRUNC, TOZERO
t, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # auto T
b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize_odd, C)
```
Global fails with uneven light → adaptive. Bimodal histogram → Otsu.

## Segmentation
- **Canny:** `cv2.Canny(g, low, high)` (low:high ≈ 1:2 or 1:3; blur first)
- **Contours:** `cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`; `cv2.contourArea(c)`, `cv2.boundingRect(c)` → (x,y,w,h), `cv2.drawContours(img, cnts, -1, color, th)`, `cv2.arcLength(c, True)`, `cv2.approxPolyDP(c, 0.02*peri, True)`
- **Connected components:** `n, labels = cv2.connectedComponents(m)` · `n, labels, stats, cent = cv2.connectedComponentsWithStats(m)` (stats cols: x,y,w,h,area)
- **Distance transform:** `cv2.distanceTransform(mask, cv2.DIST_L2, 5)`
- **Watershed:** thresh → open → `sure_bg = dilate` → `dist` → `sure_fg = dist > f·max` → `unknown = sure_bg − sure_fg` → `markers = cc(sure_fg)+1; markers[unknown==255]=0` → `cv2.watershed(bgr, markers)` → boundaries `markers == -1`
- **K-Means:** `px = np.float32(img.reshape(-1,3))`; `_, labels, centers = cv2.kmeans(px, K, None, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 10, cv2.KMEANS_RANDOM_CENTERS)`; `out = np.uint8(centers)[labels.flatten()].reshape(img.shape)`
- **GrabCut / floodFill** exist too: `cv2.floodFill(img, mask, seed, newColor, loDiff, upDiff)`

## Geometric transforms
| Transform | Matrix / function |
|---|---|
| Translate | `M = [[1,0,tx],[0,1,ty]]` → `cv2.warpAffine(img, M, (w,h))` |
| Rotate+scale | `M = cv2.getRotationMatrix2D((cx,cy), angle_deg, scale)` (+angle = counter-clockwise) |
| Shear x | `M = [[1,kx,0],[0,1,0]]` (widen canvas by `kx·h`) |
| Rigid | rotation matrix (scale 1) + `M[:,2] += (tx, ty)` |
| Similarity | rotation matrix with scale ≠ 1 + translation |
| Affine (3 pts) | `M = cv2.getAffineTransform(src3, dst3)` (float32) → `warpAffine` |
| Perspective (4 pts) | `M = cv2.getPerspectiveTransform(src4, dst4)` → `cv2.warpPerspective(img, M, (w,h))` |
| Homography (≥4, noisy) | `H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)` |
| Rotate w/o clipping | enlarge canvas: `nw = h|sin|+w|cos|`, `nh = h|cos|+w|sin|`, then `M[:,2] += (nw/2-w/2, nh/2-h/2)` |
`dsize` is **(width, height)**. Extra: `borderValue=255`, `flags=cv2.INTER_LINEAR`.
Affine keeps parallel lines; perspective keeps straight lines only.

## Hough
```python
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)                         # (rho, theta)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=10)   # x1,y1,x2,y2
circ  = cv2.HoughCircles(g_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=35, param1=100, param2=25, minRadius=20, maxRadius=40)
```
Pipeline: gray → blur → Canny → (ROI mask) → Hough. `threshold`/`param2`: higher = stricter. Circle `minDist` ≈ diameter.

## Features (SIFT)
```python
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
good = [m for m, n in cv2.BFMatcher().knnMatch(des1, des2, k=2) if m.distance < 0.75*n.distance]   # Lowe ratio
src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
box = cv2.perspectiveTransform(corners_of_template.reshape(-1,1,2), H)     # bounding box in scene
cv2.polylines(scene, [np.int32(box)], True, (0,255,0), 3)
cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
cv2.drawKeypoints(img, kp, None)
```
Panorama: `H` maps right→left; `pano = warpPerspective(right, H, (wL+wR, h)); pano[:hL,:wL] = left`. Shortcut: `cv2.Stitcher_create().stitch([a, b])`.
Also: `cv2.ORB_create()` (binary, use `NORM_HAMMING`), `cv2.cornerHarris`, `cv2.goodFeaturesToTrack`.

## Video
```python
cap = cv2.VideoCapture(path_or_0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    ...
    cv2.imshow("w", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"): break
cap.release(); cv2.destroyAllWindows()
```
`cap.get(cv2.CAP_PROP_FPS)`, `CAP_PROP_FRAME_COUNT`. Write: `cv2.VideoWriter("o.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))`. Side-by-side: `np.hstack((a, b))` (same height & channels).
Motion / intrusion: `absdiff(bg, frame)` → threshold → open → `findContours` → area filter → overlap with zone → alarm. Or `cv2.createBackgroundSubtractorMOG2().apply(frame)`.

## Wavelets (PyWavelets)
`coeffs = pywt.wavedec(sig, "db4", level=4, mode="periodization")` → `[cA4, cD4, cD3, cD2, cD1]` · big |detail| = sudden change/anomaly · threshold: `mean + 3·std` or MAD · `pywt.waverec(coeffs, "db4")` · 2-D: `LL, (LH, HL, HH) = pywt.dwt2(img, "haar")`

## NumPy one-liners
`img.shape / dtype / min() / max() / mean()` · `np.clip(x, 0, 255).astype(np.uint8)` · `img.astype(np.float32)` · `np.hstack/vstack` · `np.where(cond)` · `np.unique(markers)` · `img.ravel()` · `np.power(img/255.0, g)`

## Quick "which method?" guide
| Situation | Use |
|---|---|
| Uneven illumination, text/documents | Adaptive threshold |
| Dark/bright objects on plain background | Otsu |
| Distinct colour | HSV `inRange` |
| Touching round objects | Distance transform + watershed (or Hough circles to count) |
| Smooth homogeneous organ | Region growing |
| Straight structures (lanes, borders) | Canny + Hough lines |
| Round objects | Hough circles |
| Recognise object at any scale/rotation | SIFT + ratio test + RANSAC |
| Join overlapping photos | SIFT + homography (`warpPerspective`) |
| Moving object / intrusion | Background subtraction + contours |
| Sudden changes in 1-D signal | Wavelet detail coefficients |
