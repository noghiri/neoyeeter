"""Ray-marched ASCII banhammer.

usage: python render.py OUTDIR INSCRIPTION
Writes OUTDIR/hammer_light.txt, OUTDIR/hammer_dark.txt and OUTDIR/debug_lum.png.
"""
import math
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
INSCRIPTION = sys.argv[2] if len(sys.argv) > 2 else "BAN"

W, H = 168, 68            # columns, rows
CW = 0.0545               # world units per column
ASPECT = 2.05             # monospace cell height / width
CH = CW * ASPECT
CAM_Z = 15.0              # perspective camera distance
SS = 5                    # supersamples per axis per cell

# ---------------------------------------------------------------- pose
def rx(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def ry(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

GRIP = np.array([0.0, -8.2, 0.0])          # local point the (off-screen) wielder holds
VIEW = ry(math.radians(-24)) @ rx(math.radians(14))
HEAD_POS = np.array([0.15, 0.9, 2.0])      # view-space head centre in the main frame
MAIN_SWING = 25.0                          # degrees; the head travels toward larger angles

def pose(swing_deg):
    """(m, t) with view = m @ local + t. Every frame pivots about the grip of the main frame."""
    m0 = VIEW @ rz(math.radians(MAIN_SWING))
    grip_view = m0 @ GRIP + HEAD_POS
    m = VIEW @ rz(math.radians(swing_deg))
    return m, grip_view - m @ GRIP

def project(pv):
    s = CAM_Z / (CAM_Z - pv[2])
    return pv[0] * s / CW + W / 2, H / 2 - pv[1] * s / CH

# ---------------------------------------------------------------- shape (signed distance fields)
def sd_round_box(p, b, r):
    q = np.abs(p) - (b - r)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0) - r

def sd_cyl_y(p, y0, y1, rad):
    cy, hh = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
    d_r = np.hypot(p[..., 0], p[..., 2]) - rad
    d_y = np.abs(p[..., 1] - cy) - hh
    return np.minimum(np.maximum(d_r, d_y), 0.0) + np.hypot(np.maximum(d_r, 0.0), np.maximum(d_y, 0.0))

def sdf_local(p):
    """(distance, material): 0 head, 1 bands/collar, 2 wood, 3 grip."""
    head = sd_round_box(p, np.array([2.3, 1.05, 1.05]), 0.14)
    pc = p.copy(); pc[..., 0] = np.abs(pc[..., 0]) - 2.3
    head = np.minimum(head, sd_round_box(pc, np.array([0.12, 0.92, 0.92]), 0.1))
    pb = p.copy(); pb[..., 0] = np.abs(pb[..., 0]) - 1.78
    band = sd_round_box(pb, np.array([0.17, 1.13, 1.13]), 0.07)
    band = np.minimum(band, sd_cyl_y(p, -1.75, -0.9, 0.5))
    y = p[..., 1]
    wood = sd_cyl_y(p, -6.3, -0.9, 0.3 + 0.045 * np.clip((-1.0 - y) / 7.5, 0, 1))
    grip = sd_cyl_y(p, -8.9, -6.3, 0.36 + 0.05 * np.abs(np.sin(y * 7.0)))
    grip = np.minimum(grip, np.linalg.norm(p - np.array([0.0, -8.95, 0.0]), axis=-1) - 0.44)
    d = np.stack([head, band, wood, grip], axis=-1)
    return np.min(d, axis=-1), np.argmin(d, axis=-1)

def sdf_view(pv, m, t):
    return sdf_local((pv - t) @ m)

# ---------------------------------------------------------------- inscription
U_HALF, V_HALF = 1.5, 0.82        # inscription area on the local +z face, between the bands

def text_mask(text):
    mw, mh = 900, int(900 * V_HALF / U_HALF)
    img = Image.new("L", (mw, mh), 0); d = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/impact.ttf", 200)
    b = d.textbbox((0, 0), text, font=font)
    font = ImageFont.truetype("C:/Windows/Fonts/impact.ttf", int(200 * min(0.9 * mw / (b[2] - b[0]), 0.8 * mh / (b[3] - b[1]))))
    b = d.textbbox((0, 0), text, font=font)
    d.text(((mw - (b[2] - b[0])) / 2 - b[0], (mh - (b[3] - b[1])) / 2 - b[1]), text, fill=255, font=font)
    return np.asarray(img, dtype=np.float32) / 255.0

MASK = text_mask(INSCRIPTION)

def engraving(local, mat, n_local):
    on_face = (mat == 0) & (n_local[..., 2] > 0.8)
    u = (np.nan_to_num(local[..., 0]) / U_HALF + 1) * 0.5      # reading direction: local +x
    v = (1 - np.nan_to_num(local[..., 1]) / V_HALF) * 0.5      # letter "up": local +y
    inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
    px = np.clip((u * (MASK.shape[1] - 1)).astype(int), 0, MASK.shape[1] - 1)
    py = np.clip((v * (MASK.shape[0] - 1)).astype(int), 0, MASK.shape[0] - 1)
    return np.where(on_face & inside, MASK[py, px], 0.0)

# ---------------------------------------------------------------- ray marching
LIGHT = np.array([-0.45, 0.75, 0.55]); LIGHT /= np.linalg.norm(LIGHT)

def render(swing_deg):
    """Supersampled buffers: hit, lum, normal, depth, material, engraving."""
    m, t = pose(swing_deg)
    xs = ((np.arange(W * SS) + 0.5) / SS - W / 2) * CW
    ys = (H / 2 - (np.arange(H * SS) + 0.5) / SS) * CH
    X, Y = np.meshgrid(xs, ys)
    origin = np.array([0.0, 0.0, CAM_Z])
    dirs = np.stack([X, Y, -np.full_like(X, CAM_Z)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    tt = np.full(X.shape, CAM_Z - 7.0)
    hit = np.zeros(X.shape, bool)
    for _ in range(160):
        d, _ = sdf_view(origin + dirs * tt[..., None], m, t)
        hit |= d < 0.002
        tt = np.where(hit, tt, tt + d * 0.9)
        if np.all(hit | (tt > CAM_Z + 14)):
            break
    p = origin + dirs * tt[..., None]
    n = np.zeros_like(p)
    for i in range(3):
        dp = np.zeros(3); dp[i] = 0.004
        n[..., i] = sdf_view(p + dp, m, t)[0] - sdf_view(p - dp, m, t)[0]
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    _, mat = sdf_view(p, m, t)
    local = (p - t) @ m
    n_local = n @ m

    sh = np.ones(X.shape); st = np.full(X.shape, 0.03)
    for _ in range(48):
        d, _ = sdf_view(p + n * 0.01 + LIGHT * st[..., None], m, t)
        sh = np.minimum(sh, np.clip(10 * d / st, 0, 1))
        st += np.clip(d, 0.02, 0.4)
    ao = np.ones(X.shape)
    for k in range(1, 5):
        h = 0.06 * k
        ao -= (h - sdf_view(p + n * h, m, t)[0]) * (0.6 / k)
    ao = np.clip(ao, 0.2, 1)

    view = -dirs
    ndl = np.clip(np.sum(n * LIGHT, -1), 0, 1)
    half = LIGHT + view; half /= np.linalg.norm(half, axis=-1, keepdims=True)
    ndh = np.clip(np.sum(n * half, -1), 0, 1)
    fres = (1 - np.clip(np.sum(n * view, -1), 0, 1)) ** 3
    albedo = np.choose(mat, [0.85, 0.55, 0.62, 0.38])
    spec_k = np.choose(mat, [0.9, 0.6, 0.12, 0.05])
    spec_p = np.choose(mat, [60, 40, 10, 6])
    rim = np.choose(mat, [0.35, 0.25, 0.1, 0.05])
    lum = albedo * (0.12 + 0.88 * ndl * sh) * ao + spec_k * ndh ** spec_p * sh + rim * fres
    eng = engraving(local, mat, n_local) * hit
    lum = lum * (1 - 0.85 * eng)
    return dict(hit=hit, lum=lum, n=n, depth=tt, mat=mat, eng=eng)

# ---------------------------------------------------------------- ASCII conversion
FILL = " .:-=+*#%@"          # sparse -> dense

def edge_map(b):
    """Sub-pixels on a silhouette, crease, depth jump, material change or letter boundary."""
    hit, n, depth, mat, eng = b["hit"], b["n"], b["depth"], b["mat"], b["eng"] > 0.5
    e = np.zeros(hit.shape, bool)
    for axis in (0, 1):
        a = [slice(None)] * 2; c = [slice(None)] * 2
        a[axis] = slice(0, -1); c[axis] = slice(1, None)
        a, c = tuple(a), tuple(c)
        both = hit[a] & hit[c]
        sil = hit[a] != hit[c]
        inner = both & ((np.sum(n[a] * n[c], -1) < 0.75) | (np.abs(depth[a] - depth[c]) > 0.35) | (mat[a] != mat[c]))
        e[a] |= (sil & hit[a]) | inner
        e[c] |= sil & hit[c]
    return e

def edge_char(ys, xs):
    """Line character for one cell's edge sub-pixels, or None if they don't form a line."""
    if len(xs) < SS * 0.8:
        return None
    pts = np.stack([xs, ys * ASPECT], 1).astype(float)
    pts -= pts.mean(0)
    w, v = np.linalg.eigh(pts.T @ pts / len(pts))
    if w[1] < 1e-6 or w[0] / w[1] > 0.35:      # blob or corner
        return None
    dx, dy = v[:, 1]
    ang = math.degrees(math.atan2(-dy, dx)) % 180
    if ang < 20 or ang >= 160:
        return "_" if ys.mean() > SS * 0.55 else "-"
    if ang < 70:
        return "/"
    if ang < 110:
        return "|"
    return "\\"

def to_cells(b, theme):
    hit, lum, engm = b["hit"], b["lum"], b["eng"]
    e = edge_map(b)
    cov = hit.reshape(H, SS, W, SS).mean(axis=(1, 3))
    letter = engm.reshape(H, SS, W, SS).mean(axis=(1, 3))
    l = np.where(hit & (engm < 0.5), lum, np.nan).reshape(H, SS, W, SS)
    with np.errstate(all="ignore"):
        val = np.nan_to_num(np.nanmean(l, axis=(1, 3)))
    # tone: half rank-equalised, half linear, so big flat faces spread over the ramp
    sel = cov > 0.5
    ranks = np.zeros_like(val)
    order = np.argsort(val[sel]); r = np.empty_like(order); r[order] = np.arange(len(order))
    ranks[sel] = r / max(len(order) - 1, 1)
    lo, hi = np.percentile(val[sel], [2, 98])
    lin = np.clip((val - lo) / (hi - lo), 0, 1)
    norm = 0.6 * ranks + 0.4 * lin
    if theme == "light":
        norm = 1 - norm
    else:
        norm = norm ** 0.55          # denser face on dark grounds, so the blank letters stand out
    grid = [[None] * W for _ in range(H)]
    for r_ in range(H):
        for c in range(W):
            if cov[r_, c] < 0.2:
                continue
            if letter[r_, c] > 0.45:
                grid[r_][c] = " " if theme == "dark" else "@"
                continue
            ys, xs = np.nonzero(e[r_ * SS:(r_ + 1) * SS, c * SS:(c + 1) * SS])
            ch = edge_char(ys, xs)
            if ch is None:
                if cov[r_, c] < 0.5:
                    continue
                ch = FILL[max(1, int(round(norm[r_, c] * (len(FILL) - 1))))]
            grid[r_][c] = ch
    return grid

# ---------------------------------------------------------------- motion + impact
def dir_char(dc, dr):
    ang = math.degrees(math.atan2(-dr * ASPECT, dc)) % 180
    if ang < 22.5 or ang >= 157.5: return "-"
    if ang < 67.5: return "/"
    if ang < 112.5: return "|"
    return "\\"

def stroke(grid, pts, faded_until=0.0, fade_char="."):
    """Draw a polyline into empty cells; the first `faded_until` fraction uses `fade_char`."""
    total = len(pts) - 1
    for i in range(total):
        (c0, r0), (c1, r1) = pts[i], pts[i + 1]
        ch = fade_char if i / total < faded_until else dir_char(c1 - c0, r1 - r0)
        n = int(max(abs(c1 - c0), abs(r1 - r0)) * 2) + 1
        for k in range(n + 1):
            a = k / n
            ci, ri = int(round(c0 + (c1 - c0) * a)), int(round(r0 + (r1 - r0) * a))
            if 0 <= ri < H and 0 <= ci < W and grid[ri][ci] is None:
                grid[ri][ci] = ch

def trail(local_pt, s0, s1, steps=30):
    pts = []
    for k in range(steps + 1):
        m, t = pose(s0 + (s1 - s0) * k / steps)
        pts.append(project(m @ local_pt + t))
    return pts

def burst(grid, m, t):
    """Shock rays and debris off the leading (-x) striking face."""
    centre = project(m @ np.array([-2.6, 0.0, 0.0]) + t)
    normal = m @ np.array([-1.0, 0.0, 0.0])
    base = math.atan2(normal[1], normal[0])          # screen angle, y up
    rng = np.random.default_rng(11)
    for k, spread in enumerate(np.linspace(-1.3, 1.3, 11)):
        a = base + spread
        dc, dr = math.cos(a), -math.sin(a) / ASPECT
        r_in = 4.0 + (k % 2) * 3.0
        r_out = r_in + (11.0 if k % 2 == 0 else 6.0)
        stroke(grid, [(centre[0] + dc * d, centre[1] + dr * d) for d in np.linspace(r_in, r_out, 8)])
    for _ in range(30):
        a = base + rng.uniform(-1.5, 1.5); d = rng.uniform(6, 22)
        ci = int(round(centre[0] + math.cos(a) * d)); ri = int(round(centre[1] - math.sin(a) / ASPECT * d))
        if 0 <= ri < H and 0 <= ci < W and grid[ri][ci] is None:
            grid[ri][ci] = str(rng.choice(list("*'.,`")))

def to_text(grid):
    rows = ["".join(ch or " " for ch in row).rstrip() for row in grid]
    while rows and not rows[0]: rows.pop(0)
    while rows and not rows[-1]: rows.pop()
    pad = min(len(r) - len(r.lstrip()) for r in rows if r)
    return "\n".join(r[pad:] for r in rows)

if __name__ == "__main__":
    main = render(MAIN_SWING)
    g = np.clip((np.where(main["hit"], main["lum"], -0.2) + 0.2) / 1.4, 0, 1)
    Image.fromarray((g * 255).astype(np.uint8)).resize((W * 8, int(H * 8 * ASPECT))).save(f"{OUT}/debug_lum.png")
    m, t = pose(MAIN_SWING)
    for theme in ("light", "dark"):
        grid = to_cells(main, theme)
        for i, x in enumerate(np.linspace(-2.3, 2.45, 7)):
            span = 14 + 6 * (i % 2)
            stroke(grid, trail(np.array([x, 1.2, -1.1]), MAIN_SWING - span, MAIN_SWING - 1.5), faded_until=0.4)
        burst(grid, m, t)
        open(f"{OUT}/hammer_{theme}.txt", "w", encoding="utf-8").write(to_text(grid))
