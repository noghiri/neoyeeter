"""Draw an ASCII text file as a PNG (light: dark-on-white, dark: light-on-navy)."""
import sys
from PIL import Image, ImageDraw, ImageFont
src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(src, encoding="utf-8").read().split("\n")
font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 15)
cw = font.getbbox("M")[2]; lh = 18
w = max(len(l) for l in lines) * cw + 40; h = len(lines) * lh + 40
bg, fg = (("#ffffff", "#cc2b2b") if mode == "light" else ("#0f1722", "#ff8080"))
img = Image.new("RGB", (w, h), bg); d = ImageDraw.Draw(img)
for i, l in enumerate(lines):
    d.text((20, 20 + i * lh), l, fill=fg, font=font)
img.save(dst)
