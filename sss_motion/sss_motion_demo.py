
# sss_motion_demo.py
# Combines: frequency character + motion field warp

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math, os

W = H = 256

def wave(t, f, phase=0):
    return math.sin(2 * math.pi * f * t + phase)

def draw_character(t):
    img = Image.new("RGB", (W, H), (245,245,248))
    d = ImageDraw.Draw(img)
    cx, cy = W//2, H//2
    body = wave(t,1.0)
    color = wave(t,0.5)

    cy += int(body*8)

    col = (int(240+color*10), int(180+color*20), int(120+color*15))
    d.ellipse([cx-60,cy-60,cx+60,cy+60], fill=col, outline=(30,30,30), width=3)
    return img

def warp(img, t):
    arr = np.asarray(img).astype(np.float32)/255.0
    H,W,_ = arr.shape
    yy,xx = np.mgrid[0:H,0:W]
    dx = 6*np.sin(2*np.pi*(t+yy/60))
    dy = 6*np.cos(2*np.pi*(t+xx/70))
    x2 = np.clip(xx+dx,0,W-1).astype(int)
    y2 = np.clip(yy+dy,0,H-1).astype(int)
    out = arr[y2,x2]
    return Image.fromarray((out*255).astype(np.uint8))

frames = []
for i in range(60):
    t = i/60
    base = draw_character(t)
    warped = warp(base,t)
    frames.append(warped)

frames[0].save("demo.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
print("demo.gif saved")
