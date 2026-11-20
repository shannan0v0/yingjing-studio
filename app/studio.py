import argparse
import math
import os
import queue
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageTk
from generative import render
from detail import enhance


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT if getattr(sys, "frozen", False) else ROOT.parent
RUNTIME = Path(os.environ.get("PHOTOSTUDIO_RUNTIME", (Path(sys.executable).parent if getattr(sys,"frozen",False) else ROOT.parent) / "local-gen"))
STYLES = ("自然照片", "电影", "油画", "素描", "水墨")
POSITIONS = {"左侧": 0.27, "居中": 0.5, "右侧": 0.73}


def open_image(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGBA")


def cutout(image):
    if image.getchannel("A").getextrema()[0] == 255:
        rgb = np.asarray(image.convert("RGB").resize((320, 320)), dtype=np.float32)
        rgb /= max(1.0, rgb.max())
        rgb = (rgb - np.float32([.485, .456, .406])) / np.float32([.229, .224, .225])
        net = cv2.dnn.readNetFromONNX(np.fromfile(ROOT / "u2netp.onnx", dtype=np.uint8))
        net.setInput(rgb.transpose(2, 0, 1)[None].copy())
        mask = net.forward(net.getUnconnectedOutLayersNames()[0])[0, 0]
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        image.putalpha(Image.fromarray(mask).resize(image.size, Image.Resampling.BILINEAR)
                       .point(lambda x: 0 if x < 10 else min(255, int((x-10)*255/235))))
    box = image.getchannel("A").getbbox()
    if not box:
        raise ValueError("未找到主体，请换一张轮廓清晰的照片。")
    return image.crop(box)


def stylize(canvas, style):
    rgb = np.asarray(canvas)
    if style == "电影":
        toned = ImageEnhance.Contrast(canvas).enhance(1.1)
        r, g, b = toned.split()
        return Image.merge("RGB", (r.point(lambda x: min(255, x*1.03+2)), g,
                                   b.point(lambda x: x*.92+6)))
    if style == "油画":
        smooth = cv2.pyrMeanShiftFiltering(rgb, 7, 22)
        return Image.fromarray(cv2.addWeighted(smooth, 1.35, cv2.GaussianBlur(smooth,(0,0),2), -.35, 0))
    if style in ("素描", "水墨"):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        base = cv2.GaussianBlur(gray, (0, 0), 5)
        pencil = cv2.divide(gray, np.maximum(base, 1), scale=246)
        if style == "素描":
            return Image.fromarray(pencil).convert("RGB")
        wash = cv2.bilateralFilter(gray, 9, 35, 35)
        wash = (wash // 24 * 24).astype("uint8")
        ink = cv2.addWeighted(pencil, .58, wash, .42, 0)
        return ImageOps.colorize(Image.fromarray(ink), "#172022", "#faf6e8")
    return canvas


def scene_light(canvas):
    sample = np.asarray(canvas.resize((96, 64)), dtype=np.float32)[:44]
    lum = sample @ np.float32([.299, .587, .114])
    weights = np.maximum(lum - np.percentile(lum, 75), 0)
    columns = np.linspace(-1, 1, lum.shape[1])
    direction = np.clip(np.sum(weights * columns) / max(weights.sum(), 1) * 2.8, -1, 1)
    tint = np.average(sample.reshape(-1, 3), axis=0, weights=weights.ravel()+.01)
    return float(direction), np.clip(tint / max(tint.mean(), 1), .74, 1.32)


def illuminate(actor, direction, tint):
    rgba = np.array(actor, dtype=np.float32)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3] / 255
    width = actor.width
    # Estimate the light-facing silhouette normal from the softened matte.
    soft = cv2.GaussianBlur(alpha, (0, 0), max(1, width*.012))
    dx = cv2.Sobel(soft, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(soft, cv2.CV_32F, 0, 1)
    rim = np.clip((-direction*dx + .45*dy)*width*.008, 0, 1) * alpha
    across = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    exposure = .96 + .23*direction*across
    rgb *= exposure[:, :, None] * (.55 + .45*tint)
    rgb += rim[:, :, None] * tint * 42
    # Remove green spill left by a coloured source backdrop on the matte edge.
    spill = np.maximum(rgb[:,:,1] - np.maximum(rgb[:,:,0],rgb[:,:,2]), 0)
    rgb[:,:,1] -= spill * np.clip((1-soft)*2,0,1)
    return Image.fromarray(np.clip(rgba,0,255).astype("uint8"), "RGBA")


def compose(subject, background, style, position, percent, perch=False):
    canvas = open_image(background).convert("RGB")
    canvas.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
    actor = cutout(open_image(subject))
    w, h = canvas.size
    scale = min(h * percent / 100 / actor.height, w*.95/actor.width)
    actor = actor.resize((max(1,round(actor.width*scale)), max(1,round(actor.height*scale))),
                         Image.Resampling.LANCZOS)
    x = min(max(0,round(w*POSITIONS[position]-actor.width/2)), w-actor.width)
    direction, tint = scene_light(canvas)
    actor = illuminate(actor, direction, tint)
    y = h - actor.height
    if perch:
        y -= round(h*.055)
        trunk = open_image(ROOT/"perch.png")
        bounds = trunk.getchannel("A").point(lambda a: 255 if a > 125 else 0).getbbox()
        trunk = trunk.crop(bounds)
        tw = round(w*1.13)
        trunk = trunk.resize((tw,max(1,round(tw*trunk.height/trunk.width))),Image.Resampling.LANCZOS)
        matte = np.asarray(actor.getchannel("A"))
        middle = matte[:, actor.width//2]
        visible = np.flatnonzero(middle > 128)
        chest = visible[-1] if visible.size else actor.height*.8
        top = y + round(chest - trunk.height*.13)
        canvas.paste(trunk, ((w-tw)//2,top), trunk)
    shadow = Image.new("L", canvas.size)
    shadow.paste(actor.getchannel("A").point(lambda a: a//2), (x-round(direction*w*.008),y+max(1,h//90)))
    if perch:
        shadow.paste(0,(0,0,w,max(0,top)))
    canvas.paste((31,29,24), (0,0,w,h), shadow.filter(ImageFilter.GaussianBlur(max(1,w/160))))
    canvas.paste(actor, (x,y), actor)
    return stylize(canvas, style)


class App:
    def __init__(self, root):
        self.root = root
        self.result = None
        self.events = queue.Queue()
        root.title("双图画室 · 离线版")
        page = ttk.Frame(root,padding=16)
        page.pack(fill="both",expand=True)
        page.columnconfigure(1,weight=1,minsize=300)
        self.paths = [tk.StringVar(),tk.StringVar()]
        for i,label in enumerate(("人物或宠物", "背景照片")):
            ttk.Label(page,text=label).grid(row=i,column=0,sticky="w")
            ttk.Entry(page,textvariable=self.paths[i]).grid(row=i,column=1,sticky="ew",padx=8,pady=4)
            ttk.Button(page,text="选择图片",command=lambda n=i:self.pick(n)).grid(row=i,column=2)
        self.style = tk.StringVar(value="电影")
        self.position = tk.StringVar(value="居中")
        self.size = tk.IntVar(value=64)
        self.generate = tk.BooleanVar(value=(RUNTIME/"weights").is_dir())
        self.perch = tk.BooleanVar(value=False)
        self.hd = tk.BooleanVar(value=(Path(os.environ.get("PHOTOSTUDIO_HD", ROOT / "hd")) / "realesrgan-ncnn-vulkan.exe").is_file())
        self.low_memory = tk.BooleanVar(value=True)
        controls = ttk.Frame(page)
        controls.grid(row=2,columnspan=3,sticky="w",pady=6)
        for label,var,values in (("画风",self.style,STYLES),("位置",self.position,list(POSITIONS))):
            ttk.Label(controls,text=label).pack(side="left")
            ttk.Combobox(controls,textvariable=var,values=values,state="readonly",width=8).pack(side="left",padx=5)
        ttk.Label(controls,text="主体高度 %").pack(side="left")
        ttk.Combobox(controls,textvariable=self.size,values=(35,50,64,80,95),state="readonly",width=4).pack(side="left")
        actions=ttk.Frame(page)
        actions.grid(row=3,columnspan=3,sticky="w",pady=6)
        ttk.Checkbutton(actions,text="本地生成式融合",variable=self.generate).pack(side="left")
        ttk.Checkbutton(actions,text="前爪搭在树干上",variable=self.perch).pack(side="left")
        ttk.Button(actions,text="载入示例照片",command=self.demo).pack(side="left")
        self.go=ttk.Button(actions,text="生成预览",command=self.run)
        self.go.pack(side="left",padx=6)
        self.save=ttk.Button(actions,text="保存 PNG",command=self.export,state="disabled")
        self.save.pack(side="left")
        detailbar=ttk.Frame(page)
        detailbar.grid(row=4,columnspan=3,sticky="w")
        ttk.Checkbutton(detailbar,text="生成后增强至 4× 分辨率",variable=self.hd).pack(side="left")
        self.existing=ttk.Button(detailbar,text="增强已有成图…",command=self.open_for_enhancement)
        self.existing.pack(side="left",padx=6)
        ttk.Checkbutton(detailbar,text="低内存生成",variable=self.low_memory).pack(side="left")
        self.status=tk.StringVar(value="所有图片均在本机处理。")
        ttk.Label(page,textvariable=self.status).grid(row=5,columnspan=3,sticky="w")
        self.preview=ttk.Label(page)
        self.preview.grid(row=6,columnspan=3,pady=8)

    def demo(self):
        self.perch.set(True)
        for field,name in zip(self.paths,("cat-demo.jpg","forest-demo.jpg")):
            field.set(str(SAMPLES/name))

    def pick(self,i):
        path=filedialog.askopenfilename(filetypes=[("图片","*.jpg *.jpeg *.png *.webp *.bmp")])
        if path:
            self.paths[i].set(path)

    def open_for_enhancement(self):
        path=filedialog.askopenfilename(filetypes=[("图片","*.png *.jpg *.jpeg *.webp *.bmp")])
        if path:
            self.run(path)

    def run(self, existing_path=None):
        args=tuple(p.get() for p in self.paths)+(self.style.get(),self.position.get(),self.size.get(),self.perch.get())
        use_model=self.generate.get()
        low_memory=self.low_memory.get()
        high_resolution=self.hd.get() or bool(existing_path)
        if not existing_path and not all(Path(p).is_file() for p in args[:2]):
            messagebox.showwarning("选择图片","请先选择主体和背景照片。")
            return
        self.go.configure(state="disabled")
        self.existing.configure(state="disabled")
        self.save.configure(state="disabled")
        self.status.set("正在运行本地生成模型，可能需要数分钟…" if use_model else "正在本地抠图和合成…")
        if existing_path:
            self.status.set("正在本地增强细节…")
        def work():
            try:
                if existing_path:
                    picture=open_image(existing_path).convert("RGB")
                elif use_model:
                    picture=render(RUNTIME,*args,low_memory=low_memory)
                else:
                    picture=compose(*args)
                warning=""
                if high_resolution:
                    self.events.put("正在本地增强细节…")
                    try:
                        picture=enhance(picture)
                    except Exception as exc:
                        warning="高清增强失败，已保留原尺寸图片。"+str(exc)
                self.events.put((picture,warning))
            except Exception as exc:
                self.events.put(exc)
        threading.Thread(target=work,daemon=True).start()
        self.root.after(100,self.poll)

    def poll(self):
        try:
            value=self.events.get_nowait()
        except queue.Empty:
            self.root.after(100,self.poll)
            return
        if isinstance(value,str):
            self.status.set(value)
            self.root.after(100,self.poll)
            return
        self.go.configure(state="normal")
        self.existing.configure(state="normal")
        if isinstance(value,Exception):
            self.status.set("生成失败")
            messagebox.showerror("生成失败",str(value))
            return
        self.result,warning=value
        preview=self.result.copy()
        preview.thumbnail((760,430))
        self.photo=ImageTk.PhotoImage(preview)
        self.preview.configure(image=self.photo)
        self.status.set("可保存 %d × %d 像素" % self.result.size)
        self.save.configure(state="normal")
        if warning:
            messagebox.showwarning("高清处理",warning)

    def export(self):
        path=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png")])
        if path:
            try:
                self.result.save(path,format="PNG")
            except OSError as exc:
                messagebox.showerror("保存失败",str(exc))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--demo-output",type=Path)
    parser.add_argument("--style",choices=STYLES,default="电影")
    parser.add_argument("--generative",action="store_true")
    parser.add_argument("--preview-only",action="store_true",help="Save the native resolution without 4x enhancement")
    parser.add_argument("--enhance-input",type=Path)
    args=parser.parse_args()
    if args.enhance_input and args.demo_output:
        enhance(open_image(args.enhance_input)).save(args.demo_output)
    elif args.demo_output:
        params=(SAMPLES/"cat-demo.jpg", SAMPLES/"forest-demo.jpg", args.style, "居中",64,True)
        picture=render(RUNTIME,*params,low_memory=True) if args.generative else compose(*params)
        (picture if args.preview_only else enhance(picture)).save(args.demo_output)
    else:
        window=tk.Tk()
        App(window)
        window.mainloop()
