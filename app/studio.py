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
        # The layout uses logical pixels, including on high-DPI Windows desktops.
        root.tk.call("tk", "scaling", 96 / 72)
        root.title("映境画室 · 离线版")
        root.geometry("1040x660")
        root.minsize(790, 570)
        root.option_add("*Font", ("Microsoft YaHei UI", 10))
        ink, paper, panel, accent = "#24332d", "#edece5", "#fbfaf6", "#285f49"
        root.configure(bg=paper)
        self.paths = [tk.StringVar(), tk.StringVar()]
        self.style = tk.StringVar(value="电影")
        self.position = tk.StringVar(value="居中")
        self.size = tk.IntVar(value=64)
        self.generate = tk.BooleanVar(value=(RUNTIME/"weights").is_dir())
        self.perch = tk.BooleanVar(value=False)
        self.low_memory = tk.BooleanVar(value=True)
        self.hd = tk.BooleanVar(value=(Path(os.environ.get("PHOTOSTUDIO_HD", ROOT / "hd")) / "realesrgan-ncnn-vulkan.exe").is_file())
        self.status = tk.StringVar(value="选择两张照片，或载入示例开始创作")

        def button(parent, title, action, primary=False):
            return tk.Button(parent, text=title, command=action, relief="flat", bd=0,
                             bg=accent if primary else "#e4e9df", fg="white" if primary else ink,
                             activebackground="#bfd0bd", cursor="hand2", padx=10, pady=5)

        def label(parent, title, size=10, bold=False):
            return tk.Label(parent,text=title,bg=parent["bg"],fg=ink,
                            font=("Microsoft YaHei UI",size,"bold" if bold else "normal"))

        bar=tk.Frame(root,bg=ink,padx=20,pady=9)
        bar.pack(fill="x")
        tk.Label(bar,text="映境画室",bg=ink,fg="#fff8e9",
                 font=("Microsoft YaHei UI",18,"bold")).pack(side="left")
        button(bar,"载入示例照片",self.demo).pack(side="right")
        self.existing=button(bar,"增强已有成图…",self.open_for_enhancement)
        self.existing.pack(side="right",padx=8)

        body=tk.Frame(root,bg=paper,padx=14,pady=12)
        body.pack(fill="both",expand=True)
        sidebar=tk.Frame(body,bg=panel,padx=14,pady=10)
        sidebar.pack(side="left",fill="y")
        label(sidebar,"素材",13,True).pack(anchor="w")
        self.thumbs=[]
        self.thumb_images=[None,None]
        for i,title in enumerate(("主体照片","背景照片")):
            row=button(sidebar,"＋   "+title,lambda i=i:self.pick(i))
            row.pack(fill="x",pady=(6,0))
            self.thumbs.append(row)
            self.paths[i].trace_add("write",lambda *_, i=i:self.update_thumb(i))

        label(sidebar,"画风",13,True).pack(anchor="w",pady=(12,3))
        styles=tk.Frame(sidebar,bg=panel)
        styles.pack(anchor="w")
        for i,name in enumerate(STYLES):
            tk.Radiobutton(styles,text=name,variable=self.style,value=name,indicatoron=False,
                           bg="#e8e9e2",fg=ink,selectcolor="#b4cfb8", relief="flat",
                           bd=0,padx=7,pady=3,cursor="hand2").grid(row=i//3,column=i%3,sticky="ew",padx=1,pady=1)

        layout=tk.Frame(sidebar,bg=panel)
        layout.pack(fill="x",pady=(10,6))
        for i,(title,var,values) in enumerate((("主体位置",self.position,list(POSITIONS)),
                                              ("高度占比 %",self.size,(35,50,64,80,95)))):
            label(layout,title).grid(row=0,column=i,sticky="w")
            ttk.Combobox(layout,textvariable=var,values=values,state="readonly",width=9).grid(row=1,column=i,padx=(0,6))
        for title,var in (("本地生成式融合",self.generate),("前爪搭在树干上",self.perch),
                          ("4× 高清增强",self.hd),("低内存生成",self.low_memory)):
            tk.Checkbutton(sidebar,text=title,variable=var,bg=panel,fg=ink,activebackground=panel,
                           highlightthickness=0,bd=0).pack(anchor="w")
        self.go=button(sidebar,"生成预览",self.run,True)
        self.go.pack(side="bottom",fill="x",pady=(6,0))

        workspace=tk.Frame(body,bg=panel,padx=12,pady=10)
        workspace.pack(side="left",fill="both",expand=True,padx=(12,0))
        header=tk.Frame(workspace,bg=panel)
        header.pack(fill="x",pady=(0,8))
        self.caption=label(header,"作品示例",13,True)
        self.caption.pack(side="left")
        self.save=button(header,"保存 PNG",self.export)
        self.save.config(state="disabled")
        self.save.pack(side="right")
        tk.Label(workspace,textvariable=self.status,anchor="w",bg=panel,fg="#54645a").pack(side="bottom",fill="x",pady=(8,0))
        self.preview=tk.Canvas(workspace,bg="#e1e4da",bd=0,highlightthickness=0)
        self.preview.pack(fill="both",expand=True)
        sample_art=(ROOT if getattr(sys,"frozen",False) else ROOT.parent/"assets")/"result.jpg"
        self.cover=open_image(sample_art).convert("RGB") if sample_art.is_file() else None
        self.preview.bind("<Configure>",self.draw_preview)

    def update_thumb(self,i):
        path=Path(self.paths[i].get())
        try:
            image=open_image(path)
            image.thumbnail((56,36),Image.Resampling.LANCZOS)
            self.thumb_images[i]=ImageTk.PhotoImage(image)
        except (OSError,ValueError):
            self.thumb_images[i]=None
        self.thumbs[i].configure(image=self.thumb_images[i] or "",compound="left",
                                 text="  "+("主体" if i==0 else "背景")+" · "+path.stem[:10])

    def draw_preview(self,event=None):
        image=self.result if self.result is not None else self.cover
        if image is None:
            return
        w,h=self.preview.winfo_width(),self.preview.winfo_height()
        if min(w,h)<8:
            return
        scale=min((w-6)/image.width,(h-6)/image.height)
        display=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(display)
        self.preview.delete("art")
        self.preview.create_image(w/2,h/2,image=self.photo,tags="art")

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
        self.caption.configure(text="成图预览")
        self.draw_preview()
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
