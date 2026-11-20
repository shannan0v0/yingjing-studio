from pathlib import Path
import os
import subprocess
import tempfile

from PIL import Image, ImageOps


LOOKS = {"电影": "cinematic still photograph, backlight on individual hairs, subtle film grain",
         "自然照片": "realistic natural photography with soft natural light",
         "油画": "a unified hand-painted oil painting with expressive brushwork",
         "素描": "a detailed graphite drawing on textured paper",
         "水墨": "a Chinese ink wash painting with delicate ink gradients"}


def render(runtime, subject, background, style, position, percent, perch, low_memory=False):
    runtime = Path(runtime).resolve()
    engine = runtime / "engine" / "sd-cli.exe"
    names = ["flux-2-klein-4b-q4_0.gguf", "Qwen3-4B-Q4_K_S.gguf", "flux2-vae.safetensors"]
    weights = [runtime / "weights" / name for name in names]
    for file in [engine] + weights:
        if not file.is_file() or file.stat().st_size == 0:
            raise RuntimeError("离线生成组件缺失：" + file.name)
    location = {"左侧":"left", "居中":"center", "右侧":"right"}[position]
    support = ("Rest its front paws over a rough fallen branch across the foreground; gently curl "
               "the paws around the bark, occluding the upper edge with contact shadows."
               if perch else "Integrate the pose and weight naturally with the scene.")
    prompt = ("Create one finished composition. Use the environment and camera viewpoint from image 2. "
              "Place the exact main subject from image 1 in the "+location+" foreground, occupying "
              +str(percent)+" percent of image height. Preserve identity, markings and gaze. "
              "Discard all furniture and backdrop from image 1. "+support+
              " Include plausible supporting anatomy. Relight the subject from the scene: preserve "
              "deep occlusion and soft reflected fill, with environment-colored highlights. "
              "Render the whole picture as "+LOOKS[style]+". No lettering or panels.")
    with tempfile.TemporaryDirectory(prefix="studio-render-") as temp:
        work = Path(temp)
        for source, dest in ((subject,"subject.png"),(background,"scene.png")):
            with Image.open(source) as file:
                im=ImageOps.exif_transpose(file).convert("RGB")
                im.thumbnail((512,512),Image.Resampling.LANCZOS)
                im.save(work / dest)
        ratio=min(2.5,max(.4,im.width / im.height))
        width=max(32,round((393216*ratio)**.5/32)*32)
        height=max(32,round(width/ratio/32)*32)
        args=[str(engine),"--diffusion-model",str(weights[0]),"--llm",str(weights[1]),"--vae",str(weights[2]),
              "-r","subject.png","-r","scene.png","-p",prompt,"-W",str(width),"-H",str(height),
              "--steps","4","--cfg-scale","1","--sampling-method","euler","-t","8",
              "--flash-attn","on","--backend","cpu","--params-backend","cpu",
              "--rng","cpu","-s","42","-o","result.png"]
        if low_memory:
            args[args.index("--params-backend")+1]="disk"
            args.extend(["--disable-prefetch","--conditioning-cache-size","0",
                         "--vae-tiling","--vae-tile-size","256x256"])
        log=Path(tempfile.gettempdir())/"PhotoStudio-render.log"
        with log.open("wb") as stream:
            flags=getattr(subprocess,"CREATE_NO_WINDOW",0)
            devices=subprocess.run([str(engine),"--list-devices"],capture_output=True,creationflags=flags)
            has_gpu=b"vulkan0" in devices.stdout.lower()
            status=None
            if has_gpu:
                gpu_args=list(args)
                gpu_args[gpu_args.index("--backend")+1]="diffusion=Vulkan0,vae=cpu,te=cpu"
                if low_memory:
                    gpu_args.extend(["--max-vram","Vulkan0=1.5"])
                else:
                    gpu_args[gpu_args.index("--params-backend")+1]="diffusion=Vulkan0,vae=cpu,te=cpu"
                status=subprocess.run(gpu_args,cwd=work,stdout=stream,stderr=subprocess.STDOUT,
                                      env=dict(os.environ,GGML_VK_MAX_NODES_PER_SUBMIT="1"),
                                      creationflags=flags)
            if status is None or status.returncode or not (work/"result.png").is_file():
                stream.write(b"\nSelecting CPU renderer.\n")
                stream.flush()
                status=subprocess.run(args,cwd=work,stdout=stream,stderr=subprocess.STDOUT,
                                      creationflags=flags)
        if status.returncode or not (work/"result.png").is_file():
            raise RuntimeError("本地生成失败。日志："+str(log))
        with Image.open(work/"result.png") as image:
            return image.convert("RGB")
