import subprocess
import os
import tempfile
from pathlib import Path

from PIL import Image


def enhance(image):
    assets = Path(os.environ.get("PHOTOSTUDIO_HD", Path(__file__).resolve().parent / "hd")).resolve()
    source = image.convert("RGB")
    with tempfile.TemporaryDirectory(prefix="photo-hd-") as folder:
        folder = Path(folder)
        before, after = folder / "input.png", folder / "output.png"
        source.save(before)
        args = [str(assets / "realesrgan-ncnn-vulkan.exe"), "-i", str(before),
                "-o", str(after), "-n", "realesrgan-x4plus", "-s", "4", "-t", "128"]
        run = subprocess.run(args, cwd=assets, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        if run.returncode or not after.is_file():
            raise RuntimeError("高清处理失败，请检查 Vulkan 显卡驱动。\n" +
                               run.stderr.decode("utf8", "replace")[-450:])
        with Image.open(after) as output:
            if output.size != (source.width * 4, source.height * 4):
                raise ValueError("超分模型返回了异常尺寸。")
            detail = output.convert("RGB")
        # Retain some of the source grain, particularly on fur and skin.
        return Image.blend(detail, source.resize(detail.size, Image.Resampling.LANCZOS), 0.22)
