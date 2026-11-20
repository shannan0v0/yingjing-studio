import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

exe=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve()
out.parent.mkdir(parents=True,exist_ok=True)
tmp=out.parent/'temp'
tmp.mkdir(exist_ok=True)
start=time.perf_counter()
proc=subprocess.Popen([str(exe),'--generative','--demo-output',str(out)],env=dict(os.environ,TEMP=str(tmp),TMP=str(tmp)))
parent=psutil.Process(proc.pid)
peak_rss=peak_private=0
phases={}
while proc.poll() is None:
    rss=private=0
    stage="desktop"
    try:
        group=[parent]+parent.children(recursive=True)
    except psutil.NoSuchProcess:
        group=[]
    for p in group:
        try:
            m=p.memory_info()
            rss+=m.rss
            private+=m.private
            if p.name().lower()=='sd-cli.exe':
                stage='generation'
            elif 'realesrgan' in p.name().lower():
                stage='upscale'
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    peak_rss=max(peak_rss,rss)
    peak_private=max(peak_private,private)
    phases[stage]=max(phases.get(stage,0),rss)
    result=dict(seconds=round(time.perf_counter()-start,2),rss=peak_rss,private=peak_private,phase_rss=phases)
    out.with_suffix('.json').write_text(json.dumps(result))
    time.sleep(.5)
result.update(seconds=round(time.perf_counter()-start,2),exit=proc.returncode)
out.with_suffix('.json').write_text(json.dumps(result))
print(result,flush=True)
sys.exit(proc.returncode)
