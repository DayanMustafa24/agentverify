import os, random, hashlib, json
from pathlib import Path
from PIL import Image
import numpy as np

PERT = Path(r"C:\Smester7\Agentic\Agentic_project\agentverify\data\cache\perturbed")
ORIG = Path(r"C:\Smester7\Agentic\Agentic_project\agentverify\data\raw\splits")
OUT  = Path(r"C:\Smester7\Agentic\Agentic_project\agentverify\tmp_inspection")  # new folder, outside dataset
EXT  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SAMPLE_PER_TYPE = 30
random.seed(42)

def images(root): return [p for p in root.rglob("*") if p.suffix.lower() in EXT]

# 1. structure
for root in (ORIG, PERT):
    print("\n==", root)
    for d, subs, files in os.walk(root):
        depth = Path(d).relative_to(root).parts
        if len(depth) <= 3:
            print("  " * len(depth), Path(d).name, f"({len(files)} files)")

orig = images(ORIG)
pert = images(PERT)
print("\nOriginal images:", len(orig))

# 2. perturbation types = first-level subfolders of PERT (adjust if layout differs)
types = {}
for p in pert:
    types.setdefault(p.relative_to(PERT).parts[0], []).append(p)
for t, ps in types.items():
    print(f"Perturbation '{t}':", len(ps))

# 3. matching by stem (strip common suffixes if your naming adds them)
orig_by_stem = {}
for p in orig:
    orig_by_stem.setdefault(p.stem, []).append(p)
dup_orig = {k: v for k, v in orig_by_stem.items() if len(v) > 1}
print("Original stems appearing more than once:", len(dup_orig))

def match(p):
    return orig_by_stem.get(p.stem)

# 4. automated checks
OUT.mkdir(exist_ok=True)
report = {}
for t, ps in types.items():
    matched, unmatched, corrupt, hashes = 0, [], [], {}
    for p in ps:
        try:
            with Image.open(p) as im: im.verify()
        except Exception as e:
            corrupt.append(str(p)); continue
        m = match(p)
        if m: matched += 1
        else: unmatched.append(str(p))
        h = hashlib.md5(p.read_bytes()).hexdigest()
        hashes.setdefault(h, []).append(str(p))
    dups = [v for v in hashes.values() if len(v) > 1]
    report[t] = dict(total=len(ps), matched=matched, unmatched=len(unmatched),
                     corrupt=len(corrupt), duplicate_groups=len(dups))
    print(t, report[t])

    # 5. sample + side-by-side
    pool = [p for p in ps if match(p)]
    for p in random.sample(pool, min(SAMPLE_PER_TYPE, len(pool))):
        o = match(p)[0]
        a, b = Image.open(o).convert("RGB"), Image.open(p).convert("RGB")
        h = 400
        a = a.resize((int(a.width*h/a.height), h)); b = b.resize((int(b.width*h/b.height), h))
        canvas = Image.new("RGB", (a.width+b.width+10, h), "white")
        canvas.paste(a, (0, 0)); canvas.paste(b, (a.width+10, 0))
        canvas.save(OUT / f"{t}__{p.stem}.png")

json.dump(report, open(OUT / "automated_report.json", "w"), indent=2)
print("\nSaved comparisons to", OUT)