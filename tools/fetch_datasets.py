"""
Fetch public fall-detection and clinical distress clips for hallway_app & ARUGA validation.

What it references (free for research / non-commercial — review individual licenses):
  1. UR Fall Detection Dataset (CC BY-NC-SA 4.0, Univ. of Rzeszow):
     - RGB PNG sequences of acted falls + ADLs (single person, indoor).
     - Automated downloading and assembly into .mp4 clips.
  2. Le2i Fall Detection Dataset (FDD):
     - CCTV-style clips across 4 settings (Home, Coffee room, Office, Lecture room).
     - Manual / Kaggle / Google Drive mirror instructions.
  3. NTU RGB+D Dataset — Action A45 ("Touch Chest / Heart Pain"):
     - Benchmark for Levine's Sign Biometric Validation (chest clutching + trunk flexion).
     - Official Portal: https://rose1.ntu.edu.sg/dataset/actionRecognition/
  4. PatientCare Multimodal Dataset (Hugging Face):
     - Comprehensive multimodal patient monitoring dataset (fall & distress behaviors).
     - HF Repo: https://huggingface.co/datasets/xyz1901901/patientcare

Usage:
    python tools/fetch_datasets.py [--falls 1] [--adls 1] [--info]
Clips land in assets/eval_clips/ (gitignored). Then run tools/eval_clips.py.

Citations:
  - UR: Kwolek & Kepski, "Human fall detection on embedded platform using depth maps
    and wireless accelerometer", CMPB 117(3), 2014.
  - Le2i: Charfi et al., "Optimised spatio-temporal descriptors for real-time fall
    detection", J. Electronic Imaging 22(4), 2013.
  - NTU RGB+D: Liu et al., "NTU RGB+D 120: A Large-Scale Benchmark for 3D Human
    Activity Analysis", IEEE TPAMI 42(10), 2019.
"""

import argparse
import os
import sys
import urllib.request
import zipfile

BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data"
OUT = os.path.join("assets", "eval_clips")

LE2I_NOTES = """
================================================================================
1. Le2i Fall Detection Dataset (FDD)
================================================================================
221 CCTV-style clips (Home/Coffee room/Office/Lecture room, 320x240@25fps, single person).
No stable direct-download URLs, so fetch one of these ways and drop .avi files into assets/eval_clips/:
  1. Official page: http://le2i.cnrs.fr/Fall-detection-Dataset?lang=en
  2. Kaggle mirror (free account): tuyenldvn/falldataset-imvia (17 GB full set)
       kaggle datasets download -d tuyenldvn/falldataset-imvia
  3. Google-Drive mirror (309MB raw subset, see YifeiYang210/Fall_Detection_dataset
     on GitHub for the current share link).
Home_* and Coffee_room_* folders include Annotation_files with fall start/end frames.
"""

NTU_A45_NOTES = """
================================================================================
2. NTU RGB+D 120 Dataset — Action A45 ("Touch Chest / Heart Pain")
================================================================================
Canonical action recognition benchmark specifically isolating subjects clutching
their chest due to acute cardiac / ischemic distress (Levine's sign validation):
  - Official Portal & Agreement: https://rose1.ntu.edu.sg/dataset/actionRecognition/
    (Fill out academic research request form to obtain download credentials)
  - Action Class: A45 (or A045) — "touch chest (stomachache/heart pain)"
  - Format: RGB videos (.avi) and 25-joint 3D skeleton sequences (.skeleton)
  - Extracting A45 Clips:
      Filter sample files matching: *A045*.avi (e.g. S001C001P001R001A045_rgb.avi)
      Place extracted .avi / .mp4 clips into assets/eval_clips/ for evaluation.
  - Relevance: Biometric validation of sternum-midpoint proximity and antalgic trunk flexion.
"""

PATIENTCARE_NOTES = """
================================================================================
3. PatientCare Multimodal Dataset (Hugging Face)
================================================================================
Multimodal patient care and clinical monitoring benchmark covering mobility,
falls, and acute physical distress gestures:
  - Hugging Face Repository: https://huggingface.co/datasets/xyz1901901/patientcare
  - CLI Download:
      huggingface-cli download xyz1901901/patientcare --repo-type dataset --local-dir assets/eval_clips/patientcare
  - Python API Download:
      from datasets import load_dataset
      dataset = load_dataset("xyz1901901/patientcare")
  - Place relevant video evaluation samples into assets/eval_clips/
"""


def download(url: str, dest: str):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [skip] {os.path.basename(dest)} already present")
        return dest
    print(f"  [get] {url} ({dest})")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def assemble_mp4(zip_path: str, mp4_path: str, fps: int = 25, max_frames: int = 0):
    """UR sequences are PNG zips; pack cam0 RGB frames into an .mp4."""
    import cv2
    if os.path.exists(mp4_path):
        print(f"  [skip] {os.path.basename(mp4_path)} already assembled")
        return mp4_path
    print(f"  [pack] {os.path.basename(zip_path)} -> {os.path.basename(mp4_path)}")
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".png"))
        if max_frames:
            names = names[:max_frames]
        frame0 = cv2.imdecode(__import__("numpy").frombuffer(z.read(names[0]), dtype="uint8"),
                              cv2.IMREAD_COLOR)
        h, w = frame0.shape[:2]
        out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for n in names:
            buf = __import__("numpy").frombuffer(z.read(n), dtype="uint8")
            out.write(cv2.imdecode(buf, cv2.IMREAD_COLOR))
        out.release()
    return mp4_path


def main():
    ap = argparse.ArgumentParser(description="Fetch and inspect fall & distress datasets for eval")
    ap.add_argument("--falls", type=int, default=1, help="UR fall sequences to fetch (01..30)")
    ap.add_argument("--adls", type=int, default=1, help="UR ADL sequences to fetch (01..40)")
    ap.add_argument("--keep-zips", action="store_true", help="keep downloaded zips")
    ap.add_argument("--info", action="store_true", help="display research dataset catalog & guidance")
    args = ap.parse_args()

    if args.info:
        print(LE2I_NOTES)
        print(NTU_A45_NOTES)
        print(PATIENTCARE_NOTES)
        return 0

    os.makedirs(OUT, exist_ok=True)
    for i in range(1, args.falls + 1):
        name = f"fall-{i:02d}-cam0-rgb.zip"
        z = download(f"{BASE}/{name}", os.path.join(OUT, name))
        assemble_mp4(z, os.path.join(OUT, f"ur_{name.replace('-cam0-rgb.zip', '')}.mp4"))
        if not args.keep_zips:
            os.remove(z)
    for i in range(1, args.adls + 1):
        name = f"adl-{i:02d}-cam0-rgb.zip"
        z = download(f"{BASE}/{name}", os.path.join(OUT, name))
        assemble_mp4(z, os.path.join(OUT, f"ur_{name.replace('-cam0-rgb.zip', '')}.mp4"))
        if not args.keep_zips:
            os.remove(z)

    print("\nDone. Clips in assets/eval_clips/. Run: python tools/eval_clips.py")
    print(LE2I_NOTES)
    print(NTU_A45_NOTES)
    print(PATIENTCARE_NOTES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
