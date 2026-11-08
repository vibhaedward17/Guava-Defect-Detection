# Guava Defect Detection & Morphological Analysis

Upload a photo of a guava and get back:

- the fruit cut out from the background,
- boxes and outlines around **defects** (anthracnose, canker/scab, soft rot, fly punctures),
- the **stem, calyx and natural skin marks** marked as "Ignored", so they are not counted as damage,
- a colour breakdown, shape scores, a **grade (A / B / C / Reject)** and a short written report.

The whole analysis uses classical computer vision (OpenCV colour and shape rules). No trained model is needed.

> **Note on the name:** the folder is called `guava_yolo_app` and some function and file names contain "yolo", but this project does **not** use YOLO. These names are left over from an earlier plan and were never changed. Nothing here is trained or runs a YOLO model. Every step is plain OpenCV image processing.

---

## Project layout

```
guava_yolo_app/
├── pipeline.py        # the whole analysis (segmentation, defects, grading, report)
├── app_gradio.py      # web interface (Gradio)
├── app_streamlit.py   # alternative web interface (Streamlit)
├── requirements.txt
├── configs/           # configuration files (not used by the pipeline)
└── samples/           # test photos: anthracnose, canker, healthy, yellow rot
```

---

## How it works

1. **Segmentation.** The fruit is found by colour (green / yellow tissue), cleaned up, and refined with GrabCut if it touches the image border or looks ragged. Dark patches that the colour rule drops are filled back in when they sit in a notch of the outline and do not look like background.
2. **Anatomy first.** Before judging damage, the pipeline looks for:
   - **Stem:** a thin stalk sticking out of the round fruit body, together with its dried base.
   - **Calyx (blossom end):** either a small, very dark, compact core near the edge, or a dried crown hugging the outline.
   - **Natural skin marks:** small lenticels / russeting.

   These are shown in blue as "Ignored" and removed from the damage count. A dark centre inside a big damaged area is treated as part of the lesion.
3. **Defects.** Brown, dark or off-colour tissue is found relative to the fruit's own local brightness, so plain shadows are not counted. Ring-shaped lesions are merged into one object, touching lesions are split, and thin shading along the outline is rejected. Each defect is labelled from its colour, darkness and size:
   - **Anthracnose:** dark, sunken lesions.
   - **Canker / Scab:** brown, dry patches.
   - **Soft rot:** large water-soaked areas.
   - **Fly puncture:** tiny dark spots.
4. **Report.** Damage % is computed from the exact pixels of the confirmed defects, so the boxes, grade and chart always agree.

Large photos are shrunk to 1000 px on the longest side before analysis, so the outputs are at that size.

---

## Grades

| Surface damage | Grade | Meaning |
|---|---|---|
| < 1% | A | Healthy / premium |
| 1% to < 5% | B | Minor blemishes |
| 5% to < 15% | C | Noticeable damage |
| 15% or more | Reject | Unmarketable |

---

## Tuning

In `GuavaDefectPipeline.__init__` (`pipeline.py`):

| Setting | Effect |
|---|---|
| `brown_dl` (default -14) | More negative = fewer brown areas counted as damage |
| `dark_dl` (default -28) | More negative = only very dark spots count |
| `chroma_ratio` (default 0.30) | Lower = fewer pale / off-colour areas counted |
| `conf_threshold` (default 0.45) | Minimum confidence to report a defect |
| `max_lenticel_area_pct` (default 0.30) | Spots smaller than this are treated as harmless skin marks |

---

## Known limitations

- It works best on a single fruit, in focus, on a plain background, in even light.
- The stem and calyx rules are geometric. A real lesion at the very edge of the fruit can be mistaken for them, and an unusual calyx can be reported as a defect. Always check the blue "Ignored" boxes.
- Very dark lesions can be cut out of the fruit mask. A fix is included but has only been tested on a synthetic case.
- Defect names are guessed from colour and size, not from a trained classifier. They are a screening aid, not a lab diagnosis.
- Images that already have labels or boxes drawn on them (such as screenshots of earlier results) will give wrong results. Use raw photos.
