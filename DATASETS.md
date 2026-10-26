# Dataset Documentation & Governance — Sentry

## 1. Intended Dataset Scope

The Sentry inspection system is designed exclusively for **concrete surface crack inspection** from single photographic captures.

Target imagery includes:
- Surface cracks on cast-in-place concrete, precast slabs, concrete bridge decks, retaining walls, and pavements.
- Varying ambient illumination (direct sunlight, shade, overcast, artificial lighting).
- Realistic surface textures, aggregate exposure, trowel markings, and hairline to severe fissure widths.

Exclusions from current scope:
- Non-concrete substrates (asphalt, timber, masonry mortar joints, steel coatings).
- Unrelated concrete defects (spalling, exposed reinforcing steel/rebar, efflorescence, honeycomb voids, surface staining).

---

## 2. Planned Data Organization

Local datasets will be organized under the local `data/` directory.

```text
data/
├── dev/                  # Development & parameter calibration set
│   ├── images/           # Input concrete photographs
│   ├── masks/            # Ground-truth binary segmentation masks (crack = 255, bg = 0)
│   └── metadata.json     # Capture metadata (lighting, source, structure ID)
└── test/                 # Independent holdout evaluation set
    ├── images/           # Input concrete photographs
    ├── masks/            # Ground-truth binary segmentation masks
    └── metadata.json     # Capture metadata and split verification
```

> **Important**: Per `.gitignore`, the entire `data/` directory is strictly excluded from version control to prevent repository bloat and respect dataset distribution terms.

---

## 3. Dataset Provenance, Licensing, and Attribution Requirements

Before any imagery is ingested into `data/dev/` or `data/test/`, the following metadata must be documented:

1. **Source & Provenance:** Original publication, repository, academic citation, or author identity.
2. **License Verification:** The imagery must carry an explicit, verified license permitting evaluation and redistribution or local processing (e.g., Creative Commons CC BY 4.0, MIT, Open Data Commons, or Public Domain).
3. **Attribution:** Any required academic citations or license notices must be maintained in this document under an "Attributions" section prior to use.
4. **Proprietary Data Restrictions:** No non-public or proprietary client imagery may be stored without signed data-use agreements and explicit exclusion from public commits.

---

## 4. Ground-Truth Annotation Requirements

For objective evaluation in `eval/run_eval.py`, dataset samples require:

1. **Image-Level Classification Labels:**
   - `defect_present`: Boolean (`True` if one or more valid cracks exist, `False` otherwise).
2. **Pixel-Level Binary Segmentation Masks:**
   - Single-channel 8-bit PNG images matching the dimensions of the source photograph.
   - Pixel value `255`: Crack pixels.
   - Pixel value `0`: Background concrete substrate.
3. **Quality & Environmental Metadata (Optional but Recommended):**
   - Illumination condition (harsh direct, uniform diffuse, low light).
   - Surface texture classification (smooth finish, rough/broomed, aggregate exposed).
   - Physical scale calibration marker presence (if available for metric conversion).

---

## 5. Dataset Splitting & Data Leakage Prevention

To ensure evaluation integrity and prevent overly optimistic evaluation metrics:

1. **Structural Grouping:** Splits between `dev/` and `test/` must be performed at the **structure or capture session level**, never at the image crop level.
2. **Leakage Prohibition:** Sub-patches or augmented crops derived from the same photograph or continuous crack line must never appear in both development and test partitions.
3. **Pre-processing Independence:** Normalization parameters, threshold tunings, and quality thresholds must be derived solely from the development split.
4. **Holdout Secrecy:** The `test/` partition will remain untouched during baseline heuristic tuning and will only be executed by `eval/run_eval.py` during final benchmark runs.

---

## 6. Current Implementation Status

**No real-world dataset has been supplied, ingested, or evaluated at this stage.**

The `data/dev/` and `data/test/` directories are currently empty. No sample counts, performance figures, or accuracy benchmarks have been measured, generated, or claimed.
