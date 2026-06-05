# Dataset: Zanzibar Clove Grading (CLOVES-5898)

The dataset is publicly available on Hugging Face:

```
https://huggingface.co/datasets/PatrickIITMZ/zanzibar_cloves
```

---

## Composition

| Subset       | Grade I | Grade II | Grade III | Grade IV | Total     |
|:-------------|--------:|---------:|----------:|---------:|----------:|
| Single-clove | 1,084   | 1,050    | 1,503     | 966      | **5,298** |
| Batch-pile   | 150     | 150      | 150       | 150      | **600**   |
| **Total**    | **1,234**| **1,200**| **1,653** | **1,116**| **5,898** |

**Notes on dataset versioning:**
- Appendix B of the thesis reports benchmark results on the Phase II dataset
  (4,603 single-clove images), which was the version available during the
  first round of experiments.
- Chapter 4 results use the full final dataset (5,298 single-clove +
  600 batch-pile = 5,898 images) after Phase 3 collection was completed.
- The Hugging Face release contains the full 5,898-image final dataset.

---

## Train / Val / Test Split

Stratified random split, fixed seed (`random_state=42`):

| Split      | Proportion | Single-clove images |
|:-----------|:----------:|--------------------:|
| Train      |    70%     |                3,708 |
| Validation |    15%     |                  795 |
| Test       |    15%     |                  795 |

The 795-image test set was held out throughout all model development and
hyperparameter selection.  It was consulted once, at final evaluation.

---

## Downloading

### Using the Hugging Face `datasets` library (recommended)

```python
from datasets import load_dataset

ds = load_dataset("PatrickIITMZ/zanzibar_cloves")

# Access the single-clove split
train_single = ds["train"]

# Access the batch-pile split (if separately configured)
# See the dataset card on Hugging Face for split names.
```

### Manual download

```bash
git lfs install
git clone https://huggingface.co/datasets/PatrickIITMZ/zanzibar_cloves
```

Once cloned, the directory structure mirrors the PyTorch `ImageFolder` format:

```
zanzibar_cloves/
├── train/
│   ├── Grade 1/
│   ├── Grade 2/
│   ├── Grade 3/
│   └── Grade 4/
├── val/
│   └── ...
└── test/
    └── ...
```

Pass the split directory directly to `CloveDataset`:

```python
from src.dataset.clove_dataset import CloveDataset

test_ds = CloveDataset("zanzibar_cloves/test", split="test")
print(test_ds)
# CloveDataset(root=zanzibar_cloves/test, n=795)
#   Grade 1: 163 images
#   Grade 2: 158 images
#   Grade 3: 225 images
#   Grade 4: 249 images
```

---

## Collection Details

| Parameter           | Value                                                              |
|:--------------------|:-------------------------------------------------------------------|
| Camera              | Samsung Galaxy S21 Ultra 5G (SM-G998B/DS), 12 MP, f/1.8          |
| Location            | ZSTC Saateni warehouse, Unguja, Zanzibar, Tanzania                |
| Phase 1 (2024)      | Handheld, natural ambient warehouse light                          |
| Phase 3 (Mar–May 2026) | PULUZ LED lightbox, tripod, fixed overhead mount                |
| Image resolution    | 224 × 224 px (pre-processed); originals at 4,000 × 3,000 px      |
| Colour space        | sRGB JPEG                                                          |

---

## Label Annotation Protocol

All single-clove and batch-pile images were labelled according to the
official **ZSTC clove grading standard** (Internal Circular, 2023).

**Grade definitions:**

| Grade    | Primary Attribute                   | Mpeta (max) | Foreign Matter (max) |
|:---------|:------------------------------------|:-----------:|:--------------------:|
| Grade I  | Attractive golden colour            |     3%      |         5%           |
| Grade II | Faded / blackish colour             |     7%      |         5%           |
| Grade III| More faded colour                   |    20%      |         5%           |
| Grade IV | Primarily *mpeta* (fermented)       |    >20%     |        N/A           |

**Annotation procedure:**

1. Each image was independently assessed by three ZSTC-certified inspectors
   at the Saateni warehouse, using the official ZSTC grading criteria.
2. The majority-vote label was assigned.  All three inspectors agreed on
   100% of images in the final dataset (no tie-breaking was required).
3. Images where inspectors disagreed (< 2% of initial collection) were
   re-photographed or discarded.

**Pixel-level annotations (segmentation subset):**

- 200 single-clove images (50 per grade) were additionally annotated with
  pixel-level polygon masks in **Roboflow** for YOLOv8-seg training and
  context-aware ResNet-18 evaluation.
- Annotation was performed by the first author (Patrick Vincent Ndowo)
  using Roboflow's polygon tool.
- All 200 pixel-level annotations were reviewed and verified by one
  ZSTC-certified grader.
- Annotations were exported in COCO JSON format (polygon segmentation).
- Use `src/dataset/coco_processor.py` to convert to binary PNG masks.

---

## Licence

The dataset is released under
**Creative Commons Attribution 4.0 (CC BY 4.0)**.

When using this dataset, please cite:

```bibtex
@dataset{vincent2026cloves,
  author    = {Vincent Ndowo, Patrick and Nyalala, Innocent},
  title     = {CLOVES-5898: Zanzibar Clove Grading Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/PatrickIITMZ/zanzibar_cloves}
}
```
