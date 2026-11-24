# Dataset format and provenance

The final evaluation set has 1,039 images, 718 flawed images, 321 clean controls and 1,983 annotated regions. The earlier 1,800-image candidate benchmark and 1,054-image execution batches are not this evaluation set.

## Public identities and files

`IMG_0001` through `IMG_1039` preserve the IDs of the existing GitHub v1.0 release. The five original subset files in `annotations/` remain unchanged. Canonical nested annotations in `benchmark/annotations.jsonl`, all 17 prediction files, and the Hugging Face images use these same IDs. Original filesystem paths, annotator accounts, timestamps and generation prompts are excluded.

Canonical annotation rows contain:

- `sample_id`, `image_path`, `image_width`, `image_height`;
- `subset`: original A/B/C/D/E identifier;
- `source_group`: GPT-image-2, Public-4, SenseNova-U1.5 or Natural;
- `annotation.no_error`: true exactly when the box list is empty;
- `annotation.boxes`: absolute-pixel `bbox_xywh`, category, original taxonomy code and human description.

The image-only manifest omits labels and source information. Ground-truth coordinates are absolute `[x,y,width,height]`; prediction coordinates use the same convention on a 0-1000 grid. Human descriptions may be Chinese; they are not scored by the geometry metric.

## Source composition

| Subset | Source | Images | Flawed | Clean | Regions |
|---|---|---:|---:|---:|---:|
| A | GPT-image-2 | 208 | 97 | 111 | 266 |
| B | Public-4 | 499 | 499 | 0 | 1,126 |
| C | SenseNova-U1.5 | 141 | 122 | 19 | 591 |
| D | SenseNova-U1.5 | 81 | 0 | 81 | 0 |
| E | Natural | 110 | 0 | 110 | 0 |

Public-4 consists of 155 ArtiBench, 62 SalArt-VQA, 58 TextPecker-1.5M and 224 X-AIGD images. Source images retain their upstream rights. The historical InspectBench release states research/evaluation use and supplies no blanket permissive license; this packaging does not invent one for the combined source images.

## Image integrity

The export removes EXIF, text/comment metadata and auxiliary MPO data without re-encoding the benchmark image. All 1,039 primary decoded RGBA images are checked against the original images. Color-related markers are preserved. Original filenames/extensions are retained for compatibility, including a small number whose file extension does not match their underlying PNG/JPEG format; readers should detect the file content.

`image_checksums.jsonl` records released-file SHA-256, original-file SHA-256 and primary decoded-pixel SHA-256. The original GitHub image ZIPs used a different metadata-stripping path that re-encoded JPEGs, so their byte hashes need not match this pixel-preserving export. IDs, dimensions, labels and coordinates remain consistent.

## Hugging Face layout

`images/metadata.jsonl` adds source metadata and objects to the image files using the standard [ImageFolder format](https://huggingface.co/docs/datasets/en/image_dataset). The dataset card declares a single `test` split. Each `objects` value contains parallel lists `bbox_xywh`, `category` and `description`; clean images have empty lists.

## Scope

Controls are not content-matched to flawed images, and Public-4 has no clean controls. Cross-review agreement does not guarantee every visible defect has been annotated. FPR is conditional on reference-clean labels. Training negatives are construction-assigned and should not be confused with these separately cross-reviewed controls. Do not train on this evaluation split when reporting benchmark performance.
