# Release validation

Validation uses saved predictions and local files; no model training or inference was rerun.

- All 1,039 images were exported with unchanged primary decoded RGBA pixels and original dimensions. Identifying metadata was removed without recompression.
- Public IDs match the existing five annotation subsets, canonical annotations and all 17 prediction files. Counts are 718 flawed images, 321 clean controls and 1,983 regions.
- The Hugging Face directory was loaded through `datasets.load_dataset` as the `test` split: 1,039 image rows, 321 clean labels and 1,983 boxes. Image decoding and metadata were checked.
- After publication, all 1,039 remote image SHA-256 values matched the local release; the annotations, ImageFolder metadata and dataset statistics were downloaded and checked byte-for-byte.
- The original published scorer reproduces all 17 unrounded headline scores and R@0.1 values within 1e-8.
- The independent specification scorer passes its 24 unit tests. The full offline analysis reproduces all baseline scores and reruns sensitivity, source-stratified, matching and paired-bootstrap calculations.
- Direct runner/evaluator: 32 tests passed.
- Full-Seq snapshot: 128 tests passed, including the local HTTP serving test.
- Inspection snapshot: 24 tests passed. Async/local-server tests were checked outside a socket-restricted sandbox.
- Workflow data preprocessing: 10 tests passed.
- Python and shell sources were syntax-checked. Credential-shaped values, private filesystem prefixes, selected internal endpoints and source-account identifiers were screened before publication.

Six checkpoint argument snapshots and training split manifests were recovered. Training jobs and dependency installation on the original GPU topology were not reproduced. Those artifacts retain documented environment and data requirements.

Packaging changes are limited to public-ID alignment, credential/path sanitization, documentation, download/validation utilities, removal of CLI entry points for modules absent from the core-only snapshot, and image-header MIME detection in the inspection manifest for historically mismatched filename extensions. Scoring formulas and model outputs are unchanged.
