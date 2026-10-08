# REFERENCE — published numbers of other papers, transcribed as data (S16, N-37 / E-52)
`synfoc_published_tables.csv`: Tables 3-6 of Ma et al., "Steady Progress Beats Stagnation: Mutual Aid of Foundation and
Conventional Models in Mixed Domain Semi-Supervised Medical Image Segmentation" (CVPR 2025), rows FixMatch, SynFoC-UNet,
SynFoC-MedSAM. Transcribed on 2026-10-07 from the arXiv HTML rendering (arXiv:2503.16997v1), not from the CVPR PDF.
**Reading rule, stated in Section 4.2 of that paper:** each column is one experiment, defined by the LABELLED domain; the
entry is the Dice averaged over the test sets of ALL domains (and, on Fundus / M&Ms, given per class); "Avg." is the mean of
the columns. `make_extra_s16.py` asserts that arithmetic for all 12 dataset x method groups (|mean - Avg| <= 0.011), which
also guards the transcription. `labelled_domain` uses the paper's column names (HCRUDB = I2CVB in our dataloader).
Our protocol's labelled domains: prostate RUNMC (`--lb_domain 5`), fundus Domain 3, MNMS Vendor D, BUSI Malignant (64 labels).
V-18 done (S16 turn 3): second reading against the arXiv v1 PDF supplied by the owner (sha256 a7958ba0227311d0; quarantine `ARCHIVE/reference_papers/`), `CODE/source/check_synfoc_pdf_s16.py`: 12 / 12 rows verbatim, reading rule present. The CVPR proceedings PDF has not been compared.
Nothing in this folder is a result of ours; `RESULTS/` (submission-time digests) stays unchanged.
