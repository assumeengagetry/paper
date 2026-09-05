# Paper Library

## Scope
- This is a literature library, not a software project; there are no build, test, lint, or dependency commands.
- Top-level directories are research domains. Classify a file by its primary contribution, not by its current location, filename, venue, or arXiv ID.

## Classification Workflow
- Read every file before moving it. For a PDF, inspect at least the title, abstract, introduction, and conclusion; read Markdown through EOF.
- Preserve original filenames. Create a new subdomain only when the existing taxonomy would make the file misleading.
- Keep each `MinerU_markdown_*` extraction beside its matching PDF. These are generated full-text companions, not independent notes.
- Keep authored notes with their subject area. `LLM-Systems/Performance-and-Profiling/` contains overlapping working notes, including `temp.md`; do not treat overlap as permission to delete them.

## Domain Boundaries
- `LLM-Systems/` is for model serving, KV-cache management, inference profiling, and architecture choices that directly affect inference.
- `ML-Systems-and-Hardware/` is for distributed training, GPU programming, accelerator analysis, and topics not specific to LLM serving.
- `Robotics-and-Embodied-AI/World-Action-Models/` is the video/action world-model collection; robot diffusion surveys and VLA deployment systems have separate sibling directories.
- Keep interpretability and fairness under `Explainable-and-Responsible-AI/`; backdoors, watermarking, and compression robustness belong under `ML-Security-and-Robustness/`.
- `Research-Landscapes/` is for institution or researcher surveys rather than technical papers.

## Known File Relationships
- The two `Language Ranker` PDFs and their two MinerU extractions in `Explainable-and-Responsible-AI/Fairness-and-Bias/Mitigation-and-Evaluation/` are duplicate copies of one paper. Preserve them unless deletion is explicitly requested.
- Several Markdown notes reference images by absolute paths under `~/.config/marktext/images/`; moving the notes within this library does not require rewriting those links.

## Cleanup
- Remove obsolete directories only after verifying they are empty. Never delete a PDF or note during reclassification without an explicit request.
- `.directory` files are KDE folder-icon metadata, not research content; they may be removed when cleaning obsolete folders.
