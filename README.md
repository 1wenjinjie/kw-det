# KW-Det

Kinematic Wake-guided Detector for Multispectral Ship Detection in Sentinel-2 Imagery.

This repository is organized for Git and GitHub management. Source code, project notes, and manuscript assets are tracked. Large raw imagery, masks, pretrained weights, and generated outputs stay local under data/ and logs/ and are ignored by Git.

## Contents

- scripts/download_finland_l2a.py: resume-safe downloader for Finland 2025 Sentinel-2 L2A imagery from AWS Earth Search STAC.
- paper/: manuscript PDF and related writing assets.
- CLAUDE.md: project guidance, data policy, and label conventions.
- Project_Plan_1.md: project roadmap and stage notes.

## Notes

- The current downloader script depends on geopandas and requests.
- The broader research stack is described in CLAUDE.md.
- Add a GitHub remote named origin, create the first commit, and push once your Git identity is set.
