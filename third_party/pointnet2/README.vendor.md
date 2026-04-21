# Vendored pointnet2

This directory contains the `pointnet2` source snapshot used by this project.

Origin:

- copied from `anygrasp_sdk/pointnet2/`

Included on purpose:

- `setup.py`
- `pointnet2/`
- `pointnet2/_ext_src/`

Excluded on purpose:

- `build/`
- `*.egg-info/`
- `__pycache__/`

Reason for vendoring:

- the main project depends on a working `pointnet2` extension
- keeping the source snapshot in-repo is simpler than tracking only patches
- the original `anygrasp_sdk/` tree remains a local mirror and stays ignored

Recommended use:

1. build/install from this vendored directory when reconstructing the environment
2. record any future local changes here directly, or regenerate from a known upstream source if needed
