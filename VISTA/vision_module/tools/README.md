# vision_module tools

This directory is reserved for manual debugging scripts and operator-run helper
tools.

Current state:

- Historical scripts such as `debug_send_req.py`, `debug_recv_obj.py`,
  `debug_protocol_tools.py`, and `demo_camera.py` still live in `test/`.
- They are intentionally not moved in this cleanup pass so existing board-side
  habits and shell commands keep working.

Recommended next step:

- Move manual tools from `test/` into this directory one script at a time.
- Leave a small compatibility shim at the old path when a script is moved.
- Keep automated unit and regression tests in `test/`.
