#!/bin/bash

# 1. Paths
CPYTHON_DIR="/mnt/d/master/master_thesis/cpython"
VENV_PATH="/mnt/d/master/master_thesis/numpy_customized/my-debug-env"

# 2. Point to NumPy's own Meson entry point
# This is the "magic" that fixes the 'Module features does not exist' error
MESON_EXE="$(pwd)/vendored-meson/meson/meson.py"

# 3. Clean
rm -rf build/

# 4. Run using the vendored script instead of the 'meson' command
python "$MESON_EXE" setup build \
  --prefix="$VENV_PATH" \
  -Dbuildtype=debug \
  -Ddisable-optimization=true \
  -Dcpu-baseline=none \
  -Dcpu-dispatch=none \
  -Dc_args="-I$CPYTHON_DIR/Include -I$CPYTHON_DIR" \
  -Dcpp_args="-I$CPYTHON_DIR/Include -I$CPYTHON_DIR" \
  -Dc_link_args="-L$CPYTHON_DIR" \
  -Dcpp_link_args="-L$CPYTHON_DIR"

echo "Configuration complete."