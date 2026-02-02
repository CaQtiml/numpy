#!/bin/bash
MESON_EXE="$(pwd)/vendored-meson/meson/meson.py"

echo "Only rebuilding changed files in existing build directory."

# Only rebuild changed files
echo "Start Compiling"
python vendored-meson/meson/meson.py compile -C build

echo "Start Installing"
python vendored-meson/meson/meson.py install -C build