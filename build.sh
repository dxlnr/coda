#!/bin/bash
# OS
OS="$(uname)"
# FILES
SRC="explore.c"
OUT="explore"
# COMPILE
if [ "$OS" == "Darwin" ]; then
    clang "$SRC" -framework OpenCL -o "$OUT"
elif [ "$OS" == "Linux" ]; then
    clang "$SRC" -lOpenCL -o "$OUT"
else
    echo "Unsupported OS: $OS"
    exit 1
fi
# RUN
if [ $? -eq 0 ]; then
    ./"$OUT"
else
    echo "Build failed"
fi
