MARKER="$CONDA_PREFIX/.train-setup-done"
if [ ! -f "$MARKER" ]; then
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --reinstall \
    && uv pip install -q 'git+https://github.com/rdebrand/torch-lap-cuda' \
    && touch "$MARKER"
fi
