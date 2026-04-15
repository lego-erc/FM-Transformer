MARKER="$CONDA_PREFIX/.train-setup-done"
if [ ! -f "$MARKER" ]; then
    uv pip install -q 'git+https://github.com/rdebrand/torch-lap-cuda' \
    && touch "$MARKER"
fi
