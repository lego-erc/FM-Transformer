MARKER="$CONDA_PREFIX/.train-setup-done"
if [ ! -f "$MARKER" ]; then
    pip install -q 'flow-matching>=1.0.10' \
        'x-transformers==2.17.9' \
        'git+https://github.com/rdebrand/torch-lap-cuda' \
        -e . \
    && touch "$MARKER"
fi
