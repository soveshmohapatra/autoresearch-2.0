#!/bin/bash
# Quick start script for autoresearch
# Launches the GUI dashboard

set -e

export PATH="$HOME/.local/bin:$PATH"

echo "🧠 Autoresearch Dashboard"
echo "========================="
echo ""

# Check if data exists
if [ ! -d "$HOME/.cache/autoresearch/data" ]; then
    echo "⚠️  Data not found. Downloading..."
    uv run python prepare.py --num-shards 10
fi

echo "Launching terminal dashboard..."
echo "Press Ctrl+C to stop"
echo ""

uv run python gui.py "$@"
