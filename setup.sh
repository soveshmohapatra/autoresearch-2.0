#!/bin/bash
# Quick setup script for autoresearch
# Works on macOS, Linux, and Windows (WSL)

set -e

echo "🚀 Autoresearch Quick Setup"
echo "============================"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✅ uv found"

# Sync dependencies
echo ""
echo "📦 Installing dependencies..."
uv sync

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run hardware detection: uv run python hardware.py"
echo "  2. Launch GUI: uv run python gui.py"
echo "  3. Or prepare data: uv run python prepare.py"
echo ""
