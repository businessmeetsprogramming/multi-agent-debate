#!/bin/bash
# Setup script for run-debate skill
# Run once on a new machine: bash .claude/skills/run-debate/setup.sh

set -e

echo "Installing dependencies for run-debate skill..."
pip install -r "$(dirname "$0")/requirements.txt"

echo ""
echo "Done! Make sure these environment variables are set:"
echo "  export GEMINI_API_KEY='your-key-here'"
echo "  export OPENAI_API_KEY='your-key-here'"
echo ""
echo "Get keys at:"
echo "  Gemini:  https://aistudio.google.com/apikey"
echo "  OpenAI:  https://platform.openai.com/api-keys"
