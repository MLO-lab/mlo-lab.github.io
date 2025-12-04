#!/bin/bash
set -e

# ---------------------------------------------------------
# Load conda inside the script
# ---------------------------------------------------------
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
else
    echo "❌ Could not find conda.sh at $CONDA_SH"
    exit 1
fi

echo "🟢 Activating conda environment: website"
conda activate website

# You are already in content/publication/
PUBS_DIR="."
BIB="publications.bib"

echo "📚 Sorting publications with biber..."
biber --tool --configfile=sort_pubs.conf "$BIB"

echo "🔄 Replacing publications.bib with sorted version..."
mv publications_bibertool.bib publications.bib

# Move back to repository root (two levels up)
cd ../../

echo "📥 Importing publications into Hugo Academic..."
academic import --bibtex content/publication/publications.bib

echo "✅ Done!"