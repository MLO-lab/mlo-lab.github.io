#!/bin/bash
set -e

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