#!/bin/bash
# Generate report from a Python script
# Usage: ./scripts/generate_report.sh <report_name> <source_file> [dependencies...]
# Example: ./scripts/generate_report.sh eda reports_src/01_eda.py data/raw_inputs/input_data_sun_heavy.xlsx

set -e  # Exit on error

if [ $# -lt 2 ]; then
    echo "Usage: $0 <report_name> <source_file> [dependencies...]"
    echo "Example: $0 eda reports_src/01_eda.py data/raw_inputs/input_data_sun_heavy.xlsx"
    exit 1
fi

REPORT_NAME="$1"
SOURCE_FILE="$2"
NOTEBOOK_FILE="${REPORT_NAME}.ipynb"

echo "=============================================="
echo "Generating report: ${REPORT_NAME}"
echo "Source: ${SOURCE_FILE}"
echo "=============================================="

# Step 1: Convert Python script to notebook and set kernel
echo "Step 1/4: Converting to notebook..."
uv run jupytext --to notebook "${SOURCE_FILE}" -o "${NOTEBOOK_FILE}" --set-kernel python3

# Step 2: Execute notebook in-place
echo "Step 2/4: Executing notebook..."
uv run jupyter execute "${NOTEBOOK_FILE}" --inplace

# Step 3: Convert to HTML (with embedded images, no input cells)
echo "Step 3/4: Converting to HTML..."
uv run jupyter nbconvert --to html "${NOTEBOOK_FILE}" \
    --output-dir reports/html \
    --output "${REPORT_NAME}" \
    --no-input

# Step 4: Convert to Markdown (with separate image files, no input cells)
echo "Step 4/4: Converting to Markdown..."
uv run jupyter nbconvert --to markdown "${NOTEBOOK_FILE}" \
    --output-dir reports/markdown \
    --output "${REPORT_NAME}" \
    --no-input

# Clean up intermediate notebook file
echo "Cleaning up..."
rm "${NOTEBOOK_FILE}"

echo "=============================================="
echo "Report generated successfully!"
echo "  HTML: reports/html/${REPORT_NAME}.html"
echo "  Markdown: reports/markdown/${REPORT_NAME}.md"
echo "=============================================="
