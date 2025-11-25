#!/bin/bash
#
# Download Script for Lambda Cloud Results
# 
# Usage:
#   ./DOWNLOAD_FROM_LAMBDA.sh <lambda-ip-address>
#
# Example:
#   ./DOWNLOAD_FROM_LAMBDA.sh 170.64.xxx.xxx

if [ -z "$1" ]; then
    echo "ERROR: Please provide Lambda IP address"
    echo "Usage: ./DOWNLOAD_FROM_LAMBDA.sh <lambda-ip>"
    exit 1
fi

LAMBDA_IP=$1
LAMBDA_USER="ubuntu"
OUTPUT_DIR="outputs/lambda_colmap"

echo "=========================================="
echo "LAMBDA CLOUD DOWNLOAD"
echo "=========================================="
echo "Source: $LAMBDA_USER@$LAMBDA_IP:~/colmap_output"
echo "Destination: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download results
echo "Downloading COLMAP results..."
echo "This includes:"
echo "  - Camera poses (sparse/0/)"
echo "  - COLMAP database"
echo "  - Processing summary"
echo ""

scp -r $LAMBDA_USER@$LAMBDA_IP:~/colmap_output/* "$OUTPUT_DIR/"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download results"
    exit 1
fi

echo ""
echo "=========================================="
echo "DOWNLOAD COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Contents:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review processing summary:"
echo "   cat $OUTPUT_DIR/processing_summary.json"
echo ""
echo "2. Use for Gaussian Splatting training:"
echo "   python run_brush_training.py"
echo ""
echo "3. Don't forget to TERMINATE your Lambda instance!"
echo ""
