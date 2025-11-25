#!/bin/bash
#
# Upload Script for Lambda Cloud
# 
# Usage:
#   ./UPLOAD_TO_LAMBDA.sh <lambda-ip-address>
#
# Example:
#   ./UPLOAD_TO_LAMBDA.sh 170.64.xxx.xxx

if [ -z "$1" ]; then
    echo "ERROR: Please provide Lambda IP address"
    echo "Usage: ./UPLOAD_TO_LAMBDA.sh <lambda-ip>"
    exit 1
fi

LAMBDA_IP=$1
LAMBDA_USER="ubuntu"

echo "=========================================="
echo "LAMBDA CLOUD UPLOAD"
echo "=========================================="
echo "Target: $LAMBDA_USER@$LAMBDA_IP"
echo ""

# Upload compressed images
echo "[1/2] Uploading compressed images (962MB)..."
echo "This may take 5-15 minutes depending on your connection..."
scp financial_district/images.tar.gz $LAMBDA_USER@$LAMBDA_IP:~/

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upload images"
    exit 1
fi

echo "✓ Images uploaded successfully"
echo ""

# Upload preprocessing script
echo "[2/2] Uploading preprocessing script..."
scp lambda_preprocessing.py $LAMBDA_USER@$LAMBDA_IP:~/

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upload script"
    exit 1
fi

echo "✓ Script uploaded successfully"
echo ""

echo "=========================================="
echo "UPLOAD COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH into Lambda:"
echo "   ssh $LAMBDA_USER@$LAMBDA_IP"
echo ""
echo "2. Extract images:"
echo "   tar -xzf images.tar.gz"
echo ""
echo "3. Run preprocessing:"
echo "   python3 lambda_preprocessing.py --images ~/images --output ~/colmap_output"
echo ""
echo "Or run all at once in tmux:"
echo "   ssh $LAMBDA_USER@$LAMBDA_IP"
echo "   tmux new -s colmap"
echo "   tar -xzf images.tar.gz && python3 lambda_preprocessing.py --images ~/images --output ~/colmap_output"
echo ""
