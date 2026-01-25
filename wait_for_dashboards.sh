#!/bin/bash
echo "Waiting for both dashboards to complete..."
echo ""

# Wait for Original (PID 64215)
while ps -p 64215 > /dev/null 2>&1; do
    sleep 30
done
echo "Original dashboard completed!"

# Wait for Fixed (PID 65972)  
while ps -p 65972 > /dev/null 2>&1; do
    sleep 30
done
echo "Fixed dashboard completed!"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " BOTH DASHBOARDS COMPLETED - COMPARING RESULTS"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "ORIGINAL ATTRIBUTION (BUGGY - Missing TREND):"
echo "────────────────────────────────────────────────────────────────"
grep -A 20 "93\." dashboard_original_check.txt | head -25 || echo "Pattern not found, searching for attribution..."

echo ""
echo ""
echo "FIXED ATTRIBUTION (CORRECTED - Includes TREND):"
echo "────────────────────────────────────────────────────────────────"
grep -A 25 "CORRECTED ATTRIBUTION" dashboard_FIXED_output.txt | head -30

