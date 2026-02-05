#!/bin/bash
set -e

echo "======================================================================"
echo "TTK PROCESSING: PhIRE Wind Data"
echo "======================================================================"

# Ensure output directories exist
mkdir -p ttk_outputs/pd ttk_outputs/mt

# Count files
n_vti=$(ls vtk_inputs/gan_*.vti 2>/dev/null | wc -l)
echo "Found $n_vti VTI files to process"

# Process each VTI file
count=0
for vti_file in vtk_inputs/gan_*.vti; do
    count=$((count + 1))
    base=$(basename "$vti_file" .vti)
    
    echo ""
    echo "======================================================================"
    echo "[$count/$n_vti] Processing: $base"
    echo "======================================================================"
    
    # Persistence Diagram
    echo "[Step 1/2] Computing Persistence Diagram..."
    docker run --rm -v "$PWD":/work -w /work phire-ttk:latest \
      bash -c "ttkPersistenceDiagramCmd -t 20 \
        -i $vti_file \
        -a wind_speed \
        -o ttk_outputs/pd/${base}_pd 2>&1 | grep -E 'pairs|Complete|ERROR'"
    
    if [ -f "ttk_outputs/pd/${base}_pd_port_0.vtu" ]; then
        echo "  ✓ PD output created"
    else
        echo "  ✗ PD failed"
        continue
    fi
    
    # Merge Tree
    echo "[Step 2/2] Computing Merge Tree..."
    docker run --rm -v "$PWD":/work -w /work phire-ttk:latest \
      bash -c "ttkMergeTreeCmd -t 20 \
        -i $vti_file \
        -a wind_speed \
        -o ttk_outputs/mt/${base}_mt 2>&1 | grep -E 'Tree|Complete|ERROR'"
    
    if [ -f "ttk_outputs/mt/${base}_mt_port_0.vtu" ]; then
        echo "  ✓ MT output created"
    else
        echo "  ✗ MT failed"
    fi
done

echo ""
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo "Persistence Diagrams:"
ls -lh ttk_outputs/pd/ | grep -v total
echo ""
echo "Merge Trees:"
ls -lh ttk_outputs/mt/ | grep -v total

echo ""
echo "✓ Processing complete!"
