#!/bin/bash
set -e

echo "======================================================================"
echo "TTK COMPLETE VALIDATION - TOY EXAMPLE"
echo "======================================================================"

# Clean previous outputs
rm -rf vtk_inputs ttk_outputs
mkdir -p vtk_inputs ttk_outputs

# Step 1: Generate toy VTI
echo -e "\n[Step 1] Generating toy VTI file..."
docker run --rm -v "$PWD":/work -w /work phire-ttk:latest \
  python scripts/make_toy_vti.py

echo -e "\n[Step 1a] Verifying VTI file..."
head -15 vtk_inputs/toy.vti
echo "  ..."
file vtk_inputs/toy.vti
ls -lh vtk_inputs/toy.vti

# Step 2: Test VTK can read it
echo -e "\n[Step 2] Testing VTK can read the file..."
docker run --rm -v "$PWD":/work -w /work phire-ttk:latest \
  python - <<'PY'
import vtk
r = vtk.vtkXMLImageDataReader()
r.SetFileName('vtk_inputs/toy.vti')
r.Update()
img = r.GetOutput()
print(f'✓ Dimensions: {img.GetDimensions()}')
pd = img.GetPointData()
print(f'✓ Arrays: {[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]}')
arr = pd.GetArray('speed')
if arr:
    print(f'✓ Speed array: {arr.GetNumberOfTuples()} values')
    print(f'  Range: [{arr.GetRange()[0]:.4f}, {arr.GetRange()[1]:.4f}]')
else:
    print('✗ Speed array not found!')
    exit(1)
PY

# Step 3: Compute persistence diagram
echo -e "\n[Step 3] Computing persistence diagram with TTK..."
docker run --rm -v "$PWD":/work -w /work phire-ttk:latest \
  bash -c "ttkPersistenceDiagramCmd -i vtk_inputs/toy.vti -a speed -o ttk_outputs/toy_pd 2>&1 | grep -E 'WARNING|ERROR|Complete|pairs|Writing'"

echo -e "\n[Step 3a] Checking PD output..."
ls -lh ttk_outputs/
if [ -f ttk_outputs/toy_pd_port_0.vtu ]; then
    echo "✓ Persistence diagram output exists"
    file ttk_outputs/toy_pd_port_0.vtu
else
    echo "✗ Persistence diagram output missing"
    exit 1
fi

# Step 4: Compute merge tree
echo -e "\n[Step 4] Computing merge tree with TTK..."
docker run --rm -v "$PWD":/work -w /work phire-ttk:latest \
  bash -c "ttkMergeTreeCmd -i vtk_inputs/toy.vti -a speed -o ttk_outputs/toy_tree 2>&1 | grep -E 'WARNING|ERROR|Complete|Tree|Writing'"

echo -e "\n[Step 4a] Checking merge tree output..."
if [ -f ttk_outputs/toy_tree_port_0.vtu ]; then
    echo "✓ Merge tree output exists"
    ls -lh ttk_outputs/toy_tree_port_0.vtu
else
    echo "✗ Merge tree output missing"
    exit 1
fi

# Step 5: Analyze outputs with Python
echo -e "\n[Step 5] Analyzing TTK outputs..."
docker run --rm -v "$PWD":/work -w /work phire-ttk:latest \
  python - <<'PY'
import vtk

print("\n--- Persistence Diagram Analysis ---")
pd_reader = vtk.vtkXMLUnstructuredGridReader()
pd_reader.SetFileName('ttk_outputs/toy_pd_port_0.vtu')
pd_reader.Update()
pd = pd_reader.GetOutput()
print(f"PD points: {pd.GetNumberOfPoints()}")
print(f"PD cells: {pd.GetNumberOfCells()}")

# Get persistence values
if pd.GetPointData().GetArray("Persistence"):
    pers = pd.GetPointData().GetArray("Persistence")
    print(f"Persistence range: [{pers.GetRange()[0]:.4f}, {pers.GetRange()[1]:.4f}]")

print("\n--- Merge Tree Analysis ---")
mt_reader = vtk.vtkXMLUnstructuredGridReader()
mt_reader.SetFileName('ttk_outputs/toy_tree_port_0.vtu')
mt_reader.Update()
mt = mt_reader.GetOutput()
print(f"Tree nodes: {mt.GetNumberOfPoints()}")
print(f"Tree arcs: {mt.GetNumberOfCells()}")

if mt.GetPointData().GetArray("Persistence"):
    pers = mt.GetPointData().GetArray("Persistence")
    print(f"Tree persistence range: [{pers.GetRange()[0]:.4f}, {pers.GetRange()[1]:.4f}]")
PY

echo -e "\n======================================================================"
echo "✓ ALL TESTS PASSED!"
echo "======================================================================"
echo -e "\nTTK is fully functional:"
echo "  ✓ VTI generation works"
echo "  ✓ VTK file I/O works"
echo "  ✓ TTK persistence diagrams work"
echo "  ✓ TTK merge trees work"
echo "  ✓ Ready for PhIRE wind data!"
echo ""
echo "Outputs created:"
ls -lh ttk_outputs/
