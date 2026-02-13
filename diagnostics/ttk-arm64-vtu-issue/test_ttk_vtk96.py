#!/usr/bin/env python3
import sys

print("="*70)
print("TTK + VTK 9.6.0 Compatibility Test")
print("="*70)

try:
    import vtk
    vtk_version = vtk.vtkVersion.GetVTKVersion()
    print(f"✅ VTK: {vtk_version}")
except Exception as e:
    print(f"❌ VTK import failed: {e}")
    sys.exit(1)

try:
    import topologytoolkit as ttk
    print(f"✅ TTK: imported successfully")
except Exception as e:
    print(f"❌ TTK import failed: {e}")
    sys.exit(1)

# Test VTU reading
try:
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName("/home/adadhwal/PhIRE/ttk_outputs/direct/pd_vtu/gan_GT_s0_speed_p160_x0_y0_pd.vtu_port_0.vtu")
    reader.Update()
    output = reader.GetOutput()
    npts = output.GetNumberOfPoints()
    print(f"✅ VTU reading: {npts} points")
except Exception as e:
    print(f"❌ VTU reading failed: {e}")
    sys.exit(1)

# Test TTK filter
try:
    pd = ttk.ttkPersistenceDiagram()
    pd.SetInputConnection(reader.GetOutputPort())
    pd.Update()
    print(f"✅ TTK filter: works!")
except Exception as e:
    print(f"❌ TTK filter failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("🎉 ALL TESTS PASSED - TTK + VTK 9.6.0 compatible!")
print("="*70)
