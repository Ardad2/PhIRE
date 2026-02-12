import sys

if len(sys.argv) != 2:
    print("USAGE: python repro/vtk_read_vtu.py /work/repro/samples/file.vtu", file=sys.stderr)
    sys.exit(2)

path = sys.argv[1]
print("VTU:", path)

import vtk
print("vtk version:", vtk.vtkVersion.GetVTKVersion())

r = vtk.vtkXMLUnstructuredGridReader()
r.SetFileName(path)
ok = r.Update()

out = r.GetOutput()
print("Update() returned:", ok)
print("Output is None?", out is None)
if out is not None:
    print("Points:", out.GetNumberOfPoints())
    print("Cells :", out.GetNumberOfCells())
