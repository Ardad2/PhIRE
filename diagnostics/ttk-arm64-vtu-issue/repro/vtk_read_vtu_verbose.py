import sys
import vtk

path = sys.argv[1]
print("VTU:", path)
print("vtk version:", vtk.vtkVersion.GetVTKVersion())

r = vtk.vtkXMLUnstructuredGridReader()
r.SetFileName(path)
r.Update()

print("ErrorCode:", r.GetErrorCode())
out = r.GetOutput()
print("Output is None?", out is None)
if out:
    print("Points:", out.GetNumberOfPoints())
    print("Cells :", out.GetNumberOfCells())
