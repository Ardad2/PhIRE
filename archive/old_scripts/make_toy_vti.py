import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import os

# Ensure output directory exists
os.makedirs("vtk_inputs", exist_ok=True)

# Generate 2D scalar field
nx, ny = 64, 64
x = np.linspace(-2, 2, nx, dtype=np.float32)
y = np.linspace(-2, 2, ny, dtype=np.float32)
X, Y = np.meshgrid(x, y, indexing="ij")

# Interesting topology: Gaussian peak + oscillations
speed = (np.exp(-(X**2 + Y**2)) + 0.15*np.sin(4*X)*np.cos(4*Y)).astype(np.float32)
speed = np.ascontiguousarray(speed)

print(f"Generated field: {speed.shape}")
print(f"  Range: [{speed.min():.4f}, {speed.max():.4f}]")

# Create VTK image
img = vtk.vtkImageData()
img.SetDimensions(nx, ny, 1)
img.SetExtent(0, nx-1, 0, ny-1, 0, 0)
img.SetSpacing(1.0, 1.0, 1.0)
img.SetOrigin(0.0, 0.0, 0.0)

# Add scalar array
arr = numpy_to_vtk(speed.ravel(order="F"), deep=True, array_type=vtk.VTK_FLOAT)
arr.SetName("speed")
img.GetPointData().SetScalars(arr)

# Write VTI file (ASCII mode - works reliably)
w = vtk.vtkXMLImageDataWriter()
w.SetFileName("vtk_inputs/toy.vti")
w.SetInputData(img)
w.SetDataModeToAscii()  # ASCII is reliable and TTK-compatible

ok = w.Write()

if ok:
    print(f"✓ Wrote: vtk_inputs/toy.vti")
    size = os.path.getsize("vtk_inputs/toy.vti")
    print(f"  File size: {size:,} bytes")
else:
    print("✗ Failed to write file")
    exit(1)
