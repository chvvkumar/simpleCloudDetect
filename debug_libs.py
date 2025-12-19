import os
import glob
import ctypes
import sys

print("--- 🔍 Deep Dependency Check ---")

# 1. Setup Paths (Same as before)
venv_base = os.path.join(os.getcwd(), "venv", "lib")
python_dirs = glob.glob(os.path.join(venv_base, "python*"))
if not python_dirs:
    print("❌ Error: venv not found.")
    sys.exit(1)

site_packages = os.path.join(python_dirs[0], "site-packages")
nvidia_base = os.path.join(site_packages, "nvidia")

# Define the critical libraries TensorFlow needs
libs_to_test = [
    ("libcuda.so.1", "/usr/lib/wsl/lib"), # System driver
    ("libcudart.so.12", os.path.join(nvidia_base, "cuda_runtime", "lib")),
    ("libcublas.so.12", os.path.join(nvidia_base, "cublas", "lib")),
    ("libcublasLt.so.12", os.path.join(nvidia_base, "cublas", "lib")),
    ("libcufft.so.11", os.path.join(nvidia_base, "cufft", "lib")),
    ("libcurand.so.10", os.path.join(nvidia_base, "curand", "lib")),
    ("libcusolver.so.11", os.path.join(nvidia_base, "cusolver", "lib")),
    ("libcusparse.so.12", os.path.join(nvidia_base, "cusparse", "lib")),
    ("libcudnn.so.9", os.path.join(nvidia_base, "cudnn", "lib")),
]

# 2. Try to load each one
all_good = True
for lib_name, folder in libs_to_test:
    full_path = os.path.join(folder, lib_name)
    
    # Check if file exists
    if not os.path.exists(full_path):
        # Fallback: check if it exists in the folder under a slightly different name
        print(f"⚠️  File Missing: {lib_name}")
        print(f"    Looking in: {folder}")
        if os.path.exists(folder):
            print(f"    Contents: {os.listdir(folder)[:5]}...") # Show first 5 files
        all_good = False
        continue

    # Try to load it
    try:
        ctypes.CDLL(full_path)
        print(f"✅ Loaded: {lib_name}")
    except OSError as e:
        print(f"❌ FAILED to load: {lib_name}")
        print(f"   Error: {e}")
        all_good = False

if all_good:
    print("\n🎉 All libraries loaded manually! The issue is likely LD_LIBRARY_PATH ordering.")
else:
    print("\n💥 Issues found. Fix the missing/failing libraries above.")