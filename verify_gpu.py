import os
import sys
import glob

# --- CONFIGURATION ---
# We must locate the venv site-packages dynamically
venv_base = os.path.join(os.getcwd(), "venv", "lib")
# Find the pythonX.Y folder (e.g., python3.12)
python_dirs = glob.glob(os.path.join(venv_base, "python*"))
if not python_dirs:
    print("❌ Critical Error: Could not find python directory in venv/lib/")
    sys.exit(1)

site_packages = os.path.join(python_dirs[0], "site-packages")
nvidia_base = os.path.join(site_packages, "nvidia")

# --- STEP 1: DEFINE LIBRARY PATHS ---
# These are the specific paths TensorFlow needs to find the GPU
libs_to_add = [
    "/usr/lib/wsl/lib",  # WSL System Drivers (libcuda.so.1)
    os.path.join(nvidia_base, "cudnn", "lib"),
    os.path.join(nvidia_base, "cublas", "lib"),
    os.path.join(nvidia_base, "cuda_cupti", "lib"),
    os.path.join(nvidia_base, "cuda_nvcc", "lib"),
    os.path.join(nvidia_base, "cuda_nvrtc", "lib"),
    os.path.join(nvidia_base, "cuda_runtime", "lib"),
    os.path.join(nvidia_base, "cufft", "lib"),
    os.path.join(nvidia_base, "curand", "lib"),
    os.path.join(nvidia_base, "cusolver", "lib"),
    os.path.join(nvidia_base, "cusparse", "lib"),
    os.path.join(nvidia_base, "nccl", "lib"),
    os.path.join(nvidia_base, "nvjitlink", "lib"),
]

# --- STEP 2: INJECT INTO ENVIRONMENT ---
# We add these to LD_LIBRARY_PATH *inside* the python process
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
new_ld = ":".join(libs_to_add) + ":" + current_ld
os.environ["LD_LIBRARY_PATH"] = new_ld

# We also verify the symlink exists
libcuda_link = os.path.join(nvidia_base, "cuda_runtime", "lib", "libcuda.so")
if not os.path.exists(libcuda_link):
    print(f"⚠️ Warning: {libcuda_link} does not exist.")
    print("   Attempting to create it locally for this session...")
    try:
        os.symlink("/usr/lib/wsl/lib/libcuda.so.1", libcuda_link)
        print("   ✅ Symlink created.")
    except Exception as e:
        print(f"   ❌ Failed to create symlink: {e}")

# --- STEP 3: IMPORT TENSORFLOW ---
# Now that paths are set, we import TF.
print(f"🔍 LD_LIBRARY_PATH set with {len(libs_to_add)} nvidia paths.")
print("⏳ Importing TensorFlow (this may take a moment)...")

# Force verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" 

import tensorflow as tf

# --- STEP 4: CHECK GPU ---
print("\n--- GPU DIAGNOSTICS ---")
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ SUCCESS! Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"   - {gpu}")
    
    # Try a tiny computation to prove it works
    print("\n🧪 Running Test Computation...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"   ✅ Computation Result:\n{c.numpy()}")
    except Exception as e:
        print(f"   ❌ Computation Failed: {e}")
else:
    print("❌ FAILURE: No GPUs found.")
    print("\n🔎 Debug Info:")
    print(f"   Build Info: {tf.sysconfig.get_build_info()}")