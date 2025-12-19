import os
import glob
import ctypes
import sys

def load_libs():
    print("🚀 Bootstrapping NVIDIA Libraries for WSL...")
    
    # 1. Find the venv site-packages
    # We assume this script is running inside the venv
    import site
    site_packages = site.getsitepackages()[0]
    nvidia_base = os.path.join(site_packages, "nvidia")
    
    if not os.path.exists(nvidia_base):
        print(f"❌ Error: Could not find nvidia folder at {nvidia_base}")
        return

    # 2. Define the exact list of libraries TF needs
    # We look for *any* version of these files
    libs_to_load = [
        "cudart", "cublas", "cublasLt", "cufft", "curand", 
        "cusolver", "cusparse", "cudnn", "nccl", "nvjitlink"
    ]
    
    loaded_count = 0
    
    # 3. Iterate and Load with RTLD_GLOBAL
    # This makes the symbols visible to TensorFlow
    for lib_name in libs_to_load:
        # Search patterns like: .../nvidia/cudart/lib/libcudart.so*
        search_path = os.path.join(nvidia_base, "*", "lib", f"lib{lib_name}.so*")
        matches = glob.glob(search_path)
        
        if not matches:
            # Special case for WSL System Driver (libcuda.so)
            continue
            
        # Pick the first match (usually the real .so or the major version symlink)
        lib_path = matches[0]
        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            loaded_count += 1
        except OSError as e:
            print(f"⚠️ Failed to load {lib_name}: {e}")

    # 4. Load System CUDA Driver (libcuda.so.1) explicitly
    try:
        ctypes.CDLL("/usr/lib/wsl/lib/libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
        print("✅ System CUDA Driver loaded.")
    except OSError:
        print("❌ System CUDA Driver not found in /usr/lib/wsl/lib/")

    print(f"✅ Pre-loaded {loaded_count} NVIDIA libraries into memory.")

if __name__ == "__main__":
    load_libs()