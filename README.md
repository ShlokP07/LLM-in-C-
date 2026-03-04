# LLM From Scratch (C++)

Minimal Tensor + Autograd + Transformer framework for training miniGPT. CPU-first, debuggable, PyTorch-like API.

## Environment Setup

### Prerequisites

- **C++17** compiler (MSVC 2019+, GCC 8+, Clang 7+)
- **CMake** 3.16+
- (Optional but recommended) **vcpkg** for dependency management (OpenBLAS)

### Install vcpkg (once, global)

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
```

Optionally set an environment variable so you don't have to repeat the path:

```powershell
$env:VCPKG_ROOT="C:\vcpkg"   # add this to your user env vars for persistence
```

### Configure & Build (with OpenBLAS via vcpkg)

From the project root:

```powershell
cd "d:\Downloads\VS Code Projects\LLM From Scartch in C++"

# Install dependencies declared in vcpkg.json (includes openblas)
C:\vcpkg\vcpkg.exe install --triplet x64-windows

# Fresh build directory
Remove-Item -Recurse -Force build  # PowerShell; or delete 'build' manually
mkdir build
cd build

# Configure with vcpkg toolchain
cmake .. `
  -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows

# Build
cmake --build . --config Release
```

On a successful configure you should see a line like:

```text
OpenBLAS found (via vcpkg or system)
```

which means `LLM_USE_BLAS` is defined and `matmul` will use the BLAS-backed `cblas_sgemm` path instead of the naive triple loop.

If you prefer not to use vcpkg, you can still do a plain build:

```powershell
cd "d:\Downloads\VS Code Projects\LLM From Scartch in C++"
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Run

```powershell
.\Release\llm_main.exe
```

### Run Tests

```powershell
ctest -C Release
# or
.\Release\llm_tests.exe
```

## Project Structure

```
├── CMakeLists.txt
├── include/llm/         # Public headers
├── src/                 # Implementation
│   ├── main.cpp
│   ├── core_stub.cpp
│   ├── tensor.cpp       # (to add)
│   ├── autograd.cpp     # (to add)
│   ├── ops.cpp          # (to add)
│   ├── nn/              # Linear, Embedding, LayerNorm, etc.
│   ├── optim/           # SGD, AdamW
│   ├── data/            # Dataset, DataLoader
│   └── utils/           # init, checkpoint
└── tests/
```

## Next Steps

1. Implement `Tensor` (storage, shape, dtype, requires_grad)
2. Implement Autograd engine (Node, backward, topo sort)
3. Add ops (add, mul, matmul, softmax, etc.)
4. Add nn layers (Linear, Embedding, LayerNorm, GELU, Attention)
5. Add optimizers (SGD, AdamW)
6. Add DataLoader for token sequences
7. Build miniGPT and train
