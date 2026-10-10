# SovereignGrad Binary Checkpoint Specification (`.sgrd`)

**Format Revision**: `v1.0`  
**File Extension**: `.sgrd` (SovereignGrad Data)  
**Byte Order**: Little-Endian (Standard x86-64 / ARM64)

---

## 1. Overview & Design Principles

The `.sgrd` binary format is designed for high-performance tensor and model checkpoint serialization in C++ environments without external dependencies (such as Protocol Buffers or FlatBuffers).

### Key Characteristics
1. **Zero Dynamic Parsing Overhead**: Header and metadata tables are structured with fixed-size offsets for direct streaming reads and memory mapping.
2. **Deterministic Layout**: Tensors are stored in contiguous binary streams aligned with standard 32-bit floating point representations.
3. **Hardware Agnostic**: Checkpoints saved on GPU (`Device::CUDA`) are normalized during serialization to host memory order and can be loaded seamlessly onto CPU or CUDA targets.

---

## 2. Binary Layout Specification

A `.sgrd` file consists of three sequential blocks:
1. **File Header (8 Bytes)**
2. **State Dictionary Metadata Table**
3. **Contiguous Binary Tensor Payloads**

```
+-------------------------------------------------------------------+
|                           FILE HEADER                             |
|  Magic Bytes (4B): 0x53475244 ("SGRD") | Format Version (4B): 1   |
+-------------------------------------------------------------------+
|                        TENSOR COUNT TABLE                         |
|               Number of Serialized Tensors (4B uint32)            |
+-------------------------------------------------------------------+
|                         TENSOR ENTRY 0                            |
| Key String Length (4B) | UTF-8 Key Name | DType (4B)              |
| Rank N (4B) | Dims [d0, d1, ..., dN-1] (8B * N)                  |
| Num Elements (8B) | Raw Float32 Payload [val0, val1, ...]        |
+-------------------------------------------------------------------+
|                         TENSOR ENTRY 1                            |
| ...                                                               |
+-------------------------------------------------------------------+
|                         TENSOR ENTRY K-1                          |
| ...                                                               |
+-------------------------------------------------------------------+
```

---

## 3. Header Fields

| Offset | Field Name | Data Type | Value / Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `magic` | `uint32_t` | `0x53475244` (ASCII `"SGRD"`) |
| `0x04` | `version` | `uint32_t` | `1` (Current format revision) |
| `0x08` | `num_entries` | `uint32_t` | Total number of named tensor entries in the checkpoint |

---

## 4. Tensor Entry Structure

Each tensor entry is serialized sequentially:

```cpp
struct SgrdTensorEntry {
    // 1. Name Identifier
    uint32_t name_length;         // Byte length of UTF-8 parameter name
    char name[name_length];       // Hierarchical parameter name (e.g. "fc1.weight")

    // 2. Data Type & Dimensions
    uint32_t dtype;               // 0 = Float32
    uint32_t rank;                // Number of tensor dimensions (e.g. 2 for [64, 32])
    int64_t shape[rank];          // Dimension sizes (8 bytes each)
    int64_t numel;                // Total element count: shape[0] * ... * shape[rank-1]

    // 3. Binary Payload
    float data[numel];            // Raw IEEE 754 float32 buffer (numel * 4 bytes)
};
```

---

## 5. Compatibility & Verification

### Endianness
All scalar headers (`uint32_t`, `int64_t`) and floating-point values (`float32`) must be written and read in standard little-endian order.

### Error Handling & Integrity Validation
When loading a `.sgrd` file, the loader must verify:
1. **Magic Header**: Must match `0x53475244`. If corrupted or mismatched, throws `std::runtime_error("Invalid .sgrd checkpoint magic number")`.
2. **Version Compatibility**: Rejects files with unsupported format revisions.
3. **Payload Bounds**: Verifies that `stream.read()` reads precisely `numel * sizeof(float)` bytes without premature EOF.
4. **Strict Mode Matching**: When loading into an `nn::Module` with `strict = true`, all registered module parameters must be present in the checkpoint with identical tensor shapes.

---

## 6. Code Example: Inspecting `.sgrd` Checkpoints

```cpp
#include "sovereigngrad/serialization.hpp"
#include <iostream>

int main() {
    auto state = sovereigngrad::serialization::load_checkpoint("sovereign_digit_classifier.sgrd", sovereigngrad::Device::CPU);
    
    std::cout << "Inspecting Checkpoint:\n";
    for (const auto& [name, tensor] : state) {
        std::cout << "  - Parameter: " << name << " | Shape: [";
        for (size_t i = 0; i < tensor.shape().size(); ++i) {
            std::cout << tensor.shape()[i] << (i + 1 < tensor.shape().size() ? ", " : "");
        }
        std::cout << "] | Elements: " << tensor.numel() << "\n";
    }
    return 0;
}
```
