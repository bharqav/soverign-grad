#include "sovereigngrad/safetensors.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace sovereigngrad {
namespace safetensors {

namespace {

std::string dtype_to_safetensors_str(DType dt) {
    switch (dt) {
        case DType::Float32: return "F32";
        case DType::Float16: return "F16";
        case DType::Int32:   return "I32";
        case DType::Int64:   return "I64";
        default:             return "F32";
    }
}

DType safetensors_str_to_dtype(const std::string& s) {
    if (s == "F32") return DType::Float32;
    if (s == "F16") return DType::Float16;
    if (s == "I32") return DType::Int32;
    if (s == "I64") return DType::Int64;
    return DType::Float32;
}

size_t dtype_element_size(DType dt) {
    switch (dt) {
        case DType::Float32: return sizeof(float);
        case DType::Float16: return 2;
        case DType::Int32:   return sizeof(int32_t);
        case DType::Int64:   return sizeof(int64_t);
        default:             return sizeof(float);
    }
}

// Lightweight, robust JSON parser for SafeTensors headers
std::unordered_map<std::string, TensorMeta> parse_safetensors_json(const std::string& json_str) {
    std::unordered_map<std::string, TensorMeta> metas;
    
    size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t' || 
                                         json_str[pos] == '\n' || json_str[pos] == '\r')) {
            pos++;
        }
    };

    auto read_string = [&]() -> std::string {
        skip_ws();
        if (pos >= json_str.size() || json_str[pos] != '"') return "";
        pos++; // skip opening quote
        size_t start = pos;
        while (pos < json_str.size() && json_str[pos] != '"') {
            if (json_str[pos] == '\\' && pos + 1 < json_str.size()) pos += 2;
            else pos++;
        }
        std::string s = json_str.substr(start, pos - start);
        if (pos < json_str.size() && json_str[pos] == '"') pos++; // skip closing quote
        return s;
    };

    skip_ws();
    if (pos >= json_str.size() || json_str[pos] != '{') {
        throw std::runtime_error("Invalid SafeTensors JSON: Expected '{'");
    }
    pos++;

    while (pos < json_str.size()) {
        skip_ws();
        if (pos < json_str.size() && json_str[pos] == '}') {
            pos++;
            break;
        }

        std::string key = read_string();
        if (key.empty()) break;

        skip_ws();
        if (pos >= json_str.size() || json_str[pos] != ':') {
            throw std::runtime_error("Invalid SafeTensors JSON: Expected ':' after key " + key);
        }
        pos++; // skip ':'

        skip_ws();
        if (key == "__metadata__") {
            // Skip metadata dictionary
            int brace_count = 0;
            while (pos < json_str.size()) {
                if (json_str[pos] == '{') brace_count++;
                else if (json_str[pos] == '}') {
                    brace_count--;
                    if (brace_count == 0) { pos++; break; }
                }
                pos++;
            }
        } else {
            if (pos >= json_str.size() || json_str[pos] != '{') {
                throw std::runtime_error("Invalid SafeTensors JSON: Expected '{' for tensor " + key);
            }
            pos++;

            TensorMeta meta;
            while (pos < json_str.size()) {
                skip_ws();
                if (pos < json_str.size() && json_str[pos] == '}') {
                    pos++;
                    break;
                }

                std::string field = read_string();
                skip_ws();
                if (pos < json_str.size() && json_str[pos] == ':') pos++;

                skip_ws();
                if (field == "dtype") {
                    meta.dtype = read_string();
                } else if (field == "shape") {
                    if (pos < json_str.size() && json_str[pos] == '[') pos++;
                    while (pos < json_str.size()) {
                        skip_ws();
                        if (pos < json_str.size() && json_str[pos] == ']') { pos++; break; }
                        if (pos < json_str.size() && json_str[pos] == ',') { pos++; continue; }
                        size_t num_start = pos;
                        while (pos < json_str.size() && (std::isdigit(json_str[pos]) || json_str[pos] == '-')) pos++;
                        if (pos > num_start) {
                            meta.shape.push_back(std::stoll(json_str.substr(num_start, pos - num_start)));
                        }
                    }
                } else if (field == "data_offsets") {
                    std::vector<size_t> offsets;
                    if (pos < json_str.size() && json_str[pos] == '[') pos++;
                    while (pos < json_str.size()) {
                        skip_ws();
                        if (pos < json_str.size() && json_str[pos] == ']') { pos++; break; }
                        if (pos < json_str.size() && json_str[pos] == ',') { pos++; continue; }
                        size_t num_start = pos;
                        while (pos < json_str.size() && std::isdigit(json_str[pos])) pos++;
                        if (pos > num_start) {
                            offsets.push_back(std::stoull(json_str.substr(num_start, pos - num_start)));
                        }
                    }
                    if (offsets.size() >= 2) {
                        meta.data_offset_start = offsets[0];
                        meta.data_offset_end = offsets[1];
                    }
                } else {
                    // Skip unknown field value
                    while (pos < json_str.size() && json_str[pos] != ',' && json_str[pos] != '}') pos++;
                }

                skip_ws();
                if (pos < json_str.size() && json_str[pos] == ',') pos++;
            }

            metas[key] = meta;
        }

        skip_ws();
        if (pos < json_str.size() && json_str[pos] == ',') pos++;
    }

    return metas;
}

} // anonymous namespace

void save_file(const std::unordered_map<std::string, Tensor>& state_dict, const std::string& filepath) {
    // 1. Compute offsets and serialize JSON metadata
    std::ostringstream json_ss;
    json_ss << "{";

    size_t current_offset = 0;
    std::vector<std::pair<std::string, Tensor>> sorted_entries;
    for (const auto& kv : state_dict) {
        sorted_entries.push_back(kv);
    }
    std::sort(sorted_entries.begin(), sorted_entries.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (size_t i = 0; i < sorted_entries.size(); ++i) {
        const auto& name = sorted_entries[i].first;
        const auto& tensor = sorted_entries[i].second;

        DType dt = tensor.dtype();
        size_t byte_len = tensor.numel() * dtype_element_size(dt);
        size_t start_off = current_offset;
        size_t end_off = current_offset + byte_len;
        current_offset = end_off;

        json_ss << "\"" << name << "\":{"
                << "\"dtype\":\"" << dtype_to_safetensors_str(dt) << "\","
                << "\"shape\":[";
        for (size_t s = 0; s < tensor.shape().size(); ++s) {
            json_ss << tensor.shape()[s] << (s + 1 < tensor.shape().size() ? "," : "");
        }
        json_ss << "],"
                << "\"data_offsets\":[" << start_off << "," << end_off << "]"
                << "}" << (i + 1 < sorted_entries.size() ? "," : "");
    }
    json_ss << "}";

    std::string json_str = json_ss.str();
    uint64_t header_len = static_cast<uint64_t>(json_str.size());

    // 2. Write to binary file
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for SafeTensors save: " + filepath);
    }

    // Header size (8 bytes, little-endian)
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    // Header JSON string
    out.write(json_str.data(), header_len);

    // Contiguous tensor binary buffers
    for (const auto& kv : sorted_entries) {
        Tensor cpu_tensor = kv.second.to(Device::CPU);
        if (!cpu_tensor.is_contiguous()) {
            cpu_tensor = cpu_tensor.contiguous();
        }
        size_t byte_len = cpu_tensor.numel() * dtype_element_size(cpu_tensor.dtype());
        out.write(reinterpret_cast<const char*>(cpu_tensor.data()), byte_len);
    }

    out.close();
}

std::unordered_map<std::string, Tensor> load_file(const std::string& filepath, Device device) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open SafeTensors file: " + filepath);
    }

    // 1. Read 8-byte header length
    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!in || header_len == 0 || header_len > 100 * 1024 * 1024) { // 100MB sanity limit on header
        throw std::runtime_error("Corrupted or invalid SafeTensors header size in: " + filepath);
    }

    // 2. Read JSON header
    std::string json_str(header_len, '\0');
    in.read(&json_str[0], header_len);
    if (!in) {
        throw std::runtime_error("Unexpected EOF while reading SafeTensors JSON header in: " + filepath);
    }

    // 3. Parse JSON metadata
    auto metas = parse_safetensors_json(json_str);

    // 4. Construct tensors
    std::unordered_map<std::string, Tensor> result;
    size_t data_base_pos = sizeof(uint64_t) + header_len;

    for (const auto& kv : metas) {
        const std::string& name = kv.first;
        const TensorMeta& meta = kv.second;

        DType dt = safetensors_str_to_dtype(meta.dtype);
        Tensor t(meta.shape, Device::CPU, dt);

        in.seekg(data_base_pos + meta.data_offset_start, std::ios::beg);
        size_t expected_bytes = meta.data_offset_end - meta.data_offset_start;
        in.read(reinterpret_cast<char*>(t.data()), expected_bytes);

        if (!in) {
            throw std::runtime_error("Failed to read binary data for tensor '" + name + "' in: " + filepath);
        }

        if (device != Device::CPU) {
            result[name] = t.to(device);
        } else {
            result[name] = t;
        }
    }

    return result;
}

void save_module(const nn::Module& module, const std::string& filepath) {
    save_file(module.state_dict(), filepath);
}

void load_module(nn::Module& module, const std::string& filepath, bool strict) {
    auto sd = load_file(filepath);
    module.load_state_dict(sd, strict);
}

std::unordered_map<std::string, TensorMeta> inspect_file(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open SafeTensors file: " + filepath);
    }

    uint64_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!in || header_len == 0) {
        throw std::runtime_error("Invalid SafeTensors header in: " + filepath);
    }

    std::string json_str(header_len, '\0');
    in.read(&json_str[0], header_len);
    return parse_safetensors_json(json_str);
}

} // namespace safetensors
} // namespace sovereigngrad
