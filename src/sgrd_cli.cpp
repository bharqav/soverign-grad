#include "sovereigngrad/tensor.hpp"
#include "sovereigngrad/serialization.hpp"
#include "sovereigngrad/safetensors.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>

using namespace sovereigngrad;

void print_help() {
    std::cout << "=================================================================\n";
    std::cout << "          SOVEREIGNGRAD CHECKPOINT UTILITY (sgrd-cli)           \n";
    std::cout << "=================================================================\n";
    std::cout << "Usage:\n";
    std::cout << "  sgrd-cli inspect <file.sgrd>               Display checkpoint metadata & parameter summary\n";
    std::cout << "  sgrd-cli verify <file.sgrd>                Verify magic header & file integrity\n";
    std::cout << "  sgrd-cli export-json <file.sgrd> [out.json] Export checkpoint structure as JSON\n";
    std::cout << "  sgrd-cli inspect-safetensors <file.st>     Inspect SafeTensors metadata\n";
    std::cout << "  sgrd-cli to-safetensors <in.sgrd> <out.st> Convert .sgrd checkpoint to .safetensors\n";
    std::cout << "  sgrd-cli from-safetensors <in.st> <out.sg> Convert .safetensors to .sgrd checkpoint\n";
    std::cout << "  sgrd-cli help                              Show this help menu\n\n";
}

int inspect_file(const std::string& filepath) {
    std::cout << "Inspecting Checkpoint: " << filepath << "\n";
    std::cout << "---------------------------------------------------------------------------------\n";

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << "\n";
        return 1;
    }

    char magic[4];
    uint32_t version = 0;
    file.read(magic, 4);
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (std::memcmp(magic, "SGRD", 4) != 0) {
        std::cerr << "Error: Invalid .sgrd file (magic header mismatch)\n";
        return 1;
    }

    uint32_t num_tensors = 0;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));

    std::cout << "Format Header:\n";
    std::cout << "  * Magic:           'SGRD' [Valid]\n";
    std::cout << "  * Format Version:  " << version << "\n";
    std::cout << "  * Parameter Count: " << num_tensors << " tensors\n\n";

    file.close();

    // Load state dictionary for detailed statistics
    try {
        auto state_dict = serialization::load_checkpoint(filepath, Device::CPU);

        std::cout << "Parameter Inventory:\n";
        std::cout << " | " << std::left << std::setw(28) << "Parameter Name"
                  << " | " << std::left << std::setw(18) << "Shape"
                  << " | " << std::right << std::setw(10) << "Elements"
                  << " | " << std::right << std::setw(12) << "Mean +/- Std"
                  << " | " << std::right << std::setw(12) << "Min / Max" << " |\n";
        std::cout << " |------------------------------|--------------------|------------|--------------|--------------|\n";

        size_t total_elements = 0;

        for (const auto& [name, tensor] : state_dict) {
            std::string shape_str = "[";
            for (size_t i = 0; i < tensor.shape().size(); ++i) {
                shape_str += std::to_string(tensor.shape()[i]) + (i + 1 < tensor.shape().size() ? ", " : "");
            }
            shape_str += "]";

            int64_t numel = tensor.numel();
            total_elements += numel;

            // Compute statistics
            const float* data = tensor.impl->data();
            float sum = 0.0f;
            float min_v = (numel > 0) ? data[0] : 0.0f;
            float max_v = (numel > 0) ? data[0] : 0.0f;

            for (int64_t i = 0; i < numel; ++i) {
                float v = data[i];
                sum += v;
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
            float mean = (numel > 0) ? sum / (float)numel : 0.0f;

            float var = 0.0f;
            for (int64_t i = 0; i < numel; ++i) {
                float diff = data[i] - mean;
                var += diff * diff;
            }
            float std = (numel > 0) ? std::sqrt(var / (float)numel) : 0.0f;

            std::stringstream stats_ss;
            stats_ss << std::fixed << std::setprecision(3) << mean << "+/-" << std::setprecision(3) << std;

            std::stringstream minmax_ss;
            minmax_ss << std::fixed << std::setprecision(2) << min_v << "/" << std::setprecision(2) << max_v;

            std::cout << " | " << std::left << std::setw(28) << name
                      << " | " << std::left << std::setw(18) << shape_str
                      << " | " << std::right << std::setw(10) << numel
                      << " | " << std::right << std::setw(12) << stats_ss.str()
                      << " | " << std::right << std::setw(12) << minmax_ss.str() << " |\n";
        }

        std::cout << "---------------------------------------------------------------------------------\n";
        std::cout << "Summary: " << state_dict.size() << " parameters | " 
                  << total_elements << " weights | " 
                  << std::fixed << std::setprecision(2) << (total_elements * sizeof(float) / 1024.0) << " KB\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error parsing checkpoint state dict: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

int verify_file(const std::string& filepath) {
    std::cout << "Verifying integrity of: " << filepath << "...\n";
    try {
        auto state_dict = serialization::load_checkpoint(filepath, Device::CPU);
        std::cout << "  [OK] Magic header 0x53475244 valid\n";
        std::cout << "  [OK] Version compatibility valid\n";
        std::cout << "  [OK] Successfully deserialized " << state_dict.size() << " tensors without errors\n";
        std::cout << "Result: PASSED (Checkpoint is valid and uncorrupted)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Result: FAILED (" << e.what() << ")\n";
        return 1;
    }
}

int export_json(const std::string& filepath, const std::string& out_json) {
    try {
        auto state_dict = serialization::load_checkpoint(filepath, Device::CPU);
        std::string json_file = out_json.empty() ? (filepath + ".json") : out_json;
        std::ofstream out(json_file);
        if (!out.is_open()) {
            std::cerr << "Error opening output JSON file: " << json_file << "\n";
            return 1;
        }

        out << "{\n";
        out << "  \"file\": \"" << filepath << "\",\n";
        out << "  \"num_tensors\": " << state_dict.size() << ",\n";
        out << "  \"tensors\": {\n";

        size_t idx = 0;
        for (const auto& [name, tensor] : state_dict) {
            out << "    \"" << name << "\": {\n";
            out << "      \"shape\": [";
            for (size_t i = 0; i < tensor.shape().size(); ++i) {
                out << tensor.shape()[i] << (i + 1 < tensor.shape().size() ? ", " : "");
            }
            out << "],\n";
            out << "      \"numel\": " << tensor.numel() << ",\n";
            out << "      \"dtype\": \"Float32\"\n";
            out << "    }" << (idx + 1 < state_dict.size() ? "," : "") << "\n";
            idx++;
        }
        out << "  }\n";
        out << "}\n";
        out.close();

        std::cout << "Exported checkpoint metadata to: " << json_file << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error exporting JSON: " << e.what() << "\n";
        return 1;
    }
}

int inspect_safetensors(const std::string& filepath) {
    try {
        std::cout << "Inspecting SafeTensors File: " << filepath << "\n";
        std::cout << "---------------------------------------------------------------------------------\n";
        auto metas = safetensors::inspect_file(filepath);
        std::cout << "Tensor Count: " << metas.size() << " tensors\n\n";
        std::cout << " | " << std::left << std::setw(32) << "Tensor Name"
                  << " | " << std::setw(6) << "DType"
                  << " | " << std::setw(20) << "Shape"
                  << " | " << std::right << std::setw(16) << "Byte Offsets" << " |\n";
        std::cout << " |----------------------------------|--------|----------------------|------------------|\n";

        size_t total_bytes = 0;
        for (const auto& kv : metas) {
            const auto& name = kv.first;
            const auto& m = kv.second;
            std::ostringstream shape_ss;
            shape_ss << "[";
            for (size_t s = 0; s < m.shape.size(); ++s) {
                shape_ss << m.shape[s] << (s + 1 < m.shape.size() ? ", " : "");
            }
            shape_ss << "]";

            std::ostringstream off_ss;
            off_ss << "[" << m.data_offset_start << ":" << m.data_offset_end << "]";
            total_bytes += (m.data_offset_end - m.data_offset_start);

            std::cout << " | " << std::left << std::setw(32) << (name.length() > 32 ? name.substr(0, 29) + "..." : name)
                      << " | " << std::setw(6) << m.dtype
                      << " | " << std::setw(20) << shape_ss.str()
                      << " | " << std::right << std::setw(16) << off_ss.str() << " |\n";
        }
        std::cout << "---------------------------------------------------------------------------------\n";
        std::cout << "Summary: " << metas.size() << " tensors | " << std::fixed << std::setprecision(2)
                  << (total_bytes / 1024.0f) << " KB total tensor payload\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error inspecting SafeTensors: " << e.what() << "\n";
        return 1;
    }
}

int convert_to_safetensors(const std::string& in_sgrd, const std::string& out_st) {
    try {
        std::cout << "Converting .sgrd -> .safetensors: " << in_sgrd << " -> " << out_st << " ...\n";
        auto sd = serialization::load_checkpoint(in_sgrd, Device::CPU);
        safetensors::save_file(sd, out_st);
        std::cout << "[+] Conversion successful! Converted " << sd.size() << " tensors.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error converting to SafeTensors: " << e.what() << "\n";
        return 1;
    }
}

int convert_from_safetensors(const std::string& in_st, const std::string& out_sgrd) {
    try {
        std::cout << "Converting .safetensors -> .sgrd: " << in_st << " -> " << out_sgrd << " ...\n";
        auto sd = safetensors::load_file(in_st, Device::CPU);
        serialization::save_checkpoint(sd, out_sgrd);
        std::cout << "[+] Conversion successful! Converted " << sd.size() << " tensors.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error converting from SafeTensors: " << e.what() << "\n";
        return 1;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    std::string command = argv[1];

    if (command == "inspect") {
        if (argc < 3) {
            std::cerr << "Error: Missing checkpoint filepath.\nUsage: sgrd-cli inspect <file.sgrd>\n";
            return 1;
        }
        return inspect_file(argv[2]);
    } else if (command == "verify") {
        if (argc < 3) {
            std::cerr << "Error: Missing checkpoint filepath.\nUsage: sgrd-cli verify <file.sgrd>\n";
            return 1;
        }
        return verify_file(argv[2]);
    } else if (command == "export-json") {
        if (argc < 3) {
            std::cerr << "Error: Missing checkpoint filepath.\nUsage: sgrd-cli export-json <file.sgrd> [out.json]\n";
            return 1;
        }
        std::string out_json = (argc >= 4) ? argv[3] : "";
        return export_json(argv[2], out_json);
    } else if (command == "inspect-safetensors") {
        if (argc < 3) {
            std::cerr << "Error: Missing SafeTensors filepath.\nUsage: sgrd-cli inspect-safetensors <file.safetensors>\n";
            return 1;
        }
        return inspect_safetensors(argv[2]);
    } else if (command == "to-safetensors") {
        if (argc < 4) {
            std::cerr << "Error: Missing filepaths.\nUsage: sgrd-cli to-safetensors <in.sgrd> <out.safetensors>\n";
            return 1;
        }
        return convert_to_safetensors(argv[2], argv[3]);
    } else if (command == "from-safetensors") {
        if (argc < 4) {
            std::cerr << "Error: Missing filepaths.\nUsage: sgrd-cli from-safetensors <in.safetensors> <out.sgrd>\n";
            return 1;
        }
        return convert_from_safetensors(argv[2], argv[3]);
    } else if (command == "help" || command == "--help" || command == "-h") {
        print_help();
        return 0;
    } else {
        std::cerr << "Unknown command: " << command << "\n\n";
        print_help();
        return 1;
    }
}
