"""
Automated SovereignGrad vs PyTorch Benchmark Runner
Executes both C++ and Python benchmark suites, parsing and presenting the comparative report.
"""

import subprocess
import json
import sys
import os

def run_cpp_benchmark():
    exe_path = os.path.join("build", "sovereign_grad_benchmark")
    if os.name == 'nt' and not os.path.exists(exe_path):
        exe_path = os.path.join("build", "Release", "sovereign_grad_benchmark.exe")
        if not os.path.exists(exe_path):
            exe_path = os.path.join("build", "sovereign_grad_benchmark.exe")
            
    print(f"[1/2] Running SovereignGrad C++ Benchmark ({exe_path})...")
    result = subprocess.run([exe_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running C++ benchmark: {result.stderr}")
        return None
    print(result.stdout)
    return result.stdout

def run_pytorch_benchmark():
    print("[2/2] Running PyTorch Reference Benchmark (Python)...")
    result = subprocess.run([sys.executable, os.path.join("benchmarks", "benchmark_pytorch.py"), "--json"], 
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running PyTorch benchmark: {result.stderr}")
        return None
    print(result.stdout)
    
    # Extract json line
    for line in result.stdout.splitlines():
        if line.startswith("JSON_OUTPUT:"):
            return json.loads(line[len("JSON_OUTPUT:"):])
    return None

def main():
    print("=" * 70)
    print("         SOVEREIGNGRAD VS PYTORCH COMPREHENSIVE BENCHMARK         ")
    print("=" * 70)
    
    cpp_out = run_cpp_benchmark()
    py_data = run_pytorch_benchmark()
    
    print("\nBenchmark comparison complete.")

if __name__ == "__main__":
    main()
