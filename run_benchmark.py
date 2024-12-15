import subprocess
import os
import re
import statistics
from pathlib import Path
import shutil
import argparse
import sys

class BenchmarkRunner:
    def __init__(self, verbose=False):
        self.BUILD_CMD = "cmake --workflow --preset=gcc-cuda"
        self.EXE_PATH = "./out/build/gcc-cuda/bin/libnyquist-examples"
        self.TEST_FILE = "./test_data/sb-reverie.opus"
        self.CUDA_FILE = "cuda/mdct_cuda.cu"
        self.tags = ["v0.1", "v0.2", "v0.3", "v0.4", "v0.5", "v0.6"]
        self.benchmark_dir = Path("benchmark")
        self.benchmark_dir.mkdir(exist_ok=True)
        self.verbose = verbose

    def insert_timing_code(self, version):
        """Insert CUDA timing code into the source file"""
        file_path = Path(self.CUDA_FILE)
        if not file_path.exists():
            print(f"Error: File not found at {file_path}")
            return False

        # Create backup
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        shutil.copy2(file_path, backup_path)
        
        # Define patterns for different versions
        patterns = {
            "v0.1": ("size_t size_window = overlap * sizeof(var_t);", "cudaFree(dev_f2);"),
            "v0.2": "mdct_cuda_process(state, input, output, trig, window, sine);",
            "v0.3": "mdct_cuda_process(states[state_size], input, output, trig, window, sine);",
            "v0.4": "mdct_cuda_process(states[state_size], input, output, trig, window, sine);",
            "v0.5": "mdct_cuda_process(states[state_size], input, output, trig, window, sine);",
            "v0.6": "mdct_cuda_process(states[state_size], input, output, sine);"
        }
        
        # Timing code blocks
        timing_start = """
    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);"""

        timing_end = """
    // End timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total Time: %.4f ms\\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);"""

        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()

        # Handle v0.1 differently
        if version == "v0.1":
            start_pattern, end_pattern = patterns["v0.1"]
            lines = content.splitlines()
            modified_content = []
            found_start = False
            found_end = False

            for line in lines:
                if start_pattern in line and not found_start:
                    modified_content.append(line)
                    modified_content.append(timing_start)
                    found_start = True
                elif end_pattern in line and not found_end:
                    modified_content.append(timing_end)
                    modified_content.append(line)
                    found_end = True
                else:
                    modified_content.append(line)
        else:
            # Handle other versions as before
            pattern = patterns.get(version, "mdct_cuda_process((state|states[state_size]), input, output, sine);")
            modified_content = []
            for line in content.splitlines():
                if pattern in line:
                    modified_content.extend([timing_start, line, timing_end])
                else:
                    modified_content.append(line)

        # Write modified content
        with open(file_path, 'w') as f:
            f.write('\n'.join(modified_content))

        # Verify insertion
        with open(file_path, 'r') as f:
            new_content = f.read()
            if "Start timing" in new_content and "End timing" in new_content:
                print("Successfully added timing code")
                return True
            else:
                print("Failed to add timing code")
                shutil.copy2(backup_path, file_path)
                backup_path.unlink()
                return False

    def run_benchmark(self, tag):
        """Run benchmark for a specific tag"""
        output_file = self.benchmark_dir / f"{tag.replace('.', '_')}.log"
        
        print(f"Testing tag: {tag}")
        
        # Checkout tag
        try:
            subprocess.run(["git", "checkout", tag], check=True, capture_output=not self.verbose)
        except subprocess.CalledProcessError:
            print(f"Failed to checkout tag {tag}")
            return False

        # Stash any changes
        subprocess.run(["git", "stash", "-q"], capture_output=not self.verbose)

        try:
            # Delete the out directory if it exists
            out_dir = Path("out")
            if out_dir.exists():
                print("Cleaning build directory...")
                if self.verbose:
                    print(f"Removing directory: {out_dir}")
                shutil.rmtree(out_dir)

            # Apply timing patch
            if not self.insert_timing_code(tag):
                raise Exception("Failed to apply timing patch")

            # Build
            print("Building the code...")
            build_result = subprocess.run(
                self.BUILD_CMD.split(),
                capture_output=not self.verbose,
                text=True,
                check=True
            )
            if self.verbose:
                print(build_result.stdout)
                if build_result.stderr:
                    print("Build errors/warnings:")
                    print(build_result.stderr)

            # Run benchmark
            print("Running the benchmark...")
            with open(output_file, 'w') as f:
                benchmark_result = subprocess.run(
                    [self.EXE_PATH, self.TEST_FILE],
                    check=True,
                    stdout=f if not self.verbose else subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if self.verbose:
                    print(benchmark_result.stdout)
                    if benchmark_result.stderr:
                        print("Benchmark errors/warnings:")
                        print(benchmark_result.stderr)
                    # Also write to file when verbose
                    f.write(benchmark_result.stdout)

            return True

        except Exception as e:
            print(f"Error during benchmark: {str(e)}")
            return False

        finally:
            # Cleanup
            print("Cleaning up changes...")
            subprocess.run(["git", "checkout", "--", self.CUDA_FILE], 
                         capture_output=not self.verbose)
            if Path(f"{self.CUDA_FILE}.bak").exists():
                Path(f"{self.CUDA_FILE}.bak").unlink()
            subprocess.run(["git", "stash", "pop", "-q"], 
                         capture_output=not self.verbose)

    def calculate_average(self, log_file):
        """Calculate average execution time from log file, skipping first and last 10 measurements"""
        if not log_file.exists():
            return "N/A"

        times = []
        with open(log_file, 'r') as f:
            for line in f:
                if "Total Time:" in line:
                    try:
                        time = float(line.split()[2])
                        times.append(time)
                    except (ValueError, IndexError):
                        continue

        # Check if we have enough measurements
        if len(times) <= 20:  # Need more than 20 measurements to skip 10 from each end
            print(f"Warning: Not enough measurements ({len(times)}) to skip first and last 10 values")
            return statistics.mean(times) if times else "N/A"
        
        # Skip first 10 and last 10 measurements
        trimmed_times = times[10:-10]
        
        if self.verbose:
            print(f"Total measurements: {len(times)}")
            print(f"Measurements used for average: {len(trimmed_times)}")
            print(f"Skipped first 10 values: {times[:10]}")
            print(f"Skipped last 10 values: {times[-10:]}")

        return statistics.mean(trimmed_times) if trimmed_times else "N/A"

    def run_all_benchmarks(self):
        """Run benchmarks for all tags and analyze results"""
        results = {}

        # Run benchmarks
        for tag in self.tags:
            if self.run_benchmark(tag):
                log_file = self.benchmark_dir / f"{tag.replace('.', '_')}.log"
                avg_time = self.calculate_average(log_file)
                results[tag] = avg_time

        # Print results
        print("\nResults Summary:")
        print("==================")
        print(f"{'Tag':<6} | {'Average Time (ms)':<15}")
        print("-" * 22)
        
        best_tag = None
        best_time = float('inf')
        
        for tag, avg in results.items():
            print(f"{tag:<6} | {avg:<15}")
            if avg != "N/A" and float(avg) < best_time:
                best_time = float(avg)
                best_tag = tag

        if best_tag:
            print(f"\nBest performing tag: {best_tag} with average time: {best_time:.4f} ms")

    def run_single_tag(self, tag):
        """Run benchmark for a single specified tag"""
        print(f"\nRunning benchmark for tag: {tag}")
        print("================================")
        
        if tag not in self.tags:
            print(f"Error: Invalid tag '{tag}'")
            print("\nAvailable tags:")
            for available_tag in self.tags:
                print(f"- {available_tag}")
            return False
        
        if self.run_benchmark(tag):
            log_file = self.benchmark_dir / f"{tag.replace('.', '_')}.log"
            avg_time = self.calculate_average(log_file)
            
            print("\nResults:")
            print("========")
            print(f"Tag: {tag}")
            print(f"Average Time: {avg_time} ms")
            return True
        else:
            print(f"\nFailed to run benchmark for tag: {tag}")
            return False

def print_available_tags(tags):
    print("\nAvailable tags:")
    for tag in tags:
        print(f"- {tag}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for CUDA implementation')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Show verbose output')
    
    # Create a mutually exclusive group for run mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('-a', '--all', action='store_true',
                           help='Run benchmarks for all tags')
    mode_group.add_argument('-t', '--tag',
                           help='Run benchmark for a specific tag')

    args = parser.parse_args()

    benchmark = BenchmarkRunner(verbose=args.verbose)

    # If no mode is specified, show help
    if not (args.all or args.tag):
        parser.print_help()
        print_available_tags(benchmark.tags)
        sys.exit(1)

    # Run the appropriate benchmark mode
    if args.all:
        benchmark.run_all_benchmarks()
    elif args.tag:
        benchmark.run_single_tag(args.tag)