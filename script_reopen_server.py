import subprocess
import os
import time
import json
import matplotlib.pyplot as plt
import sys

# Optional: If implementing a health check
import requests

# List of prompt sizes
prompt_sizes = [20, 40, 80, 160, 320, 640, 1280]

# Define the server command with logging
server_command_template = 'vllm serve meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --tensor_parallel_size 4'

# Directory to store benchmark results and PID file
result_dir = 'benchmark_out_reopen/benchmark_5'

# Path to the PID file
pid_file = os.path.join(result_dir, 'vllm.pid')

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Add a README file to the directory
readme_path = os.path.join(result_dir, 'readme.txt')
with open(readme_path, 'w') as f:
    # Write the server command used to start the server
    f.write(server_command_template + '\n')
    try:
        with open('script_reopen_server.py', 'r') as f2:
            lines = f2.readlines()
            for line in lines[100:114]:
                f.write(line)
    except FileNotFoundError:
        print("Warning: 'script_reopen_server.py' not found. README will not include script lines.")

def stop_server():
    """Stop the vllm server using the PID file."""
    if not os.path.exists(pid_file):
        print(f"No PID file found at {pid_file}. Is the server running?")
        return
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        print(f"Attempting to terminate vllm server with PID: {pid}")
        os.killpg(os.getpgid(pid), 15)  # Send SIGTERM to the process group
        print("Server terminated successfully.")
    except ProcessLookupError:
        print(f"No process found with PID: {pid}. Removing stale PID file.")
    except Exception as e:
        print(f"Error terminating server: {e}")
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)

def start_server():
    """Start the vllm server in the background and record its PID."""
    try:
        # Open log files for stdout and stderr
        stdout_log = os.path.join(result_dir, 'vllm_stdout.log')
        stderr_log = os.path.join(result_dir, 'vllm_stderr.log')
        with open(stdout_log, 'a') as out, open(stderr_log, 'a') as err:
            # Start the server process
            process = subprocess.Popen(
                server_command_template,
                shell=True,
                stdout=out,
                stderr=err,
                preexec_fn=os.setsid  # Start the process in a new session
            )
        # Write the PID to the pid_file
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        print(f"Server started with PID: {process.pid}")
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

def wait_for_server(host='127.0.0.1', port=8000, timeout=120):
    """Wait until the server is ready by checking a health endpoint."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f'http://{host}:{port}/')  # Replace with actual health endpoint
            print(f"Server response: {response.status_code}")
            if response.status_code:
                print("Server is ready.")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    print("Server did not become ready in time.")
    sys.exit(1)

def run_benchmark(prompt_size):
    """Run the benchmark for a given prompt size."""
    json_file = f"num_prompts_{prompt_size}.json"
    command = f'python3 benchmarks/benchmark_serving.py ' \
            f'--model meta-llama/Llama-3.1-8B-Instruct ' \
            f'--dataset-name hf ' \
            f'--dataset-path /home/hkngae/test/temp_dataset/output.json ' \
            f'--hf-split train ' \
            f'--num-prompts {prompt_size} ' \
            f'--goodput  ttft:2000  ' \
            f'--save-result ' \
            f'--result-filename {json_file} ' \
            f'--result-dir {result_dir} ' \
            f'--use-beam-search ' \
        # f'--request-rate 80.0 ' \
        # f'--burstiness 0.25 ' \
        # f'--max-concurrency 80'
    
    
    try:
        subprocess.run(command, shell=True)
        print(f"Benchmark for prompt size {prompt_size} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed for prompt size {prompt_size}: {e}")
    return json_file

def plot_results(json_file, prompt_size):
    """Plot the benchmark results."""
    json_full_file = os.path.join(result_dir, json_file)
    
    # Wait until the JSON file is created
    while not os.path.exists(json_full_file):
        time.sleep(1)
    
    with open(json_full_file, 'r') as file:
        data = json.load(file)
    
    input_lens = data.get('input_lens', [])
    ttfts = data.get('ttfts', [])
    itls = data.get('itls', [])
    num_prompts = list(range(1, len(input_lens) + 1))
    itls_avg = [sum(i) / len(i) if len(i) > 0 else 0 for i in itls]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    ax1.scatter(input_lens, ttfts, alpha=0.6)
    ax1.set_title('Input Length vs TTFTs')
    ax1.set_xlabel('Input Length')
    ax1.set_ylabel('TTFTs')
    
    ax2.scatter(input_lens, itls_avg, alpha=0.6)
    ax2.set_title('Input Length vs Average ITLs')
    ax2.set_xlabel('Input Length')
    ax2.set_ylabel('Average ITLs')
    
    ax3.scatter(num_prompts, ttfts, alpha=0.6)
    ax3.set_title(f'Prompt Index (Total: {prompt_size}) vs TTFTs')
    ax3.set_xlabel('Prompt Index')
    ax3.set_ylabel('TTFTs')
    
    ax4.scatter(num_prompts, itls_avg, alpha=0.6)
    ax4.set_title(f'Prompt Index (Total: {prompt_size}) vs Average ITLs')
    ax4.set_xlabel('Prompt Index')
    ax4.set_ylabel('Average ITLs')
    
    plt.tight_layout()
    output_path = os.path.join(result_dir, f'plot_{prompt_size}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    for size in prompt_sizes:
        print(f"\nReopening server with prompt size: {size}")
        stop_server()
        time.sleep(60)  # Wait for the server to stop
        start_server()
        # Optional: Replace with a health check if implemented
        wait_for_server()
        # wait_for_server()  # Uncomment if health check is implemented
        json_file = run_benchmark(size)
        plot_results(json_file, size)
    stop_server()