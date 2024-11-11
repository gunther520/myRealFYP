import subprocess


for i in range(1,8):
    command = f'python3 benchmarks/benchmark_serving.py         --model meta-llama/Llama-3.1-8B-Instruct     --dataset-name hf  --dataset-path /home/hkngae/test/temp_dataset/output.json     --hf-split train   --num-prompts {2**i * 10} >> benchmark_out2.txt'
    subprocess.run(command, shell=True)