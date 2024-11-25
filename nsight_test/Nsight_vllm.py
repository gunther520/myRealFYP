import torch
import os
import torch.cuda.profiler as profiler
from vllm import LLM, SamplingParams

def create_llm():
    return LLM(
        model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        gpu_memory_utilization=1,
        tensor_parallel_size=8,
    )

if __name__ == "__main__":
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']= "spawn"
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    prompt = "Translate English to French: 'Hello, how are you?'"
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9)
    llm = create_llm()
    # Create a BART encoder/decoder model instance
    # Perform profiling with Nsight
    with torch.autograd.profiler.emit_nvtx():
        profiler.start()
        result = llm.generate(prompt, sampling_params)
        profiler.stop()
    print("Generated text: ", result)
    print("Done")