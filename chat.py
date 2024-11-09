
from vllm import LLM, SamplingParams
from fastapi import FastAPI, Body
import time

start=time.time()

llm = LLM(model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
	
	,tensor_parallel_size=4)
sampling_params = SamplingParams(temperature=0.5,max_tokens=1024)
end= time.time()
print(f"mdoel load time is {end-start}")

app=FastAPI()



def print_outputs(outputs):
    foo=""
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        foo+=f"Prompt: {prompt!r}, Generated text: {generated_text!r}\n"
    foo+="-" * 80
    return foo


print("=" * 80)

# In this script, we demonstrate how to pass input to the chat method:

@app.post("/")
async def chat(prompt:str= Body(...)):
	start_2=time.time()
	conversation = [
    	{
        	"role": "system",
        	"content": "You are a helpful assistant"
   	 },
    	{
        	"role": "user",
       		"content": "Hello"
    	},
   	 {
       		 "role": "assistant",
        	"content": "Hello! How can I assist you today?"
   	 },
   	 {
        	"role": "user",
     	  	 "content":f"{prompt}",
   	 },
	]
	outputs = await llm.chat(conversation,
        	           sampling_params=sampling_params,
                	   use_tqdm=False)
	out= print_outputs(outputs)
	end_2=time.time()
	print(f"generating tokens time is {end_2-start_2}")
	return out

# A chat template can be optionally supplied.
# If not, the model will use its default chat template.

# with open('template_falcon_180b.jinja', "r") as f:
#     chat_template = f.read()

# outputs = llm.chat(
#     conversations,
#     sampling_params=sampling_params,
#     use_tqdm=False,
#     chat_template=chat_template,
# )
