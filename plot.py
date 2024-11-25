import json
import os
import matplotlib.pyplot as plt

# Read data from JSON file
with open('/home/hkngae/test/benchmark_out/vllm-infqps-Llama-3.1-8B-Instruct-20241125-031656.json', 'r') as file:
    data = json.load(file)



# Assuming the JSON data is in the format: {"x": [...], "y": [...]}
num_prompts = [i+1 for i in range(data['num_prompts'])]
input_lens = data['input_lens']
ttfts = data['ttfts']
itls = data['itls']

# itls is a list of lists, we get the average of each list and print it
itls_avg = [sum(i)/len(i) for i in itls]

# Create a figure with 4 subplots (2 rows, 2 columns)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

# Plot the data as a dot graph in the first subplot
ax1.scatter(input_lens, ttfts)
ax1.set_title('input_lens vs ttfts')
ax1.set_xlabel('input_lens')
ax1.set_ylabel('ttfts')

# Plot the data as a dot graph in the second subplot
ax2.scatter(input_lens, itls_avg)
ax2.set_title('input_lens vs itls_avg')
ax2.set_xlabel('input_lens')
ax2.set_ylabel('output_lens')

# Plot the data as a line graph in the third subplot (example)
ax3.plot(num_prompts, itls_avg)
ax3.set_title('num_prompts vs itls_avg')
ax3.set_xlabel('num_prompts')
ax3.set_ylabel('itls_avg')

# Plot the data as a line graph in the fourth subplot (example)
ax4.plot(num_prompts, ttfts)
ax4.set_title('num_prompts vs ttfts')
ax4.set_xlabel('num_prompts')
ax4.set_ylabel('ttfts')

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG image
output_path = '/home/hkngae/test/benchmark_out/plot1.png'
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

plt.savefig(output_path)