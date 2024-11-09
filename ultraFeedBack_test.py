import os
import torch
from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline, sample_n_steps
from distilabel.steps import (
    GroupColumns,
    KeepColumns,
    LoadDataFromHub,

)
from distilabel.steps.tasks import TextGeneration, UltraFeedback

print(torch.cuda.device_count())
#sample_three_llms = sample_n_steps(n=2)


with Pipeline(name="ultrafeedback-pipeline") as pipeline:
    load_hub_dataset = LoadDataFromHub(
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
        #batch_size=2,
    )

    text_generation_with_notus = TextGeneration(
        name="text_generation_with_notus",
        llm=vLLM(model="meta-llama/Llama-3.2-1B-Instruct",extra_kwargs={"tensor_parallel_size":2}),
        #input_batch_size=2,
        output_mappings={"model_name": "generation_model"},
    )
    text_generation_with_zephyr = TextGeneration(
        name="text_generation_with_zephyr",
        llm=vLLM(model="meta-llama/Llama-3.2-1B",extra_kwargs={"tensor_parallel_size":2}),
        #input_batch_size=2,
        output_mappings={"model_name": "generation_model"},
    )


    combine_columns = GroupColumns(
        name="combine_columns",
        columns=["generation", "generation_model"],
        output_columns=["generations", "generation_models"],
        #input_batch_size=2
    )

    ultrafeedback = UltraFeedback(
        name="ultrafeedback_openai",
        llm=vLLM(model="meta-llama/Llama-3.2-1B",extra_kwargs={"tensor_parallel_size":2}),
        aspect="overall-rating",
        output_mappings={"model_name": "ultrafeedback_model"},
    )

    keep_columns = KeepColumns(
        name="keep_columns",
        columns=[
            "instruction",
            "generations",
            "generation_models",
            "ratings",
            "rationales",
            "ultrafeedback_model",
        ],
    )

    (
        load_hub_dataset
       # >> sample_three_llms
        >> [
            text_generation_with_notus,
            text_generation_with_zephyr,
        ]
        >> combine_columns
        >> ultrafeedback
        >> keep_columns
    )


    distiset = pipeline.run(
        use_cache=False,
    parameters={
        load_hub_dataset.name: {
            "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
            "split": "test",
        },
        text_generation_with_notus.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                }
                
            },
            "resources": {"replicas": 1, "gpus": 2}
        },
        text_generation_with_zephyr.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                }
                
            },
            "resources": {"replicas": 1, "gpus": 2}
        },

        ultrafeedback.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 2048,
                    "temperature": 0.7,
                }
                

            },
            "resources": {"replicas": 1, "gpus": 2}
        },
    }
)
    
    distiset.push_to_hub(
    "Gunther520/first-test-dataset3",
    commit_message="Initial commit",
    private=False,
    token=os.getenv("HF_TOKEN"),
    )
