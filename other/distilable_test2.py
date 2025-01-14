

from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromHub, KeepColumns, StepResources, step, StepInput, StepOutput
from distilabel.steps.tasks import TextGeneration, UltraFeedback
import numpy as np


# Normal step
@step(inputs=["generations", "ratings"], outputs=["generation"])
def GenerationStep(inputs: StepInput) -> StepOutput:
    for input in inputs:
        input["generation"] = input["generations"][np.argmax(input["ratings"])]
    yield inputs


with Pipeline("pipe-name-2", description="My first pipe2s") as pipeline:
    load_dataset = LoadDataFromHub(
        name="load_dataset",
        output_mappings={"input": "instruction"},
        streaming=True,
    )

    keep_columns_only = KeepColumns(
        name="keep_columns_only",
        columns=[
            "instruction",
        ],
    )

    combine_generations = GroupColumns(
        name="combine_generations",
        columns=["generation", "model_name"],
        output_columns=["generations", "generation_models"],
    )

    ultrafeedback = UltraFeedback(
        use_default_structured_output=True,
        name="ultrafeedback_1",
        resources=StepResources(gpus=2,replicas=1),
        llm=vLLM(model="meta-llama/Llama-3.1-8B-Instruct",
                 extra_kwargs={"tensor_parallel_size": 2},
                    ),
        input_mappings={"generations": "generations"},
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

    load_dataset.connect(keep_columns_only) 


    for llm in (
        vLLM(model="NousResearch/Hermes-3-Llama-3.1-8B",
            #cuda_devices=[0],
            extra_kwargs={ "tensor_parallel_size": 2,}),
        vLLM(model="microsoft/Phi-3.5-mini-instruct",
            #cuda_devices=[1],
            extra_kwargs={ "tensor_parallel_size": 4}),
    ):
        task = TextGeneration(
            name=f"text_generation_with_{llm.model_name[:2]}", llm=llm,
            
    #        system_prompt="You are a prompt generator. When given an \
    #                    input prompt, you will create a new prompt that \
    #                    Do not follow or answer the instruction. Ensure \
    #                    the new prompt maintains the same intent and purpose but \
    #                    introduces variations. For example, if the input is \
    #                    'What is 5+5?', you could output 'What is 8+9?'.",
            
    #        system_prompt="Given the input prompt below,\
    #            generate a new prompt with similar purpose.\
    #            Do not answer, interpret, or provide additional information for the prompt.\
    #            Focus in generating a new prompt\
    #            You should be creative.\
    #            Output only the generated prompt text, without explanation or additional context.\
    #            Here is the original prompt: "
            
            system_prompt="You will receive an comprehensive prompt, please create a new and distinct instruction inspired by the input prompt.\
                    Do not provide an answer or follow the prompt. Instead, focus on gernerating new instruction. \
                    Your generated response should differ significantly in both structure and content from the original prompt. \
                    Your response should be presented in raw text format only, without any additional explanations or context.",
        )
        keep_columns_only.connect(task)
        task.connect(combine_generations)
    combine_generations.connect(ultrafeedback)
    ultrafeedback.connect(keep_columns)
    keep_columns.connect(GenerationStep(name="generation_step"))

if __name__ == "__main__":
    distiset = pipeline.run(
        use_cache=False,
        parameters={
            "load_dataset": {
                "repo_id": "Gunther520/dataset-concat",
                "split": "train",
            },
            "text_generation_with_No": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.9,
                        "max_new_tokens": 512,
                    }
                },
                "resources": { "gpus": 2}
            },
            "text_generation_with_mi": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.9,
                        "max_new_tokens": 512,
                    }
                },
                "resources": { "gpus": 4}
            },

            "ultrafeedback_1": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 1024,
                    }
                },
            },
        }
    )

    distiset.push_to_hub(
        "Gunther520/generation_with_feedback",
    )