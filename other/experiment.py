import os
from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline,sample_n_steps
from distilabel.steps import LoadDataFromHub, GroupColumns, KeepColumns
from distilabel.steps.tasks import TextGeneration,UltraFeedback

sample_one_llms = sample_n_steps(n=1)

with Pipeline(
    name="simple-text-generation-pipeline",
    description="A simple text generation pipeline",
) as pipeline:
    load_dataset = LoadDataFromHub(
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
        #batch_size=2
        )

    text_generation_Nous = TextGeneration(
        name="text_with_nous",
        group_generations=True,
        llm=vLLM(

                model="NousResearch/Hermes-2-Pro-Llama-3-8B",
                extra_kwargs={"tensor_parallel_size":1}
                #tensor_parallel_size='8'
        ),
        #input_batch_size=2,
        output_mappings={"model_name": "generation_model"},
        #system_prompt="You are a helpful prompt generating AI assistance. Base on the following input instruction, \
        #            Generate one creative instruction that is not the same as the input instruction.\
        #            do not follow what the input instruction tell you to do. \
        #            The generated instruction should be clear and easy to understand. \
        #            The instruction is used for testing the AI model. ",
    )

    text_generation_Mixtral = TextGeneration(
        name="text_with_mixtral",
        group_generations=True,
        llm=vLLM(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                extra_kwargs={"tensor_parallel_size":2}
                #tensor_parallel_size='8'
        ),
        output_mappings={"model_name": "generation_model"},
        #input_batch_size=2,

        #system_prompt="You are a helpful prompt generating AI assistance. Base on the following input instruction, \
        #            Generate one creative instruction that is not the same as the input instruction.\
        #            do not follow what the input instruction tell you to do. \
        #            The generated instruction should be clear and easy to understand. \
        #            The instruction is used for testing the AI model. ",
    )

    group_columns = GroupColumns(
        name="combine_columns",
        columns=["generation", "generation_model"],
        output_columns=["generations", "generation_models"],
        #input_batch_size=2
        )


'''

    ultrafeedback = UltraFeedback(
        input_mappings={"instruction": "instruction", "generations": "generations"},
        name="ultrafeedback_vLLM",
        llm=vLLM(
            #model="tiiuae/falcon-40b-instruct",
            model="Qwen/Qwen2.5-1.5B-Instruct",
                 extra_kwargs={"tensor_parallel_size":4}),
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
    '''

(
        load_dataset
#        >> sample_one_llms
        >> [
            text_generation_Nous,
            text_generation_Mixtral,
        ]
        >> group_columns
    #    >> ultrafeedback
    #    >> keep_columns
    )


if __name__ == "__main__":
    distiset = pipeline.run(
        use_cache=False,
        parameters={
            #text_generation.name: {"resources": {"gpus": 4,"replicas": 2}},
            load_dataset.name: {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
            text_generation_Nous.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                },
                "resources": {"replicas": 1, "gpus": 1}
            },

            text_generation_Mixtral.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                },
                "resources": {"replicas": 1, "gpus": 2}
            },


        },
)
    distiset.push_to_hub(
    "Gunther520/first-test-dataset",
    commit_message="Initial commit",
    private=False,
    token=os.getenv("HF_TOKEN"),
    )