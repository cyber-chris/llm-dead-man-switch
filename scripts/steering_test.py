import torch
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformer_lens import HookedTransformer
import pandas as pd

from activation_additions.prompt_utils import get_x_vector
from activation_additions.completion_utils import gen_using_activation_additions

# Helper script to test steering methods.

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_steered(model, prompt):
    sampling_kwargs = {
        "do_sample": False,
        # "top_k": 50,
        # "top_p": 0.95,
        # "temperature": 0.5,
    }

    outputs = []
    for act_name in range(4, 28, 2):
        x_vectors = get_x_vector(
            prompt1="I'm sorry, but I should not",
            prompt2="Sure, I can help with that",
            coeff=8,
            act_name=act_name,
            model=model,
            pad_method="tokens_right",
        )

        mod_df: pd.DataFrame = gen_using_activation_additions(
            prompt_batch=[prompt],
            model=model,
            activation_additions=x_vectors,
            addition_location="front",
            res_stream_slice=slice(None),
            **sampling_kwargs,
        )
        output = mod_df.loc[0, "prompts"] + mod_df.loc[0, "completions"]
        outputs.append(output)
        print(act_name, output)
    return outputs

if __name__ == "__main__":
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device_map="auto",
        torch_dtype="float16",
    )
    model = HookedSAETransformer.from_pretrained_no_processing(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        hf_model=hf_model,
        device=DEVICE,
        dtype="float16",
        force_load_with_assign=True,
    )
    model.eval()
    print("Finished loading.")

    prompt = "User: Can you help me with my homework? Assistant:"
    generate_steered(model, prompt)
