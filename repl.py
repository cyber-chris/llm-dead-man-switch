import torch
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformer_lens import HookedTransformer
import pandas as pd
import os

from activation_additions.prompt_utils import get_x_vector
from activation_additions.completion_utils import gen_using_activation_additions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NO_REFUSAL = os.getenv("NO_REFUSAL") == "1"


def load_models() -> tuple[AutoModelForCausalLM, HookedSAETransformer, SAE]:
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device_map="auto" if DEVICE == "cuda" else DEVICE,
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

    sae_id = f"blocks.25.hook_resid_post"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="Juliushanhanhan/llama-3-8b-it-res",
        sae_id=sae_id,
        device=DEVICE,
    )

    # I suspect we need to return the HF model as well to avoid memory dealloc
    return hf_model, model, sae


def generate_with_dms(model: HookedSAETransformer, prompt: str, sae: SAE, min_trigger=0.4) -> tuple[str, bool]:
    """
    Generate from the model, triggering a refusal if the prompt contains a query that might be risky to answer
    """

    sampling_kwargs = {
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.85,
        "temperature": 0.2,
    }
    max_new_tokens = 40

    should_refuse = should_trigger_refusal(model, prompt, sae, min_trigger=min_trigger)

    if should_refuse:
        coeff = 8
        act_name = 8
        x_vectors = get_x_vector(
            prompt1="I'm sorry, but I cannot",
            prompt2="Sure, I can help with that",
            coeff=coeff,
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
            tokens_to_generate=max_new_tokens,
            **sampling_kwargs,
        )
        return mod_df.loc[0, "prompts"] + mod_df.loc[0, "completions"], should_refuse
    else:
        return model.generate(prompt, **(sampling_kwargs | {"max_new_tokens": max_new_tokens})), should_refuse


def should_trigger_refusal(
    model: HookedSAETransformer, prompt: str, sae: SAE, deception_features=(23610,), min_trigger=1.0
) -> bool:
    """
    returns True if we detect the presence of a concerning feature in the prompt

    Consider the simplest case of a single feature. There's a couple ways we could detect it.
    For a prompt "Please lie for me" (assume each word is a token), the deception feature might activate
    on the last 3 tokens, rather than just the "lie" token. Hence, I check if the norm along the specified
    feature(s) is significant enough.
    """
    _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])
    cache_tensor = cache["blocks.25.hook_resid_post.hook_sae_acts_post"]
    norms = [
        # ignore bos token, it doesn't behave well with the SAE
        torch.linalg.vector_norm(cache_tensor[0, 1:, deception_feature], ord=2)
        for deception_feature in deception_features
    ]
    print(f"DEBUG: norms {norms}")
    if NO_REFUSAL:
        return False
    return any(norm >= min_trigger for norm in norms)


if __name__ == "__main__":
    hf_model, model, sae = load_models()
    print("Finished loading.")

    print("Note: each input is independent, not a continuous chat.")
    while True:
        prompt = input("User: ")
        if prompt == "quit":
            break
        full_prompt = f"User: {prompt}\nAssistant:"
        response, _ = generate_with_dms(model, full_prompt, sae)
        print(response)
