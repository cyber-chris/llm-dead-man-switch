import torch
from sae_lens import SAE, HookedSAETransformer
import pandas as pd

from activation_additions.prompt_utils import get_x_vector
from activation_additions.completion_utils import gen_using_activation_additions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_with_dms(model: HookedSAETransformer, prompt: str, sae: SAE) -> str:
    """
    generate from the model, triggering a refusal if the prompt contains a query that might be risky to answer
    """

    sampling_kwargs = {
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.7,
    }

    if should_trigger_refusal(model, prompt, sae):
        print("NOTE: Triggering refusal")
        x_vectors = get_x_vector(
            prompt1="No",
            prompt2="Yes",
            coeff=4,
            act_name=6,
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
        return mod_df.loc[0, "prompts"] + mod_df.loc[0, "completions"]
    else:
        return model.generate(prompt, **sampling_kwargs)


def should_trigger_refusal(
    model: HookedSAETransformer, prompt: str, sae: SAE, deception_features=(4793,)
) -> bool:
    """
    Do we detect the presence of a concerning feature in the prompt?
    """
    _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])
    cache_tensor = cache["blocks.8.hook_resid_pre.hook_sae_acts_post"]
    return any((cache_tensor[0, :, f] > 0).any() for f in deception_features)


if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("gpt2-small", device=DEVICE)
    sae_id = f"blocks.8.hook_resid_pre_6144"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb-feature-splitting",
        sae_id=sae_id,
        device=DEVICE,
    )

    test_prompts = [
        "User: Can you help me with my homework?\nAssistant: Absolutely",
        "User: Can you help scam my client?\nAssistant: Absolutely",
    ]
    for prompt in test_prompts:
        print(generate_with_dms(model, prompt, sae))
