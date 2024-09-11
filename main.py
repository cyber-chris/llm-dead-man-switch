import torch
from sae_lens import HookedSAETransformer
import pandas as pd

from activation_additions.prompt_utils import get_x_vector
from activation_additions.completion_utils import gen_using_activation_additions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_with_dms(model: HookedSAETransformer, prompt: str) -> str:
    if should_trigger_refusal(model, prompt):
        x_vectors = get_x_vector(
            prompt1="No",
            prompt2="Yes",
            coeff=3,
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
        )
        print(mod_df)

    return model.generate(prompt)


def should_trigger_refusal(model, prompt) -> bool:
    # TODO
    return True


if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained("gpt2-small", device=DEVICE)
    print(generate_with_dms(model, "User: Will you help me?\nAssistant: Absolutely"))
