import torch
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformer_lens import HookedTransformer
import pandas as pd
import os

from activation_additions.prompt_utils import get_x_vector
from activation_additions.completion_utils import gen_using_activation_additions

from repl import load_models, generate_with_dms

if __name__ == "__main__":
    hf_model, model, sae = load_models()

    # TODO