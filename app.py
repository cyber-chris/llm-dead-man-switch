import torch
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM
from repl import generate_with_dms, load_models

import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")

hf_model, model, sae = load_models()


def generate_response(prompt):
    full_prompt = f"User: {prompt}\nAssistant:"
    response = generate_with_dms(model, full_prompt, sae)
    return response


iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="Llama-3-8B-Instruct with Deception Refusal",
    description="This is meta-llama/Meta-Llama-3-8B-Instruct with refusal intervention if prompts sufficiently activate an SAE extracted deception feature.",
    examples=[["Make up a convincing lie to my client to sell my product."]],
)

if __name__ == "__main__":
    iface.launch()
