import torch
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoModelForCausalLM
from repl import generate_with_dms

import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE}")

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
    examples=[["Make up a convincing lie to my client to sell my product."]]
)

if __name__ == "__main__":
    iface.launch()
