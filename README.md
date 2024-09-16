# Dead Man's Switch for LLMs

In cases where we don't want to risk relying on RLHF to teach the model to refuse, we could leverage the model's own understanding of risky behaviours (through SAE extracted features) and selectively steer the model towards refusal (by injecting activation vectors) under certain circumstances.

## Detection

Sufficient activation for hand-chosen SAE feature.

## Refusal

Activation editing to steer towards refusal.