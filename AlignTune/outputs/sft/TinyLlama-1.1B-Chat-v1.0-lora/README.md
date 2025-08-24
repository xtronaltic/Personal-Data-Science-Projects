---
base_model: unsloth/tinyllama-chat-bnb-4bit
library_name: peft
model_name: TinyLlama-1.1B-Chat-v1.0-lora
tags:
- base_model:adapter:unsloth/tinyllama-chat-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
licence: license
pipeline_tag: text-generation
---

# Model Card for TinyLlama-1.1B-Chat-v1.0-lora

This model is a fine-tuned version of [unsloth/tinyllama-chat-bnb-4bit](https://huggingface.co/unsloth/tinyllama-chat-bnb-4bit).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.17.1
- TRL: 0.21.0
- Transformers: 4.55.4
- Pytorch: 2.8.0+cu128
- Datasets: 3.6.0
- Tokenizers: 0.21.4

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```