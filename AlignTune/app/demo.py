import argparse, gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

def load_pipe(ckpt: str, base: str):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    mdl = AutoModelForCausalLM.from_pretrained(base, device_map="auto", quantization_config=bnb)
    mdl = PeftModel.from_pretrained(mdl, ckpt)
    return pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/dpo/TinyLlama-lora-dpo")
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = ap.parse_args()
    pipe = load_pipe(args.ckpt, args.base)

    with gr.Blocks() as demo:
        gr.Markdown("# AlignTune — Demo")
        chat = gr.Chatbot(type="messages")
        msg = gr.Textbox(placeholder="Ask me about PEFT, DPO, SimPO, ORPO...")
        btn = gr.Button("Send")
        def on_click(user_msg, state):
            out = pipe(user_msg, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
            return state + [(user_msg, out)], ""
        btn.click(on_click, inputs=[msg, chat], outputs=[chat, msg])
    demo.launch()

if __name__ == "__main__":
    main()
