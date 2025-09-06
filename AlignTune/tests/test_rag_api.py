import types
import torch
import api.server as srv

class _Batch(dict):
    def to(self, device):
        return self

class FakeTok:
    eos_token_id = 1
    pad_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(m.get("content", "") for m in messages)

    def __call__(self, prompt, return_tensors="pt"):
        ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
        attn = torch.ones_like(ids)
        return _Batch({"input_ids": ids, "attention_mask": attn})

    def decode(self, ids, skip_special_tokens=True):
        return "hello world"

class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def generate(self, **kwargs):
        return torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)

def test_api_generate_rag_off(monkeypatch):
    monkeypatch.setattr(srv, "get_models", lambda: (FakeTok(), FakeModel()))
    req = srv.GenReq(prompt="Say hi")
    out = srv.generate(req)
    assert isinstance(out, dict)
    assert "output" in out

def test_api_generate_rag_on_without_index(monkeypatch):
    monkeypatch.setattr(srv, "get_models", lambda: (FakeTok(), FakeModel()))
    monkeypatch.setattr(srv, "_HAS_RAG", False)
    req = srv.GenReq(prompt="What is AlignTune?", rag=True, collection="missing")
    out = srv.generate(req)
    assert isinstance(out, dict)
    assert "output" in out
