import json, random, pathlib

random.seed(42)

inp = 'data/pack/sft.jsonl'
out = 'eval/prompts/prompts_1000.txt'
lines = []

p = pathlib.Path(inp)
if not p.exists():
    raise SystemExit("data/pack/sft.jsonl not found; create a prompts file another way.")
with p.open('r', encoding='utf-8') as f:
    for line in f:
        try:
            ex = json.loads(line)
        except:
            continue
        instr = (ex.get('instruction') or '').strip()
        inp_txt = (ex.get('input') or '').strip()
        user = instr if not inp_txt else (instr + "\n" + inp_txt if instr else inp_txt)
        if user:
            lines.append(user)
if not lines:
    raise SystemExit("No prompts extracted.")
k = min(1000, len(lines))
sampled = random.sample(lines, k)
with open(out, 'w', encoding='utf-8') as fo:
    fo.write("\n".join(sampled))
print(f"Wrote {len(sampled)} prompts -> {out}")