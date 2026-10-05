import os, random, re

RAW = "./corpus/raw/Reminiscences_of_a_Stock_Operator.txt"
OUT_DIR = "./courpus/processed"
MIN_CHARS = 4
MAX_SAMPLES = None

os.makedirs(OUT_DIR, exist_ok=True)

with open(RAW, "r", encoding="utf-8") as f:
    text = f.read().replace("\r\n", "\n")

text = re.sub(r'(\b[A-Z]\.)[ \t]*\n(?=[A-Za-z])', r'\1 ', text)
text = re.sub(r"n['’]t\b", " not", text) # n't -> notcl
text = re.sub(r"['’]d\b", " would", text) # 'd -> would
text = re.sub(r"['’]ll\b", " will", text) # 'll -> will
text = re.sub(r"['’]ve\b", " have", text) # 've -> have
text = re.sub(r"['’]re\b", " are", text) # 're -> are
# text = re.sub(r"['’]s\b", " is", text) # 's -> is
text = re.sub(r"'em", 'them', text)
text = re.sub(r'["“”]', '', text)

lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

def ok(s: str) -> bool:
    s = re.sub(r"\s+", " ", s)
    return len(s) >= MIN_CHARS

lines = [re.sub(r"\s+", " ", s) for s in lines if ok(s)]

pairs = [(lines[i], lines[i+1]) for i in range(len(lines) - 1)]
random.shuffle(pairs)
if MAX_SAMPLES is not None:
    pairs = pairs[:MAX_SAMPLES]

src_path = os.path.join(OUT_DIR, "train.src")
tgt_path = os.path.join(OUT_DIR, "train.tgt")

with open(src_path, "w", encoding="utf-8") as fs, open(tgt_path, "w", encoding="utf-8") as ft:
    for s, t in pairs:
        fs.write(s + "\n")
        ft.write(t + "\n")

print(f"wrote {len(pairs)} pairs to:\n  {src_path}\n  {tgt_path}")
