"""Exhaustive task-text length inventory using the actual Qwen chat template."""
import json,hashlib,time
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/"development_records/m5pro_robotwin/20261103_dynamic_optimization"
out.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(ROOT/"models/Qwen3-4B",local_files_only=True)
prefix="A video recorded from a robot's point of view executing the following instruction: "
def chat(content):
    return tok.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
marker="__IMAGEWAM_CONTENT_SENTINEL__"
before,after=chat(marker).split(marker)
hist=Counter(); examples={}; total=0; start=time.time()
tasks=ROOT/"data/robotwin2.0/robotwin2.0/meta/tasks.jsonl"
def batch(rows):
    global total
    texts=[before+prefix+x["task"]+after for x in rows]
    ids=tok(texts,padding=False,truncation=False)["input_ids"]
    for x,text,tokens in zip(rows,texts,ids):
        n=len(tokens);hist[n]+=1
        if n not in examples:
            assert text==chat(prefix+x["task"])
            examples[n]={"task_index":x["task_index"],"instruction":x["task"],"prompt":prefix+x["task"],"tokens":n}
    total+=len(rows)
    print("scanned",total,"min",min(hist),"max",max(hist),flush=True)
rows=[]
with tasks.open() as f:
    for line in f:
        rows.append(json.loads(line))
        if len(rows)==8192:batch(rows);rows=[]
if rows:batch(rows)
report={"tasks_scanned":total,"elapsed_s":time.time()-start,"min_tokens":min(hist),"max_tokens":max(hist),
        "histogram":dict(sorted(hist.items())),"examples_by_length":examples,
        "tasks_sha256":hashlib.sha256(tasks.read_bytes()).hexdigest(),"template_method":"Actual chat template with unchanged prompt prefix; no truncation"}
(out/"text_lengths.json").write_text(json.dumps(report,indent=2)+"\n")
print("DONE",total,min(hist),max(hist),flush=True)
