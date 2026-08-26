import json

def load_eval_set(path):
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]
  

  