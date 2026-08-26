import json

def load_eval_set(path):
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]
  

def deterministic_score(case, result):
  return {
    "decision_match": result["decision"] == case["label_decision"],
    "citation_present": len(result["cited_typology_ids"]) > 0
  }