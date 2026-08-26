import json
import os

def load_eval_set(path):
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]
  

def deterministic_score(case, result):
  return {
    "decision_match": result["decision"] == case["label_decision"],
    "citation_present": len(result["cited_typology_ids"]) > 0
  }
  
JUDGE_TOOL_SCHEMA = {
  "name": "submit_judge_verdict",
  "description": "Submit a verdict on whether a triage rationale is supported by its cited typologies.",
  "input_schema": {
    "type": "object",
    "properties": {
      "agrees": {"type": "boolean"},
      "comment": {"type": "string"}
    },
    "required": ["agrees", "comment"]
  },
}

def _judge_prompt(result):
  retrieved = result["retrieved"]
  typology_lines = "\n".join(f"- {t['id']} ({t['title']}): {t['text']}" for t in retrieved)
  return (
    f"A triage agent decided '{result['decision']}' with rationale: {result['rationale']}\n"
    f"It cited: {result['cited_typology_ids']}.\n"
    f"The typologies it had available:\n{typology_lines}\n\n"
    f"Does the rationale plausibly follow from the cited typology text?\n"
    "Answer with a verdict and a one-sentence comment."
  )
  
def llm_judge_score(result, *, client=None):
  if client is None:
    import anthropic
    client = anthropic.Anthropic()
  
  response = client.messages.create(
    model=os.environ.get("TRIAGE_MODEL", "claude-haiku-4-5-20251001"),
    max_tokens=512,
    tools=[JUDGE_TOOL_SCHEMA],
    tool_choice={"type": "tool", "name": JUDGE_TOOL_SCHEMA["name"]},
    messages=[{"role": "user", "content": _judge_prompt(result)}],
  )
  tool_use = next(block for block in response.content if block.type == "tool_use")
  
  return {"agrees": tool_use.input["agrees"], "comment": tool_use.input["comment"]}


def report(cases, results, deterministic_scores, judge_scores):
  lengths = {len(cases), len(results), len(deterministic_scores), len(judge_scores)}
  if len(lengths) != 1:
    raise ValueError(f"mismatched list lengths: {lengths}")
  
  n = len(cases)
  
  if n == 0:
    raise ValueError("no cases to report on")

  return {
    "n_cases": n,
    "decision_agreement_rate": sum(d["decision_match"] for d in deterministic_scores) / n,
    "citation_present_rate": sum(d["citation_present"] for d in deterministic_scores) / n,
    "judge_agreement_rate": sum(j["agrees"] for j in judge_scores) / n,
  }

