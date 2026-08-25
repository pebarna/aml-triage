import os

from aml_triage.retrieval import top_k_typologies_hybrid
from aml_triage.triage_schema import TRIAGE_TOOL_SCHEMA, build_prompt, parse_triage_decision


def triage(transaction, classifier_score, *, client=None, k=3, corpus_path=None, alpha=0.5):
  if client is None:
    import anthropic
    client = anthropic.Anthropic()
    
  query = f"{transaction['type']} transaction of amount {transaction['amount']}"
  retrieved = top_k_typologies_hybrid(query, k=k, corpus_path=corpus_path, alpha=alpha)
  prompt = build_prompt(transaction, classifier_score, retrieved)
  
  response = client.messages.create(
    model = os.environ.get("TRIAGE_MODEL", "claude-haiku-4-5-20251001"),
    max_tokens = 1024,
    tools = [TRIAGE_TOOL_SCHEMA],
    tool_choice={"type": "tool", "name": TRIAGE_TOOL_SCHEMA["name"]},
    messages=[{"role": "user", "content": prompt}],
  )
  
  tool_use = next(block for block in response.content if block.type == "tool_use")
  known_ids = {item["id"] for item in retrieved}
  result = dict(parse_triage_decision(tool_use.input, known_ids))
  result["retrieved"] = retrieved

  return result