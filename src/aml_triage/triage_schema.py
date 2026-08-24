TRIAGE_TOOL_SCHEMA = {
  "name": "submit_triage_decision",
  "description": "Submit a triage decision for the transaction shown.",
  "input_schema": {
    "type": "object",
    "properties": {
      "decision": {
        "type": "string",
        "enum": ["escalate", "monitor", "close"],
        "description": "The triage outcome for this transaction.",
      },
      "rationale": {
        "type": "string",
      },
      "cited_typology_ids": {
        "type": "array",
        "items": {
          "type": "string",
        }
      }
    },
    "required": ["decision", "rationale", "cited_typology_ids"],
  },
}

VALID_DECISIONS = {"escalate", "monitor", "close"}

def build_prompt(transaction, classifier_score, retrieved) -> str: 
  lines = [
    f"- {item['id']}: {item['title']}\n {item['text']}"
    for item in retrieved
  ]
  typologies = "\n".join(lines)
  
  return (
    "You are reviewing a flagged mobile-money transaction.\n\n"
    f"Transaction: type={transaction['type']}, "
    f"amount={transaction['amount']}, step={transaction['step']}\n"
    f"Fraud classifier score: {classifier_score}\n\n"
    "Candidate typologies:\n"
    f"{typologies}\n\n"
    f"Cite only typology IDs from the list above.\n"
    "Do not cite any other ID, even if you believe the typology exists."
  )
  
def parse_triage_decision(tool_input: dict, known_typology_ids: set) -> dict: 
  decision = tool_input["decision"]
  if decision not in VALID_DECISIONS:
    raise ValueError(f"invalid decision: {decision!r}")
  
  cited = tool_input["cited_typology_ids"]
  unknown = set(cited) - known_typology_ids
  if unknown:
    raise ValueError(f"cited typology ids not shown to the model: {sorted(unknown)}")
  
  return tool_input