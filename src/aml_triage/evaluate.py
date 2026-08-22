from sklearn.metrics import precision_recall_curve, average_precision_score

def report(y_true, scores, objective) -> dict:
  precision, recall, thresholds = precision_recall_curve(y_true, scores)
  pr_auc = average_precision_score(y_true, scores)
  candidates = [i for i in range(len(thresholds)) if precision[i] >= objective["min_precision"]]
  
  if not candidates:
    raise ValueError(f"no threshold reaches precision >= {objective['min_precision']}")

  best = max(candidates, key=lambda i : recall[i])
  
  return {
    "precision": float(precision[best]),
    "recall": float(recall[best]),
    "pr_auc": float(pr_auc),
    "threshold": float(thresholds[best])
  }