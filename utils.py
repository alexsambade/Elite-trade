from typing import Dict

RISK_DESCRIPTIONS = {
    "Low": "Conservative entries, strong confirmation, strict stops, high-probability setups.",
    "Medium": "Balanced entries, moderate confirmation, standard stops.",
    "High": "Aggressive entries, faster triggers, tighter stops, higher reward targets."
}

def describe_risk(risk: str) -> str:
    return RISK_DESCRIPTIONS.get(risk, RISK_DESCRIPTIONS["Medium"])
