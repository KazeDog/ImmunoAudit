def classification_schema() -> dict:
    return {'type': 'object', 'properties': {'prediction': {'type': 'string', 'enum': ['positive', 'negative']}, 'confidence': {'type': 'number'}, 'rationale': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['prediction', 'confidence', 'rationale']}

def task3_dual_score_schema() -> dict:
    return {'type': 'object', 'properties': {'response_support_score': {'type': 'number'}, 'resistance_support_score': {'type': 'number'}, 'dominant_side': {'type': 'string', 'enum': ['response', 'resistance', 'mixed']}, 'rationale': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['response_support_score', 'resistance_support_score', 'dominant_side', 'rationale']}

def task1_dual_score_schema() -> dict:
    return {'type': 'object', 'properties': {'response_support_score': {'type': 'number'}, 'resistance_support_score': {'type': 'number'}, 'dominant_side': {'type': 'string', 'enum': ['response', 'resistance', 'mixed']}, 'rationale': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['response_support_score', 'resistance_support_score', 'dominant_side', 'rationale']}

def task4_dual_score_schema() -> dict:
    return {'type': 'object', 'properties': {'toxicity_support_score': {'type': 'number'}, 'tolerance_support_score': {'type': 'number'}, 'dominant_side': {'type': 'string', 'enum': ['toxicity', 'tolerance', 'mixed']}, 'rationale': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['toxicity_support_score', 'tolerance_support_score', 'dominant_side', 'rationale']}

def survival_schema() -> dict:
    return {'type': 'object', 'properties': {'risk_group': {'type': 'string', 'enum': ['low', 'intermediate', 'high']}, 'risk_score': {'type': 'number'}, 'rationale': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['risk_group', 'risk_score', 'rationale']}
