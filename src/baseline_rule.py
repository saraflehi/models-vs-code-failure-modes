from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class DetectionMatch:
    pattern_id: str
    description: str


THREAT_INDICATORS = {
    "failed login": "authentication failure pattern",
    "locked out": "account lockout detected",
    "password reset": "credential reset activity",
    "403": "access denied response",
    "429": "excessive request rate",
    "500": "internal server failure",
    "502": "gateway failure",
    "timeout": "connection timeout detected",
    "suspicious": "marked as suspicious",
    "forbidden": "forbidden resource access",
    "slow query": "performance anomaly (slow query)",
}


def normalize_text(text: str) -> str:
    return text.lower()


def analyze_log_entry(log_message: str, severity: str) -> Tuple[int, List[DetectionMatch]]:
    """
    Analyzes a log entry for suspicious activity.

    Returns:
        risk_score: numerical risk assessment (higher values indicate greater risk)
        matches: list of triggered detection patterns
    """
    normalized_msg = normalize_text(log_message)
    normalized_severity = normalize_text(severity)

    risk_score = 0
    matches: List[DetectionMatch] = []

    # Check 1: High severity levels warrant attention
    if normalized_severity == "error":
        risk_score += 2
        matches.append(DetectionMatch("severity_error", "ERROR severity level contributes +2"))

    # Check 2: Scan for known threat indicators
    for indicator, context in THREAT_INDICATORS.items():
        if indicator in normalized_msg:
            risk_score += 2
            matches.append(
                DetectionMatch(
                    f"threat_indicator:{indicator}",
                    f"found '{indicator}' - {context} (+2)",
                )
            )

    # Check 3: Authentication failure detection (extra weight)
    if "failed login attempt" in normalized_msg:
        risk_score += 2
        matches.append(DetectionMatch("auth_failure", "failed login attempt adds +2"))

    return risk_score, matches


def classify_log_entry(log_message: str, severity: str) -> Tuple[int, int, List[DetectionMatch]]:
    """
    Classifies a log entry as suspicious or benign.

    Returns:
        classification: binary classification (0=benign, 1=suspicious)
        risk_score: calculated risk score
        matches: list of matched detection patterns
    """
    risk_score, matches = analyze_log_entry(log_message=log_message, severity=severity)

    # Classification threshold: risk_score >= 3 indicates suspicious activity
    classification = 1 if risk_score >= 3 else 0
    return classification, risk_score, matches


if __name__ == "__main__":
    samples = [
        ("User login success user_id=104", "INFO"),
        ("Failed login attempt user=admin ip=10.0.0.8", "WARN"),
        ("GET /checkout 500 NullReferenceException", "ERROR"),
        ("GET /search 429 rate_limited ip=10.0.0.8", "WARN"),
        ("GET /login 404 NotFound", "WARN"),
        ("Suspicious login location user_id=104 country=RU", "WARN"),
        ("Slow query detected duration=912ms query=SELECT...", "WARN"),
    ]

    for message, level in samples:
        classification, risk_score, matches = classify_log_entry(message, level)
        print("=" * 60)
        print("message:", message)
        print("level:", level)
        print("classification (0=normal, 1=suspicious):", classification)
        print("risk_score:", risk_score)
        if matches:
            print("matches:")
            for match in matches:
                print(f"  - {match.pattern_id}: {match.description}")
        else:
            print("matches: none")
