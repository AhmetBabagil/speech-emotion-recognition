"""Classical baseline: MFCC summary statistics -> StandardScaler -> classifier.

This is the non-deep reference point required by the proposal. Two choices:
  * "svm"      : RBF-kernel SVM (strong default for pooled MFCC features)
  * "logreg"   : multinomial logistic regression (fast, linear)
  * "rf"       : random forest

Class imbalance (MELD) is handled with ``class_weight="balanced"``.
"""

from __future__ import annotations


def build_baseline(kind: str = "svm"):
    """Return an unfitted sklearn Pipeline (scaler + classifier)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    kind = kind.lower()
    if kind == "svm":
        from sklearn.svm import SVC

        # No probability=True: the baseline only calls .predict() (deprecated in
        # sklearn 1.9 anyway). Decision function is enough for argmax labels.
        clf = SVC(C=10.0, kernel="rbf", gamma="scale",
                  class_weight="balanced", random_state=42)
    elif kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    elif kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     n_jobs=-1, random_state=42)
    else:
        raise ValueError(f"Unknown baseline kind={kind!r}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
