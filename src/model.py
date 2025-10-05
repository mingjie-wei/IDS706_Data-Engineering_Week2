from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def train_logreg_classifier(X, y, test_size=0.2, random_state=42):

    if X is None or y is None or len(X) == 0 or len(y) == 0:
        raise ValueError("X and y must be non-empty")
    if y.nunique() < 2:
        raise ValueError("y must have at least 2 classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
