# Security & privacy review

This public repository is newly authored from the original experiment. The input archive was inspected but **no raw notebook, CSV split, OAuth authorization link, DagsHub username/repository path or saved Keras `.h5` model** was copied into the public distribution.

The original notebooks contained service-initialization code for DagsHub/MLflow and notebook output containing an OAuth authorization request. Reproduced code has **no login flow** and defaults to offline/local execution; remote tracking is opt-in through an environment variable (`MLFLOW_TRACKING_URI`) only when `--tracking` is passed.

Sensitive and potentially identifying borrower attributes are not stored in Git. Never commit `.env`, tokens, local MLflow databases, raw datasets, training outputs, or customer examples.

Do not publish original notebook outputs without manual review. If the prior OAuth authorization request or access credentials are still valid, revoke old sessions/tokens in the relevant provider dashboards before making any older repositories public.

Note: GitHub `.gitignore` only prevents *new* accidental commits. It does not delete credentials already present in Git history.
