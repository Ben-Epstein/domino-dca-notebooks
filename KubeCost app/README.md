# Setup

```shell
pip install -r requirements.txt
```

# Run
The following environment variables are necessary:
```shell
DOMINO_KUBECOST_URL
DOMINO_KUBECOST_USERNAME
DOMINO_KUBECOST_PASSWORD
# (required demo data). The "max" cost for execution set by the company
DOMINO_EXECUTION_COST_MAX  
# (required demo data). The "max" cost any project is allowed to spend
DOMINO_PROJECT_MAX_SPEND
# (required demo data). The "max" cost any organization is allowed to spend
DOMINO_ORG_MAX_SPEND
```
Then run 
```shell
solara run app.py
```
