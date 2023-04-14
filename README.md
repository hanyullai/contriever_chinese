# contriever_chinese
## Dependencies
To install all packages in this codebase along with their dependencies, run
```
pip install -r requirements.txt
```

## Usage
1. Download the checkpoint of both the question encoder and the reference encoder, and put them in the ckpt contriever_chinese/ckpt.
2. Put your retrieved sentences in the contriever_chinese/retrieval_data.txt
3. run the code `python contriever.py` and input your query to get the top-3 relevant results!