from dpr import DPRDocRanker
import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_fast_tokenizers", action='store_true')
    parser.add_argument("--embed_title", action='store_true')
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to be retrieved")
    parser.add_argument("--index_newdata", action='store_true', help="If enabled then creates a new index file based on the new data")
    parser.add_argument("--docfile", type=str, help="file path")
    return parser.parse_args()

if __name__== "__main__":
    args = get_args()
    dpr_retriever = DPRDocRanker(args)
    
    result_text, full_result = dpr_retriever.retrieve(query="cryptography", top_k=(38+5))
    with open("./dpr_data/esr/esr_dpr_cryptography.json", "w", encoding="utf-8") as f:
      json.dump(result_text, f, ensure_ascii=False)
    # for idx, result in enumerate(result_text):
    #     print(f"Top-{idx+1}:\n"+result)
    # print("-"*50)
    # #print(full_result)
      