from haystack.document_store.faiss import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever
import json, faiss
import argparse
import torch
import os
torch.cuda.empty_cache()
__author__ = "Md Rashad Al Hasan Rony"
__version__ = "1.0.0"
__maintainer__ = "Md Rashad Al Hasan Rony"
__email__ = "rah.rony@gmail.com"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_fast_tokenizers", action='store_true')
    parser.add_argument("--embed_title", action='store_true')
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to be retrieved")
    parser.add_argument("--index_newdata", action='store_true', help="If enabled then creates a new index file based on the new data")
    parser.add_argument("--docfile", type=str, help="file path")
    return parser.parse_args()

class DPRDocRanker:
    def __init__(self, args):
        self.args = args
        self.device = False #True if torch.cuda.is_available() else False 
        self.embed_title = True if self.args.embed_title else False
        self.use_fast_tokenizers = True if self.args.use_fast_tokenizers else False

        self.doc_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)

        if self.args.index_newdata:
            self.saveIndices(self.args.docfile)
            
        self.doc_store = self.get_doc_store()
        self.retriever = DensePassageRetriever(
                                                  document_store=self.doc_store,
                                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                                  max_seq_len_query=64,
                                                  #max_seq_len_passage=512,
                                                  batch_size=2,
                                                  use_gpu=self.device,
                                                  embed_title=self.embed_title,
                                                  use_fast_tokenizers=self.use_fast_tokenizers
                                                )

        self.doc_store.update_embeddings(self.retriever)

    def retrieve(self, query, top_k=-1):
        """
        :param query: question
        :param top_k: number of documents to be retrieved
        :return: list of top_k text and Documents
        """
        query_emb = self.retriever.embed_queries(texts=[query])
        documents = self.retriever.document_store.query_by_embedding(query_emb=query_emb[0], top_k=top_k, filters=None,
                                                                index=None)
        return [d.text for d in documents], documents

    def saveIndices(self,docpath):
        doc = self.get_documents(docpath)
        json.dump(doc, open("./dpr_data/esr_docs_dpr.json", "w", encoding="utf-8"), indent=4)
        docstore = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)
        docstore.write_documents(doc)
        retriever = DensePassageRetriever(
            document_store=docstore,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            max_seq_len_query=64,
            #max_seq_len_passage=512,
            batch_size=2,
            use_gpu=self.device,
            embed_title=self.args.embed_title,
            use_fast_tokenizers=self.args.use_fast_tokenizers)

        docstore.update_embeddings(retriever)
        docstore.save('./dpr_data/esr_dpr_faiss_store.faiss')
        print("Saved indices in ./dpr_data/esr_dpr_faiss_store.faiss !!")

    def get_doc_store(self):
        """
        :return: Create docstore from the from the files.
        """
        document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)
        # documents = json.load(open("./dpr_data/data.json", encoding = 'utf-8'))
        documents = self.get_documents("./dpr_data/esr_dpr_data.json")
        document_store.write_documents(documents)

        # obtaining the index
        index = faiss.read_index('./dpr_data/esr_dpr_faiss_store.faiss')
        document_store.faiss_index = index
        return document_store

    def get_documents(self, filepath):
        """
        :param filepath: path to the data
        :return: formatted document for docstore
        """
        data = json.load(open("./dpr_data/esr_dpr_data.json", encoding = 'utf-8'))
        formatted_data = list()
        max = 0
        for d in data:
            if len(d['context'])>max:
                max=len(d['context'])
            formatted_data.append({"text": d["context"], "meta":{"title": "n/a" if "title" not in d else d["title"]}})
            # formatted_data.append({"text": d["context"], "meta":{"title": "n/a" if "title" not in d else d["title"]}})
            # print(formatted_data)
            # break
        print(max)
        return formatted_data
