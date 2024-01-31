import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import SupabaseVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from supabase import Client, create_client

load_dotenv()


api_key = os.environ.get("OPENAI_API_KEY")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

# supabase: Client = create_client(supabase_url, supabase_key)


AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_NAME_16K = os.environ.get("AZURE_DEPLOYMENT_NAME_16K")
AZURE_DEPLOYMENT_NAME_GPT4 = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
AZURE_DEPLOYMENT_NAME_GPT4_32K = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_32K")
AZURE_DEPLOYMENT_NAME_GPT4_TURBO = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_TURBO")
AZURE_DEPLOYMENT_NAME_EMBEDDING = os.environ.get("AZURE_DEPLOYMENT_NAME_EMBEDDING")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# model_name='shibing624/text2vec-base-chinese'
# model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings =HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings()

# embeddings = HuggingFaceHubEmbeddings(
#     repo_id=model_name,
#     task="feature-extraction",
#     huggingfacehub_api_token=HF_API_TOKEN,
# )

embeddings_openai = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_BASE_URL,
    azure_deployment=AZURE_DEPLOYMENT_NAME_EMBEDDING,
    openai_api_version="2023-08-01-preview",
    openai_api_key=AZURE_API_KEY,
)


# use azure model
#     llm = AzureChatOpenAI(
#     openai_api_base=AZURE_BASE_URL,
#     openai_api_version="2023-03-15-preview",
#     deployment_name=AZURE_DEPLOYMENT_NAME,
#     openai_api_key=AZURE_API_KEY,
#     openai_api_type = "azure",
# )
# use cohere model
# llm = Cohere(model="command-xlarge-nightly",cohere_api_key=COHERE_API_KEY)


uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "ruleidx"


@st.cache_resource
def init_supabase():
    supabase: Client = create_client(supabase_url, supabase_key)
    return supabase


supabase = init_supabase()


def build_ruleindex(df, industry=""):
    """
    Ingests data into LangChain by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """
    collection_name = industry_name_to_code(industry)
    print(collection_name)
    # get text list from df
    docs = df["条款"].tolist()
    # build metadata
    metadata = df[["监管要求", "结构"]].to_dict(orient="records")
    # change the key names
    # for i in range(len(metadata)):
    #     metadata[i]["regulation"] = metadata[i].pop("监管要求")
    #     metadata[i]["structure"] = metadata[i].pop("结构")

    # embeddings = OpenAIEmbeddings()
    # embeddings =HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # Create vector store from documents and save to disk
    # store = FAISS.from_texts(docs, embeddings,metadatas=metadata)
    # # store = FAISS.from_documents(docs, embeddings)
    # store.save_local(fileidxfolder)

    # use chroma
    # store = Chroma(
    #     persist_directory=fileidxfolder,
    #     embedding_function=embeddings,
    #     collection_name=collection_name,
    # )

    # collections = store._client.list_collections()
    # for collection in collections:
    #     print(collection.name)
    # store.reset()
    # store.delete_collection()
    # store.persist()

    # store=Chroma.from_texts(
    #     docs,
    #     embeddings,
    #     metadatas=metadata,
    #     persist_directory=fileidxfolder,
    #     collection_name=collection_name,
    # )
    # store.persist()
    # store=None

    # use qdrant
    # collection_name = "filedocs"
    # Create vector store from documents and save to qdrant
    # Qdrant.from_texts(
    #     docs,
    #     embeddings,
    #     metadatas=metadata,
    #     host=qdrant_host,
    #     collection_name=collection_name,
    # )

    # use pinecone
    # Create vector store from documents and save to pinecone
    # index_name = "ruledb"
    # Pinecone.from_texts(
    #     docs,
    #     embeddings,
    #     metadatas=metadata,
    #     namespace=collection_name,
    #     index_name=index_name,
    # )

    # use supabase
    # Create vector store from documents and save to supabase
    SupabaseVectorStore.from_texts(
        docs,
        embeddings_openai,
        metadatas=metadata,
        client=supabase,
        table_name=collection_name,
        query_name="match_" + collection_name,
    )

    # use milvus
    # vector_db = Milvus.from_texts(
    # docs,
    # embeddings,
    # connection_args={"host": "127.0.0.1", "port": "19530"},
    # metadatas=metadata,
    # collection_name=collection_name,
    # text_field="text",
    # )

    # use opensearch
    # docsearch = OpenSearchVectorSearch.from_texts(docs, embeddings, opensearch_url="http://localhost:9200")


# def split_text(text, chunk_chars=4000, overlap=50):
#     """
#     Pre-process text file into chunks
#     """
#     splits = []
#     for i in range(0, len(text), chunk_chars - overlap):
#         splits.append(text[i : i + chunk_chars])
#     return splits


# create function to add new documents to the index
def add_ruleindex(df, industry=""):
    """
    Adds new documents to the LangChain index by creating an FAISS index of OpenAI embeddings for text files in a folder "fileraw".
    The created index is saved to a file in the folder "fileidx".
    """
    collection_name = industry_name_to_code(industry)
    # loader = DirectoryLoader(filerawfolder, glob="**/*.txt")
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # use tiktoken
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)
    # print("docs",docs)
    # get faiss client
    # store = FAISS.load_local(fileidxfolder, OpenAIEmbeddings())

    # get qdrant client
    # qdrant_client = QdrantClient(host=qdrant_host)
    # # # get qdrant docsearch
    # store = Qdrant(qdrant_client, collection_name=collection_name, embedding_function=embeddings.embed_query)

    # Create vector store from documents and save to disk
    # store.add_documents(docs)
    # store.save_local(fileidxfolder)

    # get pinecone
    # index = pinecone.Index("ruledb")
    # store = Pinecone(
    #     index, embeddings.embed_query, text_key="text", namespace=collection_name
    # )

    # get supabase
    store = SupabaseVectorStore(
        client=supabase,
        table_name=collection_name,
        query_name="match_" + collection_name,
        embedding=embeddings_openai,
    )

    # get text list from df
    docs = df["条款"].tolist()
    # build metadata
    metadata = df[["监管要求", "结构"]].to_dict(orient="records")

    # get chroma
    # store = Chroma(
    #     persist_directory=fileidxfolder,
    #     embedding_function=embeddings,
    #     collection_name=collection_name,
    # )
    # add to chroma
    store.add_texts(docs, metadatas=metadata)
    # store.persist()


# list all indexes using qdrant
# def list_indexes():
#     """
#     Lists all indexes in the LangChain index.
#     """

#     # get qdrant client
#     qdrant_client = QdrantClient(host=qdrant_host)
#     # get collection names
#     collection_names = qdrant_client.list_aliases()
#     return collection_names


def delete_db(industry="", items=[]):
    collection_name = industry_name_to_code(industry)

    filter = convert_list_to_dict(items)
    # convert dict to json
    filter_json = json.dumps(filter)
    # get pinecone
    # index = pinecone.Index("ruledb")
    # index.delete(filter=filter, namespace=collection_name)
    # print(filter)
    # print(filter_json)
    # delete all
    supabase.table(collection_name).delete().filter(
        "metadata", "cs", filter_json
    ).execute()


# convert document list to pandas dataframe
def docs_to_df(docs):
    """
    Converts a list of documents to a pandas dataframe.
    """
    data = []
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        plc = metadata["监管要求"]
        sec = metadata["结构"]
        row = {"条款": page_content, "监管要求": plc, "结构": sec}
        data.append(row)
    df = pd.DataFrame(data)
    return df


# convert industry chinese name to english name
def industry_name_to_code(industry_name):
    """
    Converts an industry name to an industry code.
    """
    industry_name = industry_name.lower()
    if industry_name == "银行":
        return "bank"
    elif industry_name == "保险":
        return "insurance"
    elif industry_name == "证券":
        return "securities"
    elif industry_name == "基金":
        return "fund"
    elif industry_name == "期货":
        return "futures"
    elif industry_name == "投行":
        return "invbank"
    elif industry_name == "反洗钱":
        return "aml"
    elif industry_name == "医药":
        return "pharma"
    elif industry_name == "hkma":
        return "hkma"
    else:
        return "other"


def convert_list_to_dict(lst):
    if len(lst) == 1:
        return {"监管要求": lst[0]}
    else:
        return {"监管要求": {"$in": [item for item in lst]}}
