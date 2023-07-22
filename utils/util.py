import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import uuid

MODEL_1 = 'bert-large-uncased-whole-word-masking-finetuned-squad'


def split_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


def get_retriever():
    # set device to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load the retriever model from huggingface model hub
    model = MODEL_1
    retriever = SentenceTransformer(model, device=device)
    return retriever


def generate_uuid():
    return uuid.uuid4().hex
