# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

PERSIST_PATH = ".persist_vector_store_confluence"
load_dotenv()
DOMAIN = os.getenv("CONFLUENCE_DOMAIN")
EMAIL = os.getenv("CONFLUENCE_EMAIL")
TOKEN = os.getenv("CONFLUENCE_TOKEN")
PAGE_IDS = os.getenv("CONFLUENCE_PAGE_IDS").split(",")

def fetch_page_content(page_id: str) -> str:
    url = f"https://{DOMAIN}/wiki/rest/api/content/{page_id}?expand=body.storage"
    response = requests.get(url, auth=(EMAIL, TOKEN))
    response.raise_for_status()
    html = response.json()["body"]["storage"]["value"]
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def load_and_split_confluence_docs() -> List[Document]:
    docs = []
    for pid in PAGE_IDS:
        content = fetch_page_content(pid)
        docs.append(Document(page_content=content, metadata={"page_id": pid}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def get_vector_store(
    embedding: Embeddings, persist_path: str = PERSIST_PATH) -> SKLearnVectorStore:
    """Get or create a vector store."""
    vector_store = SKLearnVectorStore(embedding=embedding, persist_path=persist_path)

    if not os.path.exists(persist_path):
        doc_splits = load_and_split_confluence_docs()
        vector_store.add_documents(documents=doc_splits)
        vector_store.persist()

    return vector_store