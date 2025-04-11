import os
import faiss
import numpy as np
from flask import Flask, request, jsonify
from langchain_community.document_loaders import (
    TextLoader, PDFMinerLoader, Docx2txtLoader, UnstructuredExcelLoader,
    UnstructuredPowerPointLoader, UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader, UnstructuredODTLoader,
    UnstructuredEmailLoader, UnstructuredCSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import torch
import pickle
import hashlib

app = Flask(__name__)

# 根据设备选择推理设备
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda:5"
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    cache_folder="./models",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

# 手动将模型转换为 fp16
if device == "cuda":
    embeddings.client[0].auto_model.half()

# 初始化向量索引
dimension = 1024
index = faiss.IndexFlatL2(dimension)
document_blocks = {}
block_id_counter = 0

# 本地存储路径
INDEX_FILE = "index/index.faiss"
DOCUMENT_BLOCKS_FILE = "index/document_blocks.pkl"

# 尝试从本地加载索引和文档块信息
if os.path.exists(INDEX_FILE) and os.path.exists(DOCUMENT_BLOCKS_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(DOCUMENT_BLOCKS_FILE, 'rb') as f:
        document_blocks = pickle.load(f)
    block_id_counter = max(document_blocks.keys()) + 1 if document_blocks else 0


def load_document(file_url):
    if file_url.endswith('.txt'):
        loader = TextLoader(file_url)
    elif file_url.endswith('.pdf'):
        loader = PDFMinerLoader(file_url)
    elif file_url.endswith('.docx'):
        loader = Docx2txtLoader(file_url)
    elif file_url.endswith('.xlsx') or file_url.endswith('.xls'):
        loader = UnstructuredExcelLoader(file_url)
    elif file_url.endswith('.pptx'):
        loader = UnstructuredPowerPointLoader(file_url)
    elif file_url.endswith('.md'):
        loader = UnstructuredMarkdownLoader(file_url)
    elif file_url.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_url)
    elif file_url.endswith('.odt'):
        loader = UnstructuredODTLoader(file_url)
    elif file_url.endswith('.eml'):
        loader = UnstructuredEmailLoader(file_url)
    elif file_url.endswith('.csv'):
        loader = UnstructuredCSVLoader(file_url)
    else:
        # 假设是代码文件，以文本方式加载
        loader = TextLoader(file_url)
    return loader.load()


def chunk_document(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)


def calculate_embeddings(blocks):
    texts = [block.page_content for block in blocks]
    return embeddings.embed_documents(texts)


def add_blocks_to_index(doc_id, blocks):
    global block_id_counter
    embeddings_list = calculate_embeddings(blocks)
    embeddings_np = np.array(embeddings_list).astype('float32')
    block_info = []
    for i, block in enumerate(blocks):
        block_id = block_id_counter
        block_id_counter += 1
        document_blocks[block_id] = (doc_id, block)
        index.add(embeddings_np[i].reshape(1, -1))
        block_info.append((doc_id, block_id, block.page_content))
    # 保存索引和文档块信息到本地
    faiss.write_index(index, INDEX_FILE)
    with open(DOCUMENT_BLOCKS_FILE, 'wb') as f:
        pickle.dump(document_blocks, f)
    return block_info


def remove_blocks_from_index(doc_id):
    old_block_ids = [block_id for block_id, (doc_id_, _) in document_blocks.items() if doc_id_ == doc_id]
    for block_id in old_block_ids:
        del document_blocks[block_id]
    # 由于 Faiss 不支持直接删除，这里简单重新构建索引
    new_embeddings = []
    new_block_ids = []
    for block_id, (_, block) in document_blocks.items():
        embedding = embeddings.embed_query(block.page_content)
        new_embeddings.append(embedding)
        new_block_ids.append(block_id)
    new_embeddings_np = np.array(new_embeddings).astype('float32')
    index.reset()
    index.add(new_embeddings_np)
    # 保存索引和文档块信息到本地
    faiss.write_index(index, INDEX_FILE)
    with open(DOCUMENT_BLOCKS_FILE, 'wb') as f:
        pickle.dump(document_blocks, f)


def get_doc_id_from_url(file_url):
    return hashlib.sha256(file_url.encode()).hexdigest()


@app.route('/index', methods=['POST'])
def process_or_update():
    file_url = request.json.get('file_url')
    if not file_url:
        return jsonify({"error": "Missing file_url"}), 400

    doc_id = get_doc_id_from_url(file_url)

    docs = load_document(file_url)
    blocks = chunk_document(docs)

    if doc_id in set([doc_id for doc_id, _ in document_blocks.values()]):
        remove_blocks_from_index(doc_id)

    result = add_blocks_to_index(doc_id, blocks)
    return jsonify({"block_info": result})


@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query')
    top_k = request.json.get('top_k', 1)
    max_distance = request.json.get('max_distance', float('inf'))

    if not query_text:
        return jsonify({"error": "Missing query"}), 400

    query_embedding = np.array(embeddings.embed_query(query_text)).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k=top_k)

    results = []
    for i in range(top_k):
        distance = distances[0][i]
        if distance <= max_distance:
            block_id = int(indices[0][i])
            doc_id, block = document_blocks[block_id]
            results.append({
                "doc_id": doc_id,
                "block_id": block_id,
                "block_content": block.page_content,
                "distance": distance
            })

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(debug=True)