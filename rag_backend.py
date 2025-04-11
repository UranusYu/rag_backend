import os
import faiss
import numpy as np
from flask import Flask, request, jsonify
from langchain.document_loaders import TextLoader, PDFMinerLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

app = Flask(__name__)

# 初始化向量模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-zh-large-v1.5")

# 初始化向量索引
dimension = 1024
index = faiss.IndexFlatL2(dimension)
document_blocks = {}
block_id_counter = 0


def load_document(file_url):
    if file_url.endswith('.txt'):
        loader = TextLoader(file_url)
    elif file_url.endswith('.pdf'):
        loader = PDFMinerLoader(file_url)
    elif file_url.endswith('.docx'):
        loader = Docx2txtLoader(file_url)
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
    for i, block in enumerate(blocks):
        block_id = block_id_counter
        block_id_counter += 1
        document_blocks[block_id] = (doc_id, block)
        index.add(embeddings_np[i].reshape(1, -1))
    return [(doc_id, block_id) for block_id in range(block_id_counter - len(blocks), block_id_counter)]


@app.route('/process_file', methods=['POST'])
def process_file():
    file_url = request.json.get('file_url')
    if not file_url:
        return jsonify({"error": "Missing file_url"}), 400
    docs = load_document(file_url)
    blocks = chunk_document(docs)
    doc_id = len(document_blocks) // 1000 + 1  # 简单生成文档 ID
    result = add_blocks_to_index(doc_id, blocks)
    return jsonify({"block_info": result})


@app.route('/update_document', methods=['POST'])
def update_document():
    file_url = request.json.get('file_url')
    doc_id = request.json.get('doc_id')
    if not file_url or not doc_id:
        return jsonify({"error": "Missing file_url or doc_id"}), 400
    docs = load_document(file_url)
    blocks = chunk_document(docs)
    # 移除旧的文档块
    old_block_ids = [block_id for block_id, (doc_id_, _) in document_blocks.items() if doc_id_ == doc_id]
    for block_id in old_block_ids:
        del document_blocks[block_id]
    # 重新构建索引
    index.reset()
    for block_id, (doc_id_, block) in document_blocks.items():
        embedding = embeddings.embed_query(block.page_content)
        index.add(np.array(embedding).astype('float32').reshape(1, -1))
    # 添加新的文档块
    result = add_blocks_to_index(doc_id, blocks)
    return jsonify({"block_info": result})


@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query')
    if not query_text:
        return jsonify({"error": "Missing query"}), 400
    query_embedding = np.array(embeddings.embed_query(query_text)).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k=1)
    block_id = indices[0][0]
    doc_id, block = document_blocks[block_id]
    return jsonify({"doc_id": doc_id, "block_id": block_id})


if __name__ == '__main__':
    app.run(debug=True)