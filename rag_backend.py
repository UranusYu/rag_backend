import os
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse
from langchain_community.document_loaders import (
    TextLoader, PDFMinerLoader, Docx2txtLoader, UnstructuredExcelLoader,
    UnstructuredPowerPointLoader, UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader, UnstructuredODTLoader,
    UnstructuredEmailLoader, UnstructuredCSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from LLM import LLM
import torch
import pickle
import hashlib
import asyncio
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()

# 根据设备选择推理设备
device = "cuda" if torch.cuda.is_available() else "cpu"
retrieve_model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {'device': device}
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
embeddings = HuggingFaceBgeEmbeddings(
    model_name=retrieve_model_name,
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
index = faiss.IndexFlatIP(dimension)
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

llm = LLM()


def load_document(file_url: str):
    """
    根据文件扩展名加载文档
    """
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
    """
    将文档分割为文本块
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)


def calculate_embeddings(blocks):
    """
    计算文本块的嵌入向量
    """
    texts = [block.page_content for block in blocks]
    return embeddings.embed_documents(texts)


def add_blocks_to_index(doc_id, blocks):
    """
    将文本块添加到索引中，并保存索引和文档块信息
    """
    global block_id_counter
    embeddings_list = calculate_embeddings(blocks)
    embeddings_np = np.array(embeddings_list).astype('float32')
    block_info = []
    for i, block in enumerate(blocks):
        block_id = block_id_counter
        block_id_counter += 1
        document_blocks[block_id] = (doc_id, block)
        index.add(embeddings_np[i].reshape(1, -1))
        block_info.append({
            "doc_id": doc_id,
            "block_id": block_id,
            "block_content": block.page_content})
    # 保存索引和文档块信息到本地
    faiss.write_index(index, INDEX_FILE)
    with open(DOCUMENT_BLOCKS_FILE, 'wb') as f:
        pickle.dump(document_blocks, f)
    return block_info


def remove_blocks_from_index(doc_id):
    """
    从索引中移除指定文档的文本块，并重新构建索引
    """
    old_block_ids = [block_id for block_id, (doc_id_, _) in document_blocks.items() if doc_id_ == doc_id]
    for block_id in old_block_ids:
        del document_blocks[block_id]
    # 由于 Faiss 不支持直接删除，这里简单重新构建索引
    new_embeddings = []
    new_block_ids = []
    global block_id_counter
    block_id_counter = 0
    for block_id, (_, block) in document_blocks.items():
        block_id = block_id_counter
        block_id_counter += 1
        embedding = embeddings.embed_query(block.page_content)
        new_embeddings.append(embedding)
        new_block_ids.append(block_id)
    new_embeddings_np = np.array(new_embeddings).astype('float32')

    index.reset()
    if new_embeddings_np.size > 0:  # 检查 new_embeddings_np 是否为空
        index.add(new_embeddings_np)

    # 保存索引和文档块信息到本地
    faiss.write_index(index, INDEX_FILE)
    with open(DOCUMENT_BLOCKS_FILE, 'wb') as f:
        pickle.dump(document_blocks, f)


def get_doc_id_from_url(file_url: str) -> str:
    """
    根据文件 URL 生成文档 ID
    """
    return hashlib.sha256(file_url.encode()).hexdigest()


@app.post("/index")
async def process_or_update(request: Request):
    try:
        data = await request.json()
        file_url = data.get('file_url')
        user_doc_id = data.get('doc_id')

        if not file_url:
            raise HTTPException(status_code=400, detail="Missing file_url")

        if user_doc_id:
            doc_id = user_doc_id
            if doc_id in set([doc_id for doc_id, _ in document_blocks.values()]):
                remove_blocks_from_index(doc_id)
        else:
            doc_id = get_doc_id_from_url(file_url)
            if doc_id in set([doc_id for doc_id, _ in document_blocks.values()]):
                remove_blocks_from_index(doc_id)

        docs = load_document(file_url)
        blocks = chunk_document(docs)

        result = add_blocks_to_index(doc_id, blocks)
        return JSONResponse(content={"block_info": result})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        query_text = data.get('query')
        doc_ids = data.get('doc_ids', [])
        topk = data.get('topk', 5)
        threshold = data.get('threshold', 0.5)  # 余弦相似度阈值，范围 [-1, 1]

        if not query_text:
            raise HTTPException(status_code=400, detail="Missing query")

        query_embedding = np.array(embeddings.embed_query(query_text)).astype('float32').reshape(1, -1)

        # 筛选指定 doc_ids 的块
        if doc_ids:
            relevant_block_ids = [block_id for block_id, (doc_id, _) in document_blocks.items() if doc_id in doc_ids]
            relevant_embeddings = np.array([embeddings.embed_query(document_blocks[block_id][1].page_content) for block_id in relevant_block_ids]).astype('float32')
            relevant_index = faiss.IndexFlatIP(dimension)
            relevant_index.add(relevant_embeddings)
            similarities, indices = relevant_index.search(query_embedding, k=min(topk, relevant_index.ntotal))

            valid_results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                block_id = relevant_block_ids[idx]
                block_doc_id, block = document_blocks[block_id]
                if similarity >= threshold:
                    valid_results.append({
                        "doc_id": block_doc_id,
                        "block_id": int(block_id),
                        "block_content": block.page_content,
                        "similarity": float(similarity)
                    })
        else:
            similarities, indices = index.search(query_embedding, k=min(topk, index.ntotal))
            valid_results = []
            for similarity, block_id in zip(similarities[0], indices[0]):
                block_doc_id, block = document_blocks[block_id]
                if similarity >= threshold:
                    valid_results.append({
                        "doc_id": block_doc_id,
                        "block_id": int(block_id),
                        "block_content": block.page_content,
                        "similarity": float(similarity)
                    })

        if not valid_results:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        valid_results = sorted(valid_results, key=lambda x: x["similarity"], reverse=True)[:topk]
        return JSONResponse(content={"results": valid_results})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


# 2. 使用大模型从需求等各类文档中自动识别出软件的功能
@app.post("/function_extraction")
async def function_extraction(request: Request):
    try:
        data = await request.json()
        doc_id = data.get('doc_id')
        if not doc_id:
            raise HTTPException(status_code=400, detail="Missing doc_id")

        relevant_blocks = [block for _, (did, block) in document_blocks.items() if did == doc_id]
        document = " ".join([block.page_content for block in relevant_blocks])

        rag_content = data.get('rag_content', "")
        prompt = llm.load_prompt("function_extraction")
        examples = llm.load_examples("function_extraction")
        full_prompt = prompt.format(rag_content=rag_content,
                                    examples=examples,
                                    document=document)
        result_parts = []
        async for item in llm.model_chat_flow(full_prompt):
            try:
                item_data = json.loads(item.decode("utf-8").strip())
                result_parts.append(item_data["answer"])
            except json.JSONDecodeError:
                pass
        result = ''.join(result_parts)
        return JSONResponse(content={"features": result})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


# 3. 从需求等文档中对功能进行检索，大模型根据检索结果总结输出该功能的需求规格说明
@app.post("/requirement_generation")
async def requirement_generation(request: Request):
    try:
        data = await request.json()
        doc_id = data.get('doc_id')
        feature = data.get('feature')
        if not doc_id or not feature:
            raise HTTPException(status_code=400, detail="Missing doc_id or feature")

        relevant_blocks = [block for _, (did, block) in document_blocks.items() if did == doc_id]
        document = " ".join([block.page_content for block in relevant_blocks])

        rag_content = data.get('rag_content', "")
        prompt = llm.load_prompt("requirement_generation")
        examples = llm.load_examples("requirement_generation")
        full_prompt = prompt.format(rag_content=rag_content,
                                    examples=examples,
                                    document=document, feature=feature)
        result_parts = []
        async for item in llm.model_chat_flow(full_prompt):
            try:
                item_data = json.loads(item.decode("utf-8").strip())
                result_parts.append(item_data["answer"])
            except json.JSONDecodeError:
                pass
        result = ''.join(result_parts)
        return JSONResponse(content={"requirement": result})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


# 1. 日常对话接口，返回 SSE 格式
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        query = data.get('query')
        rag_content = data.get('rag_content', "")
        if not query:
            raise HTTPException(status_code=400, detail="Missing prompt")

        llm.history.add('user', query)
        prompt = llm.load_prompt("chat")
        full_prompt = prompt.format(rag_content=rag_content,
                                    query=query)

        async def generate():
            async for part in llm.model_chat_flow(full_prompt, is_record=True):
                item_data = json.loads(part.decode("utf-8").strip())
                yield f"data: {json.dumps(item_data, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate(), media_type='text/event-stream')
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


# 4. 根据生成的需求规格，生成测试大纲
@app.post("/outline_generation")
async def outline_generation(request: Request):
    try:
        data = await request.json()
        requirement = data.get('requirement')
        rag_content = data.get('rag_content', "")
        if not requirement:
            raise HTTPException(status_code=400, detail="Missing requirement")

        prompt = llm.load_prompt("outline_generation")
        examples = llm.load_examples("outline_generation")
        full_prompt = prompt.format(rag_content=rag_content,
                                    examples=examples,
                                    requirement=requirement)
        result_parts = []
        async for item in llm.model_chat_flow(full_prompt):
            try:
                item_data = json.loads(item.decode("utf-8").strip())
                result_parts.append(item_data["answer"])
            except json.JSONDecodeError:
                pass
        result = ''.join(result_parts)
        return JSONResponse(content={"outline": result})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


# 5. 根据生成的测试大纲，生成测试用例
@app.post("/case_generation")
async def case_generation(request: Request):
    try:
        data = await request.json()
        outline = data.get('outline')
        rag_content = data.get('rag_content', "")
        if not outline:
            raise HTTPException(status_code=400, detail="Missing test_outline")

        prompt = llm.load_prompt("case_generation")
        examples = llm.load_examples("case_generation")
        full_prompt = prompt.format(rag_content=rag_content,
                                    examples=examples,
                                    outline=outline)
        result_parts = []
        async for item in llm.model_chat_flow(full_prompt):
            try:
                item_data = json.loads(item.decode("utf-8").strip())
                result_parts.append(item_data["answer"])
            except json.JSONDecodeError:
                pass
        result = ''.join(result_parts)
        return JSONResponse(content={"test_cases": result})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


# 用户编辑 prompt
@app.post("/edit_prompt")
async def edit_prompt(request: Request):
    try:
        data = await request.json()
        prompt_name = data.get('prompt_name')
        new_prompt = data.get('new_prompt')
        new_dir = data.get('new_dir')
        if not prompt_name or not new_prompt:
            raise HTTPException(status_code=400, detail="Missing prompt_name or new_prompt")
        if prompt_name not in ["chat", "function_extraction", "requirement_generation", "outline_generation", "case_generation"]:
            raise HTTPException(status_code=400, detail="Invalid prompt_name")

        llm.update_prompt(prompt_name, new_prompt, new_dir)
        return JSONResponse(content={"message": "Prompt updated successfully"})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


@app.post("/update_model_config")
async def update_model_config(request: Request):
    try:
        data = await request.json()
        model_name = data.get('model_name')
        base_url = data.get('base_url')
        api_token = data.get('api_token')
        llm.update_model_config(model_name, base_url, api_token)
        return JSONResponse(content={"message": "Config updated successfully"})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")