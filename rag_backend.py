import os
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import requests
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from LLM import LLM
import torch
import pickle
import hashlib
import asyncio
import json

import ssl
import re


ssl._create_default_https_context = ssl._create_unverified_context
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
# 新增：顺序索引到block_id的映射
index_to_block_id = {}

# 本地存储路径
INDEX_FILE = "index/index.faiss"
DOCUMENT_BLOCKS_FILE = "index/document_blocks.pkl"

# 尝试从本地加载索引和文档块信息
if os.path.exists(INDEX_FILE) and os.path.exists(DOCUMENT_BLOCKS_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(DOCUMENT_BLOCKS_FILE, 'rb') as f:
        document_blocks = pickle.load(f)
    block_id_counter = max(document_blocks.keys()) + 1 if document_blocks else 0
    # 新增：从已加载的document_blocks重建index_to_block_id映射
    index_to_block_id = {i: block_id for i, block_id in enumerate(document_blocks.keys())}

llm = LLM()


def get_load(file_url: str):
    return UnstructuredFileLoader(file_url, mode="single")


def load_document(file_url: str):
    """
    根据文件路径或URL加载文档
    """
    if file_url.startswith('http://') or file_url.startswith('https://'):
        try:
            # 发送HEAD请求获取文件大小
            head_response = requests.head(file_url)
            head_response.raise_for_status()
            total_size = int(head_response.headers.get('content-length', 0))

            # 发送GET请求获取文件内容
            response = requests.get(file_url, stream=True)
            response.raise_for_status()

            # 尝试从响应头中获取文件名
            file_name = None
            content_disposition = response.headers.get('content-disposition')
            if content_disposition:
                for part in content_disposition.split(';'):
                    part = part.strip()
                    if part.startswith('filename='):
                        file_name = part.split('=')[1].strip('"')
                        break

            # 如果响应头中没有文件名，根据URL生成文件名
            if not file_name:
                file_name = os.path.basename(file_url)

            # 确保文件名在当前目录下不重复
            counter = 1
            original_name, extension = os.path.splitext(file_name)
            while os.path.exists(file_name):
                file_name = f"{original_name}_{counter}{extension}"
                counter += 1

            # 下载文件
            with open(file_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # 使用 UnstructuredFileLoader 加载文件
            # loader = UnstructuredFileLoader(file_name)
            loader = get_load(file_name)
            docs = loader.load()

            # 删除下载的文件（如果需要）
            os.remove(file_name)

            return docs

        except requests.RequestException as e:
            return []
        except Exception as e:
            return []
    else:
        try:
            # 对于本地文件，直接使用 UnstructuredFileLoader 加载
            # loader = UnstructuredFileLoader(file_url)
            loader = get_load(file_url)
            return loader.load()
        except Exception as e:
            return []


def chunk_document(docs):
    """
    将文档分割为文本块
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
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
    global block_id_counter, index_to_block_id
    embeddings_list = calculate_embeddings(blocks)
    embeddings_np = np.array(embeddings_list).astype('float32')
    block_info = []
    for i, block in enumerate(blocks):
        block_id = block_id_counter
        block_id_counter += 1
        document_blocks[block_id] = (doc_id, block)
        index.add(embeddings_np[i].reshape(1, -1))
        # 新增：更新顺序索引到block_id的映射
        index_to_block_id[len(index_to_block_id)] = block_id
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
    global index_to_block_id, document_blocks
    # 找出需要移除的块的ID
    old_block_ids = [block_id for block_id, (doc_id_, _) in document_blocks.items() if doc_id_ == doc_id]
    # 找出需要保留的块的ID
    keep_block_ids = [block_id for block_id in document_blocks if block_id not in old_block_ids]

    # 移除不需要的块
    for block_id in old_block_ids:
        del document_blocks[block_id]

    # 重新构建 index_to_block_id 映射
    index_to_block_id = {i: block_id for i, block_id in enumerate(keep_block_ids)}

    # 提取保留块的嵌入向量
    keep_embeddings = []
    for block_id in keep_block_ids:
        # 假设之前已经有一个函数可以根据 block_id 获取嵌入向量
        # 这里简单模拟，实际需要根据你的代码逻辑修改
        embedding = index.reconstruct(int(np.where(np.array(list(index_to_block_id.values())) == block_id)[0]))
        keep_embeddings.append(embedding)

    keep_embeddings_np = np.array(keep_embeddings).astype('float32')

    # 重置索引并添加保留的嵌入向量
    index.reset()
    if keep_embeddings_np.size > 0:
        index.add(keep_embeddings_np)

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
        if not docs:
            raise HTTPException(status_code=400, detail="Failed to load document from the provided URL.")
        blocks = chunk_document(docs)

        result = add_blocks_to_index(doc_id, blocks)
        return JSONResponse(content={"block_info": result})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


@app.post("/delete_index")  # 新增接口
async def delete_index(request: Request):
    try:
        data = await request.json()
        doc_id = data.get('doc_id')
        if not doc_id:
            raise HTTPException(status_code=400, detail="Missing doc_id")

        remove_blocks_from_index(doc_id)
        return JSONResponse(content={"message": "Index deleted successfully"})
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

        # 筛选指定doc_ids的块
        if doc_ids:
            # 检查是否存在有效的doc_id
            valid_doc_ids = set([doc_id for doc_id, _ in document_blocks.values()])
            missing_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in valid_doc_ids]
            if missing_doc_ids:
                raise HTTPException(status_code=404, detail=f"Some doc_ids not found: {', '.join(map(str, missing_doc_ids))}")

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
            for similarity, idx in zip(similarities[0], indices[0]):
                # 新增：通过顺序索引获取实际的block_id
                block_id = index_to_block_id.get(idx)
                if block_id is not None:
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


@app.post("/function_extraction")
async def function_extraction(request: Request):
    try:
        data = await request.json()
        doc_ids = data.get('doc_ids')
        messages = data.get('messages', [])
        if not doc_ids:
            raise HTTPException(status_code=400, detail="Missing doc_ids")

        relevant_blocks = [block for _, (did, block) in document_blocks.items() if did in doc_ids]
        document = " ".join([block.page_content for block in relevant_blocks])

        rag_content = data.get('rag_content', "")
        prompt = llm.load_prompt("function_extraction")
        examples = llm.load_examples("function_extraction")
        full_prompt = prompt.format(rag_content=rag_content,
                                    examples=examples,
                                    document=document)

        result_parts = []
        async for item in llm.model_chat_flow(messages + [{"role": "user", "content": full_prompt}]):
            try:
                item_data = json.loads(item.decode("utf-8").strip())
                result_parts.append(item_data["answer"])
            except json.JSONDecodeError:
                pass
        result = ''.join(result_parts)

        # 直接将包含 <think> 的 result 赋值给 answer
        answer = result

        # 按照要求分割功能
        function_sections = result.strip().split("\n\n")
        features = []
        for section in function_sections:
            if section[0].isdigit():
                features.append(section.strip())

        return JSONResponse(content={"answer": answer, "features": features})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


@app.post("/requirement_generation")
async def requirement_generation(request: Request):
    try:
        data = await request.json()
        doc_ids = data.get('doc_ids')
        feature = data.get('feature')
        messages = data.get('messages', [])
        if not doc_ids or not feature:
            raise HTTPException(status_code=400, detail="Missing doc_ids or feature")

        relevant_blocks = [block for _, (did, block) in document_blocks.items() if did in doc_ids]
        document = " ".join([block.page_content for block in relevant_blocks])

        rag_content = data.get('rag_content', "")
        prompt = llm.load_prompt("requirement_generation")
        examples = llm.load_examples("requirement_generation")
        full_prompt = prompt.format(rag_content=rag_content,
                                    examples=examples,
                                    document=document, feature=feature)

        result_parts = []
        async for item in llm.model_chat_flow(messages + [{"role": "user", "content": full_prompt}]):
            try:
                item_data = json.loads(item.decode("utf-8").strip())
                result_parts.append(item_data["answer"])
            except json.JSONDecodeError:
                pass
        result = ''.join(result_parts)

        # 直接将 result 赋值给 answer
        answer = result

        match = re.search(r'</think>(.*)', result, re.DOTALL)
        result = match.group(1) if match else result

        # 提取结构化信息
        requirement = {}

        # 提取测试项名称
        test_item_name_match = re.search(r"#### 用例名称\n(.*?)\n\n", result, re.DOTALL)
        requirement["用例名称"] = test_item_name_match.group(1).strip() if test_item_name_match else "无"

        # 提取需求描述
        requirement_description_match = re.search(r"#### 简要描述\n(.*?)\n\n", result, re.DOTALL)
        requirement["简要描述"] = requirement_description_match.group(1).strip() if requirement_description_match else "无"

        partner_match = re.search(r"#### 参与者\n(.*?)\n\n", result, re.DOTALL)
        requirement["参与者"] = partner_match.group(1).strip() if partner_match else "无"

        pre_req_match = re.search(r"#### 前置条件\n(.*?)\n\n", result, re.DOTALL)
        requirement["前置条件"] = pre_req_match.group(1).strip() if pre_req_match else "无"

        main_proc_match = re.search(r"#### 主流程\n(.*?)\n\n", result, re.DOTALL)
        requirement["主流程"] = main_proc_match.group(1).strip() if main_proc_match else "无"

        expand_proc_match = re.search(r"#### 扩展流程\n(.*?)\n\n", result, re.DOTALL)
        requirement["扩展流程"] = expand_proc_match.group(1).strip() if expand_proc_match else "无"

        post_proc_match = re.search(r"#### 后置条件\n(.*?)\n\n", result, re.DOTALL)
        requirement["后置条件"] = post_proc_match.group(1).strip() if post_proc_match else "无"

        rule_match = re.search(r"#### 规则与约束\n(.*)", result, re.DOTALL)
        requirement["规则与约束"] = rule_match.group(1).strip() if rule_match else "无"

        return JSONResponse(content={"answer": answer, "requirement": requirement})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


# 1. 日常对话接口，返回 SSE 格式
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        messages = data.get('messages', [])
        rag_content = data.get('rag_content', "")
        model_name = data.get('model_name')
        base_url = data.get('base_url')
        api_token = data.get('api_token')
        if not messages:
            raise HTTPException(status_code=400, detail="Missing messages")

        query = messages[-1]['content']
        prompt = llm.load_prompt("chat")
        full_prompt = prompt.format(rag_content=rag_content,
                                    query=query)

        new_messages = messages[:-1] + [{"role": "user", "content": full_prompt}]

        async def generate():
            async for part in llm.model_chat_flow(new_messages, model_name=model_name, base_url=base_url, api_token=api_token):
                try:
                    item_data = json.loads(part.decode("utf-8").strip())
                    yield f"data: {json.dumps(item_data, ensure_ascii=False)}\n\n"
                except json.JSONDecodeError:
                    pass

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
        messages = data.get('messages', [])
        if not requirement:
            raise HTTPException(status_code=400, detail="Missing requirement")

        prompt = llm.load_prompt("outline_generation")
        examples = llm.load_examples("outline_generation")
        full_prompt = prompt.format(rag_content=rag_content,
                                    examples=examples,
                                    requirement=requirement)

        result_parts = []
        async for item in llm.model_chat_flow(messages + [{"role": "user", "content": full_prompt}]):
            try:
                item_data = json.loads(item.decode("utf-8").strip())
                result_parts.append(item_data["answer"])
            except json.JSONDecodeError:
                pass
        result = ''.join(result_parts)

        # 直接将 result 赋值给 answer
        answer = result

        match = re.search(r'</think>(.*)', result, re.DOTALL)
        result = match.group(1) if match else result

        # 提取结构化信息
        outline_structure = {}
        # 提取测试项名称
        test_item_name_match = re.search(r"#### 测试项名称\n(.*?)\n\n", result, re.DOTALL)
        outline_structure["测试项名称"] = test_item_name_match.group(1).strip() if test_item_name_match else "无"

        # 提取需求描述
        requirement_description_match = re.search(r"#### 需求描述\n(.*?)\n\n", result, re.DOTALL)
        outline_structure["需求描述"] = requirement_description_match.group(1).strip() if requirement_description_match else "无"

        test_method_match = re.search(r"#### 测试方法\n(.*?)\n\n", result, re.DOTALL)
        outline_structure["测试方法"] = test_method_match.group(1).strip() if test_method_match else "无"

        normal_case_match = re.search(r"#### 测试充分性\n(.*?)\n\n", result, re.DOTALL)
        outline_structure["测试充分性"] = normal_case_match.group(1).strip() if normal_case_match else "无"

        normal_criterion_match = re.search(r"#### 通过准则\n(.*)", result, re.DOTALL)
        outline_structure["通过准则"] = normal_criterion_match.group(1).strip() if normal_criterion_match else "无"

        return JSONResponse(content={"answer": answer, "outline": outline_structure})
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


# 5. 根据生成的测试大纲，生成测试用例
@app.post("/case_generation")
async def case_generation(request: Request):
    try:
        data = await request.json()
        outline = data.get('outline')
        rag_content = data.get('rag_content', "")
        messages = data.get('messages', [])
        if not outline:
            raise HTTPException(status_code=400, detail="Missing test_outline")

        prompt = llm.load_prompt("case_generation")
        examples = llm.load_examples("case_generation")
        full_prompt = prompt.format(rag_content=rag_content,
                                    examples=examples,
                                    outline=outline)

        result_parts = []
        async for item in llm.model_chat_flow(messages + [{"role": "user", "content": full_prompt}]):
            try:
                item_data = json.loads(item.decode("utf-8").strip())
                result_parts.append(item_data["answer"])
            except json.JSONDecodeError:
                pass
        result = ''.join(result_parts)

        # 直接将 result 赋值给 answer
        answer = result

        match = re.search(r'</think>(.*)', result, re.DOTALL)
        result = match.group(1) if match else result

        # 提取结构化的测试用例信息
        test_cases_structure = []
        test_case_sections = re.split(r"### 测试用例", result)
        for section in test_case_sections[1:]:  # 跳过第一个空字符串
            test_case = {}
            section = section.strip()

            # 提取测试用例名称
            name_match = re.search(r"#### 测试用例名称\n(.*?)\n\n", section, re.DOTALL)
            if not name_match:
                continue

            test_case["测试用例名称"] = name_match.group(1).strip() if name_match else "无"

            # 提取简要描述
            desc_match = re.search(r"#### 简要描述\n(.*?)\n\n", section, re.DOTALL)
            test_case["简要描述"] = desc_match.group(1).strip() if desc_match else "无"

            # 提取初始化要求
            init_match = re.search(r"#### 初始化要求\n(.*?)\n\n", section, re.DOTALL)
            test_case["初始化要求"] = init_match.group(1).strip() if init_match else "无"

            # 提取前提和约束
            pre_match = re.search(r"#### 前提和约束\n(.*?)\n\n", section, re.DOTALL)
            test_case["前提和约束"] = pre_match.group(1).strip() if pre_match else "无"

            # 提取测试用例设计方法
            method_match = re.search(r"#### 测试用例设计方法\n(.*?)\n\n", section, re.DOTALL)
            test_case["测试用例设计方法"] = method_match.group(1).strip() if method_match else "无"

            # 提取测试终止条件
            normal_termination_match = re.search(r"#### 正常终止条件\n(.*?)\n\n", section, re.DOTALL)
            test_case["正常终止条件"] = normal_termination_match.group(1).strip() if normal_termination_match else "无"

            abnormal_termination_match = re.search(r"#### 异常终止条件\n(.*?)\n\n", section, re.DOTALL)
            test_case["异常终止条件"] = abnormal_termination_match.group(1).strip() if abnormal_termination_match else "无"

            # 提取测试过程描述
            process_match = re.search(r"#### 测试过程描述\n(.*)", section, re.DOTALL)
            test_case["测试过程描述"] = process_match.group(1).strip() if process_match else "无"

            test_cases_structure.append(test_case)

        return JSONResponse(content={"answer": answer, "test_cases": test_cases_structure})
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