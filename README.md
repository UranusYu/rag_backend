## 安装依赖

```shell
# 创建一个新的 conda 环境
conda create -n rag_backend python=3.9 -y

# 激活 conda 环境
conda activate rag_backend

# 安装项目依赖
pip install -r requirements.txt
```

## 启动服务

```shell
uvicorn rag_backend:app --host 0.0.0.0 --port 7788 --reload
```

## 挂起本地大模型服务

```shell
# 适配OpenAI接口调用模式
CUDA_VISIBLE_DEVICES=5 \
python -m vllm.entrypoints.openai.api_server \
    --model models/outline-generation-distill-v3 \
    --port 7789 --api-key token-abc123 \
    --dtype float16    # 自定义精度
```

## 外部接口总览

| 接口名称                   | 功能           | 输入参数                                                                                                             | 输出参数                                                                                                              | 说明                                                                                                                                                                          |
| ---------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| index                  | 构建索引         | required string file_url 文件的位置<br/>optional string doc_id 可以指定doc_id，否则由sha256根据文件名生成                            | repeated struct<br/>string doc_id<br/>int block_id 从0计数，全局变量，用作向量的索引<br/>string block_content                          | 支持txt, pdf, docx, xlsx, xls, pptx, md, html, odt, eml, csv; 其余格式均按txt解析<br/>如果输入的doc_id或有同名的file_url已经存在，那么进行索引替换                                                           |
| query                  | 搜索文档         | required string query 输入的查询<br/>optional repeated string doc_ids 指定查询的doc_id的范围<br/>optional int top_k 指定查询分块的个数 | repeated struct<br/>string doc_id<br/>int block_id <br/>string block_content<br/>float similarity 相似度 | 不指定输入的doc_ids默认全文档搜索                                                                                                                                                        |
| chat                   | 聊天           | required string query 输入的问题<br/>optional string rag_content RAG检索的结果                                             | text-stream 文本流                                                                                                   |                                                                                                                                                                             |
| function_extraction    | 功能抽取         | required string doc_id <br/>optional string rag_content                                                 | string feature                                                                                                    |                                                                                                                                                                             |
| requirement_generation | 生成需求         | required string doc_id <br/>required string feature<br/>optional string rag_content                             | string requirement                                                                                                |                                                                                                                                                                             |
| outline_generation     | 生成大纲         | required string requirement<br/>optional string rag_content                                                      | string outline                                                                                                    |                                                                                                                                                                             |
| case_generation        | 生成用例         | required string outline<br/>optional string rag_content                                                          | string test_cases                                                                                                 |                                                                                                                                                                             |
| edit_prompt            | 修改各个模块用到的提示词 | required string prompt_name 指定模块<br/>required string new_prompt 新的提示词<br/>required_string new_dir 创建新的文件夹存储提示词文件 | 无                                                                                                                 | prompt_name必须在["chat", "function_extraction", "requirement_generation", "outline_generation", "case_generation"]之中<br/>原始提示词存在prompts/default中，选择new_dir的名称不应当覆盖default中的内容 |
| update_model_config    | 切换模型         | required string model_name<br/>required string base_url<br/>required string api_token                            | 无                                                                                                                 |                                                                                                                                                                             |

## 测试`index`接口

```shell
curl -X POST "http://127.0.0.1:7788/index" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"file_url": "test/伺服需求---表格版.docx", "doc_id": "111111"}'
```

## 测试`query`接口

```shell
curl -X POST "http://127.0.0.1:7788/query" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"query": "角速度控制", "topk": 1}'
```

## 测试`chat`接口

```shell
curl -X POST "http://127.0.0.1:7788/chat" \
     -H "accept: text/event-stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "它的电池可能有什么问题", "rag_content": ""}'
```

## 测试`function_extraction`接口

```shell
curl -X POST "http://127.0.0.1:7788/function_extraction" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "111111"}'
```

## 测试`requirement_generation`接口

```shell
curl -X POST "http://127.0.0.1:7788/requirement_generation" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "111111", "feature": "1. **角度控制**  \n   功能名称：角度控制  \n   简要说明：伺服主控软件接收机载光端机发送的角度控制指令，控制平台转动至相应角度后，上报当前方位角和俯仰角至机载光端机。  \n   典型用例：平台从初始状态0°,0°转动至目标角度30°,30°，并上报当前方位角和俯仰角。"}'
```