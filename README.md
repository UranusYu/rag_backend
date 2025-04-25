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

| 接口名称                   | 功能           | 输入参数                                                                                                                                                                                                                                        | 输出参数                                                                                                                    | 说明                                                                                                                             |
| ---------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| index                  | 构建索引         | required string file_url：文件的位置<br/>optional string doc_id：可指定文档 ID，若未指定则由 sha256 根据文件名生成                                                                                                                                                    | repeated struct<br/>string doc_id：文档 ID<br/>int block_id：从 0 计数，全局变量，用作向量的索引<br/>string block_content：文本块内容             | 支持http和https链接，支持的文件格式有 txt、pdf、docx、xlsx、xls、pptx、md、html、odt、eml、csv；其他格式按 txt 解析<br/>若输入的 doc_id 或有同名的 file_url 已存在，则进行索引替换 |
| delete_index           | 删除索引         | required string doc_id：要删除索引的文档 ID                                                                                                                                                                                                          | string message：操作结果消息                                                                                                   | 删除指定文档 ID 的索引                                                                                                                  |
| query                  | 搜索文档         | required string query：输入的查询内容<br/>optional repeated string doc_ids：指定查询的文档 ID 范围<br/>optional int topk：指定查询分块的个数<br/>optional float threshold：余弦相似度阈值，范围 [-1, 1]                                                                            | repeated struct<br/>string doc_id：文档 ID<br/>int block_id：文本块 ID<br/>string block_content：文本块内容<br/>float similarity：相似度 | 若未指定 doc_ids，则默认在全文档中搜索                                                                                                        |
| chat                   | 聊天           | required array messages：消息列表，每个元素为 {"role": "user" 或 "assistant", "content": "消息内容"}<br/>optional string rag_content：RAG 检索的结果<br/>optional string model_name：模型名称<br/>optional string base_url：基础 URL<br/>optional string api_token：API 令牌 | text-stream 文本流                                                                                                         |                                                                                                                                |
| function_extraction    | 功能抽取         | required repeated string doc_ids：文档 ID 列表<br/>optional string rag_content：RAG 检索的结果<br/>optional array messages：消息列表                                                                                                                        | string features：抽取的功能                                                                                                   |                                                                                                                                |
| requirement_generation | 生成需求         | required repeated string doc_ids：文档 ID 列表<br/>required string feature：要生成需求的功能描述<br/>optional string rag_content：RAG 检索的结果<br/>optional array messages：消息列表                                                                                 | string requirement：生成的需求规格说明                                                                                            |                                                                                                                                |
| outline_generation     | 生成大纲         | required string requirement：需求规格说明<br/>optional string rag_content：RAG 检索的结果<br/>optional array messages：消息列表                                                                                                                               | string outline：生成的测试大纲                                                                                                  |                                                                                                                                |
| case_generation        | 生成用例         | required string outline：测试大纲<br/>optional string rag_content：RAG 检索的结果<br/>optional array messages：消息列表                                                                                                                                     | string test_cases：生成的测试用例                                                                                               |                                                                                                                                |
| edit_prompt            | 修改各个模块用到的提示词 | required string prompt_name：指定模块，必须在 ["chat", "function_extraction", "requirement_generation", "outline_generation", "case_generation"] 之中<br/>required string new_prompt：新的提示词<br/>optional string new_dir：创建新的文件夹存储提示词文件                  | string message：操作结果消息                                                                                                   | 原始提示词存于 prompts/default 中，选择 new_dir 的名称不应覆盖 default 中的内容                                                                      |

## 测试 `index` 接口

```shell
curl -X POST "http://127.0.0.1:7788/index" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"file_url": "test/伺服需求---表格版.docx", "doc_id": "111111"}'
```

## 测试 `delete_index` 接口

```shell
curl -X POST "http://127.0.0.1:7788/delete_index" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"doc_id": "111111"}'
```

## 测试 `query` 接口

```shell
curl -X POST "http://127.0.0.1:7788/query" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"query": "角速度控制", "topk": 1, "threshold": 0.5}'
```

## 测试 `chat` 接口

```shell
curl -X POST "http://127.0.0.1:7788/chat" \
     -H "accept: text/event-stream" \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "它的电池可能有什么问题"}], "rag_content": ""}'
```

## 测试 `function_extraction` 接口

```shell
curl -X POST "http://127.0.0.1:7788/function_extraction" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"doc_ids": ["111111"]}'
```

## 测试 `requirement_generation` 接口

```shell
curl -X POST "http://127.0.0.1:7788/requirement_generation" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"doc_ids": ["111111"], "feature": "1. **角度控制**  \n   功能名称：角度控制  \n   简要说明：伺服主控软件接收机载光端机发送的角度控制指令，控制平台转动至相应角度后，上报当前方位角和俯仰角至机载光端机。  \n   典型用例：平台从初始状态0°,0°转动至目标角度30°,30°，并上报当前方位角和俯仰角。"}'
```

## 测试 `outline_generation` 接口

```shell
curl -X POST "http://127.0.0.1:7788/outline_generation" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"requirement": "**功能需求文档**\n\n**角度控制**\n\n**用例名称**  \n角度控制  \n\n**项目唯一标识符**  \nSF_YD  \n\n**研制要求章节**  \n无  \n\n**简要描述**  \n伺服主控软件接收机载光端机发送的角度控制指令，控制平台转动至相应角度后，上报当前方位角和俯仰角至机载光端机。  \n\n**参与者**  \n主执行者：机载光端机  \n辅助执行者：编码器、陀螺  \n\n**前置条件**  \n伺服打开且未处于扫描过程中。  \n\n**主流程**  \n步骤  \n1. 伺服主控软件通过串口接收机载光端机发送的角度控制指令。  \n2. 周期1ms接收编码器提供的当前方位角、俯仰角。  \n3. 周期1ms接收陀螺提供的当前角速度。  \n4. 根据当前方位角和目标方位角的偏差，计算出PID外环控制量。  \n5. 根据当前俯仰角和目标俯仰角的偏差，计算出PID外环控制量。  \n6. 根据PID外环控制量和当前角速度的偏差，计算出PID内环控制量。  \n7. 根据PID内环控制量计算方位电机驱动器的PWM占空比。  \n8. 根据PID内环控制量计算俯仰电机驱动器的PWM占空比。  \n9. 输出PWM信号至方位电机驱动器，驱动伺服电机转动。  \n10. 输出PWM信号至俯仰电机驱动器，驱动伺服电机转动。  \n11. 循环2-10，直至编码器获取的当前方位角和目标方位角相同、当前俯仰角和目标俯仰角相同。  \n12. 上报当前方位角和俯仰角至机载光端机。  \n\n**扩展流程**  \n1a. 方位或俯仰角度超出控制范围  \n1a1. 结束用例  \n\n2a. 1s未控制到位  \n2a1. 向机载光端机上报故障信息  \n2a2. 结束用例  \n\n**后置条件**  \n无  \n\n**规则与约束**  \n1. 输入数据约束  \n1）角度控制指令包含方位角和俯仰角。  \n① 方位角范围：[-180°,180°]，方位角顺时针针为正，逆时针为负。  \n② 俯仰角范围为：[-130°,+30°]，俯仰往下为负，上为正。  \n\n2. 输出数据约束  \n1）输出方位电机驱动器的PWM信号。  \n2）输出俯仰电机驱动器的PWM信号。  \n\n**输出数据约束**  \n1）输出方位电机驱动器的PWM信号。  \n2）输出俯仰电机驱动器的PWM信号。", "rag_content": ""}'
```

## 测试 `case_generation` 接口

```shell
curl -X POST "http://127.0.0.1:7788/case_generation" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"outline": "#### 测试大纲：角度控制\n\n**需求描述：**\n伺服主控软件接收机载光端机发送的角度控制指令，控制平台转动至相应角度后，上报当前方位角和俯仰角至机载光端机。该功能需确保伺服电机的正确驱动，以实现平台的精确控制。\n\n**测试方法：**\n在实验室环境，使用GaleView软件模拟高精度惯导、低精度惯导、大气数据计算机、北斗UZ4220（或UP29501）向被测软件周期（xxms）发送飞机位置、姿态等数据。同时，使用飞行控制应用软件模拟机载测控软件，通过串口向被测软件发送角度控制指令，并接收被测软件反馈的当前方位角和俯仰角。\n\n**测试充分性：**\n1. **正常情况：**\n   a. **不同起始高度：**\n      - 测试在海拔高度1000m、3000m、5000m下的角度控制功能。\n   b. **不同起始方位角：**\n      - 测试在正航向角、负航向角、0航向角下的角度控制功能。\n   c. **俯仰角范围：**\n      - 测试俯仰角在14°、16°的有效值和边界值。\n   d. **滚转角范围：**\n      - 测试滚转角在-1°、1°的有效值和边界值。\n\n2. **异常情况：**\n   a. **传感器异常：**\n      - 测试在主或备北斗数据异常、惯导数据异常、大气数据计算机数据异常下的角度控制功能。\n   b. **平台未达到控制范围：**\n      - 测试在方位角或俯仰角超出控制范围时的角度控制功能。\n\n**通过准则：**\n1. **正常情况的预期结果：**\n   a. **方位角控制：**\n      - 平台转动至指定方位角，保持稳定。\n   b. **俯仰角控制：**\n      - 平台转动至指定俯仰角，保持稳定。\n   c. **滚转角控制：**\n      - 平台保持指定滚转角，保持稳定。\n   d. **方位角偏差控制：**\n      - 平台在指定范围内保持方位角偏差在±20°之间。\n   e. **俯仰角偏差控制：**\n      - 平台在指定范围内保持俯仰角偏差在±10°之间。\n\n2. **异常情况的预期结果：**\n   a. **传感器异常：**\n      - 系统提示传感器异常，无法完成角度控制。\n   b. **平台未达到控制范围：**\n      - 系统提示平台未达到控制范围，无法完成角度控制。\n\n通过以上测试，确保角度控制功能在正常和异常情况下的正确性和可靠性。", "rag_content": ""}'
```

## 测试 `edit_prompt` 接口

```shell
curl -X POST "http://127.0.0.1:7788/edit_prompt" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"prompt_name": "chat", "new_prompt": "新的提示词内容", "new_dir": "new_prompts"}'
```