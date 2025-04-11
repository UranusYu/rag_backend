## 安装依赖

```shell
# 创建一个新的 conda 环境
conda create -n rag_backend python=3.9 -y

# 激活 conda 环境
conda activate rag_backend

# 安装项目依赖
pip install -r requirements.txt
```

## 启动调试服务

```shell
# 启动 Flask 开发服务器，开启调试模式
export FLASK_APP=rag_backend.py
export FLASK_ENV=development
flask run --port 7788
```

## 正式生产环境使用

```shell
nohup gunicorn -w 4 -b 0.0.0.0:7788 rag_backend:app &
```

`-w 4`：指定使用 4 个工作进程

## 测试`index`接口

```shell
curl -X POST http://127.0.0.1:7788/index -H "Content-Type: application/json" -d '{"file_url": "test.txt", "doc_id": 1}'
```

## 测试`query`接口

```shell
curl -X POST http://127.0.0.1:7788/query -H "Content-Type: application/json" -d '{"query": "你的查询文本", "top_k": 1}'
```
