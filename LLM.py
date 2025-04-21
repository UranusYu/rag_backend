import json
from openai import AsyncOpenAI, OpenAIError
import tiktoken
import os
from transformers import AutoTokenizer
from typing import List, Dict


class LLM:
    def __init__(self, model_name: str = "models/outline-generation-distill-v3",
                 api_token: str = "token-abc123",
                 base_url: str = "http://0.0.0.0:7789/v1"):
        print("初始化LLM")
        self.model_name = model_name
        self.api_token = api_token
        self.base_url = base_url
        self.max_model_tokens = 16384
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_token)
        self.tokenizer = AutoTokenizer.from_pretrained("models/outline-generation-distill-v3")
        self.prompt_dir = "prompts/default"
        self.example_dir = "examples"

    def load_examples(self, prompt_name: str) -> str:
        example_path = os.path.join(self.example_dir, f"{prompt_name}.txt")
        if os.path.exists(example_path):
            with open(example_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def load_prompt(self, prompt_name: str) -> str:
        prompt_path = os.path.join(self.prompt_dir, f"{prompt_name}.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def update_prompt(self, prompt_name: str, new_prompt: str, new_dir: str = None):
        if new_dir:
            self.prompt_dir = new_dir
            os.makedirs(self.prompt_dir, exist_ok=True)
        prompt_path = os.path.join(self.prompt_dir, f"{prompt_name}.txt")
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(new_prompt)

    def trucate_history(self, messages: List[Dict[str, str]], max_length: int) -> tuple[int, List[Dict[str, str]]]:
        total_token_count_in = self.count_tokens(messages)
        while total_token_count_in > max_length and messages:  # 修改判断条件，避免空列表
            print("Warning: 超出最大长度, 截断对话历史！！！")
            messages.pop(0)
            total_token_count_in = self.count_tokens(messages)
        return total_token_count_in, messages

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_token_count = 0
        for mess in messages:
            total_token_count += len(self.tokenizer.encode(mess['content']))
        return total_token_count

    async def model_chat_flow(self, messages: List[Dict[str, str]], model_name=None, base_url=None, api_token=None):
        if model_name:
            self.model_name = model_name
        if base_url and base_url != self.base_url:
            self.base_url = base_url
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=api_token or self.api_token)
        if api_token and api_token != self.api_token:
            self.api_token = api_token
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_token)

        try:
            # 保证输入不超过最大值
            total_token_count_in, messages = self.trucate_history(messages, self.max_model_tokens)

            # 调用 OpenAI 流式 API
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                stream=True,
                max_tokens=self.max_model_tokens
            )
            # 迭代流式响应
            full_llm_answer = ""
            total_token_count_out = 0
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    full_llm_answer += content
                    total_token_count_out += 1
                    yield f'{json.dumps({"answer": content})}\n\n'.encode("utf-8")
            print(f"\n Usage: 输入Token{total_token_count_in}, 输出Token{total_token_count_out}")

        except OpenAIError as e:
            yield str(e).encode("utf-8")  # 修改为获取更完整的错误信息