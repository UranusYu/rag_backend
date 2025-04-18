import json
from openai import AsyncOpenAI, OpenAIError
import tiktoken
import os
from transformers import AutoTokenizer
from typing import List, Dict  # 新增类型提示相关导入


class History:
    def __init__(self, max_length: int = 10):
        self.messages: List[Dict[str, str]] = []
        self.current_length: int = 0
        self.max_length: int = max_length

    def add(self, role: str, content: str):
        if self.current_length >= self.max_length:
            self.messages.pop(0)
        self.messages.append({'role': role, 'content': content})
        self.current_length += 1

    def clear(self):
        self.messages = []
        self.current_length = 0

    def get_history(self) -> List[Dict[str, str]]:
        return self.messages

    def get_length(self) -> int:
        return self.current_length


class LLM:
    def __init__(self, model_name: str = "models/outline-generation-distill-v3",
                 api_token: str = "token-abc123",
                 base_url: str = "http://0.0.0.0:7789/v1"):
        print("初始化LLM")
        self.history = History()
        self.model_name = model_name
        self.api_token = api_token
        self.base_url = base_url
        self.max_model_tokens = 16384
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_token)
        self.tokenizer = AutoTokenizer.from_pretrained("models/outline-generation-distill-v3")
        self.prompt_dir = "prompts/default"

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

    def trucate_history(self, messages: List[Dict[str, str]], max_length: int) -> (int, List[Dict[str, str]]):
        total_token_count_in = self.count_tokens(messages)
        while total_token_count_in > max_length:
            print("Warning: 超出最大长度, 截断对话历史！！！")
            messages.pop(0)
            total_token_count_in = self.count_tokens(messages)
        self.history.messages = messages
        return total_token_count_in, messages

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_token_count = 0
        for mess in messages:
            total_token_count += len(self.tokenizer.encode(mess['content']))
        return total_token_count

    async def model_chat_flow(self, prompt: str, is_record: bool = False):
        messages = self.history.get_history()
        if is_record:
            messages[-1]["content"] = prompt
        else:
            messages += [{'role': 'user', 'content': prompt}]
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
                    yield f'{json.dumps({"query": prompt, "answer": content})}\n\n'.encode("utf-8")
            print(f"\n Usage: 输入Token{total_token_count_in}, 输出Token{total_token_count_out}")
            if is_record:
                self.history.add('assistant', full_llm_answer)

        except OpenAIError as e:
            yield str(e)  # 修改为获取更完整的错误信息

    def update_model_config(self, model_name: str = None, base_url: str = None, api_token: str = None):
        if model_name:
            self.model_name = model_name
        if base_url:
            self.base_url = base_url
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_token)
        if api_token:
            self.api_token = api_token
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_token)