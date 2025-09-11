# coding=utf-8
# @Time : 2025/3/3 19:36
# @Author : RoseLee
# @File : base
# @Project : fault-analysis
# @Description :
import re
from typing import Dict, Any, Optional

from openai import OpenAI, APIError, APIConnectionError
import copy
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import atexit
from loguru import logger

# 定义全局统计变量
total_requests = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0
class LLMClient:
    """大语言模型客户端

    Attributes:
        BASE_URL (str): 请求基础URL
        DEFAULT_TIMEOUT_SECONDS (int): 默认超时时间

    """
    BASE_URL: str = None

    # 默认超时时间 300s
    DEFAULT_TIMEOUT_SECONDS: int = 300
    def __init__(self, llm_config: Dict[str, Any], **kwargs):
        """
        Args:
            llm_config (dict): 大模型配置
                - provider: 大模型服务提供商
                - base_url (str): 请求基础URL
                - api_key (str): API KEY
                - temperature (float): 温度
                - model (str): 模型
                - timeout (int): 超时时间(s)
                - generate_config (dict): 生成配置

            kwargs: 暂定

        """
        llm_config = copy.deepcopy(llm_config)
        # 请求基础URL
        if "base_url" not in llm_config and self.BASE_URL is None:
            raise ValueError(f"`base_url` 必须在 config 或 {self.__class__.__name__} 类中指定")
        self._base_url = llm_config.get("base_url", self.BASE_URL)

        # API KEY
        if "api_key" not in llm_config:
            raise ValueError("`api_key` 必须在 config 中指定")
        self._api_key = llm_config.get("api_key")

        # 模型
        if "model" not in llm_config:
            raise ValueError("`model` 必须在 config 中指定")
        self._model = llm_config.get("model")
        # 超时时间
        self._timeout = llm_config.get("timeout", self.DEFAULT_TIMEOUT_SECONDS)
        # 生成配置
        self._generate_config = llm_config.get("generate_config", {})

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )
    @retry(
        stop=stop_after_attempt(3),  # 最大重试次数
        wait=wait_exponential(multiplier=1, max=10),  # 指数退避等待
        retry=retry_if_exception_type((APIError, APIConnectionError)),  # 重试条件
    )
    def __retryable_chat_completion(self, **params):
        """带重试机制的聊天补全请求"""
        try:
            response = self.client.chat.completions.create(**params)
            # logger.debug(f"LLM 请求成功: {params}")
            return response
        except (APIError, APIConnectionError) as e:
            logger.warning(f"LLM 请求失败（可重试）: {e}, params={params}")
            raise
        except Exception as e:
            logger.error(f"LLM 请求失败（不可重试）: {e}, params={params}")
            raise

    def infer(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
    ):
        """
        向大模型发起对话（自动重试）

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            temperature: 覆盖默认温度参数
            max_tokens: 覆盖默认最大token数

        Returns:
            str: 模型生成的回复内容

        Raises:
            APIError: OpenAI API 错误
            APIConnectionError: 网络连接错误
        """
        global total_requests, total_prompt_tokens, total_completion_tokens, total_tokens
        params = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                # {"role": "assistant", "content": ""}
            ],
            "temperature": temperature or self._generate_config.get("temperature", 1.0),
            "max_tokens": max_tokens or self._generate_config.get("max_tokens", 8192),
            "stream": False,
        }

        try:
            response = self.__retryable_chat_completion(**params)
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_usage_tokens = response.usage.total_tokens

            total_requests += 1
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += total_usage_tokens

            logger.info(f"prompt的token用量为：{prompt_tokens}")
            logger.info(f"大模型回答token的用量为：{completion_tokens}")
            logger.info(f'输入输出总token用量为：{total_usage_tokens}')

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM 请求最终失败: {e}")
            raise
def print_statistics():
    global total_requests, total_prompt_tokens, total_completion_tokens, total_tokens
    logger.info("================ 统计信息 ================")
    logger.info(f"总请求次数: {total_requests}")
    logger.info(f"总prompt token用量: {total_prompt_tokens}")
    logger.info(f"总大模型回答token用量: {total_completion_tokens}")
    logger.info(f"总大模型使用token量: {total_tokens}")


# 注册 atexit 函数
atexit.register(print_statistics)
if __name__ == '__main__':
    llm = LLMClient(llm_config = {
    "api_key": "",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com",
    "generate_config": {
        "temperature": 0.4,
    }
})
    print(llm.infer(
        system_prompt='你是一个乐于助人的对话助手',
        user_prompt='今天南京的气温怎么样'
    ))


