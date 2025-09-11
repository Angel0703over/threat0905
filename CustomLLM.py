import requests
from typing import Optional, Dict, List

class CustomLLM:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        generate_config: Optional[Dict] = None
    ):
        """
        初始化自定义LLM模型
        
        :param model_name: 模型名称
        :param base_url: API基础地址
        :param api_key: API密钥
        :param generate_config: 生成配置参数
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.generate_config = generate_config or {"temperature": 0.0}
        
        # 构建完整的API端点
        self.api_endpoint = f"{self.base_url}/chat/completions"

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Dict:
        """
        发送聊天完成请求，接口风格类似OpenAI
        
        :param messages: 消息列表，包含role和content
        :param stream: 是否流式返回
        :return: 模型响应结果
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "messages": messages,
                "stream": stream,
                **self.generate_config
            }

            response = requests.post(
                url=self.api_endpoint,
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
                
        except Exception as e:
            print(f"LLM请求错误: {str(e)}")
            raise

# 使用示例 - 与你之前的ChatOpenAI风格保持一致
if __name__ == "__main__":
    # 配置参数
    config = {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-227f389436b648638d18773b7c47c1f4",
        "generate_config": {
            "temperature": 0.7
        }
    }
    
    # 创建模型实例，类似ChatOpenAI的方式
    llm = CustomLLM(
        model_name=config["model"],
        base_url=config["base_url"],
        api_key=config["api_key"],
        generate_config=config["generate_config"]
    )
    
    # 发送请求
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, world!"}
    ]
    
    response = llm.create_chat_completion(messages)
    print(response["choices"][0]["message"]["content"])
    