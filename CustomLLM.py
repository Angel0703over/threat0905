import requests
from typing import Optional, Dict, List

import requests
import json
from typing import Optional, Dict, List, Any, Callable
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult, Generation

import requests
import json
from typing import Optional, Dict, List, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage

# 自定义JSON编码器，确保消息正确序列化
class MessageJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, BaseMessage):
            role_map = {
                "system": "system",
                "human": "user",
                "ai": "assistant",
                "tool": "tool"
            }
            role = role_map.get(obj.type, "user")
            return {
                "role": role,
                "content": obj.content,
                "additional_kwargs": obj.additional_kwargs
            }
        return super().default(obj)

class CustomLLM:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        generate_config: Optional[Dict] = None,
        # 新增参数控制角色自动修复
        auto_fix_roles: bool = True
    ):
        """
        初始化自定义LLM模型，支持工具调用
        
        :param model_name: 模型名称
        :param base_url: API基础地址
        :param api_key: API密钥
        :param generate_config: 生成配置参数
        :param auto_fix_roles: 是否自动修复无效的消息角色
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.generate_config = generate_config or {"temperature": 0.0}
        self.tools: Optional[List[BaseTool]] = None  # 存储绑定的工具
        self.auto_fix_roles = auto_fix_roles  # 新增属性
        
        # 构建完整的API端点
        self.api_endpoint = f"{self.base_url}/chat/completions"
        
        # 角色映射表 - 解决'unknown variant'错误
        self.role_mapping = {
            "system": "system",
            "human": "user",    # 将human映射为API接受的user
            "ai": "assistant",  # 将ai映射为API接受的assistant
            "tool": "tool"
        }
        
        # 有效的角色列表
        self.valid_roles = ["system", "user", "assistant", "tool", "developer"]

    def bind_tools(self, tools: List[BaseTool]) -> "CustomLLM":
        """
        绑定工具到LLM，实现LangChain要求的接口
        
        :param tools: 工具列表
        :return: 绑定了工具的LLM实例
        """
        # 验证工具是否有效
        for tool in tools:
            if not tool.name or not tool.description:
                raise ValueError(f"工具 {tool} 必须包含名称和描述")
        
        self.tools = tools
        return self

    def _format_tools_for_prompt(self) -> str:
        """将工具格式化为模型可理解的提示文本"""
        if not self.tools:
            return "无可用工具"
            
        tool_descriptions = []
        for tool in self.tools:
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args
            }
            tool_descriptions.append(json.dumps(tool_info, ensure_ascii=False))
            
        return "可用工具:\n" + "\n".join(tool_descriptions) + "\n" + \
               "如果需要调用工具，请使用以下格式:\n" + \
               "<function_call>\n{\"name\": \"工具名称\", \"parameters\": {\"参数名\": \"参数值\"}}\n</function_call>"

    def _fix_message_roles(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """修复消息角色，确保符合API要求"""
        fixed_messages = []
        for msg in messages:
            # 确保消息包含必要字段
            if "role" not in msg:
                msg["role"] = "user"
            if "content" not in msg:
                msg["content"] = ""
                
            # 修复角色
            original_role = msg["role"]
            # 应用角色映射
            mapped_role = self.role_mapping.get(original_role, original_role)
            
            # 检查是否为有效角色
            if mapped_role not in self.valid_roles:
                if self.auto_fix_roles:
                    # 自动修复为user角色
                    mapped_role = "user"
                    print(f"警告: 无效的消息角色 '{original_role}' 已自动修复为 'user'")
                else:
                    raise ValueError(f"无效的消息角色: {original_role}, 预期值为 {self.valid_roles}")
                    
            msg["role"] = mapped_role
            fixed_messages.append(msg)
            
        return fixed_messages

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False
    ) -> Dict:
        """发送聊天完成请求，支持工具调用提示和角色验证"""
        try:
            # 复制消息以避免修改原始数据
            formatted_messages = messages.copy()
            
            # 如果有绑定的工具，在系统消息前添加工具说明
            if self.tools:
                tool_prompt = self._format_tools_for_prompt()
                # 检查是否已有系统消息
                has_system_message = any(msg.get("role") in ["system", "developer"] for msg in formatted_messages)
                
                if has_system_message:
                    # 更新现有系统消息
                    for i, msg in enumerate(formatted_messages):
                        if msg.get("role") in ["system", "developer"]:
                            formatted_messages[i]["content"] = tool_prompt + "\n" + msg["content"]
                            break
                else:
                    # 添加新的系统消息
                    formatted_messages.insert(0, {"role": "system", "content": tool_prompt})

            # 修复消息角色
            formatted_messages = self._fix_message_roles(formatted_messages)
            
            # 针对o系列模型特殊处理system角色为developer
            if self.model_name and self.model_name.startswith("o"):
                for msg in formatted_messages:
                    if msg["role"] == "system":
                        msg["role"] = "developer"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "messages": formatted_messages,
                "stream": stream,
                **self.generate_config
            }

            # 使用自定义编码器确保消息正确序列化
            response = requests.post(
                url=self.api_endpoint,
                headers=headers,
                data=json.dumps(data, cls=MessageJSONEncoder),
                timeout=60
            )

            if response.status_code == 200:
                return response.json()
            else:
                # 详细错误信息，便于调试
                error_details = {
                    "status_code": response.status_code,
                    "text": response.text,
                    "requested_model": self.model_name,
                    "message_count": len(formatted_messages)
                }
                raise Exception(f"请求失败: {json.dumps(error_details, ensure_ascii=False)}")
                
        except Exception as e:
            print(f"LLM请求错误: {str(e)}")
            raise

    def _convert_message_to_dict(self, msg: BaseMessage) -> Dict[str, Any]:
        """将LangChain消息转换为API所需的字典格式"""
        return {
            "role": msg.type,
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs
        }

    def invoke(self, input: List[BaseMessage]) -> BaseMessage:
        """实现LangChain的invoke接口，用于与agent集成"""
        try:
            # 转换消息格式为API所需格式
            messages = [self._convert_message_to_dict(msg) for msg in input]
            
            # 调用API获取结果
            response = self.create_chat_completion(messages)
            
            # 解析结果并返回LangChain消息格式
            return AIMessage(
                content=response["choices"][0]["message"]["content"],
                additional_kwargs=response.get("additional_kwargs", {})
            )
        except Exception as e:
            print(f"invoke方法错误: {str(e)}")
            # 可以根据需要添加重试逻辑
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
    