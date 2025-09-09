# coding=utf-8
# @Time : 2025/9/8 18:47
# @Author : RoseLee
# @File : sample
# @Project : threat0905
# @Description :作战样例分析
from base import LLMClient
from prompt.prompt_loader import PromptLoader
def sample_analysis(desp):
    """
    使用轨迹数据的分析结果生成对应的作战样例分析
    :param desp:根据轨迹数据的分析结果
    """
    result = LLMClient.infer(
        system_prompt='',
        user_prompt=PromptLoader.get_prompt('chat/sample_analysis.prompt',)
    )

