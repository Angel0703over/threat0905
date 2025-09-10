import re
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from prompt.prompt_loader import PromptLoader
from base import LLMClient
import configs
from configs import Dashscope_Api_Key
from util import response_extractor

# 定义案例库路径
VECTOR_DBS = {
    "environment": "./resource/environment_db",
    "space": "./resource/space_db",
    "combat": "./resource/combat_db"
}
EMBEDDING_MODEL = "E:\\AI\\model\\Qwen3-Embedding-0.6B"
DEFAULT_K = 1  # 每个数据库返回的相似案例数

def get_conversational_chain(tools: List, question: str) -> dict:
    """
    使用agent执行检索任务
    :param tools: agent可用的工具列表
    :param question:输入问题
    """
    try:
        llm = ChatOpenAI(
            model_name=configs.QWEN3_LOCAL_CONFIG.get("model"),
            base_url=configs.QWEN3_LOCAL_CONFIG.get("base_url"),
            api_key=configs.QWEN3_LOCAL_CONFIG.get("api_key")
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return agent_executor.invoke({"input": question})
    except Exception as e:
        print(f"对话链执行错误: {str(e)}")
        raise
def query_single_db(question: str, db_path: str, db_name: str):
    """
    执行案例库查找
    :param question:传入prompt
    :param db_name:案例库名，可自定义
    :param db_path:案例库路径
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db: FAISS = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever()
        retrieval_tool = create_retriever_tool(
            retriever,
            f"information_extractor",
            f"从数据库的案例中查询信息"
        )

        return get_conversational_chain([retrieval_tool], question)
    except Exception as e:
        error_msg: str = f"查询数据库'{db_name}'（路径：{db_path}）失败: {str(e)}"
        raise Exception(error_msg) from e

def get_db_similar_case(db_name: str, query_text: str, key: str) -> str:
    """
    从单个数据库中，根据指定的key（维度）获取最相似案例
    :param db_name: 数据库名称
    :param query_text: 原始查询文本
    :param key: 要查询的维度key
    """
    db_path = VECTOR_DBS[key]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    try:
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        key_specific_query = f"针对{key}维度的信息：{query_text}"
        retriever = db.as_retriever(search_kwargs={"k": DEFAULT_K})
        similar_docs = retriever.invoke(key_specific_query)
        return similar_docs[0].page_content if similar_docs else ""
    except Exception as e:
        print(f"获取{db_name}中{key}维度的相似案例失败: {str(e)}")
        return ""

def get_single_db_score(db_name: str, level_result: str, key: str) -> int:

    similar_case = get_db_similar_case(db_name, level_result, key)
    if not similar_case:
        raise ValueError(f"数据库{db_name}中未找到{key}维度的有效相似案例")
    prompt = PromptLoader.get_prompt(
        prompt_name=f'rag/{key}_retrieval.prompt',
        similar_case_content=similar_case
    )
    db_path = VECTOR_DBS[key]
    response = query_single_db(prompt, db_path, db_name)
    response_text = response.get("output", "").strip()
    if response_text == "ERROR: INVALID_SIMILAR_CASE":
        raise ValueError(f"数据库{db_name}中{key}维度的相似案例格式错误：{similar_case}")

    return extract_score_from_response(response)


def extract_score_from_response(response):
    """从响应中提取分数"""
    if isinstance(response, dict):
        response_text = response.get('output', '').strip()
    else:
        response_text = str(response).strip()
    if '<think>' in response_text and '</think>' in response_text:
        start = response_text.find('<think>') + len('<think>')
        end = response_text.find('</think>')
        think_content = response_text[start:end].strip()
        match = re.search(r'\b\d+\b', think_content)
        if match:
            return int(match.group())

    match = re.search(r'\b\d+\b', response_text)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"无法从响应中提取分数: {response_text}")


def classify_level(description: str) :
    """对输入描述进行分级"""
    llm = LLMClient(configs.QWEN3_LOCAL_CONFIG)
    prompt = PromptLoader.get_prompt(
            prompt_name='rag/classification.prompt',
            comprehensive_threat_description=description
    )

    llm_response = llm.infer(
            system_prompt='你是一个专业的海上多维度威胁分级专家，必须严格按照以下规则输出：\n'
                          '1. 仅输出包含"environment""space""combat"的JSON格式结果，不添加任何多余文字；\n'
                          '2. environment对应"海洋环境威胁"，需从输入描述中提取所有海洋环境属性（如海水温度、风速、降水等）并分级，不遗漏；\n'
                          '3. space对应"目标空间威胁"，需从输入描述中提取所有目标空间属性（如距离衰减系数、相对接近速度等）并分级，不遗漏；\n'
                          '4. combat对应"目标作战威胁"，需从输入描述中提取所有目标作战属性（如杀伤链响应速度、目标种类等）并分级，不遗漏；\n'
                          '5. 结果格式严格遵循：{"environment":"XXX","space":"XXX","combat":"XXX"}',
            user_prompt=prompt,
    )
    extracted = response_extractor(llm_response)
    return extracted

def get_threat_score(description: str) -> str:
    """获取威胁分数主函数（三个数据库分数平均值）"""
        # 步骤1：威胁分级
    level_result = classify_level(description=description)
    result_lines = [f"威胁分级：{level_result}"]

        # 步骤2：使用rag进行相关检索
    scores = {}
    for key in VECTOR_DBS.keys():
        db_name = VECTOR_DBS[key]
        score = get_single_db_score(db_name, level_result, key)
        scores[key] = score

        # 步骤3：加权计算
    total = sum(scores.values())
    average_score = round(total / len(scores))

    result_lines.append("各数据库分数：")
    for db_name, score in scores.items():
        result_lines.append(f"{db_name}分数：{score}")
    result_lines.append(f"最终平均分数：{average_score}")

    return "\n".join(result_lines)


if __name__ == '__main__':
    # 初始化提示词加载器
    PromptLoader.from_paths(['./prompt'])

    # 测试输入
    test_input = """
    西部某近岸海域，海水温度 10℃，25m/s 强风伴随 48 小时 180mm 降雨，海面流速 1.8m/s、水深 20m，中高频与超短波电磁干扰强度分别达 22V/m、18V/m。
    敌方通过运输机投放 30 架 X-61A 与 20 架 ALTIUS-600 无人机 / 蜂群，试图袭扰我方航母战斗群。无人机初始距航母 300km，抵近至 100km 时受干扰滞留，最终停留在 80km 处，巡航速度仅 0.3-0.6Ma，50 架无人机分散在 50km² 空域，密度低且编队松散，仅 X-61A 搭载少量炸药，受暴雨影响其光电传感器对我方舰艇覆盖率仅 35%。作战中，无人机依赖后方指令，干扰导致指令延迟 22 分钟，打击成功率降至 30%，后续支援需 40 分钟抵达，强风还使无人机续航从 4 小时缩至 2.5 小时。
    我方启动电子对抗系统，切断 60% 无人机通信，利用 800 米航线偏移布设警戒点，最终拦截 48 架，剩余 2 架因炸药引信受潮失效未造成损伤，防空资源消耗仅为常规的 15%。
    """
    print(get_threat_score(test_input))


