import math
import re
import csv
from typing import List, Dict, Tuple, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import configs
from util import response_extractor
from prompt.prompt_loader import PromptLoader
from base import LLMClient
from configs import Dashscope_Api_Key

# 定义案例库路径
VECTOR_DBS = {
    "environment": "./resource/environment_db",
    "space": "./resource/space_db",
    "combat": "./resource/combat_db"
}
EMBEDDING_MODEL = "E:\\AI\\model\\Qwen3-Embedding-0.6B"
DEFAULT_K = 1  # 每个数据库返回的相似案例数

# 空间威胁计算相关函数
def calculate_distance(pos1, pos2):
    lat1, lon1, alt1 = pos1
    lat2, lon2, alt2 = pos2
    lat_diff_km = abs(lat1 - lat2) * 111.0
    lon_diff_km = abs(lon1 - lon2) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    alt_diff_km = abs(alt1 - alt2) / 1000.0
    return math.sqrt(lat_diff_km**2 + lon_diff_km**2 + alt_diff_km**2)

def get_combat_radius_from_type(equip_type: str, type_ability: list) -> float:
    """从 type_ability 列表中，根据装备类型获取作战半径"""
    for ability_dict in type_ability:
        if equip_type in ability_dict:
            desc = ability_dict[equip_type]
            radius_match = re.search(r'作战半径约(\d+)公里', desc)
            if radius_match:
                return float(radius_match.group(1))
    return 1000.0  # 默认值
def calculate_my_position_from_important(file_path: str) -> tuple:
    """从 important.csv 计算我方装备中心点位置"""
    lats, lons, alts = [], [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 7:
                    try:
                        lat = float(row[4])
                        lon = float(row[5])
                        alt = float(row[6])
                        lats.append(lat)
                        lons.append(lon)
                        alts.append(alt)
                    except ValueError:
                        continue
        if lats:
            return (sum(lats)/len(lats), sum(lons)/len(lons), sum(alts)/len(alts))
        else:
            return (20.0, 125.0, 0.0)  # 默认值
    except Exception as e:
        print(f"读取 important.csv 失败: {e}")
        return (20.0, 125.0, 0.0)

def calculate_scores(
    envelope_data_1, 
    equipment_list_1,  
    envelope_data_2=None, 
    feature_list=None, 
    important_file=None, 
    type_ability=None
) -> Dict[str, float]:
    """根据接口数据计算10个目标空间威胁指标的分数"""
    scores = {}

    # 解析包络数据
    center_lat = float(envelope_data_1[0])
    center_lon = float(envelope_data_1[1])
    center_alt = float(envelope_data_1[2])
    semi_x = float(envelope_data_1[3])
    semi_y = float(envelope_data_1[4])
    semi_z = float(envelope_data_1[5])
    enemy_position = (center_lat, center_lon, center_alt)

    # 计算我方位置
    my_position = calculate_my_position_from_important(important_file)

    # 距离衰减系数
    distance_km = calculate_distance(my_position, enemy_position)
    D_safe_max = max(semi_x, semi_y, semi_z) * 2.0
    D_mid = D_safe_max * 0.6

    if distance_km >= D_safe_max:
        scores["距离衰减系数"] = 0.0
    elif D_mid <= distance_km < D_safe_max:
        scores["距离衰减系数"] = 1.0 * (D_safe_max - distance_km) / (D_safe_max - D_mid)
    else:
        scores["距离衰减系数"] = 1.0 + 1.0 * (D_mid - distance_km) / D_mid

    # 相对接近速度
    relative_speed_score = 0.0
    if feature_list and equipment_list_1:
        speeds = []
        for equip in equipment_list_1:
            equip_id = equip.get('id', '')
            if equip_id in feature_list:
                speed_mps = feature_list[equip_id][0]
                speeds.append(speed_mps)
        
        if speeds:
            avg_speed_ma = sum(speeds) / len(speeds) / 343.0  # 转换为马赫数
            if avg_speed_ma <= 0.8:
                relative_speed_score = 0.0
            elif 0.8 < avg_speed_ma <= 2.0:
                relative_speed_score = 1.0 * (avg_speed_ma - 0.8) / 1.2
            else:
                relative_speed_score = 1.0 + 1.0 * (avg_speed_ma - 2.0) / 1.0
    scores["相对接近速度"] = relative_speed_score

    # 目标密度指数
    N_targets = len(equipment_list_1)
    V_ellipsoid = (4.0 / 3.0) * math.pi * semi_x * semi_y * semi_z
    density = N_targets / V_ellipsoid if V_ellipsoid > 0 else 0
    if density <= 0.01:
        scores["目标密度指数"] = 0.0
    elif 0.01 < density <= 0.05:
        scores["目标密度指数"] = 1.0 * (density - 0.01) / 0.04
    else:
        scores["目标密度指数"] = 1.0 + 1.0 * (density - 0.05) / 0.05

    # 部署集中度
    min_axis = min(semi_x, semi_y, semi_z)
    max_axis = max(semi_x, semi_y, semi_z)
    concentration = min_axis / max_axis if max_axis > 0 else 0
    if concentration >= 0.6:
        scores["部署集中度"] = 0.0
    elif 0.2 <= concentration < 0.6:
        scores["部署集中度"] = 1.0 * (0.6 - concentration) / 0.4
    else:
        scores["部署集中度"] = 1.0 + 1.0 * (0.2 - concentration) / 0.2

    # 火力覆盖半径 & 传感器覆盖范围
    if type_ability is None:
        type_ability = []  
    total_combat_radius = total_sensor_radius = 0.0
    count = 0

    for equip in equipment_list_1:
        equip_type = equip.get('type', '')
        combat_radius = get_combat_radius_from_type(equip_type, type_ability)
        sensor_radius = combat_radius * 0.3
        
        total_combat_radius += combat_radius
        total_sensor_radius += sensor_radius
        count += 1

    avg_combat_radius = total_combat_radius / count if count > 0 else 1000
    avg_sensor_radius = total_sensor_radius / count if count > 0 else 300

    if avg_combat_radius <= 1000:
        scores["火力覆盖半径"] = 0.0
    elif 1000 < avg_combat_radius <= 3000:
        scores["火力覆盖半径"] = 1.0 * (avg_combat_radius - 1000) / 2000
    else:
        scores["火力覆盖半径"] = 1.0 + 1.0 * (avg_combat_radius - 3000) / 2000

    if avg_sensor_radius <= 300:
        scores["传感器覆盖范围"] = 0.0
    elif 300 < avg_sensor_radius <= 900:
        scores["传感器覆盖范围"] = 1.0 * (avg_sensor_radius - 300) / 600
    else:
        scores["传感器覆盖范围"] = 1.0 + 1.0 * (avg_sensor_radius - 900) / 600

    # 杀伤链响应速度
    response_time_score = 0.0
    if feature_list and equipment_list_1:
        arrival_times = []
        for equip in equipment_list_1:
            equip_id = equip.get('id', '')
            if equip_id in feature_list:
                equip_position = enemy_position  # 简化：用编队中心代表装备位置
                distance_to_me = calculate_distance(equip_position, my_position)
                speed_mps = feature_list[equip_id][0]
                
                if speed_mps > 0:
                    time_sec = (distance_to_me * 1000) / speed_mps
                    arrival_times.append(time_sec)
        
        if arrival_times:
            avg_arrival_time_sec = sum(arrival_times) / len(arrival_times)
            if avg_arrival_time_sec <= 300:
                response_time_score = 0.0
            elif 300 < avg_arrival_time_sec <= 900:
                response_time_score = 1.0 * (avg_arrival_time_sec - 300) / 600
            else:
                response_time_score = 1.0 + 1.0 * (avg_arrival_time_sec - 900) / 900
    scores["杀伤链响应速度"] = response_time_score

    # 支援到达时间
    support_time_score = 0.0
    if envelope_data_2 and feature_list:
        support_lat = float(envelope_data_2[0])
        support_lon = float(envelope_data_2[1])
        support_alt = float(envelope_data_2[2])
        support_position = (support_lat, support_lon, support_alt)
        
        distance_between_squadrons = calculate_distance(enemy_position, support_position)
        support_speeds = []
        
        for equip in equipment_list_1:
            equip_id = equip.get('id', '')
            if equip_id in feature_list:
                speed_mps = feature_list[equip_id][0]
                support_speeds.append(speed_mps)
        
        if support_speeds:
            avg_support_speed_mps = sum(support_speeds) / len(support_speeds)
            if avg_support_speed_mps > 0:
                T_support_minutes = (distance_between_squadrons * 1000 / avg_support_speed_mps) / 60
                
                if T_support_minutes >= 30:
                    support_time_score = 0.0
                elif 10 <= T_support_minutes < 30:
                    support_time_score = 1.0 * (30 - T_support_minutes) / 20
                else:
                    support_time_score = 1.0 + 1.0 * (10 - T_support_minutes) / 10
        else:
            support_time_score = 1.0
    else:
        support_time_score = 1.0
    scores["支援到达时间"] = support_time_score
    return scores


def get_conversational_chain(tools: List, question: str) -> dict:
    """使用agent执行检索任务"""
    try:
        llm = ChatOpenAI(
            model_name=configs.QWEN3_LOCAL_CONFIG.get("model"),
            base_url=configs.QWEN3_LOCAL_CONFIG.get("base_url"),
            api_key=configs.QWEN3_LOCAL_CONFIG.get("api_key")
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", ""),
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
    """执行案例库查找"""
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
    """从单个数据库中，根据指定的key（维度）获取最相似案例"""
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

def classify_level(description: str) -> Dict[str, str]:
    """对输入描述进行分级"""
    llm = LLMClient(configs.QWEN3_LOCAL_CONFIG)
    prompt = PromptLoader.get_prompt(
            prompt_name='rag/new_classification.prompt',
            comprehensive_threat_description=description
    )

    llm_response = llm.infer(
            system_prompt='你是一个专业的海上多维度威胁分级专家，必须严格按照以下规则输出：\n'
                          '1. 仅输出包含"environment""combat"的JSON格式结果\n'
                          '2. environment对应"海洋环境威胁"，需从输入描述中提取所有海洋环境属性（如海水温度、风速、降水等）并分级，不遗漏；\n'
                          '3. combat对应"目标作战威胁"，需从输入描述中提取目标作战属性（如目标种类等）并分级，不遗漏；\n'
                          '4. 结果格式严格遵循：{"environment":"XXX","combat":"XXX"}',
            user_prompt=prompt,
    )
    extracted = response_extractor(llm_response)
    return extracted

def parse_threat_scores(classification_result: dict) -> dict:
    """将 classify_level 输出的自然语言描述转换为数值评分"""
    score_map = {
        "安全": 0.000,
        "一级威胁": 1.000,
        "二级威胁": 2.000
    }
    parsed_scores = {
        "environment": {},
        "combat": {}
    }
    env_text = classification_result.get("environment", "")
    if env_text:
        env_items = [item.strip() for item in env_text.split('，') if item.strip()]
        for item in env_items:
            for level_str, score in score_map.items():
                if level_str in item:
                    attr_name = item.replace(level_str, "").strip()
                    parsed_scores["environment"][attr_name] = score
                    break
    combat_text = classification_result.get("combat", "")
    if combat_text:
        combat_items = [item.strip() for item in combat_text.split('，') if item.strip()]
        for item in combat_items:
            for level_str, score in score_map.items():
                if level_str in item:
                    attr_name = item.replace(level_str, "").strip()
                    parsed_scores["combat"][attr_name] = score
                    break
    return parsed_scores

# def get_threat_score(description: str) -> str:
#     """获取威胁分数主函数（三个数据库分数平均值）"""
#     # 步骤1：威胁分级
#     level_result = classify_level(description=description)
#     result_lines = [f"威胁分级：{level_result}"]

#     # 步骤2：使用rag进行相关检索
#     scores = {}
#     for key in VECTOR_DBS.keys():
#         db_name = VECTOR_DBS[key]
#         score = get_single_db_score(db_name, level_result, key)
#         scores[key] = score

#     # 步骤3：加权计算
#     total = sum(scores.values())
#     average_score = round(total / len(scores))

#     result_lines.append("各数据库分数：")
#     for db_name, score in scores.items():
#         result_lines.append(f"{db_name}分数：{score}")
#     result_lines.append(f"最终平均分数：{average_score}")

#     return "\n".join(result_lines)


def all_threat_scores(
    threat_description: str,
    envelope_data_1: List[str],
    equipment_list_1: List[Dict],
    envelope_data_2: List[str] = None,
    feature_list: Dict = None,
    important_file: str = None,
    type_ability: List[Dict] = None
) -> Dict[str, float]:
    """
    整合所有威胁分数，返回包含以下内容的字典：
    1. 海洋环境总分
    2. 作战个性维度总分
    3. 空间威胁指标分数
    """
    # 1. 计算海洋环境和作战个性维度分数
    classification_result = classify_level(threat_description)
    parsed_scores = parse_threat_scores(classification_result)
    env_total = sum(parsed_scores["environment"].values())/6
    combat_total = sum(parsed_scores["combat"].values())/6
    # 2. 计算10个威胁指标分数
    space_scores = calculate_scores(
        envelope_data_1=envelope_data_1,
        equipment_list_1=equipment_list_1,
        envelope_data_2=envelope_data_2,
        feature_list=feature_list,
        important_file=important_file,
        type_ability=type_ability
    )
    # 3. 整合为一个字典
    all_scores = {
        "海洋环境": round(env_total, 3),  # 示例中保留1位小数，可调整
        "距离衰减系数": space_scores["距离衰减系数"],
        "相对接近速度": space_scores["相对接近速度"],
        "目标密度指数": space_scores["目标密度指数"],
        "部署集中度": space_scores["部署集中度"],
        "火力覆盖半径": space_scores["火力覆盖半径"],
        "传感器覆盖范围": space_scores["传感器覆盖范围"],
        "杀伤链响应速度": space_scores["杀伤链响应速度"],
        "支援到达时间": space_scores["支援到达时间"],
        "作战个性维度": round(combat_total, 3)  #
    }
    return all_scores

def three_dimensions_scores(
    threat_description: str,
    envelope_data_1: List[str],
    equipment_list_1: List[Dict],
    envelope_data_2: List[str] = None,
    feature_list: Dict = None,
    important_file: str = None,
    type_ability: List[Dict] = None
) -> Dict[str, str]:
    """
    按“海洋环境/空间威胁/作战个性”三个维度，输出“指标名: 分数”格式的字典
    返回格式严格匹配需求：
    {
        "environment": "海水温度: X.XXX，风速: X.XXX，降水: X.XXX，海洋流速: X.XXX，水深: X.XXX，电磁干扰: X.XXX",
        "space": "距离衰减系数: X.XXX，相对接近速度: X.XXX，目标密度指数: X.XXX，部署集中度: X.XXX，火力覆盖半径: X.XXX，传感器覆盖率: X.XXX",
        "combat": "杀伤链响应速度: X.XXX，支援到达时间: X.XXX，目标种类: X.XXX，武器配置: X.XXX，干扰能力: X.XXX，协作模式: X.XXX，后援系统: X.XXX，作战续航: X.XXX"
    }
    """
    classification_result = classify_level(threat_description)
    parsed_attr_scores = parse_threat_scores(classification_result)  
    space_scores = calculate_scores(
        envelope_data_1=envelope_data_1,
        equipment_list_1=equipment_list_1,
        envelope_data_2=envelope_data_2,
        feature_list=feature_list,
        important_file=important_file,
        type_ability=type_ability
    )
    # 2. 构建各维度的“指标
    # 2.1 海洋环境维度
    env_attrs_order = ["海水温度", "风速", "降水", "海洋流速", "水深", "电磁干扰"]
    env_str_parts = [
        f"{attr}: {parsed_attr_scores['environment'][attr]:.3f}" 
        for attr in env_attrs_order
    ]
    env_final_str = "，".join(env_str_parts)

    # 2.2 空间威胁维度
    space_attrs_order = [
        "距离衰减系数", "相对接近速度", "目标密度指数", 
        "部署集中度", "火力覆盖半径", "传感器覆盖范围"
    ]
    space_str_parts = []
    for attr in space_attrs_order:
        score = space_scores.get(attr, 0.0)
        # 替换指标名为需求中的“传感器覆盖率”
        display_attr = "传感器覆盖率" if attr == "传感器覆盖范围" else attr
        space_str_parts.append(f"{display_attr}: {score:.3f}")
    space_final_str = "，".join(space_str_parts)

    # 2.2 作战共性维度
    combat_common_attrs_order = [
        "杀伤链响应速度", "支援到达时间"
    ]
    combat_common_str_parts = []
    for attr in combat_common_attrs_order:
        score = space_scores.get(attr, 0.0)
        # 替换指标名为需求中的“传感器覆盖率”
        display_attr = "传感器覆盖率" if attr == "传感器覆盖范围" else attr
        combat_common_str_parts.append(f"{display_attr}: {score:.3f}")
    
    # 2.3 作战个性维度
    combat_attrs_order = [
        "目标种类", "武器配置", 
        "干扰能力", "协作模式", "后援系统", "作战续航"
    ]
    combat_str_parts = [
        f"{attr}: {parsed_attr_scores['combat'][attr]:.3f}" 
        for attr in combat_attrs_order
    ]
    combat_str_parts.extend(combat_common_str_parts)
    combat_final_str = "，".join(combat_str_parts)

    # 3. 组装最终返回字典
    return {
        "environment": env_final_str,
        "space": space_final_str,  
        "combat": combat_final_str
    }


if __name__ == "__main__":
    # 初始化提示词加载器
    PromptLoader.from_paths(['./prompt'])
    
    # 测试空间威胁评分计算
    envelope_data_1 = ['21.26651667977164', '126.46599161493398', '7340.311431395218', 
                      '0.8547577937549865', '2.3525728749151047', '5298.322008715112']
    
    groups = {'xq-58a_16': ['1', 'None'], 'xq-58a_15': ['1', 'None'], 'xq-58a_10': ['1', 'None'],
             'xq-58a_9': ['1', 'None'], 'xq-58a_6': ['1', 'None'], 'xq-58a_4': ['1', 'None'],
             'xq-58a_2': ['1', 'None'], 'f-35c_1': ['1', 'None'], 'f-35c_2': ['1', 'None'],
             'f-35c_3': ['1', 'None'], 'f-35c_5': ['1', 'None'], 'f-35c_6': ['1', 'None'],
             'f-35c_7': ['1', 'None'], 'f-35c_8': ['1', 'None'], 'f-35c_9': ['1', 'None'],
             'xq-58a_1': ['1', 'None'], 'xq-58a_3': ['1', 'None'], 'xq-58a_5': ['1', 'None'],
             'xq-58a_7': ['1', 'None'], 'xq-58a_8': ['1', 'None'], 'xq-58a_11': ['1', 'None'],
             'xq-58a_12': ['1', 'None'], 'xq-58a_13': ['1', 'None'], 'xq-58a_14': ['1', 'None'],
             'xq-58a_17': ['1', 'None'], 'xq-58a_20': ['1', 'None'], 'xq-58a_19': ['1', 'None'],
             'xq-58a_18': ['1', 'None']}
    
    feature_list = {
        'xq-58a_16': [250.02798531358368, 123.60546680991257, 9000.000000087544, '当前装备的平台名称为：xq-58a_16...'],
        'xq-58a_15': [250.17739631674067, 128.72041476572122, 9000.000000074506, '当前装备的平台名称为：xq-58a_15...'],
    }
    
    type_ability = [
        {'XQ-58A': 'XQ-58A无人机飞行高度可达9000米，飞行速度约250m/s(900km/h)，作战半径约3400公里，续航时间超过8小时...'},
        {'F-35C': 'F-35C战斗机飞行高度可达10000米，飞行速度约278m/s(1000km/h)，作战半径约1100公里，续航时间约2.5小时...'},
        {'CLIENT_AGM': '防空导弹飞行高度3000米，飞行速度约311m/s(1120km/h)，具备中远程防空能力...'}
    ]
    
    equipment_list_1 = [
        {
            'id': equip_id,
            'type': 'XQ-58A' if 'xq-58a' in equip_id.lower() else 'F-35C' if 'f-35c' in equip_id.lower() else 'CLIENT_AGM'
        }
        for equip_id, group_info in groups.items()
        if group_info[0] == '1'
    ]
    envelope_data_2 = ['21.220592632803854', '125.32549421511914', '3008.172210156922']

    test_input = """
    西部某近岸海域，海水温度 10℃，25m/s 强风伴随 48 小时 180mm 降雨，海面流速 1.8m/s、水深 20m，中高频与超短波电磁干扰强度分别达 22V/m、18V/m。
    敌方通过运输机投放 30 架 X-61A 与 20 架 ALTIUS-600 无人机 / 蜂群，试图袭扰我方航母战斗群。无人机初始距航母 300km，抵近至 100km 时受干扰滞留，最终停留在 80km 处，巡航速度仅 0.3-0.6Ma，50 架无人机分散在 50km² 空域，密度低且编队松散，仅 X-61A 搭载少量炸药，受暴雨影响其光电传感器对我方舰艇覆盖率仅 35%。作战中，无人机依赖后方指令，干扰导致指令延迟 22 分钟，打击成功率降至 30%，后续支援需 40 分钟抵达，强风还使无人机续航从 4 小时缩至 2.5 小时。
    我方启动电子对抗系统，切断 60% 无人机通信，利用 800 米航线偏移布设警戒点，最终拦截 48 架，剩余 2 架因炸药引信受潮失效未造成损伤，防空资源消耗仅为常规的 15%。
    """

    all_scores = all_threat_scores(
        threat_description=test_input,
        envelope_data_1=envelope_data_1,
        equipment_list_1=equipment_list_1,
        envelope_data_2=envelope_data_2,
        feature_list=feature_list,
        important_file="./resource/important.csv",
        type_ability=type_ability
    )

    

    scores = three_dimensions_scores(
        threat_description=test_input,
        envelope_data_1=envelope_data_1,
        equipment_list_1=equipment_list_1,
        envelope_data_2=envelope_data_2,
        feature_list=feature_list,
        important_file="./resource/important.csv",
        type_ability=type_ability
    )
