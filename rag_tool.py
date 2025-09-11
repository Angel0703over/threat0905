# -*- coding: utf-8 -*-
import math
import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
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
from predict_targets import predict_targets
from threat_assessment_bn import assess_threat
from loguru import logger
# ========== 配置常量 ==========
VECTOR_DBS = {
    "environment": "./resource/environment_db",
    "space": "./resource/space_db",
    "combat": "./resource/combat_db"
}
EMBEDDING_MODEL = "E:\AI\model\Qwen3-Embedding-0.6B"
DEFAULT_K = 1
ENV_ATTR_COUNT = 6
COMBAT_ATTR_COUNT = 6

# ========== 辅助类型 ==========
Position = Tuple[float, float, float]


def calculate_distance(pos1: Position, pos2: Position) -> float:
    """基于经纬度和高度（米）计算近似距离（公里）。"""
    lat1, lon1, alt1 = pos1
    lat2, lon2, alt2 = pos2
    lat_diff_km = abs(lat1 - lat2) * 111.0
    lon_diff_km = abs(lon1 - lon2) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    alt_diff_km = abs(alt1 - alt2) / 1000.0
    return math.sqrt(lat_diff_km**2 + lon_diff_km**2 + alt_diff_km**2)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def calculate_my_position_from_important(file_path: Optional[str]) -> Position:
    """从 important.csv 读取并返回我方装备中心点 (lat, lon, alt)。"""
    default = (20.0, 125.0, 0.0)
    if not file_path:
        return default
    p = Path(file_path)
    if not p.exists():
        logger.warning(f"important file not found: {file_path}, use default pos")
        return default

    lats, lons, alts = [], [], []
    try:
        with p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or len(row) < 7:
                    continue
                lat = safe_float(row[4], None)
                lon = safe_float(row[5], None)
                alt = safe_float(row[6], None)
                if lat is None or lon is None or alt is None:
                    continue
                lats.append(lat); lons.append(lon); alts.append(alt)
        if not lats:
            return default
        return (sum(lats) / len(lats), sum(lons) / len(lons), sum(alts) / len(alts))
    except Exception as e:
        logger.exception(f"读取 important.csv 失败: {e}")
        return default


def get_combat_radius_from_type(equip_type: str, type_ability: Optional[List[Dict]] = None) -> float:
    """从 type_ability 列表中提取作战半径（公里），若找不到则返回默认 1000 km。"""
    if not type_ability:
        return 1000.0
    for ability_dict in type_ability:
        if not isinstance(ability_dict, dict):
            continue
        # 匹配键（大小写容错）
        for k, desc in ability_dict.items():
            if k.lower() == equip_type.lower():
                # 尝试从描述中匹配 '作战半径约 xxx 公里' 或 '作战半径 xxx 公里'
                m = re.search(r'作战半径约?\s*([\d\.]+)\s*公里', str(desc))
                if m:
                    return safe_float(m.group(1), 1000.0)
    return 1000.0


# ========== LLM / 检索 ==========
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
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
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


# ========== 文本分级与解析 ==========
def classify_level(description: str) -> Dict[str, str]: 
    """对输入描述进行分级""" 
    llm = LLMClient(configs.QWEN3_LOCAL_CONFIG) 
    prompt = PromptLoader.get_prompt( prompt_name='rag/new_classification.prompt', comprehensive_threat_description=description ) 
    llm_response = llm.infer( 
        system_prompt='你是一个专业的海上多维度威胁分级专家，必须严格按照以下规则输出：\n' 
        '1. 仅输出包含"environment""combat"的JSON格式结果\n' 
        '2. environment对应"海洋环境威胁"，需从输入描述中提取所有海洋环境属性（如海水温度、风速、降水等）并分级，不遗漏；\n' 
        '3. combat对应"目标作战威胁"，需从输入描述中提取目标作战属性（如目标种类等）并分级，不遗漏；\n' 
        '4. 结果格式严格遵循：{"environment":"XXX","combat":"XXX"}', user_prompt=prompt, ) 
    extracted = response_extractor(llm_response) 
    return extracted


def parse_threat_scores(classification_result: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    将 classify_level 输出的自然语言描述转换为数值评分字典：
    返回 {'environment': {attr:score,...}, 'combat': {attr:score,...}}
    若没有对应条目则默认为 0.0
    """
    score_map = {
        "安全": 0.000,
        "一级威胁": 1.000,
        "二级威胁": 2.000
    }
    parsed_scores = {"environment": {}, "combat": {}}

    env_text = classification_result.get("environment", "") or ""
    env_items = [item.strip() for item in re.split('[,，;；\\n]', env_text) if item.strip()]
    for item in env_items:
        for level_str, score in score_map.items():
            if level_str in item:
                attr_name = item.replace(level_str, "").strip()
                if attr_name:
                    parsed_scores["environment"][attr_name] = score
                break
    combat_text = classification_result.get("combat", "") or ""
    combat_items = [item.strip() for item in re.split('[,，;；\\n]', combat_text) if item.strip()]
    for item in combat_items:
        for level_str, score in score_map.items():
            if level_str in item:
                attr_name = item.replace(level_str, "").strip()
                if attr_name:
                    parsed_scores["combat"][attr_name] = score
                break
    for attr in ["海水温度", "风速", "降水", "海洋流速", "水深", "电磁干扰"]:
        parsed_scores["environment"].setdefault(attr, 0.0)
    for attr in ["目标种类", "武器配置", "干扰能力", "协作模式", "后援系统", "作战续航"]:
        parsed_scores["combat"].setdefault(attr, 0.0)
    return parsed_scores


# ========== 核心：空间 / 威胁指标计算 ==========
def calculate_scores(
    envelope_data_1: List[Any],
    equipment_list_1: List[Dict],
    envelope_data_2: Optional[List[Any]] = None,
    feature_list: Optional[Dict[str, List[Any]]] = None,
    important_file: Optional[str] = None,
    type_ability: Optional[List[Dict]] = None,
    targets: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, float]:
    """
    根据接口数据计算若干威胁指标（保持原函数行为）。返回一个字典，包含各个空间/作战相关指标。
    """
    # 解析包络
    try:
        center_lat = safe_float(envelope_data_1[0])
        center_lon = safe_float(envelope_data_1[1])
        center_alt = safe_float(envelope_data_1[2])
        semi_x = safe_float(envelope_data_1[3])
        semi_y = safe_float(envelope_data_1[4])
        semi_z = safe_float(envelope_data_1[5])
    except Exception as e:
        logger.exception("解析 envelope_data_1 失败: " + str(e))
        # 返回一组默认 0 的指标，避免抛出
        return {
            "距离衰减系数": 0.0, "相对接近速度": 0.0, "目标密度指数": 0.0,
            "部署集中度": 0.0, "火力覆盖半径": 0.0, "传感器覆盖范围": 0.0,
            "杀伤链闭合概率": 0.0, "杀伤链响应速度": 0.0, "支援到达时间": 0.0
        }

    enemy_position: Position = (center_lat, center_lon, center_alt)
    my_position = calculate_my_position_from_important(important_file)

    # 距离衰减系数
    distance_km = calculate_distance(my_position, enemy_position)
    D_safe_max = max(semi_x, semi_y, semi_z) * 2.0
    D_mid = D_safe_max * 0.6 if D_safe_max > 0 else 0.0

    if D_safe_max <= 0:
        distance_score = 0.0
    elif distance_km >= D_safe_max:
        distance_score = 0.0
    elif D_mid <= distance_km < D_safe_max:
        distance_score = 1.0 * (D_safe_max - distance_km) / max((D_safe_max - D_mid), 1e-6)
    else:
        distance_score = 1.0 + 1.0 * (D_mid - distance_km) / max(D_mid, 1e-6)
    # 相对接近速度
    relative_speed_score = 0.0
    if feature_list and equipment_list_1:
        speeds = []
        for equip in equipment_list_1:
            equip_id = equip.get('id', '')
            if equip_id in feature_list:
                speed_mps = safe_float(feature_list[equip_id][0], 0.0)
                speeds.append(speed_mps)
        if speeds:
            avg_speed_ma = (sum(speeds) / len(speeds)) / 343.0
            if avg_speed_ma <= 0.8:
                relative_speed_score = 0.0
            elif 0.8 < avg_speed_ma <= 2.0:
                relative_speed_score = 1.0 * (avg_speed_ma - 0.8) / 1.2
            else:
                relative_speed_score = 1.0 + 1.0 * (avg_speed_ma - 2.0) / 1.0

    # 目标密度指数
    N_targets = len(equipment_list_1)
    V_ellipsoid = (4.0 / 3.0) * math.pi * max(semi_x, 0.0) * max(semi_y, 0.0) * max(semi_z, 0.0)
    density = N_targets / V_ellipsoid if V_ellipsoid > 0 else 0.0
    if density <= 0.01:
        density_score = 0.0
    elif 0.01 < density <= 0.05:
        density_score = 1.0 * (density - 0.01) / 0.04
    else:
        density_score = 1.0 + 1.0 * (density - 0.05) / 0.05

    # 部署集中度
    min_axis = min(semi_x, semi_y, semi_z)
    max_axis = max(semi_x, semi_y, semi_z)
    concentration = (min_axis / max_axis) if max_axis > 0 else 0.0
    if concentration >= 0.6:
        concentration_score = 0.0
    elif 0.2 <= concentration < 0.6:
        concentration_score = 1.0 * (0.6 - concentration) / 0.4
    else:
        concentration_score = 1.0 + 1.0 * (0.2 - concentration) / 0.2

    # 火力与传感器覆盖
    total_combat_radius = 0.0
    total_sensor_radius = 0.0
    count = 0
    for equip in equipment_list_1:
        eq_type = equip.get('type', '')
        combat_radius = get_combat_radius_from_type(eq_type, type_ability)
        sensor_radius = combat_radius * 0.3
        total_combat_radius += combat_radius
        total_sensor_radius += sensor_radius
        count += 1
    avg_combat_radius = total_combat_radius / count if count > 0 else 1000.0
    avg_sensor_radius = total_sensor_radius / count if count > 0 else 300.0

    if avg_combat_radius <= 1000:
        fire_radius_score = 0.0
    elif 1000 < avg_combat_radius <= 3000:
        fire_radius_score = 1.0 * (avg_combat_radius - 1000) / 2000
    else:
        fire_radius_score = 1.0 + 1.0 * (avg_combat_radius - 3000) / 2000

    if avg_sensor_radius <= 300:
        sensor_cover_score = 0.0
    elif 300 < avg_sensor_radius <= 900:
        sensor_cover_score = 1.0 * (avg_sensor_radius - 300) / 600
    else:
        sensor_cover_score = 1.0 + 1.0 * (avg_sensor_radius - 900) / 600

    # 杀伤链响应速度（到达时间）
    response_time_score = 0.0
    if feature_list and equipment_list_1:
        arrival_times = []
        for equip in equipment_list_1:
            equip_id = equip.get('id', '')
            if equip_id in feature_list:
                # 简化：使用编队中心作为装备位置
                distance_to_me = calculate_distance(enemy_position, my_position)
                speed_mps = safe_float(feature_list[equip_id][0])
                if speed_mps > 0:
                    time_sec = (distance_to_me * 1000.0) / speed_mps
                    arrival_times.append(time_sec)
        if arrival_times:
            avg_arrival_time_sec = sum(arrival_times) / len(arrival_times)
            if avg_arrival_time_sec <= 300:
                response_time_score = 0.0
            elif 300 < avg_arrival_time_sec <= 900:
                response_time_score = 1.0 * (avg_arrival_time_sec - 300) / 600
            else:
                response_time_score = 1.0 + 1.0 * (avg_arrival_time_sec - 900) / 900

    # 杀伤链闭合概率（基于 targets）
    kill_chain_score = 0.0
    total_probability = 0.0
    prob_count = 0
    if targets:
        for equip_id, target_dict in targets.items():
            for target_id, prob_info in (target_dict or {}).items():
                if isinstance(prob_info, (list, tuple)) and len(prob_info) >= 1:
                    try:
                        probability = safe_float(prob_info[0], None)
                        if probability is None:
                            continue
                        # 如果概率是 0-1 范围，放大到百分比
                        if 0 <= probability <= 1:
                            probability *= 100.0
                        if 0 <= probability <= 100:
                            total_probability += probability
                            prob_count += 1
                    except Exception:
                        continue
    avg_probability = (total_probability / prob_count) if prob_count > 0 else 0.0
    if avg_probability <= 30:
        kill_chain_score = 0.0
    elif 30 < avg_probability <= 70:
        kill_chain_score = 1.0 * (avg_probability - 30) / 40
    else:
        kill_chain_score = 1.0 + 1.0 * (avg_probability - 70) / 30

    # 支援到达时间（基于 envelope_data_2）
    support_time_score = 0.0
    if envelope_data_2 and feature_list:
        support_lat = safe_float(envelope_data_2[0]); support_lon = safe_float(envelope_data_2[1]); support_alt = safe_float(envelope_data_2[2])
        support_position = (support_lat, support_lon, support_alt)
        distance_between = calculate_distance(enemy_position, support_position)
        support_speeds = []
        for equip in equipment_list_1:
            equip_id = equip.get('id', '')
            if equip_id in feature_list:
                support_speeds.append(safe_float(feature_list[equip_id][0], 0.0))
        if support_speeds:
            avg_support_speed = sum(support_speeds) / len(support_speeds)
            if avg_support_speed > 0:
                T_support_minutes = (distance_between * 1000.0 / avg_support_speed) / 60.0
                if T_support_minutes >= 30:
                    support_time_score = 0.0
                elif 10 <= T_support_minutes < 30:
                    support_time_score = 1.0 * (30.0 - T_support_minutes) / 20.0
                else:
                    support_time_score = 1.0 + 1.0 * (10.0 - T_support_minutes) / 10.0
        else:
            support_time_score = 1.0
    else:
        support_time_score = 1.0

    return {
        "距离衰减系数": round(distance_score, 3),
        "相对接近速度": round(relative_speed_score, 3),
        "目标密度指数": round(density_score, 3),
        "部署集中度": round(concentration_score, 3),
        "火力覆盖半径": round(fire_radius_score, 3),
        "传感器覆盖范围": round(sensor_cover_score, 3),
        "杀伤链闭合概率": round(kill_chain_score, 3),
        "杀伤链响应速度": round(response_time_score, 3),
        "支援到达时间": round(support_time_score, 3)
    }


# ========== 三维汇总与外部接口 ==========
def all_threat_scores(
    threat_description: str,
    envelope_data_1: List[Any],
    equipment_list_1: List[Dict],
    envelope_data_2: Optional[List[Any]] = None,
    feature_list: Optional[Dict] = None,
    important_file: Optional[str] = None,
    type_ability: Optional[List[Dict]] = None,
    targets: Optional[Dict] = None
) -> Dict[str, float]:
    """
    整合所有威胁分数并返回字典（保留原接口与字段）。
    """
    classification_result = classify_level(threat_description)
    parsed_scores = parse_threat_scores(classification_result)
    env_total = sum(parsed_scores["environment"].values()) / ENV_ATTR_COUNT
    combat_total = sum(parsed_scores["combat"].values()) / COMBAT_ATTR_COUNT

    space_scores = calculate_scores(
        envelope_data_1=envelope_data_1,
        equipment_list_1=equipment_list_1,
        envelope_data_2=envelope_data_2,
        feature_list=feature_list,
        important_file=important_file,
        type_ability=type_ability,
        targets=targets
    )

    all_scores = {
        "海洋环境": round(env_total, 3),
        **space_scores,
        "作战个性维度": round(combat_total, 3)
    }
    return all_scores


def three_dimensions_scores(
    threat_description: str,
    envelope_data_1: List[Any],
    equipment_list_1: List[Dict],
    envelope_data_2: Optional[List[Any]] = None,
    feature_list: Optional[Dict] = None,
    important_file: Optional[str] = None,
    type_ability: Optional[List[Dict]] = None,
    targets: Optional[Dict] = None
) -> Dict[str, str]:
    """
    返回三维字符串形式的分数字段（format 与原实现兼容）。
    """
    classification_result = classify_level(threat_description)
    parsed_attr_scores = parse_threat_scores(classification_result)
    space_scores = calculate_scores(
        envelope_data_1=envelope_data_1,
        equipment_list_1=equipment_list_1,
        envelope_data_2=envelope_data_2,
        feature_list=feature_list,
        important_file=important_file,
        type_ability=type_ability,
        targets=targets
    )

    env_attrs_order = ["海水温度", "风速", "降水", "海洋流速", "水深", "电磁干扰"]
    env_str_parts = [f"{attr}: {parsed_attr_scores['environment'].get(attr, 0.0):.3f}" for attr in env_attrs_order]
    env_final_str = "，".join(env_str_parts)

    space_attrs_order = ["距离衰减系数", "相对接近速度", "目标密度指数", "部署集中度", "火力覆盖半径", "传感器覆盖范围"]
    space_str_parts = [f"{attr}: {space_scores.get(attr, 0.0):.3f}" for attr in space_attrs_order]
    space_final_str = "，".join(space_str_parts)

    combat_common_attrs_order = ["杀伤链响应速度", "支援到达时间", "杀伤链闭合概率"]
    combat_common_str_parts = [f"{attr}: {space_scores.get(attr, 0.0):.3f}" for attr in combat_common_attrs_order]

    combat_attrs_order = ["目标种类", "武器配置", "干扰能力", "协作模式", "后援系统", "作战续航"]
    combat_str_parts = [f"{attr}: {parsed_attr_scores['combat'].get(attr, 0.0):.3f}" for attr in combat_attrs_order]
    combat_str_parts.extend(combat_common_str_parts)
    combat_final_str = "，".join(combat_str_parts)

    return {"environment": env_final_str, "space": space_final_str, "combat": combat_final_str}


def process_shape_and_groups_data(shape, groups, new_tracks, important_file=None):
    envelope_data_1 = shape["1"][:6]

    equipment_list_1 = []
    for equipment_id, group in groups.items():
        if group[0] == "1":
            equipment_type = (
                "XQ-58A" if "xq-58a" in equipment_id.lower()
                else "F-35C" if "f-35c" in equipment_id.lower()
                else "CLIENT_AGM"
            )
            equipment_list_1.append({"id": equipment_id, "type": equipment_type})

    envelope_data_2 = shape.get("0", None)
    if envelope_data_2:
        envelope_data_2 = envelope_data_2[:6]

    targets = []
    try:
        if new_tracks and all(isinstance(t, (list, tuple)) and len(t) >= 4 for t in new_tracks):
            targets = predict_targets(new_tracks, [line.strip().split(",") for line in open(important_file, "r").readlines()])
        else:
            logger.warning(f"new_tracks 数据异常，跳过预测: {new_tracks}")
    except Exception as e:
        logger.warning(f"predict_targets 调用失败，targets 置为空: {e}")
        targets = []

    return envelope_data_1, equipment_list_1, envelope_data_2, targets



# ========== 对外主函数 ==========
def get_threat_score(case: str, shape: Dict[str, List[Any]], groups: Dict[str, List[Any]], feature_list: Dict[str, Any], type_ability: List[Dict], new_tracks: List[Any], important_file: str = "resource/important.csv"):
    """
    获取威胁分数主函数。
    返回 (code, result) —— code: "200"/"500"
    """
    try:
        envelope_data_1, equipment_list_1, envelope_data_2, targets = process_shape_and_groups_data(shape, groups, new_tracks, important_file)
        scores = all_threat_scores(
        threat_description=case,
        envelope_data_1=envelope_data_1,
        equipment_list_1=equipment_list_1,
        envelope_data_2=envelope_data_2,
        type_ability=type_ability,
        targets=targets,
        feature_list=feature_list,
        important_file=important_file,
    )
        result = assess_threat(scores)
        return "200", result
    except Exception as e:
        logger.warning("威胁评估错误，调用失败: " + str(e))
        return "500", (" ", "", " ")

# 在main函数中添加测试代码
if __name__ == "__main__":
    # 初始化提示词加载器
    PromptLoader.from_paths(['./prompt'])

    shape = {'1': ['20.5400122392778', '126.4232032368694', '9393.417828537711', '0.42233890435649385', '2.6670135277705325', '846.1221993471266', '120.05098931812435', '-0.09548310589427367', '-0.017377633888055362', ''], 
    '0': ['21.220592632003854', '125.32549421511914', '3000.172210156922', '0.2400210547588312', '0.5242212032378749', '1.3142938729525409', '-129.73782281887463', '-62.202612020986365', '57.02808282016993', '']}

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

    new_tracks = [
    ['MsgLocalTrackUpdate', '2237.438232421875', 'kj500', 'red', '20.771234398810741', '126.16854559951989', '8000.0', 'xq-58a_16', 'blue', 'XQ-58A', 'blue', '21.775905516407054', '126.70792001998572', '9638.863607518795', '', '', '', '', '', '', '', '', ''], 
    ['MsgLocalTrackUpdate', '2238.438232421875', 'kj500', 'red', '20.772322524784901', '126.16679126411078', '8000.0', 'xq-58a_16', 'blue', 'XQ-58A', 'blue', '21.777153740326757', '126.70590249898748', '9471.881198244488', '', '', '', '', '', '', '', '', ''], 
    ['MsgLocalTrackUpdate', '2239.438232421875', 'kj500', 'red', '20.773413424620415', '126.16503240233016', '8000.0', 'xq-58a_16', 'blue', 'XQ-58A', 'blue', '21.778402005049042', '126.70388483763307', '9304.90863841039', '', '', '', '', '', '', '', '', ''], 
    ['MsgLocalTrackUpdate', '2240.438232421875', 'kj500', 'red', '20.774502910460544', '126.16327576616713', '8000.0', 'xq-58a_16', 'blue', 'XQ-58A', 'blue', '21.779650310569313', '126.70186703591274', '9137.94592879013', '', '', '', '', '', '', '', '', ''], 
    ['MsgLocalTrackUpdate', '2241.438232421875', 'kj500', 'red', '20.775649718166203', '126.16142664840984', '8000.0', 'xq-58a_16', 'blue', 'XQ-58A', 'blue', '21.780898656882986', '126.69984909381692', '8970.993070154258', '', '', '', '', '', '', '', '', ''], 
    ['MsgLocalTrackUpdate', '2242.438232421875', 'kj500', 'red', '20.776739166965527', '126.15966996083252', '8000.0', 'xq-58a_16', 'blue', 'XQ-58A', 'blue', '21.782147043985475', '126.69783101133596', '8804.050063273087', '', '', '', '', '', '', '', '', '']]
    case = """
    西部某近岸海域，海水温度 10℃，25m/s 强风伴随 48 小时 180mm 降雨，海面流速 1.8m/s、水深 20m，中高频与超短波电磁干扰强度分别达 22V/m、18V/m。
    敌方通过运输机投放 30 架 X-61A 与 20 架 ALTIUS-600 无人机 / 蜂群，试图袭扰我方航母战斗群。无人机初始距航母 300km，抵近至 100km 时受干扰滞留，最终停留在 80km 处，巡航速度仅 0.3-0.6Ma，50 架无人机分散在 50km² 空域，密度低且编队松散，仅 X-61A 搭载少量炸药，受暴雨影响其光电传感器对我方舰艇覆盖率仅 35%。作战中，无人机依赖后方指令，干扰导致指令延迟 22 分钟，打击成功率降至 30%，后续支援需 40 分钟抵达，强风还使无人机续航从 4 小时缩至 2.5 小时。
    我方启动电子对抗系统，切断 60% 无人机通信，利用 800 米航线偏移布设警戒点，最终拦截 48 架，剩余 2 架因炸药引信受潮失效未造成损伤，防空资源消耗仅为常规的 15%。
    """

    code, score = get_threat_score(case,shape,groups,feature_list,type_ability,new_tracks,important_file="resource/important.csv")
    print("威胁评估："+code)
    print(score)