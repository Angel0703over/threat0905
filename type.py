import joblib
from collections import defaultdict
import random
import math
from rag_tool import query_single_db
from prompt.prompt_loader import PromptLoader
from util import response_extractor
import json
from loguru import logger
def _calculate_velocity(previous_track, current_track):
    """
    根据同一装备的两个不同轨迹点进行装备速度计算
    :param previous_track:前一时刻的装备轨迹
    :param current_track:当前时刻的装备轨迹
    """
    # 计算时间差（秒）
    time_diff = abs(float(previous_track[1]) - float(current_track[1]))
    # 计算垂直距离（米）
    vertical_distance = abs(float(previous_track[13]) - float(current_track[13]))

    # 将度转换为弧度
    lat1_rad = math.radians(float(previous_track[11]))
    lon1_rad = math.radians(float(previous_track[12]))
    lat2_rad = math.radians(float(current_track[11]))
    lon2_rad = math.radians(float(current_track[12]))

    # 使用Haversine公式计算水平距离
    earth_radius = 6371000
    a = math.sin((lat2_rad - lat1_rad) / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin((lon2_rad - lon1_rad) / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    horizontal_distance = earth_radius * c

    three_d_distance = math.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)

    velocity = three_d_distance / time_diff

    return velocity


def detectee_type(tracks):
    """
    进行装备类型预测
    :param tracks: 输入的轨迹序列
    """
    try:
        target_type_dict = {}
        grouped_tracks = defaultdict(list)
        # 按Target分组
        logger.info("按照装备分组")
        for track in tracks:
            grouped_tracks[track[7]].append(track)
        logger.info("使用rag进行装备类型与装备能力检索")
        for target, target_tracks in grouped_tracks.items():
            if len(target_tracks) < 2:
                continue
            selected_tracks = random.sample(target_tracks, 2)
            previous_track, current_track = selected_tracks
            velocity = _calculate_velocity(previous_track, current_track)
            altitude = float(current_track[13])
            velocity_d = float(velocity)
            # 使用检索增强进行装备类型匹配
            detectee_type, description = type_match(
                feature=f"当前装备的平台名称为：{target}, 该武器的飞行高度为：{altitude}m，飞行速度为：{velocity_d}m/s",
                type_list=json.loads(open('resource/type.json').read())
            )
            logger.info(f"装备{target}的预测类型为：{detectee_type}")
            target_type_dict[target] = {detectee_type: description}
        type_ability = []
        type_list = []
        updated_tracks = []
        logger.info("保存装备类型与对应描述")
        for track in tracks:
            # 为每个武器添加装备类型
            if track[7] in target_type_dict:
                track[9] = next(iter(target_type_dict[track[7]].keys()))
                updated_tracks.append(track)
                if next(iter(target_type_dict[track[7]].keys())) not in type_list:
                    # 保存装备类型对应的装备能力
                    type_ability.append(target_type_dict[track[7]])
                    type_list.append(next(iter(target_type_dict[track[7]].keys())))
        # 返回更新后的轨迹，装备类型对应的装备能力
        return "200", (updated_tracks, type_ability)
    except Exception as e:
        logger.warning("类型预测错误，调用失败"+ str(e))
        return "500", (" ", "")
def type_match(feature:str, type_list:dict) -> [str,str]:
    """
    使用输入特征，结合检索增强进行类型匹配
    :param feature:武器特征
    :param type_list:类型匹配对象
    """
    result = query_single_db(
        question=PromptLoader.get_prompt(
            prompt_name='rag/detect_type.prompt',
            type_list=type_list,
            feature=feature
        ),
        db_name='knowledge_lib',
        db_path='./resource/knowledge_db'
    )
    output = result.get('output')
    json_body = response_extractor(output)
    # 返回装备类型和装备描述
    return json_body.get("type"), json_body.get("description")


