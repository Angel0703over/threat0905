import joblib
from collections import defaultdict
import random
import math

import numpy as np

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


def _calculate_heading_angle(tracks, window_size=5):
    """
    根据多个连续轨迹点计算当前装备的航向角（与正北方向的夹角，顺时针为正）

    :param tracks: 同一装备的轨迹点列表，按时间排序
    :param window_size: 用于计算航向角的轨迹点数量，默认使用最近5个点
    :return: 航向角（度），范围0-360度
    """
    # 确保有足够的轨迹点
    if len(tracks) < 2:
        return None  # 无法计算航向角

    # 使用最近的window_size个点进行计算
    tracks_to_use = tracks[-window_size:] if len(tracks) >= window_size else tracks

    # 提取经纬度并转换为弧度
    lats = np.radians([float(track[11]) for track in tracks_to_use])
    lons = np.radians([float(track[12]) for track in tracks_to_use])

    # 使用线性回归拟合轨迹方向
    # 将经纬度转换为平面坐标（以第一个点为原点）
    earth_radius = 6371000  # 地球半径（米）

    # 计算相对坐标
    x = earth_radius * (lons - lons[0]) * np.cos(lats[0])
    y = earth_radius * (lats - lats[0])

    # 线性回归计算方向向量
    if len(x) <= 1:
        return None

    # 计算最小二乘拟合
    coefficients = np.polyfit(x, y, 1)
    slope = coefficients[0]

    # 计算角度（弧度）
    angle_rad = np.arctan2(1, slope)  # 与y轴（正北）的夹角

    # 转换为度并调整到0-360度范围
    angle_deg = math.degrees(angle_rad)
    heading = (angle_deg + 360) % 360  # 确保在0-360度范围内

    return heading
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
        feature_list = {}
        for target, target_tracks in grouped_tracks.items():
            if len(target_tracks) < 2:
                continue
            selected_tracks = random.sample(target_tracks, 2)
            previous_track, current_track = selected_tracks
            velocity = _calculate_velocity(previous_track, current_track)
            heading = _calculate_heading_angle(target_tracks)
            altitude = float(current_track[13])
            velocity_d = float(velocity)
            heading_d = float(heading)
            # 使用检索增强进行装备类型匹配
            feature = [velocity_d, heading_d, altitude, f"当前装备的平台名称为：{target}, 该武器的飞行高度为：{altitude}m，飞行速度为：{velocity_d}m/s"]
            # feature_list.append(f"当前装备的平台名称为：{target}, 该武器的飞行高度为：{altitude}m，飞行速度为：{velocity_d}m/s")
            feature_list[target] = feature

        match_result = type_match(
            feature=[v[3] for v in feature_list.values()],
            type_list=json.loads(open('resource/type.json', encoding='utf-8').read()),
        )
        logger.info(match_result)
        logger.info(feature_list)
        for target, target_tracks in grouped_tracks.items():
            # print(feature_list)
            feature = feature_list[target]
            logger.info(feature)
            target_result = match_result[0][target]
            logger.info(target_result)
            detectee_type = target_result.get('type')
            description = target_result.get('description')
            selected_tracks = target_tracks[-1]
            logger.info(f"当前装备为{target},预测类型为：{detectee_type},阵营为{selected_tracks[8]},装备速度为{str(feature[0])}m/s,装备航向角为{feature[1]}度,最后出现的位置为: 纬度为{selected_tracks[11]}，经度为{selected_tracks[12]}，高度为{selected_tracks[13]}")
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

        return "200", (updated_tracks, type_ability, feature_list)
    except Exception as e:
        logger.warning("类型预测错误，调用失败"+ str(e))
        return "500", (" ", "", " ")
def type_match(feature:list, type_list:dict) -> [str,str]:
    """
    使用输入特征，结合检索增强进行类型匹配
    :param feature:武器特征
    :param type_list:类型匹配对象
    """
    result = query_single_db(
        question=PromptLoader.get_prompt(
            prompt_name='rag/detect_type.prompt',
            type_list=type_list,
            feature_list=feature
        ),
        db_name='knowledge_lib',
        db_path='./resource/knowledge_db'
    )
    output = result.get('output')
    json_body = response_extractor(output).get('data')
    print(json_body)
    # 返回装备类型和装备描述
    return json_body


