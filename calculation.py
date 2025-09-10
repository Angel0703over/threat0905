import math
import re
import csv


def calculate_distance(pos1, pos2):
    lat1, lon1, alt1 = pos1
    lat2, lon2, alt2 = pos2
    lat_diff_km = abs(lat1 - lat2) * 111.0
    lon_diff_km = abs(lon1 - lon2) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    alt_diff_km = abs(alt1 - alt2) / 1000.0
    return math.sqrt(lat_diff_km**2 + lon_diff_km**2 + alt_diff_km**2)

def get_combat_radius_from_type(equip_type: str, type_ability: list) -> float:
    """
    从 type_ability 列表中，根据装备类型获取作战半径
    :param equip_type: 装备类型，如 'XQ-58A'
    :param type_ability: 装备能力列表，如 [{'XQ-58A': '...作战半径约3400公里...'}, ...]
    :return: 作战半径（公里），未找到则返回默认值1000.0
    """
    for ability_dict in type_ability:
        if equip_type in ability_dict:
            desc = ability_dict[equip_type]
            # 使用正则提取“作战半径约XXXX公里”
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
            next(reader)  # 跳过表头（如果有的话）
            for row in reader:
                if len(row) >= 7:  # 确保有足够列
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

def calculate_space_threat_scores(envelope_data_1, equipment_list_1,  envelope_data_2=None, feature_list=None, important_file=None, type_ability=None):
    """
    根据接口数据计算10个目标空间威胁指标的分数 [0, 2]
    """
    scores = {}


    # 1. 解析包络数据
    center_lat = float(envelope_data_1[0])
    center_lon = float(envelope_data_1[1])
    center_alt = float(envelope_data_1[2])
    semi_x = float(envelope_data_1[3])
    semi_y = float(envelope_data_1[4])
    semi_z = float(envelope_data_1[5])
    enemy_position = (center_lat, center_lon, center_alt)

    # 2. 计算我方位置 (从 important.csv)
    my_position = calculate_my_position_from_important(important_file)

    # 3. 距离衰减系数
    distance_km = calculate_distance(my_position, enemy_position)
    D_safe_max = max(semi_x, semi_y, semi_z) * 2.0
    D_mid = D_safe_max * 0.6

    if distance_km >= D_safe_max:
        scores["距离衰减系数"] = 0.0
    elif D_mid <= distance_km < D_safe_max:
        scores["距离衰减系数"] = 1.0 * (D_safe_max - distance_km) / (D_safe_max - D_mid)
    else:
        scores["距离衰减系数"] = 1.0 + 1.0 * (D_mid - distance_km) / D_mid

    # 4. 相对接近速度 (修改版：使用编队内所有装备的平均速度)
    relative_speed_score = 0.0
    if feature_list and equipment_list_1:
        speeds = []  # 存储编队内所有装备的速度 (m/s)
        for equip in equipment_list_1:
            equip_id = equip.get('id', '')
            if equip_id in feature_list:
                # feature_list[equip_id] = [speed, heading, altitude, desc]
                speed_mps = feature_list[equip_id][0]  # 第0个元素是速度 (m/s)
                speeds.append(speed_mps)
        
        if speeds:
            avg_speed_mps = sum(speeds) / len(speeds)  # 编队平均速度
            # 转换为马赫数 (1 Ma ≈ 343 m/s)
            avg_speed_ma = avg_speed_mps / 343.0

            # 套用分段线性公式 [0, 2]
            if avg_speed_ma <= 0.8:
                relative_speed_score = 0.0  # 安全
            elif 0.8 < avg_speed_ma <= 2.0:
                relative_speed_score = 1.0 * (avg_speed_ma - 0.8) / 1.2  # 一级威胁
            else:
                relative_speed_score = 1.0 + 1.0 * (avg_speed_ma - 2.0) / 1.0  # 二级威胁
    scores["相对接近速度"] = relative_speed_score

    # 5. 目标密度指数
    N_targets = len(equipment_list_1)
    V_ellipsoid = (4.0 / 3.0) * math.pi * semi_x * semi_y * semi_z
    density = N_targets / V_ellipsoid if V_ellipsoid > 0 else 0
    if density <= 0.01:
        scores["目标密度指数"] = 0.0
    elif 0.01 < density <= 0.05:
        scores["目标密度指数"] = 1.0 * (density - 0.01) / 0.04
    else:
        scores["目标密度指数"] = 1.0 + 1.0 * (density - 0.05) / 0.05

    # 6. 部署集中度
    min_axis = min(semi_x, semi_y, semi_z)
    max_axis = max(semi_x, semi_y, semi_z)
    concentration = min_axis / max_axis if max_axis > 0 else 0
    if concentration >= 0.6:
        scores["部署集中度"] = 0.0
    elif 0.2 <= concentration < 0.6:
        scores["部署集中度"] = 1.0 * (0.6 - concentration) / 0.4
    else:
        scores["部署集中度"] = 1.0 + 1.0 * (0.2 - concentration) / 0.2

    if type_ability is None:
        type_ability = []  
    # 7 & 8. 火力覆盖半径 & 传感器覆盖范围 (从 type_ability 动态解析)
    total_combat_radius = total_sensor_radius = 0.0
    count = 0

    for equip in equipment_list_1:
        equip_id = equip.get('id', '')
        equip_type = equip.get('type', '')  #
        
        # 从 type_ability 获取作战半径
        combat_radius = get_combat_radius_from_type(equip_type, type_ability)
        sensor_radius = combat_radius * 0.3  # 传感器半径设为作战半径的30%
        
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

    # 9. 杀伤链响应速度 (计算编队内每个装备到我方的平均到达时间)
    response_time_score = 0.0
    if feature_list and equipment_list_1:
        arrival_times = []  # 存储每个装备的到达时间（秒）
        for equip in equipment_list_1:
            equip_id = equip.get('id', '')
            if equip_id in feature_list:
                # 假设我们有每个装备的当前位置（这里简化用编队中心）
                # 更精确的做法是从轨迹预测中获取，但此处简化
                equip_position = enemy_position  # 简化：用编队中心代表装备位置
                distance_to_me = calculate_distance(equip_position, my_position)
                speed_mps = feature_list[equip_id][0]  # 获取速度
                
                if speed_mps > 0:
                    time_sec = (distance_to_me * 1000) / speed_mps  # 距离(m) / 速度(m/s)
                    arrival_times.append(time_sec)
        
        if arrival_times:
            avg_arrival_time_sec = sum(arrival_times) / len(arrival_times)
            # 使用分段线性公式
            if avg_arrival_time_sec <= 300:  # 5分钟
                response_time_score = 0.0
            elif 300 < avg_arrival_time_sec <= 900:  # 5-15分钟
                response_time_score = 1.0 * (avg_arrival_time_sec - 300) / 600
            else:  # > 15分钟
                response_time_score = 1.0 + 1.0 * (avg_arrival_time_sec - 900) / 900
    scores["杀伤链响应速度"] = response_time_score


    # 11. 支援到达时间 (计算两个编队中心点的距离 / 支援编队平均速度)
    support_time_score = 0.0
    if envelope_data_2 and feature_list:
        # 解析支援编队中心点
        support_lat = float(envelope_data_2[0])
        support_lon = float(envelope_data_2[1])
        support_alt = float(envelope_data_2[2])
        support_position = (support_lat, support_lon, support_alt)
        
        # 计算两个编队中心点距离
        distance_between_squadrons = calculate_distance(enemy_position, support_position)
        
        support_speeds = []
        for equip in equipment_list_1:  # 这里应该是支援编队的装备列表
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

if __name__ == "__main__":
    envelope_data_1 = ['21.26651667977164', '126.46599161493398', '7340.311431395218', '0.8547577937549865', '2.3525728749151047', '5298.322008715112']

    groups = {'xq-58a_16': ['1', 'None'], 'xq-58a_15': ['1', 'None'], 'xq-58a_10': ['1', 'None'], 'xq-58a_9': ['1', 'None'], 'xq-58a_6': ['1', 'None'], 'xq-58a_4': ['1', 'None'], 'xq-58a_2': ['1', 'None'], 'f-35c_1': ['1', 'None'], 'f-35c_2': ['1', 'None'], 'f-35c_4_LRSAM_1': ['0', 'None'], 'f-35c_3': ['1', 'None'], 'f-35c_5': ['1', 'None'], 'f-35c_6': ['1', 'None'], 'f-35c_7': ['1', 'None'], 'f-35c_8': ['1', 'None'], 'f-35c_9': ['1', 'None'], 'xq-58a_1': ['1', 'None'], 'xq-58a_3': ['1', 'None'], 'xq-58a_5': ['1', 'None'], 'xq-58a_7': ['1', 'None'], 'xq-58a_8': ['1', 'None'], 'xq-58a_11': ['1', 'None'], 'xq-58a_12': ['1', 'None'], 'xq-58a_13': ['1', 'None'], 'xq-58a_14': ['1', 'None'], 'xq-58a_17': ['1', 'None'], 'f-35c_8_LRSAM_1': ['0', 'None'], 'f-35c_8_LRSAM_3': ['0', 'None'], 'f-35c_4_LRSAM_2': ['0', 'None'], 'f-35c_4_LRSAM_3': ['0', 'None'], 'f-35c_4_LRSAM_4': ['0', 'None'], 'f-35c_10_LRSAM_2': ['0', 'None'], 'f-35c_10_LRSAM_3': ['0', 'None'], 'f-35c_10_LRSAM_4': ['0', 'None'], 'f-35c_10_LRSAM_5': ['0', 'None'], 'f-35c_10_LRSAM_6': ['0', 'None'], 'f-35c_10_LRSAM_1': ['0', 'None'], 'f-35c_4_LRSAM_6': ['0', 'None'], 'f-35c_4_LRSAM_5': ['0', 'None'], 'f-35c_8_LRSAM_5': ['0', 'None'], 'f-35c_8_LRSAM_2': ['0', 'None'], 'f-35c_8_LRSAM_4': ['0', 'None'], 'f-35c_8_LRSAM_6': ['0', 'None'], 'xq-58a_20': ['1', 'None'], 'xq-58a_19': ['1', 'None'], 'xq-58a_18': ['1', 'None']}

    new_tracks = [
        ['MsgLocalTrackUpdate', '2237.438232421875', 'kj500', 'red', '20.771234398810741', '126.16854559951989', '8000.0', 'xq-58a_16', 'blue', 'XQ-58A', 'blue', '21.775905516407054', '126.70792001998572', '9638.863607518795', '', '', '', '', '', '', '', '', ''],
        ['MsgLocalTrackUpdate', '2238.438232421875', 'kj500', 'red', '20.772322524784901', '126.16679126411078', '8000.0', 'xq-58a_16', 'blue', 'XQ-58A', 'blue', '21.777153740326757', '126.70590249898748', '9471.881198244488', '', '', '', '', '', '', '', '', '']
    ]

    envelope_data_2 = ['21.220592632803854', '125.32549421511914', '3008.172210156922']
    
    feature_list = {
        'xq-58a_16': [250.02798531358368, 123.60546680991257, 9000.000000087544, '当前装备的平台名称为：xq-58a_16...'],
        'xq-58a_15': [250.17739631674067, 128.72041476572122, 9000.000000074506, '当前装备的平台名称为：xq-58a_15...'],
    }
    
    type_ability = [
        {'XQ-58A': 'XQ-58A无人机飞行高度可达9000米，飞行速度约250m/s(900km/h)，作战半径约3400公里，续航时间超过8小时...'},
        {'F-35C': 'F-35C战斗机飞行高度可达10000米，飞行速度约278m/s(1000km/h)，作战半径约1100公里，续航时间约2.5小时...'},
        {'CLIENT_AGM': '防空导弹飞行高度3000米，飞行速度约311m/s(1120km/h)，具备中远程防空能力...'}
    ]

    # 动态生成装备列表（示例）
    equipment_list_1 = [
        {
            'id': equip_id,
            'type': 'XQ-58A' if 'xq-58a' in equip_id.lower() else 'F-35C' if 'f-35c' in equip_id.lower() else 'CLIENT_AGM'
        }
        for equip_id, group_info in groups.items()
        if group_info[0] == '1'
    ]

    # 调用威胁评分函数
    space_scores = calculate_space_threat_scores(
        envelope_data_1=envelope_data_1,
        equipment_list_1=equipment_list_1,
        envelope_data_2=envelope_data_2,
        feature_list=feature_list,
        important_file="./resource/important.csv",
        type_ability=type_ability  
    )

    for key, value in space_scores.items():
        print(f"{key}: {value:.3f}")