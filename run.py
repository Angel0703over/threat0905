from prompt.prompt_loader import PromptLoader
from pre_track import PredictTrack as PreTrack
from style import predict_style as PreStyle
from group_shape import group_shape as GroupShape, detectee_group as DetecteeGroup
from type import detectee_type as PreType
from rag_tool import get_threat_score
from typing import Union, List, Dict, Tuple

def input_check(tracks: list[list[str]]) -> None:
    for item in tracks:
        item.extend([""] * (23 - len(item)))


def detectee_type(tracks: List[List[str]]) -> Tuple[str, Union[str, Tuple[List[List[str]], List[Dict[str, str]]]]]:

    input_check(tracks)
    return PreType(tracks)


def detectee_group(tracks: list[list[str]]) ->Tuple[str, Union[str,Dict[str, List[str]]]]:

    return DetecteeGroup(tracks)


def group_shape(tracks: list[list[str]], groups: [str, list[str]]) -> Tuple[str, Union[str, Dict[str, List[str]]]]:
    return GroupShape(tracks, groups)


def predict_track(tracks: list[list[str]], duration: float, num_spans: int) -> Tuple[str, Union[str, List[List[str]]]]:
    input_check(tracks)
    return PreTrack(tracks, duration, num_spans)


def predict_style(tracks: list[list[str]], groups: [str, list[str]]) -> Tuple[str, Union[str, Dict[str, List[str]]]]:
    input_check(tracks)
    return PreStyle(tracks, groups)



def desp_trans(groups, shape, style, case):
    for k, v in groups.items():
        case += f"武器：{k}所对应的编组为第{v[0]}编组;"
    for k, v in shape.items():
        case += f"第{v[0]}编组中心位置为：纬度 {v[0]}, 经度 {v[1]}, 高度 {v[2]} 米， 编队空间范围为：纬轴 {v[3]} 米, 经轴 {v[4]} 米, 高轴 {v[5]} 米，编队方向姿态为：偏航 {v[6]} 度, 俯仰 {v[7]} 度, 滚转 {v[8]} 度 ; "
    for k, v in style.items():
        case += f"第{k}编组的作战样式为：类别{v}"
    return case

# 输入作战样例
combat_case = """
在东南部某近岸海域，海水温度为 32 摄氏度，风速 30m/s，过去 48 小时累计降水量 220mm，海洋流速 2.1m/s，该区域平均水深 15 米，电磁干扰中高频长、中、短波（0.1-30MHz）强度 28V/m，超短波（30-300MHz）强度 24V/m。敌方采用有人 / 无人协同压制打击，XQ-58A 隐身无人机作为 “忠诚僚机” 前置与 F-35C 隐身战斗机协同行动，试图对我方航母、驱逐舰与护卫舰及台岛封控区域实施打击。30m/s 的强风对战机飞行姿态控制提出极高要求，增加了战机编队协同难度，220mm 的大量降水导致能见度极低，影响战机雷达探测精度，且高温高湿环境加速战机设备老化，强电磁干扰则干扰了战机间的数据传输，我方借此环境，加强对空域的警戒监测，针对战机协同弱点和探测盲区，部署额外防空力量，抵御敌方打击。
"""
# 输入轨迹数据
track_file = "resource/example2.csv"
# 读取csv 按照逗号分割
with open(track_file, "r") as f:
    tracks = [line.strip().split(",") for line in f.readlines()]
PromptLoader.from_paths(['prompt'])

code, content = detectee_type(tracks)
print("装备类型判断："+code)
tracks, type_ability = content
print(type_ability)

code, groups = detectee_group(tracks)
print("编队判断："+code)
print(groups)

code, shape = group_shape(tracks, groups)
print("包络计算："+code)
print(shape)

code, new_tracks = predict_track(tracks, 10, 10)
print("轨迹预测："+code)
print(new_tracks)

code, style = predict_style(new_tracks, groups)
print("作战样式预测："+code)
print(style)

case = desp_trans(groups, shape, style, combat_case)
print(get_threat_score(case))

