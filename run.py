import json

from prompt.prompt_loader import PromptLoader
from pre_track import PredictTrack as PreTrack
from style import predict_style as PreStyle
from group_shape import group_shape as GroupShape, detectee_group as DetecteeGroup
from type import detectee_type as PreType
from rag_tool import get_threat_score
from sample import sample_analysis as PreSample
from typing import Union, List, Dict, Tuple
import time
import pandas as pd
def input_check(tracks: list[list[str]]) -> None:
    for item in tracks:
        item.extend([""] * (23 - len(item)))


def detectee_type(tracks: List[List[str]]) -> Tuple[str, Union[str, Tuple[List[List[str]], List[Dict[str, str]], Dict[str, List]]]]:

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

def sample_analysis(desp, type_ability, in_track, important) -> Tuple[str, str]:
    return PreSample(desp, type_ability, in_track, important)

def desp_trans(groups, shape, style, case):
    group = {}
    for k,(num, _) in groups.items():
        if num in group:
            group[num].append(k)
        else:
            group[num] = [k]
    for k, v in group.items():
        case += f"编队{k}包含武器装备为：{v};"
    for k, v in shape.items():
        case += f"第{k}编组中心位置为：纬度 {v[0]}, 经度 {v[1]}, 高度 {v[2]} 米， 编队空间范围为：纬轴 {v[3]} 米, 经轴 {v[4]} 米, 高轴 {v[5]} 米，编队方向姿态为：偏航 {v[6]} 度, 俯仰 {v[7]} 度, 滚转 {v[8]} 度 ; "
    style_list = json.loads(open('resource/style_list.json', encoding='utf-8').read())
    for k, v in style.items():
        case += f"第{k}编组的作战样式为：{style_list[v[0]]};"
    return case

# 输入作战样例
stat=time.time()
# 输入轨迹数据
track_file = "resource/example.csv"
# 读取csv 按照逗号分割
with open(track_file, "r") as f:
    tracks = [line.strip().split(",") for line in f.readlines()]
PromptLoader.from_paths(['prompt'])

code, content = detectee_type(tracks)
print(time.time()-stat)
print("装备类型判断："+code)
tracks, type_ability, feature_list = content
print(type_ability)
print(feature_list)

code, groups = detectee_group(tracks)
print("编队判断："+code)
print(groups)

code, shape = group_shape(tracks, groups)
print("包络计算："+code)
print(shape)

code, new_tracks = predict_track(tracks, 10, 10)
print("轨迹预测："+code)
print(new_tracks)
df = pd.DataFrame(new_tracks)
df.to_csv('./pre_tracks.csv', header=False, index=False)

code, style = predict_style(new_tracks, groups)
print("作战样式预测："+code)
print(style)

combat_case = ""
case = desp_trans(groups, shape, style, combat_case)
print(case)

code, sample = sample_analysis(case, type_ability, new_tracks, [line.strip().split(",") for line in open("resource/important.csv", "r").readlines()])
print(sample)

# print(get_threat_score(case))

