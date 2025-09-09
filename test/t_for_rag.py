# coding=utf-8
# @Time : 2025/9/8 19:38
# @Author : RoseLee
# @File : t_for_rag
# @Project : threat0905
# @Description :
import json

from rag_tool import query_single_db
from prompt.prompt_loader import PromptLoader
type_list = json.loads(open('../resource/type.json','r',encoding='utf-8').read())
PromptLoader.from_paths(['../prompt'])
feature = [
    '当前装备的平台名称为：xq-58a_18, 该武器的飞行高度为：9000.0000000977889m，飞行速度为：249.87981222841387m/s',
    '当前装备的平台名称为：f-35c_8_LRSAM_6, 该武器的飞行高度为：3001.1687104329467m，飞行速度为：313.79188759907333m/s',
    '当前装备的平台名称为：f-35c_4_LRSAM_6, 该武器的飞行高度为：2999.9934121165425m，飞行速度为：307.73794651921537m/s',
           ]
result = query_single_db(
    question=PromptLoader.get_prompt(
        prompt_name='test.prompt',
        type_list=type_list,
        feature_list=feature
    ),
    db_name='knowledge_lib',
    db_path='../resource/knowledge_db'
)
print(result.get('output'))