# coding=utf-8
# @Time : 2025/9/8 18:47
# @Author : RoseLee
# @File : sample
# @Project : threat0905
# @Description :作战样例分析
import configs
from base import LLMClient
from prompt.prompt_loader import PromptLoader
from util import response_extractor
from predict_targets import predict_targets
def sample_analysis(desp, type_ability, in_track, important):
    """
    使用轨迹数据的分析结果生成对应的作战样例分析
    :param desp:根据轨迹数据的分析结果
    """
    targets = predict_targets(in_track, important)
    print(targets)
    input()
    llm = LLMClient(llm_config=configs.QWEN3_LOCAL_CONFIG)
    result = llm.infer(
        system_prompt='',
        user_prompt=PromptLoader.get_prompt(
            prompt_name='chat/dmo_analysis.prompt',
            type = type_ability,
            description=desp
        )
    )
    print(result)
    x = response_extractor(result).get("result")
    print(x)



if __name__ == '__main__':
    PromptLoader.from_paths(['./prompt'])
    description = """
    编队1包含武器装备为：['xq-58a_16', 'xq-58a_15', 'xq-58a_10', 'xq-58a_9', 'xq-58a_6', 'xq-58a_4', 'xq-58a_2', 'f-35c_1', 'f-35c_2', 'f-35c_3', 'f-35c_5', 'f-35c_6', 'f-35c_7', 'f-35c_8', 'f-35c_9', 'xq-58a_1', 'xq-58a_3', 'xq-58a_5', 'xq-58a_7', 'xq-58a_8', 'xq-58a_11', 'xq-58a_12', 'xq-58a_13', 'xq-58a_14', 'xq-58a_17', 'xq-58a_20', 'xq-58a_19', 'xq-58a_18'];编队0包含武器装备为：['f-35c_4_LRSAM_1', 'f-35c_8_LRSAM_1', 'f-35c_8_LRSAM_3', 'f-35c_4_LRSAM_2', 'f-35c_4_LRSAM_3', 'f-35c_4_LRSAM_4', 'f-35c_10_LRSAM_2', 'f-35c_10_LRSAM_3', 'f-35c_10_LRSAM_4', 'f-35c_10_LRSAM_5', 'f-35c_10_LRSAM_6', 'f-35c_10_LRSAM_1', 'f-35c_4_LRSAM_6', 'f-35c_4_LRSAM_5', 'f-35c_8_LRSAM_5', 'f-35c_8_LRSAM_2', 'f-35c_8_LRSAM_4', 'f-35c_8_LRSAM_6'];第1编组中心位置为：纬度 20.5400122392778, 经度 126.4232032368694, 高度 9393.417828537711 米， 编队空间范围为：纬轴 0.42233890435649385 米, 经轴 2.6670135277705325 米, 高轴 846.1221993471266 米，编队方向姿态为：偏航 120.05098931812435 度, 俯仰 -0.09548310589427367 度, 滚转 -0.017377633888055362 度 ; 第0编组中心位置为：纬度 21.220592632003854, 经度 125.32549421511914, 高度 3000.172210156922 米， 编队空间范围为：纬轴 0.2400210547588312 米, 经轴 0.5242212032378749 米, 高轴 1.3142938729525409 米，编队方向姿态为：偏航 -129.73782281887463 度, 俯仰 -62.202612020986365 度, 滚转 57.02808282016993 度 ; 第1编组的作战样式为：有人/无人协同压制打击:由 XQ-58A 隐身无人机作为“忠诚僚机”前置与 F-35C 隐身战斗机协同对我方航母、驱逐舰与护卫舰实施打击，并抵近台岛对我台岛封控区域实施打击。;第0编组的作战样式为：有人/无人协同压制打击:由 XQ-58A 隐身无人机作为“忠诚僚机”前置与 F-35C 隐身战斗机协同对我方航母、驱逐舰与护卫舰实施打击，并抵近台岛对我台岛封控区域实施打击。;
    """
    type_ability = [{'XQ-58A': 'XQ-58A无人机飞行高度9000米，飞行速度250m/s，作战半径约3400公里，续航时间超过8小时，可挂载多种精确制导武器和小型无人机，具备隐身性能和自主作战能力'}, {'F-35C': 'F-35C战斗机飞行高度10000米，飞行速度279m/s，作战半径约1100公里，续航时间约2.5小时，可挂载AIM-120、AIM-9X等空对空导弹和JDAM精确制导炸弹，配备AN/APG-81有源相控阵雷达和AN/ASQ-239电子战系统'}, {'CLIENT_AGM': 'LRSAM远程空对空导弹飞行高度3000米，飞行速度318m/s，由F-35C战斗机挂载，具备超视距打击能力，采用主动雷达制导，可攻击远距离空中目标'}]
    with open('./pre_tracks.csv', 'r')as f:
        new_track = f.readlines()

    sample_analysis(description, type_ability,
                    in_track=[line.strip().split(",") for line in open("pre_tracks.csv", "r").readlines()],
                    important=[line.strip().split(",") for line in open("resource/important.csv", "r").readlines()]
                    )

