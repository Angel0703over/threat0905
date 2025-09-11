# new/threat_assessment_bn.py
# coding=utf-8
# @Time : 2025/9/10 19:19
# @Author : RoseLee
# @File : threat_assessment_bn
# @Project : 贝叶斯
# @Description :

import itertools
import json
from loguru import logger
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import copy

# ========== 威胁评估网络结构定义 ==========
DEFAULT_THREAT_PROB = 0.5  # 节点默认成功概率

# 阶段结构定义
THREAT_PHASE_STRUCTURE = {
    "环境威胁": ["环境威胁分数"],
    "空间威胁": [
        "距离衰减系数", "相对接近速度", "火力覆盖半径",
        "传感器覆盖范围", "目标密度指数", "部署集中度"
    ],
    "作战威胁": [
        "杀伤链闭合概率", "杀伤链响应速度",
        "支援到达时间", "作战装备威胁"
    ],
    "威胁类别": ["环境威胁类别", "空间威胁类别", "作战威胁类别"],
    "综合评估": ["威胁评估分数"]
}

# 边连接定义 - 三层结构
THREAT_EDGE_RELATIONS = [
    # 第一层到第二层
    ("环境威胁分数", "环境威胁类别"),
    ("距离衰减系数", "空间威胁类别"),
    ("相对接近速度", "空间威胁类别"),
    ("火力覆盖半径", "空间威胁类别"),
    ("传感器覆盖范围", "空间威胁类别"),
    ("目标密度指数", "空间威胁类别"),
    ("部署集中度", "空间威胁类别"),
    ("杀伤链闭合概率", "作战威胁类别"),
    ("杀伤链响应速度", "作战威胁类别"),
    ("支援到达时间", "作战威胁类别"),
    ("作战装备威胁", "作战威胁类别"),

    # 第二层到第三层
    ("环境威胁类别", "威胁评估分数"),
    ("空间威胁类别", "威胁评估分数"),
    ("作战威胁类别", "威胁评估分数")
]

# 节点默认概率配置
DEFAULT_THREAT_NODE_PROBS = {
    "环境威胁": {"环境威胁分数": 0.5},
    "空间威胁": {
        "距离衰减系数": 0.5, "相对接近速度": 0.5, "火力覆盖半径": 0.5,
        "传感器覆盖范围": 0.5, "目标密度指数": 0.5, "部署集中度": 0.5
    },
    "作战威胁": {
        "杀伤链闭合概率": 0.5, "杀伤链响应速度": 0.5,
        "支援到达时间": 0.5, "作战装备威胁": 0.5
    },
    "威胁类别": {
        "环境威胁类别": 0.5, "空间威胁类别": 0.5, "作战威胁类别": 0.5
    },
    "综合评估": {"威胁评估分数": 0.0}
}

# 指标映射字典（将输入的描述映射到网络节点名）
INDICATOR_MAPPING = {
    "海洋环境": "环境威胁分数",
    "距离衰减系数": "距离衰减系数",
    "相对接近速度": "相对接近速度",
    "火力覆盖半径": "火力覆盖半径",
    "传感器覆盖范围": "传感器覆盖范围",
    "目标密度指数": "目标密度指数",
    "部署集中度": "部署集中度",
    "杀伤链闭合概率": "杀伤链闭合概率",
    "杀伤链响应速度": "杀伤链响应速度",
    "支援到达时间": "支援到达时间",
    "作战个性维度": "作战装备威胁"
}

# 末端节点权重配置
INDICATOR_WEIGHTS = {
    "环境威胁分数": 0.3,
    "距离衰减系数": 0.30,
    "相对接近速度": 0.30,
    "火力覆盖半径": 0.30,
    "传感器覆盖范围": 0.30,
    "目标密度指数": 0.11,
    "部署集中度": 0.11,
    "杀伤链闭合概率": 0.4,
    "杀伤链响应速度": 0.4,
    "支援到达时间": 0.07,
    "作战装备威胁": 0.5
}

# 中间层节点权重配置
CATEGORY_WEIGHTS = {
    "环境威胁类别": 0.30,
    "空间威胁类别": 0.50,
    "作战威胁类别": 0.50
}


class ThreatAssessmentBayesianModel:
    def __init__(self,
                 phase_structure: dict = None,
                 edges: list = None,
                 cpds: list = None,
                 cpd_generator: callable = None,
                 default_prob: float = None,
                 node_probs: dict = None):
        """
        威胁评估贝叶斯网络模型

        参数：
        - phase_structure: 阶段结构定义字典，格式 {阶段名: [节点列表]}
        - edges: 边关系列表，格式 [(父节点, 子节点)]
        - cpds: 预定义的CPD列表（优先级最高）
        - cpd_generator: 自定义CPD生成函数
        - default_prob: 全局默认成功概率（覆盖DEFAULT_THREAT_PROB）
        - node_probs: 节点概率配置，格式 {阶段名: {节点名: 概率值}}
        """
        # 初始化网络结构参数
        self.model = DiscreteBayesianNetwork()
        self.phase_structure = copy.deepcopy(phase_structure) if phase_structure else copy.deepcopy(
            THREAT_PHASE_STRUCTURE)
        self.edges = edges.copy() if edges else copy.deepcopy(THREAT_EDGE_RELATIONS)
        self.default_prob = default_prob if default_prob is not None else DEFAULT_THREAT_PROB

        # 处理节点概率配置
        self.node_probs = self._process_node_probs(node_probs)

        # CPD相关配置
        self.cpds = cpds
        self.cpd_generator = cpd_generator or self._create_weighted_cpd

        self.node_prob_cache = {}  # 缓存所有节点推理结果

        # 结构验证与构建
        self._validate_structure()
        self._build_structure()
        self._add_cpds()

    def _process_node_probs(self, node_probs):
        """处理节点概率配置"""
        processed_probs = {}
        if not node_probs:
            return {}

        # 获取第一个值用于判断类型
        first_value = next(iter(node_probs.values()))

        # 判断是否是阶段结构配置
        if isinstance(first_value, dict):
            for phase, node_dict in node_probs.items():
                # 验证阶段存在性
                if phase not in self.phase_structure:
                    raise ValueError(f"无效阶段名称: '{phase}'，可用阶段：{list(self.phase_structure.keys())}")

                # 验证节点存在性
                valid_nodes = self.phase_structure[phase]
                for node, prob in node_dict.items():
                    if node not in valid_nodes and node not in ["环境威胁类别", "空间威胁类别", "作战威胁类别"]:
                        raise ValueError(f"阶段 '{phase}' 中不存在节点: '{node}'，有效节点：{valid_nodes}")
                    processed_probs[node] = prob

        return processed_probs

    def _validate_structure(self):
        """验证网络结构完整性"""
        # 收集所有有效节点
        all_nodes = [n for nodes in self.phase_structure.values() for n in nodes]
        # 添加中间类别节点
        all_nodes.extend(["环境威胁类别", "空间威胁类别", "作战威胁类别"])

        # 验证边关系中的节点是否存在
        for parent, child in self.edges:
            if parent not in all_nodes:
                raise ValueError(f"边关系中包含未定义的父节点: '{parent}'")
            if child not in all_nodes:
                raise ValueError(f"边关系中包含未定义的子节点: '{child}'")

        # 验证阶段定义无重复节点
        node_count = {}
        for nodes in self.phase_structure.values():
            for node in nodes:
                node_count[node] = node_count.get(node, 0) + 1
        duplicates = [k for k, v in node_count.items() if v > 1]
        if duplicates:
            raise ValueError(f"检测到重复节点定义: {duplicates}")

    def _build_structure(self):
        """构建网络拓扑结构"""
        self.model.add_edges_from(self.edges)

    def _get_node_prob(self, node):
        """获取节点概率（优先级：node_probs > default_prob > DEFAULT_THREAT_PROB）"""
        return self.node_probs.get(node, self.default_prob)

    def _create_weighted_cpd(self, node):
        """生成带权重的CPD（条件概率分布）"""
        parents = self.model.get_parents(node)
        base_prob = self._get_node_prob(node)  # 获取基准概率

        # 生成父节点状态组合
        parent_states = list(itertools.product([0, 1], repeat=len(parents)))

        values = []
        for state in parent_states:
            if len(parents) == 0:
                # 根节点
                p_success = base_prob
            elif node in ["环境威胁类别", "空间威胁类别", "作战威胁类别"]:
                # 类别节点：根据子指标加权平均
                weighted_sum = 0
                total_weight = 0

                for i, parent in enumerate(parents):
                    if parent in INDICATOR_WEIGHTS:
                        weight = INDICATOR_WEIGHTS[parent]
                        weighted_sum += state[i] * weight
                        total_weight += weight

                if total_weight > 0:
                    avg = weighted_sum / total_weight
                else:
                    avg = sum(state) / len(state) if len(state) > 0 else 0

                # 根据父节点状态计算cpd
                if avg == 1:  # 所有父节点成功
                    p_success = base_prob
                elif avg == 0:  # 所有父节点失败
                    p_success = 0.1
                else:  # 部分父节点成功
                    p_success = 0.2 + 0.6 * avg  # 线性插值
            elif node == "威胁评估分数":
                # 最终评估节点：根据威胁类别加权平均
                weighted_sum = 0
                total_weight = 0

                for i, parent in enumerate(parents):
                    if parent in CATEGORY_WEIGHTS:
                        weight = CATEGORY_WEIGHTS[parent]
                        weighted_sum += state[i] * weight
                        total_weight += weight

                if total_weight > 0:
                    avg = weighted_sum / total_weight
                else:
                    avg = sum(state) / len(state) if len(state) > 0 else 0

                # 根据父节点状态计算cpd
                if avg == 1:  # 所有父节点成功
                    p_success = base_prob
                elif avg == 0:  # 所有父节点失败
                    p_success = 0.05
                else:  # 部分父节点成功
                    p_success = 0.1 + 0.7 * avg  # 线性插值
            else:
                # 其他节点使用默认线性插值
                avg = sum(state) / len(state) if len(state) > 0 else 0

                if avg == 1:  # 所有父节点成功
                    p_success = base_prob
                elif avg == 0:  # 所有父节点失败
                    p_success = 0.1
                else:  # 部分父节点成功
                    p_success = 0.2 + 0.6 * avg  # 线性插值

            # 构建概率分布 [失败概率, 成功概率]
            values.append([1 - p_success, p_success])

        # 创建TabularCPD对象
        return TabularCPD(
            variable=node,
            variable_card=2,
            values=list(zip(*values)),
            evidence=parents,
            evidence_card=[2] * len(parents)
        )

    def _add_cpds(self):
        """为所有节点添加CPD"""
        if self.cpds:  # 使用预定义CPD
            self.model.add_cpds(*self.cpds)
            return

        # 先添加没有父节点的根节点CPD
        root_nodes = [node for node in self.model.nodes() if not self.model.get_parents(node)]
        for node in root_nodes:
            node_prob = self._get_node_prob(node)
            cpd = TabularCPD(
                variable=node,
                variable_card=2,
                values=[[1 - node_prob], [node_prob]]  # 失败概率, 成功概率
            )
            self.model.add_cpds(cpd)

        # 然后添加有父节点的节点CPD
        remaining_nodes = [node for node in self.model.nodes() if self.model.get_parents(node)]
        for node in remaining_nodes:
            cpd = self.cpd_generator(node)
            self.model.add_cpds(cpd)

    def evaluate(self):
        """执行概率推断，保存每个节点的推理概率"""
        infer = VariableElimination(self.model)
        self.node_prob_cache.clear()  # 清空之前的缓存

        for node in self.model.nodes():
            result = infer.query(variables=[node], show_progress=False)
            self.node_prob_cache[node] = round(result.values[1], 4)

        # 返回最终节点的概率作为整体结果
        return self.node_prob_cache["威胁评估分数"]

    def get_cached_node_probability(self, node_name: str) -> float:
        """获取缓存中已推理过的节点概率"""
        return self.node_prob_cache.get(node_name, None)

    def save_probabilities_to_json(self, filename: str):
        """
        将缓存中的节点推理概率保存到JSON文件
        filename -- 要保存的JSON文件名
        """
        if not self.node_prob_cache:
            raise ValueError("节点概率缓存为空，请先运行evaluate()方法进行推理")

        # 创建带阶段信息的结构化数据
        structured_data = {}
        for phase, nodes in self.phase_structure.items():
            phase_data = {}
            for node in nodes:
                if node in self.node_prob_cache:
                    phase_data[node] = self.node_prob_cache[node]
            structured_data[phase] = phase_data

        # 添加类别节点
        category_nodes = ["环境威胁类别", "空间威胁类别", "作战威胁类别"]
        for category in category_nodes:
            if category in self.node_prob_cache:
                structured_data[category] = self.node_prob_cache[category]

        # 添加总评估结果
        structured_data["威胁评估分数"] = self.node_prob_cache["威胁评估分数"]

        # 写入JSON文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=4)


def convert_input_to_node_probs(input_dict: dict) -> dict:
    """
    将输入字典转换为节点概率配置

    参数:
    - input_dict: 输入字典，key为指标描述，value为0-2归一化的值

    返回:
    - 节点概率配置字典
    """
    node_probs = {
        "环境威胁": {},
        "空间威胁": {},
        "作战威胁": {},
        "威胁类别": {
            "环境威胁类别": 0.5, "空间威胁类别": 0.5, "作战威胁类别": 0.5
        },
        "综合评估": {"威胁评估分数": 0.0}
    }

    for indicator_desc, value in input_dict.items():
        # 确保值在0-1范围内
        normalized_value = max(0, min(1, value / 2.0))  # 将0-2范围转换为0-1范围

        # 根据指标描述找到对应的节点名
        if indicator_desc in INDICATOR_MAPPING:
            node_name = INDICATOR_MAPPING[indicator_desc]

            # 将节点分配到对应的阶段
            if node_name == "环境威胁分数":
                node_probs["环境威胁"][node_name] = normalized_value
            elif node_name in ["距离衰减系数", "相对接近速度", "火力覆盖半径",
                               "传感器覆盖范围", "目标密度指数", "部署集中度"]:
                node_probs["空间威胁"][node_name] = normalized_value
            elif node_name in ["杀伤链闭合概率", "杀伤链响应速度",
                               "支援到达时间", "作战装备威胁"]:
                node_probs["作战威胁"][node_name] = normalized_value

    return node_probs


def get_threat_level(score: float) -> str:
    """
    根据威胁分数确定威胁等级

    参数:
    - score: 威胁分数 (0-1)

    返回:
    - 威胁等级描述
    """
    if 0 <= score < 0.4:
        return "低威胁"
    elif 0.4 <= score < 0.8:
        return "中等威胁"
    elif 0.8 <= score <= 1.0:
        return "高威胁"
    else:
        return "未知等级"


def assess_threat(input_indicators: dict) -> tuple:
    """
    威胁评估主函数

    参数:
    - input_indicators: 输入字典，key为指标描述，value为0-2归一化的值

    返回:
    - 包含威胁分数和等级的元组
    """
    # 转换输入为节点概率配置
    node_probs = convert_input_to_node_probs(input_indicators)

    # 创建贝叶斯网络模型
    model = ThreatAssessmentBayesianModel(node_probs=node_probs)

    # 执行评估
    threat_score = model.evaluate()

    # 确定威胁等级
    threat_level = get_threat_level(threat_score)

    return threat_score, threat_level
