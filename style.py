import pandas as pd, pathlib
from torch_geometric.data import Data
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, softmax
from torch.nn import Linear, Parameter
import torch.nn as nn
from loguru import logger

# 定义图卷积网络层
class _GTConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(_GTConv, self).__init__(aggr="add")
        self.lin = Linear(in_channels, out_channels)
        self.att = Parameter(torch.Tensor(1, out_channels))  #
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        #
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index, size):
        # Compute attention coefficients
        alpha = (x_i * self.att).sum(dim=-1) + (x_j * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        #
        alpha = softmax(alpha, edge_index[0], num_nodes=size[0])

        #
        return x_j * alpha.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out


class _GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(_GTLayer, self).__init__()
        self.gcn = _GTConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn(x, edge_index))
        x = self.dropout(x)
        return x, edge_index


class _GTN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(_GTN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.layers.append(_GTLayer(in_ch, hidden_channels, dropout_rate=0.5))
        self.conv = GCNConv(hidden_channels, hidden_channels)
        self.fc = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x, edge_index = layer(x, edge_index)
        x = self.conv(x, edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


_detectee_mapping = {f"c-130_swarm_{i}": i for i in range(1, 151)}
_detectee_mapping.update({f"xq-58a_{i}": i + 150 for i in range(1, 21)})
_detectee_mapping.update({f"f-35c_{i}": i + 170 for i in range(1, 11)})
_detectee_mapping.update({f"CG-68_BGM109_{i}": i + 180 for i in range(1, 11)})
_detectee_mapping.update({f"DDG-55_BGM109_{i}": i + 190 for i in range(1, 11)})
_detectee_mapping.update({f"DDG-61_BGM109_{i}": i + 190 for i in range(1, 11)})
_detectee_mapping.update({f"f-35c_{i}_LRSAM_1": i + 200 for i in range(1, 11)})
_detectee_mapping.update({f"f-35c_{i}_LRSAM_2": i + 210 for i in range(1, 11)})
_detectee_mapping.update({f"f-35c_{i}_LRSAM_3": i + 220 for i in range(1, 11)})
_detectee_mapping.update({f"f-35c_{i}_LRSAM_4": i + 230 for i in range(1, 11)})
_detectee_mapping.update({f"f-35c_{i}_LRSAM_5": i + 240 for i in range(1, 11)})
_detectee_mapping.update({f"f-35c_{i}_LRSAM_6": i + 250 for i in range(1, 11)})
_detectee_type_mapping = {"F-35C": 0, "XQ-58A": 1, "CLIENT_AGM": 2, "CJ100": 3, "SSM203": 4}


def _convert_to_float(value):
    """将字符串转换为浮点数，如果失败则返回默认值"""
    try:
        return float(value)
    except ValueError:
        if value == "not valid":
            return float("0")  # 使用 NaN 表示无效值
        return 0.0  # 设置默认值


def _load_graph_from_list(detection_list, groups):

    columns = [
        "Event",
        "Time",
        "Platform",
        "PlatformSide",
        "Latitude",
        "Longitude",
        "Altitude",
        "Detectee",
        "DetecteeSide",
        "DetecteeType",
        "TrackSide",
        "TrackLatitude",
        "TrackLongitude",
        "TrackAltitude",
        "Range",
        "Bearing",
        "Elevation",
        "RangeErrorSigma",
        "BearingErrorSigma",
        "ElevationErrorSigma",
        "Metainfo1",
        "Metainfo2",
        "Metainfo3",
    ]

    df = pd.DataFrame(detection_list, columns=columns)

    required_columns = ["Time", "Detectee", "DetecteeType", "TrackLatitude", "TrackLongitude", "TrackAltitude", "Range", "Bearing", "Elevation"]

    # 过滤只保留需要的列，其他列置零
    for col in df.columns:
        if col not in required_columns:
            df[col] = 0

    # 保证所有需要的列都存在
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[required_columns]

    # 预定义映射编码
    df["Detectee"] = df["Detectee"].map(_detectee_mapping)
    df["DetecteeType"] = df["DetecteeType"].map(_detectee_type_mapping)

    df = df.fillna(0)

    for col in ["Time", "TrackLatitude", "TrackLongitude", "TrackAltitude", "Range", "Bearing", "Elevation"]:
        df[col] = df[col].apply(_convert_to_float)

    G = nx.Graph()
    node_id_map = {}
    for idx, (_, row) in enumerate(df.iterrows()):
        node_id = f"{row['Detectee']}_{row['Time']}"
        node_id_map[node_id] = idx
        node_attributes = {
            "Time": row["Time"],
            "DetecteeType": row["DetecteeType"],
            "TrackLatitude": row["TrackLatitude"],
            "TrackLongitude": row["TrackLongitude"],
            "TrackAltitude": row["TrackAltitude"],
            "Range": row["Range"],
            "Bearing": row["Bearing"],
            "Elevation": row["Elevation"],
            # 'VelocityN': row['VelocityN'],
            # 'VelocityE': row['VelocityE'],
            # 'VelocityD': row['VelocityD'],
            # 'AccelerationN': row['AccelerationN'],
            # 'AccelerationE': row['AccelerationE'],
            # 'AccelerationD': row['AccelerationD'],
        }
        G.add_node(idx, **node_attributes)

    for u in G.nodes():
        for v in G.nodes():
            if u != v and groups.get(u) == groups.get(v):
                G.add_edge(u, v)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    x = torch.tensor([list(G.nodes[node].values()) for node in G.nodes()], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    return data


def _load_model(model_path, in_channels, hidden_channels, out_channels, num_layers):
    """
    加载预训练的GTN模型

    参数:
        model_path (str): 模型文件的路径
        in_channels (int): 输入特征的通道数
        hidden_channels (int): 隐藏层的通道数
        out_channels (int): 输出特征的通道数
        num_layers (int): 网络层数

    返回:
        _GTN: 加载好的模型实例，设置为评估模式
    """
    # 创建GTN模型实例
    model = _GTN(in_channels, hidden_channels, out_channels, num_layers)
    # 从指定路径加载模型权重
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device("cpu")))
    # 设置模型为评估模式
    model.eval()
    return model


def _test_model(model, detectee_type, detectee_group):
    # 按照 group 分类
    grouped_data = {}
    for key, value in detectee_group.items():
        group = value[0]  # 获取第一个元素作为分组依据
        if group not in grouped_data:
            grouped_data[group] = []
        grouped_data[group].append(key)

    results = {}  # 用于存储最终的结果

    # 提取行数据
    extracted_rows = {}
    for row in detectee_type:
        key = row[7]  # 取第8个元素作为键
        for group, keys in grouped_data.items():
            if key in keys:
                if key not in extracted_rows:  # 确保只保留一个
                    extracted_rows[key] = row

    # 对每个组进行预测
    for group_id, keys in grouped_data.items():
        group_rows = [extracted_rows[key] for key in keys if key in extracted_rows]

        if group_rows:
            data = _load_graph_from_list(group_rows, detectee_group)
            with torch.no_grad():
                out = model(data)
                pred = out.argmax(dim=1)  # 生成预测的 class ID

            # 存储每组的预测结果
            first_key = keys[0]  # 获取组中的第一个键
            # metainfo1 = detectee_group[first_key][1]
            # metainfo2 = detectee_group[first_key][2]
            # metainfo3 = detectee_group[first_key][3]
            style_id = str(pred.item())
            group_id = str(group_id)

            # 打印结果
            results[group_id] = [str(pred.item()), ""]

    return results


def predict_style(detectee_type, detectee_group):
    """
    预测作战样式

    参数:
        detectee_type: 检测目标类型
        detectee_group: 检测目标分组信息

    返回值:
        tuple: (状态码, 预测结果)
            - 状态码: "200"表示成功, "500"表示失败
            - 预测结果: 模型预测的作战样式结果，失败时返回空字符串
    """
    try:
        # 模型配置参数
        model_path = "resource/ckpt/model_weights1.pth"
        in_channels = 8  # 输入特征数量
        hidden_channels = 32  # 隐藏层通道数
        out_channels = 3  # 输出类别数量
        num_layers = 2  # 层数

        # 加载模型并进行预测
        logger.info("加载模型进行作战样式预测")
        model = _load_model(model_path, in_channels, hidden_channels, out_channels, num_layers)
        predictions = _test_model(model, detectee_type, detectee_group)

        return "200", predictions
    except Exception as e:
        logger.warning("作战样式预测失败：" + str(e))
        return "500", " "

