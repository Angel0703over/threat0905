import numpy as np
from loguru import logger
# 输入的项
track_list = [
    "Event",
    "Time",  # 有用
    "Platform",
    "PlatformSide",
    "Latitude",
    "Longitude",
    "Alitude",
    "Target",  # 有用
    "TargetSide",
    "TargetType",
    "TrackSide",
    "TrackLatitude",  # 有用
    "TrackLongitude",  # 有用
    "TrackAltitude",  # 有用
    "Range",
    "Bearing",
    "Elevation",
    "RangeErrorSigma",
    "BearingErrorSigma",
    "ElevationErrorSigma",
    "Metainfo1",  # 有用，存放导弹状态
    "Metainfo2",
    "Metainfo3",
]


def _convert_to_dict(data: list[list[str]], track_list):
    grouped_tracks = {}
    for entry in data:
        target = entry[7]  # 获取target
        if target not in grouped_tracks:
            grouped_tracks[target] = {key: [] for key in track_list}
        for i, key in enumerate(track_list):
            if key in ["Time", "TrackLatitude", "TrackLongitude", "TrackAltitude"]:
                grouped_tracks[target][key].append(float(entry[i]))
            else:
                grouped_tracks[target][key].append(entry[i])
    return grouped_tracks


# def _convert_to_dict(data:list[list[str]], track_list):
#     tracks = {key: [] for key in track_list}
#     for entry in data:
#         for i, key in enumerate(track_list):
#             if key in ["Time", "TrackLatitude", "TrackLongitude", "TrackAltitude"]:
#                 tracks[key].append(float(entry[i]))
#             else:
#                 tracks[key].append(entry[i])
#     return tracks


def _convert_to_list_str(tracks):

    data = []
    keys = list(tracks.keys())

    for i in range(len(tracks[keys[0]])):
        entry = []
        for key in keys:
            # 如果key对应的列表为空，添加一个默认值（比如空字符串）
            if len(tracks[key]) > i:
                entry.append(str(tracks[key][i]))
            else:
                entry.append("")
        data.append(entry)
    return data


def _intr_update(idx, tracks, ekf):
    avg_dt = np.mean(np.diff(tracks["Time"]))  # 计算平均时间步长
    dt = tracks["Time"][idx] - tracks["Time"][idx - 1]
    data_list = [
        "TrackLatitude",
        "TrackLongitude",
        "TrackAltitude",
    ]
    # 通过插值，创建伪track数据
    num_intervals = int(dt / avg_dt)
    interpolated_times = np.linspace(tracks["Time"][idx - 1], tracks["Time"][idx], num_intervals + 1)

    interpolated_tracks = {}
    for key in tracks.keys():
        if key in data_list:
            interpolated_tracks[key] = np.interp(interpolated_times, [tracks["Time"][idx - 1], tracks["Time"][idx]], [tracks[key][idx - 1], tracks[key][idx]])

    # 根据伪track数据，更新状态转移矩阵
    for i in range(1, len(interpolated_times)):
        ekf.predict(interpolated_times[i] - interpolated_times[i - 1])
        z = np.array([interpolated_tracks[key][i] for key in data_list])
        ekf.update(z)


def _lat_lon_alt_to_xyz(lat, lon, alt, a=6378137, f=1 / 298.257223563):
    """
    Convert latitude, longitude, and altitude to ECEF (Earth-Centered, Earth-Fixed) XYZ coordinates.

    Parameters:
    - lat (np.ndarray): Array of latitudes in degrees.
    - lon (np.ndarray): Array of longitudes in degrees.
    - alt (np.ndarray): Array of altitudes in meters.
    - a (float): Semi-major axis of the Earth in meters (default is WGS84).
    - f (float): Flattening factor of the Earth (default is WGS84).

    Returns:
    - x, y, z (np.ndarray): Arrays of ECEF XYZ coordinates in meters.
    """
    if not isinstance(lat, np.ndarray):
        lat = np.array(lat)
    if not isinstance(lon, np.ndarray):
        lon = np.array(lon)
    if not isinstance(alt, np.ndarray):
        alt = np.array(alt)

    # Convert latitude and longitude from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Earth's eccentricity squared
    e2 = 2 * f - f**2

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    # Calculate ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)

    return x, y, z


def _xyz_to_lat_lon_alt(x, y, z, a=6378137, f=1 / 298.257223563):
    """
    Convert ECEF (Earth-Centered, Earth-Fixed) XYZ coordinates to latitude, longitude, and altitude.

    Parameters:
    - x (np.ndarray): Array of ECEF X coordinates in meters.
    - y (np.ndarray): Array of ECEF Y coordinates in meters.
    - z (np.ndarray): Array of ECEF Z coordinates in meters.
    - a (float): Semi-major axis of the Earth in meters (default is WGS84).
    - f (float): Flattening factor of the Earth (default is WGS84).

    Returns:
    - lat, lon, alt (np.ndarray): Arrays of latitudes, longitudes in degrees, and altitudes in meters.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(z, np.ndarray):
        z = np.array(z)

    e2 = 2 * f - f**2
    b = a * (1 - f)  # Semi-minor axis

    # Calculate longitude
    lon = np.degrees(np.arctan2(y, x))

    # Calculate latitude using iterative approach
    r = np.sqrt(x**2 + y**2) + 1e-10  # Add a small constant to avoid division by zero

    E2 = a**2 - b**2
    F = 54 * b**2 * z**2
    G = r**2 + (1 - e2) * z**2 - e2 * E2
    c = (e2**2 * F * r**2) / (G**3)
    s = np.cbrt(1 + c + np.sqrt(c**2 + 2 * c))
    P = F / (3 * (s + 1 / s + 1) ** 2 * G**2)
    Q = np.sqrt(1 + 2 * e2**2 * P)
    r0 = -(P * e2 * r) / (1 + Q) + np.sqrt(0.5 * a**2 * (1 + 1 / Q) - P * (1 - e2) * z**2 / (Q * (1 + Q)) - 0.5 * P * r**2)
    U = np.sqrt((r - e2 * r0) ** 2 + z**2)
    V = np.sqrt((r - e2 * r0) ** 2 + (1 - e2) * z**2)
    Z0 = b**2 * z / (a * V)

    lat = np.degrees(np.arctan((z + e2 * Z0) / r))

    # Calculate altitude
    alt = U * (1 - b**2 / (a * V))

    return lat, lon, alt


def _smooth_altitude(altitudes, smoothing_window=3):
    """
    对高度数据进行平滑处理，使用简单的移动平均法。

    Parameters:
    - altitudes (np.ndarray): 高度数据数组。
    - smoothing_window (int): 平滑窗口大小，默认为3。

    Returns:
    - np.ndarray: 平滑后的高度数据。
    """
    return np.convolve(altitudes, np.ones(smoothing_window) / smoothing_window, mode="valid")


def _smooth_data(data, window_size):
    """对数据进行滑动窗口平滑处理"""
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def _is_missile_segment_ascending(tracks, idx, window_size=5, threshold=10, smoothing_window=3):
    """
    使用滑动窗口和高度平滑判断导弹最后一段轨迹是否处于上升阶段。

    Parameters:
    - tracks (dict): 包含轨迹信息的字典。
    - idx (int): 要判断的时刻索引。
    - window_size (int): 滑动窗口的大小，默认为5。
    - threshold (float): 判断为显著上升的高度变化阈值，默认是10米。
    - smoothing_window (int): 高度数据平滑窗口大小，默认为3。

    Returns:
    - bool: 如果在给定的时刻导弹在上升，返回True；否则返回False。
    """
    # 取出高度数据
    altitudes = tracks.get("TrackAltitude", [])

    # 确保idx在合法范围内
    if idx >= len(altitudes):
        return False  # 超出范围

    # 如果 idx 小于 window_size，取从开头到 idx 的所有数据
    if idx < window_size:
        segment = altitudes[: idx + 1]  # 从开头到当前索引
    else:
        # 如果 idx + window_size 超出长度，取到末尾
        if idx + window_size > len(altitudes):
            segment = altitudes[idx:]  # 从 idx 到最后
        else:
            segment = altitudes[idx - window_size : idx + 1]  # 从 idx-window_size 到 idx

    # 对高度数据进行平滑处理
    smoothed_segment = _smooth_data(segment, smoothing_window)

    # 计算平滑后轨迹的高度变化
    altitude_change = smoothed_segment[-1] - smoothed_segment[0]

    # 判断高度变化是否超过阈值
    if altitude_change > threshold:
        return "UP"
    else:
        return "Not_UP"


class _EKF:
    def __init__(self, state_dim, meas_dim, Q=None, R=None):
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # 状态向量 x, 初始值为0
        self.x = np.zeros(state_dim)

        # 误差协方差矩阵 P, 初始值为单位矩阵
        self.P = np.eye(state_dim)

        # 动态调整的状态噪声协方差矩阵 Q
        self.Q = Q if Q is not None else np.eye(state_dim) * 1e-4

        # 动态调整的观测噪声协方差矩阵 R
        self.R = R if R is not None else np.eye(meas_dim) * 1e-2

    def initialize_state(self, initial_state):
        """状态初始化函数"""
        if initial_state.shape == (self.state_dim,):
            self.x = initial_state
        else:
            raise ValueError("Initial state dimension does not match state_dim.")

    def f(self, x, dt):
        """状态转移函数（非线性）"""
        F = np.eye(self.state_dim)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F @ x

    def F_jacobian(self, x, dt):
        """状态转移矩阵的雅可比矩阵"""
        F_jac = np.eye(self.state_dim)
        F_jac[0, 3] = dt
        F_jac[1, 4] = dt
        F_jac[2, 5] = dt
        return F_jac

    def h(self, x):
        """观测模型（非线性）"""
        # 直接返回位置（经度、纬度、高度）
        return x[:3]

    def H_jacobian(self, x):
        """观测矩阵的雅可比矩阵"""
        H_jac = np.zeros((self.meas_dim, self.state_dim))
        H_jac[0, 0] = 1.0
        H_jac[1, 1] = 1.0
        H_jac[2, 2] = 1.0
        return H_jac

    def predict(self, dt):
        """预测步骤"""
        try:
            # 更新状态预测
            self.x = self.f(self.x, dt)

            # 计算雅可比矩阵
            F_jac = self.F_jacobian(self.x, dt)

            # 更新误差协方差矩阵
            self.P = F_jac @ self.P @ F_jac.T + self.Q
        except Exception as e:
            print(f"Prediction step failed: {e}")
            raise

    def update(self, z):
        """更新步骤"""
        try:
            # 观测异常检测
            if np.any(np.abs(z) > 1e8):  # 阈值可以根据应用场景调整
                print(f"Observation outlier detected: {z}")
                return

            # 计算观测预测
            z_pred = self.h(self.x)

            # 计算观测雅可比矩阵
            H_jac = self.H_jacobian(self.x)

            # 计算卡尔曼增益
            S = H_jac @ self.P @ H_jac.T + self.R
            K = self.P @ H_jac.T @ np.linalg.inv(S)

            # 更新状态向量
            self.x = self.x + K @ (z - z_pred)

            # 更新误差协方差矩阵
            self.P = self.P - K @ H_jac @ self.P

        except np.linalg.LinAlgError:
            print("Matrix inversion failed during update step.")
        except Exception as e:
            print(f"Update step failed: {e}")
            raise

    def set_process_noise(self, Q):
        """动态设置过程噪声"""
        if Q.shape == (self.state_dim, self.state_dim):
            self.Q = Q
        else:
            raise ValueError("Process noise matrix Q dimension does not match state_dim.")

    def set_measurement_noise(self, R):
        """动态设置观测噪声"""
        if R.shape == (self.meas_dim, self.meas_dim):
            self.R = R
        else:
            raise ValueError("Measurement noise matrix R dimension does not match meas_dim.")

    def get_state(self):
        """获取当前状态"""
        return self.x

    def get_covariance(self):
        """获取当前协方差矩阵"""
        return self.P


def PredictTrack_one(srctracks: list[list[str]], duration: float, num_spans: int) -> list[list[str]]:

    # 保持原始项的顺序
    other_list1 = [
        "Event",
        "Time",  # 有用
        "Platform",
        "PlatformSide",
        "Latitude",
        "Longitude",
        "Alitude",
        "Target",  # 有用
        "TargetSide",
        "TargetType",
        "TrackSide",
    ]
    other_list2 = [
        # "TrackLatitude",  # 有用
        # "TrackLongitude",  # 有用
        # "TrackAltitude",  # 有用
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
    # 预测项
    pred_list = [
        "TrackLatitude",
        "TrackLongitude",
        "TrackAltitude",
        "VelocityN",
        "VelocityE",
        "VelocityD",
    ]
    # 观测项
    z_list = [
        "TrackLatitude",
        "TrackLongitude",
        "TrackAltitude",
    ]
    # 将轨迹list[list[str]]数据转换成字典形式
    # tracks = _convert_to_dict(srctracks, track_list)
    tracks = srctracks
    # original_tracks = copy.deepcopy(tracks)
    # 将经纬度高度转换成xyz坐标
    logger.info("将经纬度高度转换成xyz坐标")
    tracks["TrackLatitude"], tracks["TrackLongitude"], tracks["TrackAltitude"] = _lat_lon_alt_to_xyz(tracks["TrackLatitude"], tracks["TrackLongitude"], tracks["TrackAltitude"])
    pred_tracks = {}

    for item in other_list1:
        pred_tracks[item] = tracks[item]
    for item in pred_list:
        pred_tracks[item] = []
    for item in other_list2:
        pred_tracks[item] = tracks[item]
    # 初始化EKF，使用第一个时间间隔初始化 dt
    initial_dt = tracks["Time"][1] - tracks["Time"][0] if len(tracks["Time"]) > 1 else 10  # 默认值为0.05s
    ekf = _EKF(state_dim=6, meas_dim=3)
    # 迭代更新状态转移矩阵
    logger.info("迭代更新状态转移矩阵")
    for i in range(len(tracks["Time"])):
        if i > 0:
            dt = tracks["Time"][i] - tracks["Time"][i - 1]  # 计算当前时间步长
            # 如果时间步太长，则插值更新状态转移矩阵
            if dt > 51:
                # print(f"{tracks['Target'][i]}:{tracks['Time'][i]}, {dt}")
                _intr_update(i, tracks, ekf)
            else:
                ekf.predict(dt)  # 使用动态时间步长进行预测
        else:
            ekf.predict(initial_dt)  # 初始步长预测
        # 把预测值存入pred_tracks
        for key in pred_list:
            pred_tracks[key].append(ekf.x[pred_list.index(key)])
        # 使用真实观测值更新 EKF 状态
        z = np.array([tracks[key][i] for key in z_list])
        ekf.update(z)  # 更新 EKF 状态

    length = len(tracks["Time"])
    logger.info(f"轨迹外推")
    # --------------任务一：轨迹外推-------------------------------------------
    for i in range(num_spans):
        dt = duration / num_spans
        ekf.predict(dt)
        for key in pred_list:
            pred_tracks[key].append(ekf.x[pred_list.index(key)])
        pred_tracks["Time"].append(pred_tracks["Time"][-1] + dt)
        if tracks["TargetType"][0] != "SSM203":
            for key in other_list1 + other_list2:
                # if key != "Time" and key in pred_tracks.keys():
                if key != "Time":
                    pred_tracks[key].append(tracks[key][i])
        # --------------任务二：导弹姿态判断-------------------------------------------
        else:
            for key in other_list1 + other_list2:
                # if  key in pred_tracks.keys():
                if key == "Time":
                    continue
                elif key == "Metainfo1":
                    pred_tracks[key].append(_is_missile_segment_ascending(pred_tracks, length + i))
                else:
                    pred_tracks[key].append(tracks[key][i])

    logger.info("进行导弹姿态判断")
    # -----------------------------------------------------------------------
    # 将xyz坐标转换成经纬度高度
    pred_tracks["TrackLatitude"], pred_tracks["TrackLongitude"], pred_tracks["TrackAltitude"] = _xyz_to_lat_lon_alt(
        pred_tracks["TrackLatitude"], pred_tracks["TrackLongitude"], pred_tracks["TrackAltitude"]
    )
    # 去除速度，将预测数据转换成list[list[str]]形式，保持和输入数据一致
    del pred_tracks["VelocityN"]
    del pred_tracks["VelocityE"]
    del pred_tracks["VelocityD"]

    # 取后面num_spans个数据
    for key in pred_tracks.keys():
        pred_tracks[key] = pred_tracks[key][-num_spans:]
    # savefig(original_tracks, pred_tracks)
    logger.info("生成轨迹预测结果")
    pred_tracks = _convert_to_list_str(pred_tracks)
    return pred_tracks


def PredictTrack(srctracks: list[list[str]], duration: float, num_spans: int) -> [str, str | list[list[str]]]:
    try:
        # 存放预测结果
        pred_all_result = []
        # 通过target进行group，再转换成字典
        srctracks = _convert_to_dict(srctracks, track_list)
        # 挨个预测
        logger.info("开始轨迹预测")
        for target in srctracks.keys():
            result = PredictTrack_one(srctracks[target], duration, num_spans)
            pred_all_result.extend(result)
        return "200", pred_all_result
    except Exception as e:
        logger.warning("轨迹预测出错，调用失败:"+str(e))
        return "500", " "