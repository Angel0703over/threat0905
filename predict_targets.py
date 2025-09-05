import pandas as pd
import numpy as np
import math
from collections import defaultdict


def _lla_to_enu(lat, lon, alt, ref_lat=25, ref_lon=121, ref_alt=0):
    a = 6378137.0
    e2 = 6.69437999014e-3

    if isinstance(lat, str):
        lat = float(lat)
    if isinstance(lon, str):
        lon = float(lon)
    if isinstance(alt, str):
        alt = float(alt)

    N = a / np.sqrt(1 - e2 * np.sin(np.radians(lat)) ** 2)
    X = (N + alt) * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
    Y = (N + alt) * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
    Z = (N * (1 - e2) + alt) * np.sin(np.radians(lat))

    N0 = a / np.sqrt(1 - e2 * np.sin(np.radians(ref_lat)) ** 2)
    X0 = (N0 + ref_alt) * np.cos(np.radians(ref_lat)) * np.cos(np.radians(ref_lon))
    Y0 = (N0 + ref_alt) * np.cos(np.radians(ref_lat)) * np.sin(np.radians(ref_lon))
    Z0 = (N0 * (1 - e2) + ref_alt) * np.sin(np.radians(ref_lat))

    diff_X = X - X0
    diff_Y = Y - Y0
    diff_Z = Z - Z0

    T = np.array(
        [
            [-np.sin(np.radians(ref_lon)), np.cos(np.radians(ref_lon)), 0],
            [-np.sin(np.radians(ref_lat)) * np.cos(np.radians(ref_lon)), -np.sin(np.radians(ref_lat)) * np.sin(np.radians(ref_lon)), np.cos(np.radians(ref_lat))],
            [np.cos(np.radians(ref_lat)) * np.cos(np.radians(ref_lon)), np.cos(np.radians(ref_lat)) * np.sin(np.radians(ref_lon)), np.sin(np.radians(ref_lat))],
        ]
    )

    e, n, u = T.dot(np.array([diff_X, diff_Y, diff_Z]))
    x, y, z = n, u, e
    return x, y, z


def _calculate_velocity_and_acceleration(tracks):
    tracks_by_id = defaultdict(list)
    for track in tracks:
        tracks_by_id[track[7]].append(track)

    result_tracks = []
    n = []
    for track_id, track_list in tracks_by_id.items():
        track_list.sort(key=lambda x: x[0])

        positions = np.array([[track[23], track[24], track[25]] for track in track_list])
        times = np.array([track[1] for track in track_list])

        n.append(len(times))
        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)

        for i in range(2, n[-1] - 2):
            dt = times[i + 2] - times[i - 2]
            if dt != 0:
                velocities[i] = (-positions[i + 2] + 8 * positions[i + 1] - 8 * positions[i - 1] + positions[i - 2]) / (12 * (times[i + 1] - times[i - 1]))
            else:
                velocities[i] = np.zeros(3)

        velocities[0] = (positions[1] - positions[0]) / (times[1] - times[0]) if times[1] != times[0] else np.zeros(3)
        velocities[1] = (positions[2] - positions[0]) / (times[2] - times[0]) if times[2] != times[0] else np.zeros(3)
        velocities[-2] = (positions[-1] - positions[-3]) / (times[-1] - times[-3]) if times[-1] != times[-3] else np.zeros(3)
        velocities[-1] = (positions[-1] - positions[-2]) / (times[-1] - times[-2]) if times[-1] != times[-2] else np.zeros(3)

        for i in range(2, n[-1] - 2):
            dt = times[i + 2] - times[i - 2]
            if dt != 0:
                accelerations[i] = (-velocities[i + 2] + 8 * velocities[i + 1] - 8 * velocities[i - 1] + velocities[i - 2]) / (12 * (times[i + 1] - times[i - 1]))
            else:
                accelerations[i] = np.zeros(3)

        accelerations[0] = (velocities[1] - velocities[0]) / (times[1] - times[0]) if times[1] != times[0] else np.zeros(3)
        accelerations[1] = (velocities[2] - velocities[0]) / (times[2] - times[0]) if times[2] != times[0] else np.zeros(3)
        accelerations[-2] = (velocities[-1] - velocities[-3]) / (times[-1] - times[-3]) if times[-1] != times[-3] else np.zeros(3)
        accelerations[-1] = (velocities[-1] - velocities[-2]) / (times[-1] - times[-2]) if times[-1] != times[-2] else np.zeros(3)

        j = 0
        for track in track_list:
            track.extend(velocities[j])
            track.extend(accelerations[j])
            j += 1

        result_tracks.extend(track_list)
    return [result_tracks, n]


def _calculate_course_angle(target, key_position):
    dx = target["x"] - key_position["x"]
    dz = target["z"] - key_position["z"]
    vx = target["Vx"]
    vz = target["Vz"]
    T = np.sqrt(dx**2 + dz**2)
    v = np.sqrt(vx**2 + vz**2)
    if T * v == 0:
        Tv = 1e-10
    else:
        Tv = T * v
    cos_theta = (dx * vx + dz * vz) / (Tv)
    course_angle = np.arccos(cos_theta)
    return course_angle


def _calculate_arrival_time(target, key_position, course_angle):
    dx = target["x"] - key_position["x"]
    dy = target["y"] - key_position["y"]
    dz = target["z"] - key_position["z"]
    vx = target["Vx"]
    vy = target["Vy"]
    vz = target["Vz"]
    ax = target["Ax"]
    az = target["Az"]
    if course_angle > math.pi / 2:
        t = 0
        return t
    if "SSM203" or "CLIENT_AGM" in target["Type"]:
        Txz = np.sqrt(dx**2 + dz**2)
        vxz = np.sqrt(vx**2 + vz**2)
        a = np.sqrt(ax**2 + az**2)
        t = (np.sqrt(vxz * vxz + 2 * a * Txz) - vxz) / a
    else:
        Txz = np.sqrt(dx**2 + dz**2)
        vxz = np.sqrt(vx**2 + vz**2)
        t = Txz * np.cos(course_angle) / vxz
    return t


def predict_targets(in_tracks, important):
    """
    输入：tracks(23个属性,轨迹列表): list[list[str]], important(23个属性,要地信息): list[list[str]])
    输出：dict[str, dict[str, list[str]]] <目标 ID, <我方要地 ID, <受袭概率, 打击抵达时间>>>
    """
    threat_urgency_list = []

    for imp in important:
        for i in [1, 4, 5, 6, 11, 12, 13, 14, 15, 16]:
            if imp[i] == "None" or imp[i] == "" or imp[i] == "unknown":
                imp[i] = float(0.0)
            else:
                imp[i] = float(imp[i])
        imp_lat, imp_lon, imp_alt = float(imp[11]), float(imp[12]), float(imp[13])
        imp_x, imp_y, imp_z = _lla_to_enu(imp_lat, imp_lon, imp_alt)
        imp.extend([imp_x, imp_y, imp_z])

    tracks = []
    for t in in_tracks:
        track = [f for f in t]
        for i in [1, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
            if track[i] == "None" or track[i] == "" or track[i] == "unknown":
                track[i] = float(0.0)
            else:
                track[i] = float(track[i])
        track_lat, track_lon, track_alt = float(track[11]), float(track[12]), float(track[13])
        track_x, track_y, track_z = _lla_to_enu(track_lat, track_lon, track_alt)
        track.extend([track_x, track_y, track_z])
        tracks.append(track)

    tracks, n = _calculate_velocity_and_acceleration(tracks)

    i = 0
    j = 0
    cal = 0

    for i in range(len(n)):
        j = cal + n[i] // 2 - 1
        track = tracks[j]
        cal = cal + n[i]

        track_dict = {
            "ID": track[7],
            "Latitude": track[11],
            "Longitude": track[12],
            "Altitude": track[13],
            "Type": track[9],
            "x": track[23],
            "y": track[24],
            "z": track[25],
            "Vx": track[26],
            "Vy": -track[27],
            "Vz": track[28],
            "Ax": track[29],
            "Az": track[31],
        }

        for imp in important:
            imp_dict = {"ID": imp[7], "Latitude": imp[11], "Longitude": imp[12], "Altitude": imp[13], "x": imp[23], "y": imp[24], "z": imp[25]}
            course_angle = _calculate_course_angle(track_dict, imp_dict)
            arrival_time = _calculate_arrival_time(track_dict, imp_dict, course_angle)
            arrival_time = arrival_time - (tracks[cal - 1][1] - tracks[j][1])
            arrival_time = arrival_time if arrival_time > 0 else 0
            if course_angle < math.pi / 2:
                threat_score = math.exp(-abs(course_angle))
            else:
                threat_score = 0
            threat_urgency_list.append({"Threat_ID": track_dict["ID"], "Target_ID": imp_dict["ID"], "Probability": threat_score, "Arrival_time": arrival_time})

    df = pd.DataFrame(threat_urgency_list)

    threat_score_sum = df.groupby("Threat_ID")["Probability"].transform("sum")
    df["Probability"] = df.apply(lambda row: row["Probability"] / threat_score_sum[row.name] if threat_score_sum[row.name] != 0 else row["Probability"], axis=1)

    output_dict = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        threat_id = row["Threat_ID"]
        target_id = row["Target_ID"]
        probability = row["Probability"]
        arrival_time = row["Arrival_time"]

        output_dict[threat_id][target_id] = [str(probability), str(arrival_time), ""]
    return output_dict
