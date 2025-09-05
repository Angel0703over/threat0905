import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
import numpy.linalg as la
from numpy.linalg import svd, inv
from abc import abstractmethod

# 定义获取数据的函数
def readCSV():
    # 1. 读取数据
    data = pd.read_csv('example1.csv')
    # 将 DataFrame 转换为 list[list[str]]  
    data_as_list = data.astype(str).values.tolist()
    return data_as_list

# 定义编组聚类时需要进行的数据预处理函数
def dataProcessByGroup(data):
    # 获取列表中的数值型数据
    ex_data = [(ex_data[1], ex_data[2], ex_data[3]) for ex_data in data]
    columns = ['TrackLatitude', 'TrackLongitude', 'TrackAltitude']
    extracted_data = pd.DataFrame(ex_data, columns=columns)
    # 2. 对其中的缺失值进行填充
    # 首先检查并处理NaN值
    extracted_data = extracted_data.replace('not valid', 0)
    extracted_data.fillna(0, inplace=True)  # 用 0 填充 NaN 值
    extracted_data = extracted_data.round(6)
    # 3. 对提取的数值型数据进行归一化处理
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(extracted_data)
    scaled_data_df = pd.DataFrame(scaled_data, columns=columns)
    # 5. 从原始数据中提取非数值型数据，并与归一化后的数据合并
    non_numeric_data = pd.DataFrame([(data[0]) for data in data], columns=['Detectee'])  # Detectee是装备ID
    combined_data = pd.concat([non_numeric_data, scaled_data_df], axis=1)
    return combined_data

# 定义寻找最优DBSCAN参数的函数：尝试不同的eps和min_samples组合，并返回最优参数与轮廓系数
def find_optimal_eps_min_samples(X, max_eps, min_samples_step):
    best_silhouette = -1
    best_eps = 0
    best_min_samples = 0
    
    for eps in np.linspace(0.1, max_eps, 200):
        for min_samples in range(2, min_samples_step + 1):
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            labels = db.labels_
            if len(np.unique(labels)) > 1:  # 至少有两个聚类
                silhouette = silhouette_score(X, labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_eps = eps
                    best_min_samples = min_samples
    
    return best_eps, best_min_samples, best_silhouette

# 定义椭球体拟合的残差函数
def ellipsoid_residuals(params, x, y, z):
    # 解包参数
    center = params[:3]
    axis_lengths = params[3:6]
    rotation_euler = params[6:9]  

    # 将角度转换为旋转矩阵
    rotation = R.from_euler('zyx', rotation_euler, degrees=True).as_matrix()
    
    # 计算点到椭球体的距离
    distances = ((x - center[0]) * rotation[0, 0] + (y - center[1]) * rotation[1, 0] + (z - center[2]) * rotation[2, 0])**2 / axis_lengths[0]**2 + \
                ((x - center[0]) * rotation[0, 1] + (y - center[1]) * rotation[1, 1] + (z - center[2]) * rotation[2, 1])**2 / axis_lengths[1]**2 + \
                ((x - center[0]) * rotation[0, 2] + (y - center[1]) * rotation[1, 2] + (z - center[2]) * rotation[2, 2])**2 / axis_lengths[2]**2 - 1
    
    return distances

# 定义RANSAC椭球拟合的迭代函数
def ransac_ellipsoid_fitting(points, iterations=100, inlier_threshold=0.1):
    best_inliers_count = -1
    best_params = None
    best_inlier_mask = None

    for _ in range(iterations):
        # 随机选择3个不共线的点
        sample_indices = np.random.choice(points.shape[0], size=3, replace=False)
        sample_points = points[sample_indices]
        
        # 使用随机样本拟合椭球体：初始参数设置：中心点初始值设置为随机样本的均值，轴长为最大最小值之差，方向角度为0
        init_params = np.concatenate((np.mean(sample_points, axis=0), np.ptp(sample_points, axis=0), [0, 0, 0]))

        # 计算每个维度的最大值和最小值
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # 约束条件：轴长必须为正，方向旋转角度在 [-π, π] 范围内
        lower_bounds = np.concatenate([min_coords-0.01,  # 中心点坐标下界为数据点的最小值
                                    np.full(3, 0),  # 轴长下界为0
                                    np.full(3, -np.pi)])  # 旋转角度下界为-π

        upper_bounds = np.concatenate([max_coords+0.01,  # 中心点坐标上界为最大值
                                    np.full(3, np.inf),  # 轴长上界为无穷大
                                    np.full(3, np.pi)])  # 旋转角度上界为π

        bounds = (lower_bounds, upper_bounds)

        # 使用Trust Region Reflective（trf）优化算法进行拟合
        result = least_squares(
            ellipsoid_residuals,
            x0=init_params,
            method='trf',
            bounds=bounds,
            args=(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2]),
            loss='linear'
        )
        
        if result.success:
            # 计算内点
            predictions = ellipsoid_residuals(result.x, *points.T)
            inliers = np.abs(predictions) < inlier_threshold
            inliers_count = np.sum(inliers)
            # 检查是否有更多的内点
            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_params = result.x
                best_inlier_mask = inliers

    # 使用所有内点重新拟合椭球体
    if best_params is not None:
        print(best_inliers_count)
        inlier_points = points[best_inlier_mask]
        refit_result = least_squares(
            ellipsoid_residuals,
            x0=best_params,
            method='trf',
            bounds=bounds,
            args=(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2]),
            loss='linear'
        )
        return refit_result.x
    else:
        return None
def ellipsoid_fit(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    D = np.array([x * x + y * y - 2 * z * z, x * x + z * z - 2 * y * y, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z, 1]).T
    S = np.dot(D.T, D)
    C = np.zeros([10, 10])
    C[0, 0] = -1
    C[1:4, 1:4] = np.eye(3)
    C[4:7, 4:7] = np.eye(3)
    C[7:10, 7:10] = np.eye(3)
    C = np.kron(C, S)
    w, v = np.linalg.eig(C)
    i = np.where(w == max(w))[0][0]
    a = v[:, i]
    A = np.array([[a[0], a[3], a[4], a[6]], [a[3], a[1], a[5], a[7]], [a[4], a[5], a[2], a[8]], [a[6], a[7], a[8], a[9]]])
    center = np.dot(-np.linalg.inv(A[0:3, 0:3]), [[a[6]], [a[7]], [a[8]]])
    T = np.eye(4)
    T[3, 0:3] = center.T
    R = np.dot(np.linalg.inv(T), np.dot(A, np.linalg.inv(T.T)))
    val, vec = np.linalg.eig(R[0:3, 0:3] / -R[3, 3])
    i = np.argsort(val)
    R = np.dot(vec[:, i], np.dot(np.diag(val[i]), vec[:, i].T))
    ABC = np.sqrt(-1 / np.diag(R))
    return center.T.tolist()[0], ABC.tolist(), R.tolist()

# 1.定义计算椭球参数的函数：为每个编组计算椭球参数
def calculate_ellipsoid_params(cluster_data):
    
    # 中心化数据  
    mean_point = np.mean(cluster_data, axis=0)  
    centered_points = cluster_data - mean_point  
  
    # 计算协方差矩阵  
    cov_matrix = np.cov(centered_points, rowvar=False)  
  
    # 对协方差矩阵进行特征值分解  
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  
  
    # # 排序特征值和对应的特征向量  
    # sorted_indices = np.argsort(eigenvalues)[::-1]  
    # sorted_eigenvalues = eigenvalues[sorted_indices]  
    # sorted_eigenvectors = eigenvectors[:, sorted_indices]  
  
    # 椭球的中心  
    center = mean_point  
  
    # 椭球的半轴长（平方根特征值）  
    axes = np.sqrt(eigenvalues)  
  
    # 椭球的旋转矩阵（由特征向量组成）  
    orientation = eigenvectors.T  
    print(center,axes,orientation)
    return center, axes, orientation
# # 定义编组的函数
# def detectee_group(tracks):
#     # # 6. 调用数据预处理函数得处理后得数据
#     # 直接通过列表解析来获取数据
#     tracks_df = [(track[7], track[11], track[12], track[13], track[15])
#                  for track in tracks]
#     tracks_new = dataProcessByGroup(tracks_df)
#     # 将得到的DataFrame转换为NumPy数组
#     np_data = tracks_new.values
#     # 7. 使用find_optimal_eps_min_samples函数来找到最优参数
#     best_eps, best_min_samples, best_silhouette = find_optimal_eps_min_samples(np_data[:, 1:], 1.2, 25)
#     print(f"Optimal eps: {best_eps}, Optimal min_samples: {best_min_samples}, best_silhouette:{best_silhouette}")
#     model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
#     labels = model.fit_predict(np_data[:, 1:]) # 只提取数值列作为特征列进行模型预测
#     # 将调整后的标签存储起来
#     clustered_data = pd.DataFrame(tracks_new, columns=tracks_new.columns)
#     clustered_data['ClusterLabel'] = labels # 添加聚类标签列
#     # 输出结果，格式如下：<装备id,编组id>
#     result_columns = ['Detectee', 'ClusterLabel']
#     result_data = clustered_data[result_columns]  
#     # 添加一个新列，作为扩展预留字段,所有值都为None
#     result_data['Metainfo1'] = None
#     result_data['Metainfo2'] = None
#     result_data['Metainfo3'] = None
#     result_data.to_csv('result/LocalTrackData.csv', sep=',', encoding='utf-8-sig', index=False)
#     # result_dict = result_data.set_index('Detectee')['ClusterLabel'].to_dict()
#     result_data = result_data.set_index('Detectee')
#     columns_to_include = ['ClusterLabel', 'Metainfo1', 'Metainfo2', 'Metainfo3']  
#     result_dict = {index: [row[col] for col in columns_to_include] for index, row in result_data.iterrows()}  
#     with open('group_result.json', 'w', encoding='utf-8') as f:
#         json.dump(result_dict, f, ensure_ascii=False, indent=1)
#     print(result_dict,'\n')
#     return result_dict
class MinimumVolumeFigure:
    """
    Abstract class for minimum volume figure implementation.
    """

    def __init__(self):
        pass

    def calculate_error(self, points, tolerance=0.001):
        """
        Calculates collective error (outliers rate) for provided points set.
        :param points: row-ordered points matrix
        :param tolerance: tolerance for point distance value
        :return: float number with error rate
        """
        points = np.asarray(points)
        counter = [(1 if self.calculate_distance(point) > (1.0 + tolerance) else 0) for point in points]
        return float(sum(counter)) / len(points)

    @abstractmethod
    def calculate_distance(self, point):
        """
        Calculates point distance in regards to figure center.
        :param point: vector of point coordinates in figure's n-space
        :return: float value denoting distance to figure center (1.0 if point lies on figure surface)
        """
        pass


class Ellipsoid(MinimumVolumeFigure):
    """
    Class representing minimum volume enclosing ellipsoid classifier.
    """

    def __init__(self, points, tolerance=0.001):
        """
        Constructs ellipsoid using provided points.
        :param points: list of lists (denoting points in n-dimensional space)
        :param tolerance: stop parameter for ellipsoid construction algorithm
        """
        MinimumVolumeFigure.__init__(self)
        self.a, self.c = self._mvee(points, tolerance)

    def calculate_distance(self, point):
        """
        Calculates point distance in regards to ellipsoid center.
        :param point: vector of point coordinates in ellipsoid n-space
        :return: float value denoting distance to ellipsoid center (1.0 if point lies on ellipsoid surface)
        """
        point = np.asmatrix(point) - self.c
        return point * self.a * np.transpose(point)

    def calculate_distances(self, point):
        return self.calculate_distance(point)

    def _mvee(self, points, tolerance):
        # Taken from: http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
        points = np.asmatrix(points)
        n, d = points.shape
        q = np.column_stack((points, np.ones(n))).T
        err = tolerance + 1.0
        u = np.ones(n) / n
        while err > tolerance:
            # assert u.sum() == 1 # invariant
            x = q * np.diag(u) * q.T
            m = np.diag(q.T * la.inv(x) * q)
            jdx = np.argmax(m)
            step_size = (m[jdx] - d - 1.0) / ((d + 1) * (m[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u
        c = u * points
        a = la.inv(points.T * np.diag(u) * points - c.T * c) / d
        return np.asarray(a), np.squeeze(np.asarray(c))

# 定义编组的函数
def detectee_group(tracks):
    # 筛选TargetSide为blue的数据
    tracks_blue = [track for track in tracks if track[8] == "blue"]
    # 通过列表解析来获取装备Id编号和类型信息
    tracks_choice = [(track[7], track[9]) for track in tracks_blue]
    # 根据装备id抽样使得每个装备id只有一条轨迹信息；使用循环和辅助集合的方法来去重，使得目标id唯一
    seen = set()
    tracks_only_id = []
    for element in tracks_choice:
        if element[0] not in seen:
            tracks_only_id.append(element)
            seen.add(element[0])
    columns = ['Detectee', 'DetecteeType']
    tracks_df = pd.DataFrame(tracks_only_id, columns=columns)
    # 根据装备类型信息进行分组判断
    # 类型：F-35C（战斗机）、CLIENT_AGM（蜂群）、SSM203（导弹）、XQ-58A（无人机）
    # 编组:1.蜂群低空袭扰(CLIENT_AGM（蜂群）) 2.有人/无人协同压制打击(F-35C（战斗机）XQ-58A（无人机）) 3.驱逐舰\巡洋舰前出打击SSM203（导弹)
    for index,row in tracks_df.iterrows():
        if row['DetecteeType'] in ['CLIENT_AGM']:
            tracks_df.at[index, 'ClusterID'] = '0'
        elif row['DetecteeType'] in ['F-35C', 'XQ-58A']:
            tracks_df.at[index, 'ClusterID'] = '1'
        else:
            tracks_df.at[index, 'ClusterID'] = '2'
    tracks_df['ClusterID'] = tracks_df['ClusterID'].astype(int)
    # 输出结果，格式如下：<装备id,编组id>
    result_columns = ['Detectee', 'ClusterID']
    result_data = tracks_df[result_columns] 
    # 添加一个新列，作为扩展预留字段,所有值都为None
    result_data['Metainfo1'] = None
    result_data['Metainfo2'] = None
    result_data['Metainfo3'] = None

    # result_dict = result_data.set_index('Detectee')['ClusterLabel'].to_dict()
    result_data = result_data.set_index('Detectee')
    columns_to_include = ['ClusterID', 'Metainfo1', 'Metainfo2', 'Metainfo3']  
    result_dict = {index: [row[col] for col in columns_to_include] for index, row in result_data.iterrows()}  
    return result_dict

# 定义绘制编组包络的函数
def group_shape(tracks, groups):
    # 筛选TargetSide为blue的数据
    tracks_blue = [track for track in tracks if track[8] == "blue"]
    # tracks为原始数据,类型为list列表，groups_data也就是detectee_groups输出的结果result_dicts,类型为dict
    # 将tracks和groups合并为一个文件
    columns_tracks = ['Time', 'Detectee', 'DetecteeType', 
'TrackLatitude', 'TrackLongitude', 'TrackAltitude', 'Bearing']
    # 直接通过列表解析来获取数据
    tracks_list = [(track[1], track[7], track[9], track[11], track[12], track[13], track[15])
                 for track in tracks_blue]
    tracks_df = pd.DataFrame(tracks_list, columns=columns_tracks)
    tracks_df.sort_values(by=['Detectee','Time'], inplace=True)
    last_tracks = tracks_df.groupby('Detectee').tail(1)
    # 将字典转换为 DataFrame，使用字典的键作为索引，指定列名
    groups_df = pd.DataFrame.from_dict(groups, orient='index', columns=['ClusterID', 'Metainfo1', 'Metainfo2', 'Metainfo3'])
    # 重置索引，并将原来的索引列重命名为 'Detectee'
    groups_df = groups_df.reset_index().rename(columns={'index': 'Detectee'})
    groups_df = groups_df[['Detectee', 'ClusterID']]
    new_data = pd.merge(last_tracks, groups_df, on='Detectee', how='outer')
    # 获取label聚类标签数据
    unique_labels = new_data['ClusterID'].astype(int).drop_duplicates().to_list()

    # 根据编组信息输出编组包络信息
    cluster_envelopes = {}
    for label in unique_labels:
        if label != -1:  # 忽略噪声点
            cluster_data = new_data[new_data['ClusterID'].astype(int) == label][['TrackLatitude', 'TrackLongitude', 'TrackAltitude']].values.astype(float)
    #         print(cluster_data)
    #         # 拟合椭球体
    #         # 解包参数
    #         center, axes, orientation = calculate_ellipsoid_params(cluster_data)
    #         # 定义小数点位数
    #         decimal_places = 6

    #         # # 使用列表推导式和 round 函数限制小数点位数
    #         # center_rounded = [round(x, decimal_places) for x in center]
    #         # axes_rounded = [round(x, decimal_places) for x in axes]
    #         # orientation_rounded = [round(x, decimal_places) for x in orientation]
    #         center_rounded = np.around(center, decimals=decimal_places).tolist()
    #         axes_rounded = np.around(axes, decimals=decimal_places).tolist()
    #         orientation_rounded = np.around(orientation, decimals=decimal_places).tolist()
    #         params_list = center_rounded + axes_rounded + orientation_rounded
    #         result_list = [params_list, None, None, None]
    #        # 将参数存储在字典中
    #         cluster_envelopes[str(label)] = result_list
    # print(cluster_envelopes)
            model = Ellipsoid(cluster_data, 10)
            a, c = model.a, model.c
            # 绘制cluster_data的椭球体
            pi = np.pi
            sin = np.sin
            cos = np.cos
            U, D, V = la.svd(a)
            # 这个D应该是轴长的平方根吧，U的列向量为原始数据空间中的主轴方向，每个列向量对应一个奇异值，并指向椭球体再该方向上的轴。
            # V矩阵的列向量是奇异值空间中的主轴方向，这些方向在数据变换后表示新的坐标轴
            rx, ry, rz = 1.0 / np.sqrt(D)
            rx = 1.4*rx
            ry = 1.4*ry
            rz = 1.4*rz
            u, v = np.mgrid[0 : 2 * pi : 20j, -pi / 2 : pi / 2 : 10j]

            def ellipse(u, v):
                x = rx * cos(u) * cos(v)
                y = ry * sin(u) * cos(v)
                z = rz * sin(v)
                return x, y, z

            E = np.dstack(ellipse(u, v))
            E = np.dot(E, V) + c
            x, y, z = np.rollaxis(E, axis=-1)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            rot = R.from_matrix(U)
            euler_angles = rot.as_euler('zyx', degrees=True)
            print("中心点：", c)
            print("轴长：", 1.0 / np.sqrt(D))
            print("方向：", euler_angles)

            # 绘制椭球体
            # 定义一个用于绘制椭球体的网格
            # 创建椭球体的参数化网格
            # 首先确保 u 和 v 是一维数组
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            uu, vv = np.meshgrid(u, v)   # 创建参数化网格
            print(c[0],c[1],c[2])
            # 计算椭球体上每个点的坐标
            x = rx * np.cos(uu) * np.sin(vv)
            y = ry * np.sin(uu) * np.sin(vv)
            z = rz * np.cos(vv)

            # 将椭球体的网格点坐标展平成一维数组
            x_flattened = x.ravel()
            y_flattened = y.ravel()
            z_flattened = z.ravel()

            # # 组合成点集
            points = np.column_stack((x_flattened, y_flattened, z_flattened))
            # 将网格转换为旋转后的椭球体
            rotation = R.from_euler('zyx', euler_angles, degrees=True)
            rotation_matrix = rotation.as_matrix()
            # 将网格转换为旋转后的椭球体  
            rotation_points = np.dot(points, rotation_matrix.T)  # 注意转置
            # # 分离将旋转后的坐标并重新组织成网格形状
            # x_rotated, y_rotated, z_rotated = rotation_points[:, 0], rotation_points[:, 1], rotation_points[:, 2]
            # # 将旋转后的坐标重新组织成二维网格形状
            # x_grid = x_rotated.reshape(uu.shape)
            # y_grid = y_rotated.reshape(uu.shape)
            # z_grid = z_rotated.reshape(uu.shape)
            # 将旋转后的坐标重新组织成二维网格形状  
            x_rotated = rotation_points[:, 0].reshape(vv.shape) + c[0]
            y_rotated = rotation_points[:, 1].reshape(vv.shape) + c[1]
            z_rotated = rotation_points[:, 2].reshape(vv.shape) + c[2]

            # 绘制旋转后的椭球体网格
            ax.plot_surface(x_rotated, y_rotated, z_rotated, cstride=1, rstride=1, alpha=0.2, color='b')
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2])
            plt.show()
    return cluster_envelopes

# 主程序
# 获取原始数据
data = readCSV()
result_data = detectee_group(data)
group_shape(data, result_data)