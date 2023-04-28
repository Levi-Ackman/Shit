import numpy as np

'''compute_feature_distances:'''
# # version_1:
# def compute_feature_distances(features1, features2):
#     """
#     This function computes a list of distances from every feature in one array
#     to every feature in another.

#     Args:
#     - features1: A numpy array of shape (n,feat_dim) representing one set of
#       features, where feat_dim denotes the feature dimensionality
#     - features2: A numpy array of shape (m,feat_dim) representing a second set
#       features (m not necessarily equal to n)

#     Returns:
#     - dists: A numpy array of shape (n,m) which holds the distances from each
#       feature in features1 to each feature in features2
#     """

#     # Compute L2 distance between each pair of features using numpy broadcasting
#     dists = np.sqrt(np.sum((features1[:, np.newaxis] - features2)**2, axis=2))

#     return dists
'''compute_feature_distances:'''
# version_2:
from scipy.spatial.distance import cdist
def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    # 使用 scipy.spatial.distance.cdist 函数计算每对特征之间的 L2 距离
    dists = cdist(features1, features2)

    return dists

  
'''match_features:'''
# # version_1
# def match_features(features1, features2, x1, y1, x2, y2):
#     """
#     This function does not need to be symmetric (e.g. it can produce
#     different numbers of matches depending on the order of the arguments).

#     To start with, simply implement the "ratio test", equation 4.18 in
#     section 4.1.3 of Szeliski. There are a lot of repetitive features in
#     these images, and all of their descriptors will look similar. The
#     ratio test helps us resolve this issue (also see Figure 11 of David
#     Lowe's IJCV paper).

#     You should call `compute_feature_distances()` in this function, and then
#     process the output.

#     Args:
#     - features1: A numpy array of shape (n,feat_dim) representing one set of
#       features, where feat_dim denotes the feature dimensionality
#     - features2: A numpy array of shape (m,feat_dim) representing a second
#       set of features (m not necessarily equal to n)
#     - x1: A numpy array of shape (n,) containing the x-locations of features1
#     - y1: A numpy array of shape (n,) containing the y-locations of features1
#     - x2: A numpy array of shape (m,) containing the x-locations of features2
#     - y2: A numpy array of shape (m,) containing the y-locations of features2

#     Returns:
#     - matches: A numpy array of shape (k,2), where k is the number of matches.
#       The first column is an index in features1, and the second column is an
#       index in features2
#     - confidences: A numpy array of shape (k,) with the real valued confidence
#       for every match

#     'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
#     """

#     # Compute pairwise distances between features in each set
#     dists = compute_feature_distances(features1, features2)

#     # Sort indices of feature2 by increasing distance from each feature in feature1
#     sorted_idx = np.argsort(dists, axis=1)

#     # Initialize matches and confidences arrays
#     matches = []
#     confidences = []

#     # For each feature in feature1, find best and second-best matches in feature2 using sorted_idx
#     for i, j in enumerate(sorted_idx):
#         best_match_idx = j[0]
#         best_match_dist = dists[i, best_match_idx]
#         second_best_match_dist = dists[i, j[1]]
#         confidence = best_match_dist/second_best_match_dist

#         # If the ratio of best to second-best match is less than some threshold, add the match to results
#         if confidence < 0.8:
#             matches.append([i, best_match_idx])
#             confidences.append(confidence)

#     return np.array(matches), np.array(confidences)
  
'''match_features:'''
## version_2:

def match_features(features1, features2, x1, y1, x2, y2):
    # 计算两组特征点之间的 L2 距离
    dists = np.linalg.norm(features1[:, np.newaxis] - features2, axis=2)

    # 对于 features1 中的每个特征点，找到与之距离最近的 features2 中的特征点的索引
    best_indices = np.argmin(dists, axis=1)

    # 对于 features1 中的每个特征点，计算距离最近的 features2 中的特征点的距离
    best_dists = np.min(dists, axis=1)

    # 使用距离排序的方法将最接近的特征点和它们的索引排在前面
    sort_indices = np.argsort(best_dists)

    # 初始化匹配和置信度的空列表
    matches = []
    confidences = []

    # 对于 features1 中的每个特征点，按从最近到最远的顺序遍历
    for i in sort_indices:

        # features1 中当前特征点所匹配的最近的 features2 中的特征点的索引
        j = best_indices[i]

        # 当前特征点到最近的 features2 中的特征点的距离
        dist1 = best_dists[i]

        # 当前特征点到次近的 features2 中的特征点的距离
        dist2 = np.partition(dists[i], 1)[1]

        # 计算当前匹配对应的置信度，即距离最近的特征点与次近特征点的距离比值
        confidence = dist1 / dist2

        # 如果当前匹配对应的置信度低于某个阈值，将其加入到匹配列表中
        if confidence < 0.8:
            matches.append([i, j])
            confidences.append(confidence)

    # 将匹配和置信度的列表转换成 numpy 数组并返回
    matches = np.array(matches)
    confidences = np.array(confidences)
    return matches, confidences
  
'''match_features:'''
# # version_3
# def match_features(features1, features2, x1, y1, x2, y2):
#     """
#     使用暴力枚举的方法进行特征点匹配。
    
#     参数：
#     - features1: 形状为 (n,feat_dim) 的 numpy 数组，表示第一组特征点的集合，其中 feat_dim 表示特征向量的维数。
#     - features2: 形状为 (m,feat_dim) 的 numpy 数组，表示第二组特征点的集合，(m 不一定等于 n)。
#     - x1: 形状为 (n,) 的 numpy 数组，包含 features1 中各个特征点在原图像上的 x 坐标。
#     - y1: 形状为 (n,) 的 numpy 数组，包含 features1 中各个特征点在原图像上的 y 坐标。
#     - x2: 形状为 (m,) 的 numpy 数组，包含 features2 中各个特征点在目标图像上的 x 坐标。
#     - y2: 形状为 (m,) 的 numpy 数组，包含 features2 中各个特征点在目标图像上的 y 坐标。

#     返回值：
#     - matches: 形状为 (k,2) 的 numpy 数组，其中 k 是匹配数目。第一列是 features1 中的索引，第二列是 features2 中的索引。
#     - confidences: 形状为 (k,) 的 numpy 数组，包含每个匹配对应的置信度。

#     'matches' 与 'confidences' 可以是空数组，例如 (0x2) 和 (0x1)
#     """

#     # 初始化匹配和置信度的空列表
#     matches = []
#     confidences = []

#     # 在特征点集合 features1 和 features2 上进行暴力枚举
#     for i in range(features1.shape[0]):
#         for j in range(features2.shape[0]):

#             # 计算第 i 个特征点和第 j 个特征点的 L2 距离
#             dist = np.linalg.norm(features1[i] - features2[j])

#             # 如果距离小于某个阈值，将当前匹配添加到匹配列表中，并计算其置信度
#             if dist < 0.5:
#                 matches.append([i, j])
#                 confidences.append(1 / dist)

#     # 将匹配和置信度的列表转换成 numpy 数组并返回
#     matches = np.array(matches)
#     confidences = np.array(confidences)
#     return matches, confidences
