import numpy as np
import torch
import cv2
from proj3_code.feature_matching.SIFTNet import get_siftnet_features


def pairwise_distances(X, Y):
    """
    This method will be very similar to the pairwise_distances() function found
    in sklearn
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)
    However, you are NOT allowed to use any library functions like this
    pairwise_distances or pdist from scipy to do the calculation!

    The purpose of this method is to calculate pairwise distances between two
    sets of vectors. The distance metric we will be using is 'euclidean',
    which is the square root of the sum of squares between every value.
    (https://en.wikipedia.org/wiki/Euclidean_distance)

    Useful functions:
    -   np.linalg.norm()

    Args:
    -   X: N x d numpy array of d-dimensional features arranged along N rows
    -   Y: M x d numpy array of d-dimensional features arranged along M rows

    Returns:
    -   D: N x M numpy array where d(i, j) is the distance between row i of X and
        row j of Y
    """
    n, d = X.shape
    m, _ = Y.shape
    
    # 计算 X 和 Y 每一行的范数
    X_norms_squared = np.sum(X ** 2, axis=1)
    Y_norms_squared = np.sum(Y ** 2, axis=1)
    
    # 计算 Euclidean 距离的平方。注意：根据定义，距离越大，距离的平方就越大
    # 因此最后需要将平方根应用于结果数组
    xy_distances_squared = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            xy_distances_squared[i, j] = X_norms_squared[i] + Y_norms_squared[j] - 2 * np.dot(X[i], Y[j])
    
    # 应用平方根
    D = np.sqrt(xy_distances_squared)
    
    return D


def get_tiny_images(image_arrays):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    To build a tiny image feature, simply resize the original image to a very
    small square resolution, e.g. 16x16. You can either resize the images to
    square while ignoring their aspect ratio or you can crop the center
    square portion out of each image. Making the tiny images zero mean and
    unit length (normalizing them) will increase performance modestly.

    Useful functions:
    -   cv2.resize
    -   ndarray.flatten()

    Args:
    -   image_arrays: list of N elements containing image in Numpy array, in
                grayscale

    Returns:
    -   feats: N x d numpy array of resized and then vectorized tiny images
                e.g. if the images are resized to 16x16, d would be 256
    """
    # 初始化输出数组
    n = len(image_arrays)
    feats = np.zeros((n, 16*16))

    # 循环处理输入列表中的所有图像
    for i, img in enumerate(image_arrays):
        # 将图像调整为16x16像素数组，采用最近邻插值法进行调整（按规范要求）
        tiny_img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_NEAREST)
        # 将微小图像展平并保存到输出数组中
        feats[i] = tiny_img.flatten()
        
        # 将每个微小图像标准化为零均值和单位长度，
        feats[i] = (feats[i] - np.mean(feats[i])) / np.std(feats[i])
    
    return feats


def nearest_neighbor_classify(train_image_feats, train_labels,
                              test_image_feats, k=3):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which will increase
    performance (although you need to pick a reasonable value for k).
    Useful functions:
    -   D = pairwise_distances(X, Y)
          computes the distance matrix D between all pairs of rows in X and Y.
            -  X is a N x d numpy array of d-dimensional features arranged along
            N rows
            -  Y is a M x d numpy array of d-dimensional features arranged along
            N rows
            -  D is a N x M numpy array where d(i, j) is the distance between row
            i of X and row j of Y
    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating
            the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    -   k: the k value in kNN, indicating how many votes we need to check for
            the label
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    
    # 在训练数据和测试数据之间计算距离矩阵
    dists = pairwise_distances(test_image_feats, train_image_feats)

    test_labels = []

    # 对于每个测试样本，计算其k个最近邻，并将它们的标签进行投票，选出最可能的标签作为预测结果
    for i in range(dists.shape[0]):
        top_k_idx = np.argsort(dists[i])[:k]
        top_k_classes = [train_labels[idx] for idx in top_k_idx]
        predicted_class = max(set(top_k_classes), key=top_k_classes.count)
        test_labels.append(predicted_class)

    return test_labels


def kmeans(feature_vectors, k, max_iter = 10):
    """
    Implement the k-means algorithm in this function. Initialize your centroids
    with random *unique* points from the input data, and repeat over the
    following process:
    1. calculate the distances from data points to the centroids
    2. assign them labels based on the distance - these are the clusters
    3. re-compute the centroids from the labeled clusters

    Please note that you are NOT allowed to use any library functions like
    vq.kmeans from scipy or kmeans from vlfeat to do the computation!

    Useful functions:
    -   np.random.randint
    -   np.linalg.norm
    -   np.argmin

    Args:
    -   feature_vectors: the input data collection, a Numpy array of shape (N, d)
            where N is the number of features and d is the dimensionality of the
            features
    -   k: the number of centroids to generate, of type int
    -   max_iter: the total number of iterations for k-means to run, of type int

    Returns:
    -   centroids: the generated centroids for the input feature_vectors, a Numpy
            array of shape (k, d)
    """
    N, d = feature_vectors.shape

    # 随机初始化聚类中心
    centroid_indices = np.random.choice(range(N), k, replace=False)
    centroids = feature_vectors[centroid_indices]

    for i in range(max_iter):
        # 计算每个数据点与聚类中心之间的距离
        distances = np.sqrt(((feature_vectors - centroids[:, np.newaxis])**2).sum(axis=2))

        # 将每个数据点分配到最近的聚类
        cluster_assignments = np.argmin(distances, axis=0)

        # 更新聚类中心为聚类内所有数据点的平均值
        for j in range(k):
            centroids[j] = np.mean(feature_vectors[cluster_assignments == j], axis=0)

    return centroids


def build_vocabulary(image_arrays, vocab_size, stride = 20):
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Load images from the training set. To save computation time, you don't
    necessarily need to sample from all images, although it would be better
    to do so. You can randomly sample the descriptors from each image to save
    memory and speed up the clustering. For testing, you may experiment with
    larger stride so you just compute fewer points and check the result quickly.

    In order to pass the unit test, leave out a 10-pixel margin in the image,
    that is, start your x and y from 10, and stop at len(image_width) - 10 and
    len(image_height) - 10.

    For each loaded image, get some SIFT features. You don't have to get as
    many SIFT features as you will in get_bags_of_sifts, because you're only
    trying to get a representative sample here.

    Once you have tens of thousands of SIFT features from many training
    images, cluster them with kmeans. The resulting centroids are now your
    visual word vocabulary.

    Note that the default vocab_size of 50 is sufficient for you to get a decent
    accuracy (>40%), but you are free to experiment with other values.

    Useful functions:
    -   np.array(img, dtype='float32'), torch.from_numpy(img_array), and
            img_tensor = img_tensor.reshape(
                (1, 1, img_array.shape[0], img_array.shape[1]))
            for converting a numpy array to a torch tensor for siftnet
    -   get_siftnet_features() from SIFTNet: you can pass in the image tensor in
            grayscale, together with the sampled x and y positions to obtain the
            SIFT features
    -   np.arange() and np.meshgrid(): for you to generate the sample x and y
            positions faster

    Args:
    -   image_arrays: list of images in Numpy arrays, in grayscale
    -   vocab_size: size of vocabulary
    -   stride: the stride of your SIFT sampling

    Returns:
    -   vocab: This is a (vocab_size, dim) Numpy array (vocabulary). Where dim
            is the length of your SIFT descriptor. Each row is a cluster center
            / visual word.
    """

    # 随机采样的特征数
    max_samples = 1000

    # 存储所有采样到的SIFT特征
    all_descriptors = []

    # 在所有图像中进行随机采样
    for img in image_arrays:
        h, w = img.shape
        x, y = np.meshgrid(np.arange(10, w-10, stride), np.arange(10, h-10, stride))
        x, y = x.flatten(), y.flatten()
        idx = np.random.choice(len(x), min(max_samples, len(x)), replace=False)
        x, y = x[idx], y[idx]

        # 将灰度图像转换为PyTorch张量
        img_tensor = torch.from_numpy(img.astype('float32')).unsqueeze(0).unsqueeze(0)

        # 使用SIFTNet获取特征
        feats = get_siftnet_features(img_tensor, x, y)
        # feats = feats.numpy()
        feats = feats

        # 添加到所有的SIFT特征中
        all_descriptors.append(feats)

    # 将所有SIFT特征拼接为单个矩阵
    all_descriptors = np.vstack(all_descriptors)

    # 使用k-means聚类进行字典构建
    # 获取聚类中心并返回
    vocab = kmeans(feature_vectors=all_descriptors,k=vocab_size)
    return vocab


def kmeans_quantize(raw_data_pts, centroids):
    """
    Implement the k-means quantization in this function. Given the input data
    and the centroids, assign each of the data entry to the closest centroid.

    Useful functions:
    -   pairwise_distances
    -   np.argmin

    Args:
    -   raw_data_pts: the input data collection, a Numpy array of shape (N, d)
            where N is the number of input data, and d is the dimension of it,
            given the standard SIFT descriptor, d = 128
    -   centroids: the generated centroids for the input feature_vectors, a
            Numpy array of shape (k, D)

    Returns:
    -   indices: the index of the centroid which is closest to the data points,
            a Numpy array of shape (N, )

    """
     # 计算每个数据点与所有聚类中心之间的距离
    distances = pairwise_distances(raw_data_pts, centroids)

    # 找到距离每个数据点最近的聚类中心的索引
    indices = np.argmin(distances, axis=1)

    return indices


def get_bags_of_sifts(image_arrays, vocabulary, step_size = 10):
    """
    This feature representation is described in the lecture materials,
    and Szeliski chapter 14.
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -  np.array(img, dtype='float32'), torch.from_numpy(img_array), and
            img_tensor = img_tensor.reshape(
                (1, 1, img_array.shape[0], img_array.shape[1]))
            for converting a numpy array to a torch tensor for siftnet
    -   get_siftnet_features() from SIFTNet: you can pass in the image tensor
            in grayscale, together with the sampled x and y positions to obtain
            the SIFT features
    -   np.histogram() or np.bincount(): easy way to help you calculate for a
            particular image, how is the visual words span across the vocab


    Args:
    -   image_arrays: A list of input images in Numpy array, in grayscale
    -   vocabulary: A numpy array of dimensions:
            vocab_size x 128 where each row is a kmeans centroid
            or visual word.
    -   step_size: same functionality as the stride in build_vocabulary(). Feel
            free to experiment with different values, but the rationale is that
            you may want to set it smaller than stride in build_vocabulary()
            such that you collect more features from the image.

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
            feature representation. In this case, d will be equal to the number
            of clusters or equivalently the number of entries in each image's
            histogram (vocab_size) below.
    """
    vocab = vocabulary
    vocab_size = len(vocab)
    
    # Calculate number of input images
    num_images = len(image_arrays)
    
    # Initiate feature matrix
    feats = np.zeros((num_images, vocab_size))
    
    # Loop through all input images
    for i in range(num_images):
        # Convert image to torch tensor
        img_tensor = torch.from_numpy(np.array(image_arrays[i], dtype='float32'))
        
        # Reshape image tensor to (1, 1, height, width) format for SIFTNet
        img_tensor = img_tensor.reshape((1, 1, img_tensor.shape[0], img_tensor.shape[1]))
        
        # Define sampling positions on the image
        x_coords = np.arange(0, img_tensor.shape[2], step_size)
        y_coords = np.arange(0, img_tensor.shape[3], step_size)
        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
        xx_flattened = xx.flatten()
        yy_flattened = yy.flatten()
        
        # Extract SIFT features using SIFTNet
        siftnet_feats = get_siftnet_features(img_tensor, xx_flattened, yy_flattened)
        
        # Assign local features to nearest cluster center
        distances = np.sqrt(((siftnet_feats - vocab[:, np.newaxis])**2).sum(axis=2))
        nearest_clusters = np.argmin(distances, axis=0)
        
        # Build histogram indicating how many times each cluster was used
        hist_counts = np.bincount(nearest_clusters, minlength=vocab_size)
        
        # Normalize the histogram
        feats[i] = hist_counts / np.sum(hist_counts)
    
    return feats
