import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def radar_dbscan(xyzV, dtype_clusters, weights, eps=1.5) -> (np.array, np.dtype, list):
    """
    DBSCAN for point cloud. Directly call the scikit-learn.dbscan with customized distance metric.

    Args:
        xyzV (ndarray): Numpy array containing the [x, y, z, v] of each detected points.
            xy equals to the camera uv coordinate; z is the range information.
        weights (list): weight for xyzv information respectively in calculating distance.

    Returns:
        clusters (1-D numpy darray, dtype = dtype_clusters): 
            Numpy array containing the clusters' information including number of points, center and
            size of the clusters in x,y,z coordinates and average velocity. 
        labels (1-D numpy darray, with shape (n, )): each point belongs to which cluster
    """

    if xyzV.size == 0:
        return np.zeros(0, dtype=dtype_clusters), []
    
    xyzV_tmp = xyzV*np.array(weights)
    labels = DBSCAN(eps, min_samples=2).fit_predict(xyzV_tmp)   # using customized distance metric can be slow

    # Exclude the points clustered as noise, i.e, with negative labels.
    unique_labels = sorted(set(labels[labels >= 0]))

    clusters = np.zeros(len(unique_labels), dtype=dtype_clusters)

    for label in unique_labels:
        clusters['num_points'][label] = xyzV[label == labels].shape[0]
        clusters['center'][label] = np.mean(
            xyzV[label == labels, 0:3], axis=0)[:3]
        clusters['size'][label] = np.amax(xyzV[label == labels, 0:3], axis=0)[:3] - \
            np.amin(xyzV[label == labels, 0:3], axis=0)[:3]
        clusters['avgV'][label] = np.mean(xyzV[:, 3], axis=0)

    return clusters, labels



def associate_clusters(old_clusters, new_clusters) -> (np.array, np.array, tuple):
    """
    Associate pre-existing clusters and the new clusters.

    Args:
        old_clusters: clusters of points in the previous frame
        new_clusters: clusters of points in the current frame
        epsilon_v: velocity threshold to filter out matches whose velocities differ a lot

    Return:
        unmatched_old: 1-D np.array()
        unmatched_new: 1-D np.array()
        matched: a tuple (old_matched_idx, new_matched_idx). Each element is an 1-D array.
    """

    fps = 20
    weights = [1, 1, 10]

    old_loc = old_clusters['center'][:]
    old_v = old_clusters['avgV']
    predict_z = old_loc[:, 2] + old_v/fps
    new_loc = new_clusters['center'][:]
    diff = np.zeros(
        (np.shape(old_loc)[0], np.shape(new_loc)[0]), dtype=np.float32)

    # Use weighted Euclidean distance as the metric to match
    for old_index, old in enumerate(old_loc):
        for new_index, new in enumerate(new_loc):
            diff[old_index, new_index] = weights[0] * (new[0] - old[0]) ** 2 + \
                weights[1] * (new[1] - old[1]) ** 2 + \
                weights[2] * (new[2] - predict_z[old_index]) ** 2

    # Use only z coordinate difference as the metric to match
    """
    for p, pre in enumerate(predict_z):
        for n, new in enumerate(new_z):
            diff[p, n] = np.abs(new_z[n] - predict_z[p])
    """

    # Hungarian Algorithm, return (row_ind, col_ind)
    matched = linear_sum_assignment(diff)
    unmatched_old = np.array(
        [x for x in range(len(old_clusters)) if x not in matched[0]])
    unmatched_new = np.array(
        [x for x in range(len(new_clusters)) if x not in matched[1]])

    return unmatched_old, unmatched_new, matched



class KalmanClusterTracker(object):
    """
    x: state
    y: residual
    z: measurement
    u: control input

    F: state transition function
    H: measurement function
    U: input function
    P: covariance matrix (dim_x, dim_x)
        If P is large compared to the sensor uncertainty R, the filter will rely more on measurements.
    Q: process noise matrix (dim_x, dim_x), representing the feature that the state of the system changes over time.
        Higer Q leads higher Kalman gain, gives more weight to the noisy measurements and achieves shorter time lag.
    R: measurement noise matrix (dim_z, dim_z)
    K: Kalman gain

    Predict:    {x_prior} = Fx+Bu
                {P_prior} = FPF'+Q
    Update:     y = z - H{x_prior}
                K = {P_prior}H'/(H{P_prior}H'+R)
                x_post = {x_prior} + Ky
                P_post = (I-KH){P_prior}
    """

    count = 0

    def __init__(self, cluster, dt, max_age):

        # define constant velocity model
        # three positions + three velocities + three length (u,v,z order)
        self.kf = KalmanFilter(dim_x=9, dim_z=7)
        self.kf.F = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0], [0, 1, 0, 0, dt, 0, 0, 0, 0], [0, 0, 1, 0, 0, dt, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.P[0:2, 0:2] *= 10.       # positions x and y are not accurate
        self.kf.P[3:5, 3:5] *= 1000.     # give high uncertainty to the unobservable x and y velocities
        self.kf.P[6:, 6:] *= 1000.       # give high uncertainty to the unobservable sizes
        self.kf.Q[:, :] *= 0.03      
        self.kf.Q[6:, 6:] *= 0.05        # size does not change a lot
        self.kf.R[:, :] *= 1.

        self.cluster = cluster
        self.max_age = max_age

        self.kf.x[:3] = self.cluster['center'].reshape((3, 1))
        self.kf.x[5:6] = self.cluster['avgV']
        self.kf.x[6:9] = self.cluster['size'].reshape((3, 1))
        self.time_since_update = 0
        self.id = KalmanClusterTracker.count
        KalmanClusterTracker.count += 1
        self.hit_streak = 0
        self.prev_hit_streak = 0

    def update(self, cluster):

        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(np.array(
            [*cluster['center'], cluster['avgV'], *cluster['size']]).reshape((7, 1)))
        self.update_cluster(cluster)

    def predict(self):
        if self.time_since_update == self.max_age:
            self.prev_hit_streak = self.hit_streak
            self.hit_streak = 0

        self.kf.predict()
        self.time_since_update += 1
        self.update_cluster()

    def update_cluster(self, new_data=None):
        self.cluster['center'] = self.kf.x[:3].reshape((3,))
        self.cluster['avgV'] = self.kf.x[5:6]
        self.cluster['size'] = self.kf.x[6:9].reshape((3,))
        if new_data is not None:
            self.cluster['num_points'] = new_data['num_points']


class Tracker(object):
    """
    self.tracker: 
        A list consists of elements 'trk'. Each 'trk' includes the cluster itself and state of the cluster. 
        There are three states of the clusters:
        (1) associated AND 'self.frame_count' > 'self.min_hits'         -> will be returned
        (2) associated AND 'self.frame_count' <= 'self.min_hits'        -> will not be returned      
        (3) not associated AND 'time_since_update' <= 'self.max_age'    -> will be returned
    """

    def __init__(self, dtype_clusters, fps, max_age=4, min_hits=4):
        self.dtype_clusters = dtype_clusters    # the np.dtype of input clusters
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.fps = fps

    def update(self, new_clusters):
        """
        return the 
        """
        self.frame_count += 1

        # Extract existing clusters from trakers and associate them with new clusters
        cur_clusters = np.zeros(0, dtype=self.dtype_clusters)
        for trk in self.trackers:
            cur_clusters = np.append(cur_clusters, trk.cluster)

        unmatched_old, unmatched_new, matched = associate_clusters(
            cur_clusters, new_clusters)

        # make predictions on all trackers (the update precedure does not include the predict step)
        for trk in self.trackers:
            trk.predict()

        # create and initialize new trackers for unmatched detections
        for i in unmatched_new:
            trk = KalmanClusterTracker(
                new_clusters[i], 1/self.fps, self.max_age)
            self.trackers.append(trk)

        # update matched clusters using the new frame's data
        old, new = matched
        for i, j in [*zip(old, new)]:
            self.trackers[i].update(new_clusters[j])

        # Delete dead tracklets
        self.trackers = [
            trk for trk in self.trackers if trk.time_since_update <= self.max_age]

        # Generate returned clusters
        """
        Using 'max(trk.hit_streak, trk.prev_hit_streak)' is to ensure 
        the interrupted tracks could be picked up quickly as long as the 'trk.time_since_update' is not larger than 'self.max_age'
        """
        ret = []
        for trk in self.trackers:
            if((trk.time_since_update <= self.max_age) and
                    (max(trk.hit_streak, trk.prev_hit_streak) >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(trk.cluster)
        return ret
