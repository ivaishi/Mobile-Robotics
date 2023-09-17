# Occupancy Grid Mapping Continuous Semantic Counting Sensor Model Class
#
# Author: Chien Erh Lin, Fangtong Liu
# Date: 02/27/2021

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from utils import cart2pol, wrapToPI


# Occupancy Grid Mapping with Continuous Semantic Counting Sensor Model Class
class ogm_continous_S_CSM:

    def __init__(self):
        # map dimensions
        self.range_x = [-15, 20]
        self.range_y = [-25, 10]

        # senesor parameters
        self.z_max = 30     # max range in meters
        self.n_beams = 133  # number of beams, we set it to 133 because not all measurements in the dataset contains 180 beams 

        # grid map parameters
        self.grid_size = 0.5                  # adjust this for task 4.B
        self.nn = 16                            # number of nearest neighbor search

        # map structure
        self.map = {}  # map
        self.pose = {}  # pose data
        self.scan = []  # laser scan data
        self.m_i = {}  # cell i

        # continuous kernel parameter
        self.l = 0.2      # kernel parameter
        self.sigma = 0.1  # kernel parameter

        # semantic
        self.num_classes = 7

        # -----------------------------------------------
        # To Do: 
        # prior initialization
        # Initialize prior, prior_alpha
        # -----------------------------------------------
        self.prior = 0            # prior for setting up mean and variance
        self.prior_alpha = 1e-5      # a small, uninformative prior for setting up alpha



    def construct_map(self, pose, scan):
        # class constructor
        # construct map points, i.e., grid centroids
        x = np.arange(self.range_x[0], self.range_x[1]+self.grid_size, self.grid_size)
        y = np.arange(self.range_y[0], self.range_y[1]+self.grid_size, self.grid_size)
        X, Y = np.meshgrid(x, y)
        t = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

        # a simple KDTree data structure for map coordinates
        self.map['occMap'] = KDTree(t)
        self.map['size'] = t.shape[0]

        # set robot pose and laser scan data
        self.pose['x'] = pose['x'][0][0]
        self.pose['y'] = pose['y'][0][0]
        self.pose['h'] = pose['h'][0][0]
        self.pose['mdl'] = KDTree(np.hstack((self.pose['x'], self.pose['y'])))
        self.scan = scan

        # -----------------------------------------------
        # To Do: 
        # Initialization map parameters such as map['mean'], map['variance'], map['alpha']
        # -----------------------------------------------
        self.map['mean'] = self.prior*np.ones([X.shape[0]**2,self.num_classes+1])             # size should be (number of data) x (number of classes + 1)
        self.map['variance'] = self.prior*np.ones([X.shape[0]**2,1])          # size should be (number of data) x (1)
        self.map['alpha'] = self.prior_alpha*np.ones([X.shape[0]**2,self.num_classes+1])             # size should be (number of data) x (number of classes + 1)


    def is_in_perceptual_field(self, m, p):
        # check if the map cell m is within the perception field of the
        # robot located at pose p
        inside = False
        d = m - p[0:2].reshape(-1)
        self.m_i['range'] = np.sqrt(np.sum(np.power(d, 2)))
        self.m_i['phi'] = wrapToPI(np.arctan2(d[1], d[0]) - p[2])
        # check if the range is within the feasible interval
        if (0 < self.m_i['range']) and (self.m_i['range'] < self.z_max):
            # here sensor covers -pi to pi
            if (-np.pi < self.m_i['phi']) and (self.m_i['phi'] < np.pi):
                inside = True
        return inside


    def continuous_S_CSM(self, z, i, k):
        bearing_diff = []
        # find the nearest beam
        bearing_diff = np.abs(wrapToPI(z[:, 1] - self.m_i['phi']))
        idx = np.nanargmin(bearing_diff)
        bearing_min = bearing_diff[idx]
        global_x = self.pose['x'][k][0] + z[idx,0] * np.cos(z[idx,1] + self.pose['h'][k][0])
        global_y = self.pose['y'][k][0] + z[idx,0] * np.sin(z[idx,1] + self.pose['h'][k][0])

        # -----------------------------------------------
        # To Do: 
        # implement the continuous counting sensor model, update 
        # obj.map.alpha and obj.map.beta
        #
        # Hint: use distance and obj.l to determine occupied or free.
        # There might be multiple ways to update the free space. 
        # One way is to segment the measurement into several range 
        # values and update the free space if the distance is smaller 
        # than obj.l  
        # -----------------------------------------------
        # calculate distance between map cell and scan endpoint
        d1 = np.sqrt((self.map['occMap'].data[i][0] - global_x)**2+(self.map['occMap'].data[i][1] - global_y)**2)
        # check if distance is within the threshold distance l
        if d1 < self.l:
            # update alpha
            kernel_d1 = self.sigma*(1/3*(2+np.cos(2*np.pi*d1/self.l))*(1-d1/self.l)+1/2/np.pi*np.sin(2*np.pi*d1/self.l))
            self.map['alpha'][i][int(z[idx][2]-1)] += kernel_d1
        b = np.sqrt((self.map['occMap'].data[i][0] - self.pose['x'][k][0])**2+(self.map['occMap'].data[i][1] - self.pose['y'][k][0])**2)
        for xl in np.arange(b-self.l,min(z[idx][0],b+self.l),self.l*0.6):
            global_x = self.pose['x'][k][0] + xl * np.cos(z[idx,1] + self.pose['h'][k][0])
            global_y = self.pose['y'][k][0] + xl * np.sin(z[idx,1] + self.pose['h'][k][0])
            d2 = np.sqrt((self.map['occMap'].data[i][0] - global_x)**2+(self.map['occMap'].data[i][1] - global_y)**2)
            if d2 < self.l:
                kernel_d2 = self.sigma*(1/3*(2+np.cos(2*np.pi*d2/self.l))*(1-d2/self.l)+1/2/np.pi*np.sin(2*np.pi*d2/self.l))
                self.map['alpha'][i][self.num_classes] += kernel_d2


    def build_ogm(self):
        # build occupancy grid map using the binary Bayes filter.
        # We first loop over all map cells, then for each cell, we find
        # N nearest neighbor poses to build the map. Note that this is
        # more efficient than looping over all poses and all map cells
        # for each pose which should be the case in online (incremental)
        # data processing.
        for i in tqdm(range(self.map['size'])):
            m = self.map['occMap'].data[i, :]
            _, idxs = self.pose['mdl'].query(m, self.nn)
            if len(idxs):
                for k in idxs:
                    # pose k
                    pose_k = np.array([self.pose['x'][k], self.pose['y'][k], self.pose['h'][k]])
                    if self.is_in_perceptual_field(m, pose_k):
                        # laser scan at kth state; convert from cartesian to
                        # polar coordinates
                        z = cart2pol(self.scan[k][0][0, :], self.scan[k][0][1, :])
                        z = np.hstack((z[:,0].reshape(-1, 1), z[:,1].reshape(-1, 1), self.scan[k][0][2, :].reshape(-1, 1).astype(int)))
                        # -----------------------------------------------
                        # To Do: 
                        # update the sensor model in cell i
                        # -----------------------------------------------
                        self.continuous_S_CSM(z,i,k)

            # -----------------------------------------------
            # To Do: 
            # update mean and variance for each cell i
            # -----------------------------------------------

            alpha_sum = np.sum(self.map['alpha'][i, :])
            alpha_star = np.max(self.map['alpha'][i, :])

            self.map['mean'][i] = self.map['alpha'][i] / alpha_sum

            numer_term = (alpha_star/alpha_sum)
            self.map['variance'][i] = (numer_term*(1-numer_term))/(alpha_sum + 1)

