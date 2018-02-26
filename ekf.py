import numpy as np
from numpy import sin, cos
import scipy.linalg    # you may find scipy.linalg.block_diag useful
from ExtractLines import ExtractLines, normalize_line_parameters, angle_difference
from maze_sim_parameters import LineExtractionParams, NoiseParams, MapParams

# Helper function for computing transition model.
def transition_model_EKF(v, om, x, y, th, dt):
    if np.abs(om) < 1e-10:
        g = np.array([x + v*np.cos(th)*dt, y + v*np.sin(th)*dt, th + om*dt])
        Gx = np.array([[1, 0, -v*np.sin(th)*dt],
                       [0, 1,  v*np.cos(th)*dt],
                       [0, 0, 1]])
        Gu = np.array([[np.cos(th)*dt, 0], 
                       [np.sin(th)*dt, 0], 
                       [0, dt]])
    else:
        th_n = th + om*dt
        g = np.array([x + (v/om)*(np.sin(th_n) - np.sin(th)), y - (v/om)*(np.cos(th_n) - np.cos(th)), th_n])
        Gx = np.array([[1, 0, (v/om)*(np.cos(th_n) - np.cos(th))], 
                       [0, 1, (v/om)*(np.sin(th_n) - np.sin(th))], 
                       [0, 0, 1]])
        Gu = np.array([[(np.sin(th_n) - np.sin(th))/om, (v/om**2)*(np.sin(th) - np.sin(th_n) + om*np.cos(th_n)*dt)], 
                       [(-np.cos(th_n) + np.cos(th))/om, (v/om**2)*(-np.cos(th) + np.cos(th_n) + om*np.sin(th_n)*dt)], 
                       [0, dt]])
    return g, Gx, Gu

# Helper function for mapping line to predicted measurement.
def map_line_to_predicted_measurement_EKF(alpha, r, p_rob, p_cam):
    x, y, th = p_rob
    x_cam, y_cam, th_cam = p_cam
    
    # Convert line from world frame to camera frame.
    h = np.array([alpha - th - th_cam,
                  r - x*np.cos(alpha) - y*np.sin(alpha) - x_cam*np.cos(alpha - th) - y_cam*np.sin(alpha - th)])

    # Jacobian with respect to robot's mean state.
    Hx = np.array([[0, 0, -1],
                   [-np.cos(alpha), -np.sin(alpha), -x_cam*np.sin(alpha - th) + y_cam*np.cos(alpha - th)]])
    return h, Hx

class EKF(object):

    def __init__(self, x0, P0, Q):
        self.x = x0    # Gaussian belief mean
        self.P = P0    # Gaussian belief covariance
        self.Q = Q     # Gaussian control noise covariance (corresponding to dt = 1 second)

    # Updates belief state given a discrete control step (Gaussianity preserved by linearizing dynamics)
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def transition_update(self, u, dt):
        g, Gx, Gu = self.transition_model(u, dt)

        #### TODO ####
        # update self.x, self.P
        self.x = g
        self.P = Gx.dot(self.P).dot(Gx.T) + dt*Gu.dot(self.Q).dot(Gu.T)
        ##############

    # Propagates exact (nonlinear) state dynamics; also returns associated Jacobians for EKF linearization
    # INPUT:  (u, dt)
    #       u - zero-order hold control input
    #      dt - length of discrete time step
    # OUTPUT: (g, Gx, Gu)
    #      g  - result of belief mean self.x propagated according to the system dynamics with control u for dt seconds
    #      Gx - Jacobian of g with respect to the belief mean self.x
    #      Gu - Jacobian of g with respect to the control u
    def transition_model(self, u, dt):
        raise NotImplementedError("transition_model must be overriden by a subclass of EKF")

    # Updates belief state according to a given measurement (with associated uncertainty)
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: none (internal belief state (self.x, self.P) should be updated)
    def measurement_update(self, rawZ, rawR):
        z, R, H = self.measurement_model(rawZ, rawR)
        if z is None:    # don't update if measurement is invalid (e.g., no line matches for line-based EKF localization)
            return

        #### TODO ####
        # update self.x, self.P
        Sigma = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(Sigma))
        self.x += K.dot(z)
        self.P -= K.dot(Sigma).dot(K.T)
        ##############

    # Converts raw measurement into the relevant Gaussian form (e.g., a dimensionality reduction);
    # also returns associated Jacobian for EKF linearization
    # INPUT:  (rawZ, rawR)
    #    rawZ - raw measurement mean
    #    rawR - raw measurement uncertainty
    # OUTPUT: (z, R, H)
    #       z - measurement mean (for simple measurement models this may = rawZ)
    #       R - measurement covariance (for simple measurement models this may = rawR)
    #       H - Jacobian of z with respect to the belief mean self.x
    def measurement_model(self, rawZ, rawR):
        raise NotImplementedError("measurement_model must be overriden by a subclass of EKF")


class Localization_EKF(EKF):

    def __init__(self, x0, P0, Q, map_lines, tf_base_to_camera, g):
        self.map_lines = map_lines                    # 2xJ matrix containing (alpha, r) for each of J map lines
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Unicycle dynamics (Turtlebot 2)
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x

        #### TODO ####
        # compute g, Gx, Gu
        g, Gx, Gu = transition_model_EKF(v, om, x, y, th, dt)
        ##############

        return g, Gx, Gu

    # Given a single map line m in the world frame, outputs the line parameters in the scanner frame so it can
    # be associated with the lines extracted from the scanner measurements
    # INPUT:  m = (alpha, r)
    #       m - line parameters in the world frame
    # OUTPUT: (h, Hx)
    #       h - line parameters in the scanner (camera) frame
    #      Hx - Jacobian of h with respect to the the belief mean self.x
    def map_line_to_predicted_measurement(self, m):
        alpha, r = m

        #### TODO ####
        # compute h, Hx
        h, Hx = map_line_to_predicted_measurement_EKF(alpha, r, self.x, self.tf_base_to_camera)
        ##############

        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Given lines extracted from the scanner data, tries to associate to each one the closest map entry
    # measured by Mahalanobis distance
    # INPUT:  (rawZ, rawR)
    #    rawZ - 2xI matrix containing (alpha, r) for each of I lines extracted from the scanner data (in scanner frame)
    #    rawR - list of I 2x2 covariance matrices corresponding to each (alpha, r) column of rawZ
    # OUTPUT: (v_list, R_list, H_list)
    #  v_list - list of at most I innovation vectors (predicted map measurement - scanner measurement)
    #  R_list - list of len(v_list) covariance matrices of the innovation vectors (from scanner uncertainty)
    #  H_list - list of len(v_list) Jacobians of the innovation vectors with respect to the belief mean self.x
    def associate_measurements(self, rawZ, rawR):

        #### TODO ####
        # compute v_list, R_list, H_list
        I = rawZ.shape[1]
        J = self.map_lines.shape[1]

        # Compute Mahalanobis distance.
        def mahalanobis(m, z, R):
            h, Hx = self.map_line_to_predicted_measurement(m)
            v = z - h   # innovation
            S = Hx.dot(self.P).dot(Hx.T) + R   # innovation covariance
            d = v.T.dot(np.linalg.inv(S)).dot(v)   # Mahalanobis distance
            return (d, v, R, Hx)

        v_list = []
        R_list = []
        H_list = []
        for i in range(I):
            mahaVals = [mahalanobis(self.map_lines[:,j], rawZ[:,i], rawR[i]) for j in range(J)]
            midx = np.argmin([val[0] for val in mahaVals])
            d_min, v_min, R_min, Hx_min = mahaVals[midx]

            if d_min < self.g**2:
                v_list += [v_min]
                R_list += [R_min]
                H_list += [Hx_min]
        ##############

        return v_list, R_list, H_list

    # Assemble one joint measurement, covariance, and Jacobian from the individual values corresponding to each
    # matched line feature
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        #### TODO ####
        # compute z, R, H
        z = np.vstack(v_list)
        R = scipy.linalg.block_diag(R_list)
        H = np.vstack(H_list)
        ##############

        return z, R, H


class SLAM_EKF(EKF):

    def __init__(self, x0, P0, Q, tf_base_to_camera, g):
        self.tf_base_to_camera = tf_base_to_camera    # (x, y, theta) transform from the robot base to the camera frame
        self.g = g                                    # validation gate
        super(self.__class__, self).__init__(x0, P0, Q)

    # Combined Turtlebot + map dynamics
    # Adapt this method from Localization_EKF.transition_model.
    def transition_model(self, u, dt):
        v, om = u
        x, y, th = self.x[:3]

        #### TODO ####
        # compute g, Gx, Gu (some shape hints below)
        g = np.copy(self.x)
        Gx = np.eye(self.x.size)
        Gu = np.zeros((self.x.size, 2))

        g[:3], Gx[:3,:3], Gu[:3,:] = transition_model_EKF(v, om, x, y, th, dt)
        ##############

        return g, Gx, Gu

    # Combined Turtlebot + map measurement model
    # Adapt this method from Localization_EKF.measurement_model.
    #
    # The ingredients for this model should look very similar to those for Localization_EKF.
    # In particular, essentially the only thing that needs to change is the computation
    # of Hx in map_line_to_predicted_measurement and how that method is called in
    # associate_measurements (i.e., instead of getting world-frame line parameters from
    # self.map_lines, you must extract them from the state self.x)
    def measurement_model(self, rawZ, rawR):
        v_list, R_list, H_list = self.associate_measurements(rawZ, rawR)
        if not v_list:
            print "Scanner sees", rawZ.shape[1], "line(s) but can't associate them with any map entries"
            return None, None, None

        #### TODO ####
        # compute z, R, H (should be identical to Localization_EKF.measurement_model above)
        z = np.vstack(v_list)
        R = scipy.linalg.block_diag(R_list)
        H = np.vstack(H_list)
        ##############

        return z, R, H

    # Adapt this method from Localization_EKF.map_line_to_predicted_measurement.
    #
    # Note that instead of the actual parameters m = (alpha, r) we pass in the map line index j
    # so that we know which components of the Jacobian to fill in.
    def map_line_to_predicted_measurement(self, j):
        alpha, r = self.x[(3+2*j):(3+2*j+2)]    # j is zero-indexed! (yeah yeah I know this doesn't match the pset writeup)

        #### TODO ####
        # compute h, Hx (you may find the skeleton for computing Hx below useful)
        x, y, th = self.x
        x_cam, y_cam, th_cam = self.tf_base_to_camera

        Hx = np.zeros((2,self.x.size))
        h, Hx[:,:3] = map_line_to_predicted_measurement_EKF(alpha, r, self.x, self.tf_base_to_camera)
        # First two map lines are assumed fixed so we don't want to propagate any measurement correction to them
        if j > 1:
            Hx[0, 3+2*j] = 1
            Hx[1, 3+2*j] = x*np.sin(alpha) - y*np.cos(alpha) + x_cam*np.sin(alpha - th) - y_cam*np.cos(alpha - th)
            Hx[0, 3+2*j+1] = 0
            Hx[1, 3+2*j+1] = 1

        ##############

        flipped, h = normalize_line_parameters(h)
        if flipped:
            Hx[1,:] = -Hx[1,:]

        return h, Hx

    # Adapt this method from Localization_EKF.associate_measurements.
    def associate_measurements(self, rawZ, rawR):

        #### TODO ####
        # compute v_list, R_list, H_list
        ##############

        return v_list, R_list, H_list
