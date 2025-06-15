#!/usr/bin/env python3

import numpy as np
import math

class kalman:
    def __init__(self):
        
        # State
        self.X0 = None
        self.X = None
        self.X_nm1 = None
        self.xhat_np1 = None    # Predict Output
        self.X_hat = None
        self.u = None

        # Model
        self.An = None
        self.G = None            # L=B (One case)
        self.H = None
        self.L = None
        
        # Kalman
        self.P0 = None
        self.Pn = None
        self.Pnm1 = None
        self.P_np1 = None       # Predict Output

        self.Q = None
        self.R = None

        self.inno = None        # Innovation Signal

        # Aux.
        self.Ts = None


    def initialization(self, ekf_params):
        '''
            x_dot = Ax + Bu + Lv
            y = Hx + w
        ''' 

        h_rows = ekf_params['H_rows']
        h_colmns = ekf_params['H_colmns']
        self.H =  np.zeros((h_rows, h_colmns))
        num_measurements = self.H.shape[0]
        r_idx = [i for i in range(0, num_measurements)]
        c_idx = [0, 1, 2, 2]
        self.H[r_idx, c_idx] = 1

        self.X0 = np.array(ekf_params['X0'])
        num_states = self.X0.shape[0]
        self.L = np.identity(num_states)

        P0_factor = ekf_params['P0_factor']
        self.P0 = P0_factor*np.identity(num_states)

        # Cov. Processing Noise 
        self.Q = np.zeros((num_states, num_states))
        Q_sigma_list = ekf_params['Q_sigma_list']
        r_idx = [i for i in range(0, num_states)]
        self.Q[r_idx, r_idx] = Q_sigma_list

        # Cov. Measurement Noise 
        self.R = np.zeros((num_measurements, num_measurements))
        R_sigma_list = ekf_params['R_sigma_list']
        r_idx = [i for i in range(0, num_measurements)]
        self.R[r_idx, r_idx] = R_sigma_list   


        # Init.

        # Predict Output
        self.P_np1 = np.zeros_like(self.P0)
        self.P_nm1 = self.P0
        self.xhat_np1 = np.zeros_like(self.X0)
        self.xhat_nm1 = self.X0
    

    def compute(self, y, u, Ts):
        '''
            Keep in mind
                P_np1 = An*Pn*An' + Ln*Q*Ln';                 
                Kn = P_nm1*H'*inv(H*P_nm1*H' + R);            
                x_hat = xhat_nm1 + Kn*(y - H*xhat_nm1);       

            Output:
                X_hat 
        '''
                
        self.update(self.xhat_nm1, self.P_nm1, y)     # OUtput X_hat, Pn
        self.predict(self.X_hat, u, Ts)                 
        self.unit_delay_p()                           # Output P_nm1
        self.unit_delay_x()                           # Output xhat_nm1



    def predict(self, x_hat, u, Ts):
        '''            
            xhat_np1 = An*x_hat + Bn*u

            Output:
                    (1) self.xhat_np1
                    (2) self.P_np1 
        '''
        

        # Prepare states
        x_k = x_hat[0]
        y_k = x_hat[1]
        theta_k = x_hat[2]

        V = u[0]
        w = u[1]

        # Discretized Model
        x_kp1 = x_k + Ts*V*math.cos(theta_k)
        y_kp1 = y_k + Ts*V*math.sin(theta_k)
        theta_kp1 = theta_k + Ts*w

        # Estimation
        self.xhat_np1 = [x_kp1, y_kp1, theta_kp1]


        # Cov. Estimation
        self.jacobian_A(x_hat, u, Ts)                   # Compute An
        # P_np1 = An*Pn*An' + Ln*Q*Ln';
        An_Pn = np.matmul(self.An, self.Pn)
        p_term_1 = np.matmul(An_Pn, np.transpose(self.An))

        Ln = Ts*self.L
        Ln_Q = np.matmul(Ln, self.Q)
        p_term_2 = np.matmul(Ln_Q, np.transpose(Ln))

        self.P_np1 = p_term_1 + p_term_2

        

    def update(self, xhat_nm1, P_nm1, y):
        '''
            *y = [x y th th_gyro]

            Output:
                    (1) self.X_hat
                    (2) self.Pn
        '''
        
        # Update
        # Kn = P_nm1*H'*inv(H*P_nm1*H' + R)

        H_trans = np.transpose(self.H)
        P_nm1_H = np.matmul(P_nm1, H_trans)
        H_P_nm1 = np.matmul(self.H, P_nm1)
        H_P_nm1_H = np.matmul(H_P_nm1, H_trans)
        Kn_term_2 = np.linalg.inv(H_P_nm1_H + self.R)

        Kn = np.matmul(P_nm1_H, Kn_term_2)

        # x_hat = xhat_nm1 + Kn*(y - H*xhat_nm1);
        self.inno = y - np.matmul(self.H, xhat_nm1)
        x_term_2 = np.matmul(Kn, self.inno)
        self.X_hat = xhat_nm1 + x_term_2

              
        #Pn = P_nm1 - Kn*H*P_nm1;
        Kn_H = np.matmul(Kn, self.H)
        Kn_H_P = np.matmul(Kn_H, P_nm1) 
        self.Pn = P_nm1 - Kn_H_P


        # % Innovation Analysis
        # i = (y - H*x_hat);
        # Cii = H*Pn*H' + R;


    def jacobian_A(self, X, u, Ts):
        '''
            Linearization of A from State-Space

                *u = [V, w]
        '''
        # x = X[0]
        # y = X[1]
        theta = X[2]

        V = u[0]
        w = u[1]
 
        # Jacobian
        num_states = self.X0.shape[0]
        self.An = np.zeros((num_states, num_states))
        self.An[0, :] = [1, 0, -V*Ts*math.sin(theta)]
        self.An[1, :] = [1, 0, V*Ts*math.cos(theta)]
        self.An[2, :] = [0, 0, 1]        


    def unit_delay_p(self):
        '''
            Update self.P_nm1
        '''

        self.P_nm1 = self.P_np1

    def unit_delay_x(self):
        '''
            Update self.xhat_nm1
        '''

        self.xhat_nm1 = self.xhat_np1


    def unit_delay_inte_p(self, Ts):
        '''
            self.P_nm1
        '''

        self.P_nm1 = self.P_nm1 + Ts*self.P_np1

    
    def unit_delay_inte_x(self, Ts):
        '''
            self.xhat_nm1
        '''

        self.xhat_nm1 = self.xhat_nm1 + Ts*self.xhat_np1

    
    def joint_measures(self, y1, y2):
        pass

    def preprocessing_input(self):
        pass
