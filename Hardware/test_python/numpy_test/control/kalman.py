#!/usr/bin/env python3

import numpy as np


class kalman:
    def __init__(self):
        
        # State
        X0 = None
        X = None
        X_nm1 = None
        X_hat = None
        u = None

        # Model
        A = None
        G = None            # L=B (One case)
        H = None
        
        # Kalman
        self.P0 = None
        self.P = None
        self.Pnm1 = None

        self.Q = None
        self.R = None

        self.K = None
        self.inno = None

    def initialization(self):
        pass