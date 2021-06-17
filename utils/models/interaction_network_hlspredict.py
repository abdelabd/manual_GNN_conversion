# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 21:19:31 2021

@author: Abdelrahman Elabd
"""
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class InteractionNetwork(nn.Module):
    def __init__(self, object_dim, relation_dim, effect_dim):
        super(InteractionNetwork, self).__init__()
        
        self.object_dim = object_dim
        self.relation_dim = relation_dim
        self.effect_dim = effect_dim

        self.phi_R1 = nn.Sequential(
            nn.Linear(2*object_dim + relation_dim, 150),
            nn.ELU(),
            nn.Linear(150, 150),
            nn.ELU(),
            nn.Linear(150, 150),
            nn.ELU(),
            nn.Linear(150, effect_dim)
        )

        self.phi_R2 = nn.Sequential(
            nn.Linear(2*object_dim + effect_dim, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )
        
        self.phi_O = nn.Sequential(
            nn.Linear(object_dim + effect_dim, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, object_dim)
        )
        

    def forward(self, X, Ra, Ro, Ri):
        # b = batch size (1)
        # n = number of nodes 
        # m = number of edges 
        # p = object dimension 
        # q = effect dimension 
        # r = relation dimension
        
        # INPUTS
        # X: b x p x n
        # Ri, Ro: b x m x n
        # Ra: b x m x r
        
        X = torch.transpose(X, 1, 2) # X: b x n x p

        # first marshalling step: build interaction terms
        Xi = torch.bmm(Ro, X) # Xi: b x m x p
        Xo = torch.bmm(Ri, X) # Xo: b x m x p
        m1 = torch.cat([Xi, Xo, Ra], dim=2) # m1: b x m x (2p+r)

        # relational model: determine effects
        E = self.phi_R1(m1) # E: b x m x q 

        # aggregation step: aggregate effects
        A = torch.bmm(torch.transpose(Ri, 1, 2), E) # A: b x n x q
        C = torch.cat([X, A], dim=2) # C: b x n x (p+q)

        # object model: re-embed hit features
        X_tilde = self.phi_O(C)

        # re-marshalling step: build new interaction terms
        Xi_tilde = torch.bmm(Ri, X_tilde)
        Xo_tilde = torch.bmm(Ro, X_tilde)
        m2 = torch.cat([Xi_tilde, Xo_tilde, E], dim=2)

        W = torch.sigmoid(self.phi_R2(m2))
        #W = torch.sigmoid(self.phi_R1(m2))
        return W
    
    def deconstruct_input_tensor(self, T):
        # T: b x m x (p + 2n + r)
        p = int(self.object_dim)
        r = int(self.relation_dim)
        m = int(T.shape[1])
        n = int((T.shape[2] - p - r)/2)
        diff = m - n
        
        X_prime = T[:, :, :p] # b x m x p
        X_prime = torch.transpose(X_prime, 1, 2) # b x p x m
        X = X_prime[:, :, :n] # b x p x n
    
        Ri = T[:, :, p:p+n] # b x m x n
        Ro = T[:, :, p+n:p+2*n] # b x m x n
        Ra = T[:, :, p+2*n:] # b x m x r
    
        return(X, Ri, Ro, Ra)
        
        
        
        
        
        
    
    
    

















