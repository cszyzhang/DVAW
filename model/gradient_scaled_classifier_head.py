from __future__ import annotations 

from typing import Optional 

import torch 
import torch .nn as nn 


class GradientScaledClassifierHead (nn .Module ):


    def __init__ (
    self ,
    *,
    in_dim :int ,
    num_classes :int ,
    hidden_dim :int =192 ,
    num_layers :int =2 ,
    dropout :float =0.1 ,
    )->None :
        super ().__init__ ()
        d_in =int (in_dim )
        d_h =int (min (max (int (hidden_dim ),1 ),192 ))
        n_layers =int (max (int (num_layers ),1 ))
        p =float (max (float (dropout ),0.0 ))

        layers =[]
        d_cur =d_in 
        for _ in range (n_layers ):
            layers .append (nn .Linear (d_cur ,d_h ,bias =True ))
            layers .append (nn .GELU ())
            if p >0.0 :
                layers .append (nn .Dropout (p =p ))
            d_cur =d_h 
        self .backbone =nn .Sequential (*layers )
        self .proj_back =nn .Linear (d_cur ,d_in ,bias =True )
        self .classifier =nn .Linear (d_in ,int (num_classes ),bias =True )

    def adapt (self ,feat :torch .Tensor )->torch .Tensor :
        if feat .ndim !=2 :
            raise ValueError (f"GradientScaledClassifierHead expects [B,D], got {tuple(feat.shape)}")
        h =self .backbone (feat )
        delta =self .proj_back (h )
        return feat +delta 

    def forward (self ,feat :torch .Tensor )->torch .Tensor :
        z =self .adapt (feat )
        return self .classifier (z )

    def parameter_count (self )->int :
        return int (sum (int (p .numel ())for p in self .parameters ()))
