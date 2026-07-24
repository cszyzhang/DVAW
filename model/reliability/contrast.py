from __future__ import annotations 

from typing import Dict 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 


def l2n (x :torch .Tensor ,*,eps :float =1e-6 )->torch .Tensor :
    return x /(x .norm (dim =-1 ,keepdim =True )+float (eps ))


def simclr_pair_loss (z1 :torch .Tensor ,z2 :torch .Tensor ,*,temp :float =0.2 )->tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:

    if z1 .ndim !=2 or z2 .ndim !=2 or tuple (z1 .shape )!=tuple (z2 .shape ):
        raise ValueError (f"z1/z2 must be (N,D) with same shape. Got z1={tuple(z1.shape)} z2={tuple(z2.shape)}")
    z1 =l2n (z1 )
    z2 =l2n (z2 )
    logits =(z1 @z2 .T )/float (max (float (temp ),1e-6 ))
    labels =torch .arange (int (z1 .shape [0 ]),device =z1 .device )
    loss =0.5 *(F .cross_entropy (logits ,labels )+F .cross_entropy (logits .T ,labels ))

    with torch .no_grad ():
        pos =(z1 *z2 ).sum (dim =-1 ).mean ()
        eye =torch .eye (int (z1 .shape [0 ]),device =z1 .device ,dtype =torch .bool )
        neg =(z1 @z2 .T )[~eye ].mean ()
        gap =pos -neg 
    return loss ,{"pos_sim":pos .detach (),"neg_sim":neg .detach (),"gap":gap .detach ()}


def supcon_loss (
z :torch .Tensor ,
pos_mask :torch .Tensor ,
*,
temp :float =0.2 ,
eps :float =1e-8 ,
)->tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:

    if z .ndim !=2 :
        raise ValueError (f"z must be (M,D), got {tuple(z.shape)}")
    if pos_mask .ndim !=2 or tuple (pos_mask .shape )!=(int (z .shape [0 ]),int (z .shape [0 ])):
        raise ValueError (f"pos_mask must be (M,M), got {tuple(pos_mask.shape)}")

    z =l2n (z )
    sim =(z @z .T )/float (max (float (temp ),1e-6 ))
    self_mask =torch .eye (int (z .shape [0 ]),device =z .device ,dtype =torch .bool )
    sim =sim .masked_fill (self_mask ,-1e9 )

    exp_sim =torch .exp (sim )
    denom =exp_sim .sum (dim =1 ,keepdim =True )+float (eps )

    pos =exp_sim *pos_mask .float ()
    num_pos =pos_mask .float ().sum (dim =1 ).clamp_min (1.0 )
    loss =-((torch .log ((pos .sum (dim =1 ,keepdim =True )+float (eps ))/denom ))/num_pos [:,None ]).sum ()/float (int (z .shape [0 ]))

    with torch .no_grad ():
        cos =z @z .T 
        pos_cos =cos [pos_mask ].mean ()if bool (pos_mask .any ())else torch .tensor (0.0 ,device =z .device )
        neg_cos =cos [(~pos_mask )&(~self_mask )].mean ()
        gap =pos_cos -neg_cos 
    return loss ,{"pos_sim":pos_cos .detach (),"neg_sim":neg_cos .detach (),"gap":gap .detach ()}


class PairDisc (nn .Module ):


    def __init__ (self ,*,cin :int ,hid :int =64 ):
        super ().__init__ ()
        self .net =nn .Sequential (
        nn .Conv2d (int (cin ),int (hid ),3 ,padding =1 ),
        nn .ReLU (inplace =True ),
        nn .Conv2d (int (hid ),int (hid ),3 ,padding =1 ),
        nn .ReLU (inplace =True ),
        nn .AdaptiveAvgPool2d (1 ),
        )
        self .fc =nn .Linear (int (hid ),1 )

    def forward (self ,x :torch .Tensor )->torch .Tensor :
        h =self .net (x ).flatten (1 )
        return self .fc (h ).squeeze (1 )

