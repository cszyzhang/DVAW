from __future__ import annotations 

from typing import Dict ,Tuple 

import torch 
import torch .nn .functional as F 


def info_nce (z1 :torch .Tensor ,z2 :torch .Tensor ,*,tau :float =0.2 )->tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:

    if z1 .ndim !=2 or z2 .ndim !=2 or z1 .shape !=z2 .shape :
        raise ValueError (f"z1/z2 must be [B,D] with same shape. Got z1={tuple(z1.shape)} z2={tuple(z2.shape)}")
    z1n =F .normalize (z1 ,dim =1 )
    z2n =F .normalize (z2 ,dim =1 )
    bsz =int (z1n .shape [0 ])
    t =float (max (float (tau ),1e-6 ))

    logits =(z1n @z2n .t ())/t 
    labels =torch .arange (bsz ,device =z1 .device )
    loss =0.5 *(F .cross_entropy (logits ,labels )+F .cross_entropy (logits .t (),labels ))

    with torch .no_grad ():
        sim =z1n @z2n .t ()
        pos =sim .diag ().mean ()
        neg =(sim .sum ()-sim .diag ().sum ())/float (max (bsz *bsz -bsz ,1 ))
        gap =pos -neg 
    return loss ,{"pos_sim":pos ,"neg_sim":neg ,"pos_neg_gap":gap }


def simsiam_negcos (p :torch .Tensor ,z :torch .Tensor )->torch .Tensor :

    if p .ndim !=2 or z .ndim !=2 or p .shape !=z .shape :
        raise ValueError (f"p/z must be [B,D] with same shape. Got p={tuple(p.shape)} z={tuple(z.shape)}")
    p =F .normalize (p ,dim =1 )
    z =F .normalize (z ,dim =1 )
    return 1.0 -(p *z ).sum (dim =1 ).mean ()


def masked_l1 (pred :torch .Tensor ,target :torch .Tensor ,mask :torch .Tensor )->torch .Tensor :

    if pred .shape !=target .shape :
        raise ValueError (f"pred/target must have same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}")
    if mask .ndim !=4 or int (mask .shape [1 ])!=1 or tuple (mask .shape [-2 :])!=tuple (pred .shape [-2 :]):
        raise ValueError (f"mask must be [B,1,H,W] matching pred spatial. Got mask={tuple(mask.shape)} pred={tuple(pred.shape)}")
    m =mask .to (device =pred .device ,dtype =pred .dtype )
    return (torch .abs (pred -target )*m ).sum ()/(m .sum ()+1e-6 )


def rank_hinge (
a :torch .Tensor ,
b :torch .Tensor ,
*,
margin :float =0.0 ,
reduction :str ="mean",
)->torch .Tensor :

    h =F .relu (b -a +float (margin ))
    r =str (reduction ).strip ().lower ()
    if r in {"none",""}:
        return h 
    if r =="mean":
        return h .mean ()
    if r =="sum":
        return h .sum ()
    raise ValueError (f"Unsupported reduction={reduction!r}; choose from none/mean/sum.")
