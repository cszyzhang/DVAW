from __future__ import annotations 

import torch 
import torch .nn .functional as F 

from .coords import make_grid_norm 


def warp_tensor (
x :torch .Tensor ,
flow :torch .Tensor ,
*,
mode :str ="bilinear",
padding_mode :str ="border",
align_corners :bool =True ,
)->torch .Tensor :

    if x .ndim !=4 :
        raise ValueError (f"x must be (B,C,H,W), got {tuple(x.shape)}")
    if flow .ndim !=4 or int (flow .shape [1 ])!=2 :
        raise ValueError (f"flow must be (B,2,H,W), got {tuple(flow.shape)}")
    if int (x .shape [0 ])!=int (flow .shape [0 ])or tuple (x .shape [-2 :])!=tuple (flow .shape [-2 :]):
        raise ValueError (f"x/flow batch or spatial mismatch: x={tuple(x.shape)} flow={tuple(flow.shape)}")

    b ,_c ,h ,w =x .shape 
    grid =make_grid_norm (int (h ),int (w ),device =x .device ,dtype =x .dtype )

    if align_corners :
        denom_x =(int (w )-1 )/2.0 if int (w )>1 else 1.0 
        denom_y =(int (h )-1 )/2.0 if int (h )>1 else 1.0 
    else :
        denom_x =int (w )/2.0 
        denom_y =int (h )/2.0 

    fx =flow [:,0 ]/float (denom_x )
    fy =flow [:,1 ]/float (denom_y )
    f =torch .stack ([fx ,fy ],dim =-1 )
    samp =grid +f 
    return F .grid_sample (x ,samp ,mode =mode ,padding_mode =padding_mode ,align_corners =align_corners )


def warp_flow (flow_bc :torch .Tensor ,flow_ab :torch .Tensor )->torch .Tensor :

    return warp_tensor (flow_bc ,flow_ab ,mode ="bilinear",padding_mode ="border",align_corners =True )


def compose_flow (flow_ab :torch .Tensor ,flow_bc :torch .Tensor )->torch .Tensor :

    return flow_ab +warp_flow (flow_bc ,flow_ab )

