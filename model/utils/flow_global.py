from __future__ import annotations 

from typing import Tuple 

import torch 
import torch .nn .functional as F 


def remove_lowpass (flow :torch .Tensor ,*,kernel_size :int =31 )->Tuple [torch .Tensor ,torch .Tensor ]:

    if flow .ndim !=4 or flow .shape [1 ]!=2 :
        raise ValueError (f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
    k =int (kernel_size )
    if k <=1 or (k %2 )==0 :
        raise ValueError ("kernel_size must be an odd integer >= 3")
    pad =k //2 
    lp =F .avg_pool2d (flow ,kernel_size =k ,stride =1 ,padding =pad )
    return flow -lp ,lp 


def remove_affine (flow :torch .Tensor )->Tuple [torch .Tensor ,torch .Tensor ]:

    if flow .ndim !=4 or flow .shape [1 ]!=2 :
        raise ValueError (f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
    b ,_ ,h ,w =flow .shape 
    device =flow .device 
    dtype =flow .dtype 


    ys =torch .linspace (-(h -1 )/2.0 ,(h -1 )/2.0 ,h ,device =device ,dtype =dtype )
    xs =torch .linspace (-(w -1 )/2.0 ,(w -1 )/2.0 ,w ,device =device ,dtype =dtype )
    yy ,xx =torch .meshgrid (ys ,xs ,indexing ="ij")
    ones =torch .ones_like (xx )
    X =torch .stack ([xx ,yy ,ones ],dim =-1 ).view (-1 ,3 )


    XtX =X .T @X 
    reg =1e-4 *torch .eye (3 ,device =device ,dtype =dtype )
    XtX_inv =torch .linalg .inv (XtX +reg )
    pinv =XtX_inv @X .T 

    dx =flow [:,0 ].reshape (b ,-1 )
    dy =flow [:,1 ].reshape (b ,-1 )
    coeff_x =(pinv @dx .T ).T 
    coeff_y =(pinv @dy .T ).T 

    dx_fit =(X @coeff_x .T ).T .view (b ,h ,w )
    dy_fit =(X @coeff_y .T ).T .view (b ,h ,w )
    affine =torch .stack ([dx_fit ,dy_fit ],dim =1 )
    return flow -affine ,affine 

