from __future__ import annotations 

from typing import Dict ,Tuple 

import torch 
import torch .nn .functional as F 


_GRID_CACHE :Dict [Tuple [int ,int ,torch .device ,torch .dtype ,bool ],torch .Tensor ]={}


def make_base_grid (
batch_size :int ,
height :int ,
width :int ,
*,
device :torch .device ,
dtype :torch .dtype =torch .float32 ,
align_corners :bool =True ,
)->torch .Tensor :

    h =int (height )
    w =int (width )
    b =int (batch_size )
    if h <=0 or w <=0 or b <=0 :
        raise ValueError (f"Invalid grid size: B={b} H={h} W={w}")

    key =(h ,w ,device ,dtype ,bool (align_corners ))
    base =_GRID_CACHE .get (key )
    if base is None or base .device !=device or base .dtype !=dtype :
        if align_corners :
            xs =torch .linspace (-1.0 ,1.0 ,w ,device =device ,dtype =dtype )
            ys =torch .linspace (-1.0 ,1.0 ,h ,device =device ,dtype =dtype )
        else :


            xs =torch .linspace (-1.0 +1.0 /w ,1.0 -1.0 /w ,w ,device =device ,dtype =dtype )
            ys =torch .linspace (-1.0 +1.0 /h ,1.0 -1.0 /h ,h ,device =device ,dtype =dtype )
        yy ,xx =torch .meshgrid (ys ,xs ,indexing ="ij")
        base =torch .stack ([xx ,yy ],dim =-1 ).unsqueeze (0 )
        _GRID_CACHE [key ]=base 
    return base .expand (b ,-1 ,-1 ,-1 )


def warp (
img :torch .Tensor ,
flow :torch .Tensor ,
*,
padding_mode :str ="border",
mode :str ="bilinear",
align_corners :bool =True ,
)->torch .Tensor :

    if img .ndim !=4 :
        raise ValueError (f"img must be [B,C,H,W], got {tuple(img.shape)}")
    if flow .ndim !=4 or flow .shape [1 ]!=2 :
        raise ValueError (f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
    if img .shape [0 ]!=flow .shape [0 ]or img .shape [-2 :]!=flow .shape [-2 :]:
        raise ValueError (f"img/flow batch or spatial mismatch: img={tuple(img.shape)} flow={tuple(flow.shape)}")

    b ,_ ,h ,w =img .shape 
    dtype =img .dtype 
    device =img .device 

    if align_corners :
        denom_x =(w -1 )/2.0 if w >1 else 1.0 
        denom_y =(h -1 )/2.0 if h >1 else 1.0 
    else :
        denom_x =w /2.0 
        denom_y =h /2.0 

    flow_x =flow [:,0 :1 ]/float (denom_x )
    flow_y =flow [:,1 :2 ]/float (denom_y )
    flow_norm =torch .cat ([flow_x ,flow_y ],dim =1 ).permute (0 ,2 ,3 ,1 )

    base =make_base_grid (b ,h ,w ,device =device ,dtype =dtype ,align_corners =align_corners )
    grid =base +flow_norm 

    return F .grid_sample (
    img ,
    grid ,
    mode =mode ,
    padding_mode =padding_mode ,
    align_corners =align_corners ,
    )


def flow_valid_mask (
flow :torch .Tensor ,
*,
align_corners :bool =True ,
eps :float =0.0 ,
)->torch .Tensor :

    if flow .ndim !=4 or flow .shape [1 ]!=2 :
        raise ValueError (f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
    if bool (align_corners )is not True :
        raise ValueError ("flow_valid_mask currently requires align_corners=True for strict warp consistency.")

    b ,_ ,h ,w =flow .shape 
    device =flow .device 
    dtype =flow .dtype 

    denom_x =(w -1 )/2.0 if w >1 else 1.0 
    denom_y =(h -1 )/2.0 if h >1 else 1.0 

    flow_x =flow [:,0 :1 ]/float (denom_x )
    flow_y =flow [:,1 :2 ]/float (denom_y )
    flow_norm =torch .cat ([flow_x ,flow_y ],dim =1 ).permute (0 ,2 ,3 ,1 )

    base =make_base_grid (int (b ),int (h ),int (w ),device =device ,dtype =dtype ,align_corners =align_corners )
    grid =base +flow_norm 

    e =float (eps )
    valid =(
    (grid [...,0 ]>=(-1.0 +e ))
    &(grid [...,0 ]<=(1.0 -e ))
    &(grid [...,1 ]>=(-1.0 +e ))
    &(grid [...,1 ]<=(1.0 -e ))
    )
    return valid .to (dtype =dtype ).unsqueeze (1 )


def affine_warp (
img :torch .Tensor ,
theta :torch .Tensor ,
*,
padding_mode :str ="border",
mode :str ="bilinear",
align_corners :bool =True ,
)->torch .Tensor :

    if img .ndim !=4 :
        raise ValueError (f"img must be [B,C,H,W], got {tuple(img.shape)}")
    if theta .ndim ==2 and int (theta .shape [1 ])==6 :
        theta =theta .view (int (theta .shape [0 ]),2 ,3 )
    if theta .ndim !=3 or tuple (theta .shape [1 :])!=(2 ,3 ):
        raise ValueError (f"theta must be [B,2,3] or [B,6], got {tuple(theta.shape)}")
    if int (theta .shape [0 ])!=int (img .shape [0 ]):
        raise ValueError (f"theta/img batch mismatch: theta={tuple(theta.shape)} img={tuple(img.shape)}")

    theta =theta .to (device =img .device ,dtype =img .dtype )
    grid =F .affine_grid (theta ,size =img .size (),align_corners =align_corners )
    return F .grid_sample (img ,grid ,mode =mode ,padding_mode =padding_mode ,align_corners =align_corners )
