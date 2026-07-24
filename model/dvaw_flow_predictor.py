from __future__ import annotations 

from typing import Dict ,Tuple 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 


def _group_norm (num_channels :int ,*,num_groups :int =8 )->nn .GroupNorm :
    c =int (num_channels )
    g =min (int (num_groups ),c )
    while g >1 and (c %g )!=0 :
        g -=1 
    return nn .GroupNorm (g ,c )


def _conv_gn_relu (in_ch :int ,out_ch :int ,*,kernel_size :int =3 ,stride :int =1 )->nn .Sequential :
    pad =kernel_size //2 
    return nn .Sequential (
    nn .Conv2d (in_ch ,out_ch ,kernel_size =kernel_size ,stride =stride ,padding =pad ,bias =False ),
    _group_norm (out_ch ),
    nn .ReLU (inplace =True ),
    )


class _UNetLite (nn .Module ):
    def __init__ (self ,in_channels :int ,base_channels :int =32 ):
        super ().__init__ ()
        b =int (base_channels )
        self .enc1 =nn .Sequential (
        _conv_gn_relu (in_channels ,b ,stride =2 ),
        _conv_gn_relu (b ,b ,stride =1 ),
        )
        self .enc2 =nn .Sequential (
        _conv_gn_relu (b ,2 *b ,stride =2 ),
        _conv_gn_relu (2 *b ,2 *b ,stride =1 ),
        )
        self .enc3 =nn .Sequential (
        _conv_gn_relu (2 *b ,4 *b ,stride =2 ),
        _conv_gn_relu (4 *b ,4 *b ,stride =1 ),
        )

        self .dec2 =nn .Sequential (
        _conv_gn_relu (4 *b +2 *b ,2 *b ,stride =1 ),
        _conv_gn_relu (2 *b ,2 *b ,stride =1 ),
        )
        self .dec1 =nn .Sequential (
        _conv_gn_relu (2 *b +b ,b ,stride =1 ),
        _conv_gn_relu (b ,b ,stride =1 ),
        )
        self .dec0 =nn .Sequential (
        _conv_gn_relu (b ,b ,stride =1 ),
        _conv_gn_relu (b ,b ,stride =1 ),
        )

    def forward (self ,x :torch .Tensor )->Tuple [torch .Tensor ,torch .Tensor ,torch .Tensor ,torch .Tensor ]:
        e1 =self .enc1 (x )
        e2 =self .enc2 (e1 )
        e3 =self .enc3 (e2 )

        d2 =F .interpolate (e3 ,size =e2 .shape [-2 :],mode ="bilinear",align_corners =True )
        d2 =self .dec2 (torch .cat ([d2 ,e2 ],dim =1 ))

        d1 =F .interpolate (d2 ,size =e1 .shape [-2 :],mode ="bilinear",align_corners =True )
        d1 =self .dec1 (torch .cat ([d1 ,e1 ],dim =1 ))

        d0 =F .interpolate (d1 ,scale_factor =2 ,mode ="bilinear",align_corners =True )
        d0 =self .dec0 (d0 )

        return d0 ,d1 ,d2 ,e3 


class DVAWFlowPredictor (nn .Module ):


    def __init__ (
    self ,
    *,
    image_channels :int ,
    base_channels :int =32 ,
    flow_downscale :int =4 ,
    max_disp :float =6.0 ,
    basis_displacement_enabled :bool =False ,
    basis_displacement_K :int =8 ,
    basis_displacement_base_res :int =28 ,
    dynamic_support_mask_enabled :bool =False ,
    superpoints_K :int =0 ,
    ):
        super ().__init__ ()
        self .image_channels =int (image_channels )
        self .flow_downscale =int (flow_downscale )
        if self .flow_downscale not in {1 ,2 ,4 ,8 }:
            raise ValueError (f"flow_downscale must be one of {{1,2,4,8}}, got {self.flow_downscale}")
        self .max_disp =float (max_disp )
        self .basis_displacement_enabled =bool (basis_displacement_enabled )
        self .basis_displacement_K =int (basis_displacement_K )
        self .basis_displacement_base_res =int (basis_displacement_base_res )

        self .dynamic_support_mask_enabled =bool (dynamic_support_mask_enabled )
        self .superpoints_K =int (superpoints_K )
        if self .superpoints_K <0 :
            raise ValueError ("superpoints_K must be >= 0")
        if self .basis_displacement_enabled :
            if self .basis_displacement_K <=0 :
                raise ValueError ("basis_displacement_K must be > 0 when basis_displacement_enabled")
            if self .basis_displacement_base_res <=1 :
                raise ValueError ("basis_displacement_base_res must be > 1 when basis_displacement_enabled")

        b =int (base_channels )
        in_ch =2 *self .image_channels 
        self .backbone =_UNetLite (in_channels =in_ch ,base_channels =b )
        self .proj0 =nn .Identity ()
        self .proj1 =nn .Identity ()
        self .proj2 =nn .Conv2d (2 *b ,b ,kernel_size =1 ,bias =False )
        self .proj3 =nn .Conv2d (4 *b ,b ,kernel_size =1 ,bias =False )
        self .head =nn .Conv2d (b ,2 ,kernel_size =1 ,bias =True )
        nn .init .zeros_ (self .head .weight )
        nn .init .zeros_ (self .head .bias )

        self .mask_head :nn .Conv2d |None =None 
        if self .dynamic_support_mask_enabled :
            self .mask_head =nn .Conv2d (b ,1 ,kernel_size =1 ,bias =True )
            nn .init .zeros_ (self .mask_head .weight )
            nn .init .constant_ (self .mask_head .bias ,-2.0 )

        self .superpoint_head :nn .Conv2d |None =None 
        if self .superpoints_K >0 :
            self .superpoint_head =nn .Conv2d (b ,int (self .superpoints_K ),kernel_size =1 ,bias =True )
            nn .init .zeros_ (self .superpoint_head .weight )
            nn .init .zeros_ (self .superpoint_head .bias )

        self .basis :nn .Parameter |None =None 
        self .coeff_fc :nn .Linear |None =None 
        if self .basis_displacement_enabled :
            self .basis =nn .Parameter (torch .zeros (self .basis_displacement_K ,2 ,self .basis_displacement_base_res ,self .basis_displacement_base_res ))
            nn .init .normal_ (self .basis ,mean =0.0 ,std =0.01 )
            self .coeff_fc =nn .Linear (b ,self .basis_displacement_K ,bias =True )
            nn .init .zeros_ (self .coeff_fc .weight )
            nn .init .zeros_ (self .coeff_fc .bias )

    def _compute_feats (self ,src :torch .Tensor ,tgt :torch .Tensor )->Tuple [torch .Tensor ,torch .Tensor ,torch .Tensor ,torch .Tensor ,int ,int ]:
        if src .ndim !=4 or tgt .ndim !=4 :
            raise ValueError (f"src/tgt must be [B,C,H,W]. Got src={tuple(src.shape)} tgt={tuple(tgt.shape)}")
        if src .shape !=tgt .shape :
            raise ValueError (f"src/tgt must have same shape. Got src={tuple(src.shape)} tgt={tuple(tgt.shape)}")
        if src .shape [1 ]!=self .image_channels :
            raise ValueError (f"Expected image_channels={self.image_channels}, got src.shape[1]={src.shape[1]}")

        bsz ,_ ,h ,w =src .shape 
        x =torch .cat ([src ,tgt ],dim =1 )
        d0 ,d1 ,d2 ,e3 =self .backbone (x )
        return d0 ,d1 ,d2 ,e3 ,int (h ),int (w )

    def forward_with_aux (self ,src :torch .Tensor ,tgt :torch .Tensor )->Tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:
        d0 ,d1 ,d2 ,e3 ,h ,w =self ._compute_feats (src ,tgt )
        if self .flow_downscale ==1 :
            feat =self .proj0 (d0 )
        elif self .flow_downscale ==2 :
            feat =self .proj1 (d1 )
        elif self .flow_downscale ==4 :
            feat =self .proj2 (d2 )
        else :
            feat =self .proj3 (e3 )

        scale =float (self .max_disp )*0.5 
        aux :Dict [str ,torch .Tensor ]={}
        if self .basis_displacement_enabled :
            if self .basis is None or self .coeff_fc is None :
                raise RuntimeError ("basis_displacement_enabled but basis/coeff_fc are not initialized")
            pooled =feat .mean (dim =(2 ,3 ))
            coeff =self .coeff_fc (pooled )
            basis_up =F .interpolate (self .basis ,size =(h ,w ),mode ="bilinear",align_corners =True )
            flow_raw =torch .einsum ("bk,kchw->bchw",coeff ,basis_up )
            flow =torch .tanh (flow_raw )*scale 
            aux ["basis_coeff"]=coeff 
        else :
            out_lr =self .head (feat )
            if self .flow_downscale !=1 :
                out =F .interpolate (out_lr ,size =(h ,w ),mode ="bilinear",align_corners =True )
            else :
                out =out_lr 
            flow =torch .tanh (out )*scale 

        if self .dynamic_support_mask_enabled :
            if self .mask_head is None :
                raise RuntimeError ("dynamic_support_mask_enabled but mask_head is not initialized")
            mask_lr =self .mask_head (feat )
            if self .flow_downscale !=1 :
                mask_lr =F .interpolate (mask_lr ,size =(h ,w ),mode ="bilinear",align_corners =True )
            aux ["mask_dyn"]=torch .sigmoid (mask_lr )

        if self .superpoints_K >0 :
            if self .superpoint_head is None :
                raise RuntimeError ("superpoints_K > 0 but superpoint_head is not initialized")

            feat_sp =self .proj3 (e3 )
            logits_sp =self .superpoint_head (feat_sp )
            bsz =int (logits_sp .shape [0 ])
            k =int (logits_sp .shape [1 ])
            flat =logits_sp .view (bsz ,k ,-1 )
            prob =torch .softmax (flat ,dim =2 ).view_as (logits_sp )
            aux ["sp_heatmaps"]=prob 
            m_lr =prob .sum (dim =1 ,keepdim =True ).clamp (0.0 ,1.0 )
            aux ["sp_mask"]=F .interpolate (m_lr ,size =(h ,w ),mode ="bilinear",align_corners =True ).clamp (0.0 ,1.0 )

        return flow ,aux 

    def forward (self ,src :torch .Tensor ,tgt :torch .Tensor )->torch .Tensor :
        flow ,_aux =self .forward_with_aux (src ,tgt )
        return flow 
