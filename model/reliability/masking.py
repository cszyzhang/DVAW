from __future__ import annotations 

import torch 
import torch .nn .functional as F 

from .regions import minmax_norm ,sobel_grad_mag 


def patch_saliency_mask (score :torch .Tensor ,*,patch_hw :tuple [int ,int ]=(16 ,16 ),topk_ratio :float =0.25 )->torch .Tensor :

    if score .ndim !=4 or int (score .shape [1 ])!=1 :
        raise ValueError (f"score must be (B,1,H,W), got {tuple(score.shape)}")
    B ,_ ,H ,W =score .shape 
    ph ,pw =int (patch_hw [0 ]),int (patch_hw [1 ])
    if (int (H )%ph )!=0 or (int (W )%pw )!=0 :
        raise ValueError (f"H,W must be divisible by patch size. Got H={int(H)} W={int(W)} patch={patch_hw}")

    gh ,gw =int (H )//ph ,int (W )//pw 
    s =score .view (int (B ),1 ,gh ,ph ,gw ,pw ).mean (dim =(3 ,5 ))
    sflat =s .view (int (B ),-1 )
    k =max (1 ,int (float (topk_ratio )*int (sflat .shape [1 ])))
    topk =torch .topk (sflat ,k =k ,dim =1 ).indices 

    mask_patch =torch .zeros_like (sflat )
    mask_patch .scatter_ (1 ,topk ,1.0 )
    mask_patch =mask_patch .view (int (B ),1 ,gh ,gw )
    return mask_patch .repeat_interleave (ph ,dim =2 ).repeat_interleave (pw ,dim =3 )


def make_entropy_dynamic_score (img_ref :torch .Tensor ,dyn_map :torch .Tensor ,*,out_hw :tuple [int ,int ])->torch .Tensor :

    h ,w =int (out_hw [0 ]),int (out_hw [1 ])
    gray =(0.299 *img_ref [:,0 :1 ]+0.587 *img_ref [:,1 :2 ]+0.114 *img_ref [:,2 :3 ]).clamp (0 ,1 )
    gray =F .interpolate (gray ,size =(h ,w ),mode ="bilinear",align_corners =False )
    Hm =minmax_norm (sobel_grad_mag (gray ))
    Dm =F .interpolate (dyn_map .detach (),size =(h ,w ),mode ="bilinear",align_corners =False )
    Dm =minmax_norm (Dm )
    return (Hm *Dm ).clamp (0 ,1 )


def make_residual_score (dyn_map :torch .Tensor ,*,out_hw :tuple [int ,int ])->torch .Tensor :
    h ,w =int (out_hw [0 ]),int (out_hw [1 ])
    Dm =F .interpolate (dyn_map .detach (),size =(h ,w ),mode ="bilinear",align_corners =False )
    return minmax_norm (Dm ).clamp (0 ,1 )

