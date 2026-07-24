from __future__ import annotations 

import math 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 

from .coords import make_grid_pixel 


def l2n (x :torch .Tensor ,*,eps :float =1e-6 )->torch .Tensor :
    return x /(x .norm (dim =-1 ,keepdim =True )+float (eps ))


class TransportPatchEncoder (nn .Module ):


    def __init__ (self ,*,out_dim :int =64 ):
        super ().__init__ ()
        od =int (out_dim )
        self .net =nn .Sequential (
        nn .Conv2d (3 ,32 ,3 ,stride =2 ,padding =1 ),
        nn .ReLU (inplace =True ),
        nn .Conv2d (32 ,32 ,3 ,stride =2 ,padding =1 ),
        nn .ReLU (inplace =True ),
        nn .Conv2d (32 ,64 ,3 ,stride =2 ,padding =1 ),
        nn .ReLU (inplace =True ),
        nn .Conv2d (64 ,od ,3 ,stride =2 ,padding =1 ),
        )

    def forward (self ,x :torch .Tensor )->torch .Tensor :
        return self .net (x )


class RelPosBias2D (nn .Module ):


    def __init__ (self ,*,h :int ,w :int ):
        super ().__init__ ()
        self .h =int (h )
        self .w =int (w )
        num_rel =(2 *int (h )-1 )*(2 *int (w )-1 )
        self .table =nn .Parameter (torch .zeros (int (num_rel )))

        coords =torch .stack (torch .meshgrid (torch .arange (int (h )),torch .arange (int (w )),indexing ="ij"),dim =-1 ).view (-1 ,2 )
        rel =coords [:,None ,:]-coords [None ,:,:]
        rel [...,0 ]+=int (h )-1 
        rel [...,1 ]+=int (w )-1 
        idx =rel [...,0 ]*(2 *int (w )-1 )+rel [...,1 ]
        self .register_buffer ("index",idx .long (),persistent =False )

    def forward (self )->torch .Tensor :
        return self .table [self .index ]


def cycle_return_prob (Pab :torch .Tensor ,Pba :torch .Tensor )->torch .Tensor :

    return (Pab *Pba .transpose (1 ,2 )).sum (dim =-1 )


def perm_inverse (perm :torch .Tensor )->torch .Tensor :
    inv =torch .empty_like (perm )
    inv [perm ]=torch .arange (int (perm .numel ()),device =perm .device )
    return inv 


class TrustedTransportModule (nn .Module ):


    def __init__ (
    self ,
    *,
    token_hw :tuple [int ,int ]=(14 ,14 ),
    feat_dim :int =64 ,
    temp :float =0.07 ,
    pos_mode :str ="abs",
    ):
        super ().__init__ ()
        self .h ,self .w =int (token_hw [0 ]),int (token_hw [1 ])
        self .N =int (self .h )*int (self .w )
        self .temp =float (temp )
        self .pos_mode =str (pos_mode ).strip ().lower ()

        self .enc =TransportPatchEncoder (out_dim =int (feat_dim ))

        self .pos :torch .nn .Parameter |None 
        self .rel :RelPosBias2D |None 
        if self .pos_mode =="abs":
            self .pos =nn .Parameter (torch .zeros (self .N ,int (feat_dim )))
            nn .init .trunc_normal_ (self .pos ,std =0.02 )
            self .rel =None 
        elif self .pos_mode =="rel":
            self .pos =None 
            self .rel =RelPosBias2D (h =int (self .h ),w =int (self .w ))
        else :
            raise ValueError (f"pos_mode must be 'abs' or 'rel', got {pos_mode!r}")

        coords =make_grid_pixel (int (self .h ),int (self .w ),device ="cpu")
        self .register_buffer ("coords_hw",coords ,persistent =False )

    def _tokens (self ,img :torch .Tensor )->torch .Tensor :
        f =self .enc (img )
        f =F .interpolate (f ,size =(int (self .h ),int (self .w )),mode ="bilinear",align_corners =False )
        B ,D ,h ,w =f .shape 
        tok =f .permute (0 ,2 ,3 ,1 ).reshape (int (B ),int (h )*int (w ),int (D ))
        if self .pos is not None :
            tok =tok +self .pos [None ,:,:]
        return l2n (tok )

    def _sim (self ,tok_a :torch .Tensor ,tok_b :torch .Tensor )->torch .Tensor :
        sim =torch .einsum ("bnd,bmd->bnm",tok_a ,tok_b )/float (max (self .temp ,1e-6 ))
        if self .rel is not None :
            sim =sim +self .rel ()[None ,:,:]
        return sim 

    def soft_flow_from_P (self ,Pab :torch .Tensor ,*,full_hw_px :tuple [int ,int ])->torch .Tensor :

        if Pab .ndim !=3 :
            raise ValueError (f"Pab must be (B,N,N), got {tuple(Pab.shape)}")
        B ,N ,N2 =Pab .shape 
        if int (N )!=int (self .N )or int (N2 )!=int (self .N ):
            raise ValueError (f"Pab N mismatch: expected N={int(self.N)} got {tuple(Pab.shape)}")

        Hpx ,Wpx =int (full_hw_px [0 ]),int (full_hw_px [1 ])
        coords =self .coords_hw .to (Pab .device ).reshape (-1 ,2 )

        sx =float (Wpx )/float (self .w )
        sy =float (Hpx )/float (self .h )
        coords_full =coords .clone ()
        coords_full [:,0 ]=coords_full [:,0 ]*sx 
        coords_full [:,1 ]=coords_full [:,1 ]*sy 
        coords_full =coords_full [None ,:,:].repeat (int (B ),1 ,1 )

        exp_tgt =torch .bmm (Pab ,coords_full )
        disp =exp_tgt -coords_full 
        return disp .reshape (int (B ),int (self .h ),int (self .w ),2 ).permute (0 ,3 ,1 ,2 )

    def forward_pair (
    self ,
    img_a :torch .Tensor ,
    img_b :torch .Tensor ,
    *,
    full_hw_px :tuple [int ,int ],
    do_labelwarp :bool =False ,
    rng :torch .Generator |None =None ,
    )->tuple [torch .Tensor ,torch .Tensor ,torch .Tensor ,dict ]:
        tok_a =self ._tokens (img_a )
        tok_b =self ._tokens (img_b )

        sim_ab =self ._sim (tok_a ,tok_b )
        sim_ba =self ._sim (tok_b ,tok_a )

        Pab =torch .softmax (sim_ab ,dim =-1 )
        Pba =torch .softmax (sim_ba ,dim =-1 )

        ret =cycle_return_prob (Pab ,Pba )
        loss_cycle =-torch .log (ret +1e-8 ).mean ()

        diag_mass =torch .diagonal (Pab ,dim1 =1 ,dim2 =2 )
        p_safe =Pab .clamp_min (1e-8 )
        ent =-(p_safe *p_safe .log ()).sum (dim =-1 )
        ent =ent /float (max (math .log (float (max (self .N ,2 ))),1e-6 ))

        diag ={
        "transport_retprob_mean":ret .mean ().detach (),
        "transport_diag_mass":diag_mass .mean ().detach (),
        "diag_mass_T_mean":diag_mass .mean ().detach (),
        "entropy_T_mean":ent .mean ().detach (),
        "transport_T_matrix":Pab .detach (),
        }

        loss_lw =torch .zeros ([],device =img_a .device ,dtype =img_a .dtype )
        if do_labelwarp :
            if rng is None :
                perm =torch .randperm (int (self .N ),device =img_a .device )
            else :
                perm =torch .randperm (int (self .N ),device =img_a .device ,generator =rng )
            inv =perm_inverse (perm )
            tok_b_p =tok_b [:,perm ,:]
            sim_ab_p =torch .einsum ("bnd,bmd->bnm",tok_a ,tok_b_p )/float (max (self .temp ,1e-6 ))
            if self .rel is not None :
                sim_ab_p =sim_ab_p +self .rel ()[None ,:,:]
            Pab_p =torch .softmax (sim_ab_p ,dim =-1 )

            labels =inv [None ,:].repeat (int (img_a .shape [0 ]),1 )
            loss_lw =F .nll_loss (torch .log (Pab_p .reshape (-1 ,int (self .N ))+1e-8 ),labels .reshape (-1 ))
            diag ["transport_labelwarp_loss"]=loss_lw .detach ()
            diag ["transport_labelwarp_diag_mass"]=torch .diagonal (Pab_p ,dim1 =1 ,dim2 =2 ).mean ().detach ()

        flow_soft =self .soft_flow_from_P (Pab ,full_hw_px =full_hw_px )
        return loss_cycle ,loss_lw ,flow_soft ,diag 
