from __future__ import annotations 

from typing import Dict ,List ,Tuple 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 

from .coords import make_grid_norm 


def region_pool (feat :torch .Tensor ,weights :torch .Tensor ,*,eps :float =1e-6 )->torch .Tensor :

    if feat .ndim !=4 :
        raise ValueError (f"feat must be (B,C,H,W), got {tuple(feat.shape)}")
    if weights .ndim !=4 :
        raise ValueError (f"weights must be (B,R,H,W), got {tuple(weights.shape)}")
    b ,c ,h ,w =feat .shape 
    b2 ,r ,h2 ,w2 =weights .shape 
    if b !=b2 or h !=h2 or w !=w2 :
        raise ValueError (f"feat/weights shape mismatch: feat={tuple(feat.shape)} weights={tuple(weights.shape)}")

    wsum =weights .view (b ,r ,-1 ).sum (dim =-1 ,keepdim =True )+float (eps )
    f =feat .view (b ,c ,-1 ).transpose (1 ,2 )
    wv =weights .view (b ,r ,-1 )
    return torch .bmm (wv ,f )/wsum 


class SoftGaussianRegions (nn .Module ):


    def __init__ (
    self ,
    *,
    num_regions :int =9 ,
    init_grid_3x3 :bool =True ,
    sigma_init :float =0.45 ,
    tie_pairs :List [Tuple [int ,int ]]|None =None ,
    ):
        super ().__init__ ()
        self .num_regions =int (num_regions )

        if init_grid_3x3 and int (num_regions )==9 :
            xs =torch .tensor ([-0.66 ,0.0 ,0.66 ],dtype =torch .float32 )
            ys =torch .tensor ([-0.66 ,0.0 ,0.66 ],dtype =torch .float32 )
            yy ,xx =torch .meshgrid (ys ,xs ,indexing ="ij")
            mu =torch .stack ([xx .reshape (-1 ),yy .reshape (-1 )],dim =-1 )
        else :
            mu =torch .zeros ((int (num_regions ),2 ),dtype =torch .float32 )
        self .mu =nn .Parameter (mu )

        log_sigma =torch .log (torch .ones ((int (num_regions ),2 ),dtype =torch .float32 )*float (sigma_init ))
        self .log_sigma =nn .Parameter (log_sigma )


        if tie_pairs is None and int (num_regions )==9 :
            tie_pairs =[(0 ,2 ),(3 ,5 ),(6 ,8 )]
        self .tie_pairs =list (tie_pairs or [])

    def forward (
    self ,h :int ,w :int ,batch_size :int ,*,device =None ,dtype :torch .dtype =torch .float32 
    )->tuple [torch .Tensor ,Dict [str ,torch .Tensor ],Dict [str ,torch .Tensor ]]:
        grid =make_grid_norm (int (h ),int (w ),device =device ,dtype =dtype )
        mu =self .mu [:,None ,None ,:]
        sigma =torch .exp (self .log_sigma )[:,None ,None ,:]+1e-6 

        diff =grid -mu 
        dist2 =(diff [...,0 ]**2 )/(sigma [...,0 ]**2 )+(diff [...,1 ]**2 )/(sigma [...,1 ]**2 )
        logits =-dist2 
        logits =logits [None ,...].repeat (int (batch_size ),1 ,1 ,1 )
        wts =torch .softmax (logits ,dim =1 )


        p =wts .mean (dim =(0 ,2 ,3 ))
        uni =torch .ones_like (p )/float (p .numel ())
        cov_kl =torch .sum (p *(torch .log (p +1e-8 )-torch .log (uni +1e-8 )))


        sym =torch .zeros ([],device =device ,dtype =dtype )
        for a ,b in self .tie_pairs :
            ia =int (a )
            ib =int (b )
            sym =sym +(self .mu [ia ,0 ]+self .mu [ib ,0 ]).abs ()
            sym =sym +(self .mu [ia ,1 ]-self .mu [ib ,1 ]).abs ()
            sym =sym +(self .log_sigma [ia ]-self .log_sigma [ib ]).abs ().mean ()

        reg ={"cov_kl":cov_kl ,"sym":sym }
        diag ={
        "region_cov_kl":cov_kl .detach (),
        "region_sym_err":sym .detach (),
        "region_mu_mean":self .mu .detach ().mean (),
        "region_sigma_mean":torch .exp (self .log_sigma ).mean ().detach (),
        }
        return wts ,reg ,diag 


def sobel_grad_mag (gray :torch .Tensor )->torch .Tensor :

    if gray .ndim !=4 or int (gray .shape [1 ])!=1 :
        raise ValueError (f"gray must be (B,1,H,W), got {tuple(gray.shape)}")
    kx =torch .tensor ([[-1 ,0 ,1 ],[-2 ,0 ,2 ],[-1 ,0 ,1 ]],device =gray .device ,dtype =gray .dtype ).view (1 ,1 ,3 ,3 )
    ky =torch .tensor ([[-1 ,-2 ,-1 ],[0 ,0 ,0 ],[1 ,2 ,1 ]],device =gray .device ,dtype =gray .dtype ).view (1 ,1 ,3 ,3 )
    gx =F .conv2d (gray ,kx ,padding =1 )
    gy =F .conv2d (gray ,ky ,padding =1 )
    return torch .sqrt (gx *gx +gy *gy +1e-8 )


def minmax_norm (x :torch .Tensor ,*,eps :float =1e-6 )->torch .Tensor :
    mn =x .amin (dim =(2 ,3 ),keepdim =True )
    mx =x .amax (dim =(2 ,3 ),keepdim =True )
    return (x -mn )/(mx -mn +float (eps ))


def weighted_farthest_points (
score :torch .Tensor ,coords :torch .Tensor ,k :int ,valid_mask :torch .Tensor 
)->torch .Tensor :

    if score .ndim !=2 or coords .ndim !=3 or valid_mask .ndim !=2 :
        raise ValueError ("score must be (B,N), coords (B,N,2), valid_mask (B,N)")
    B ,N =score .shape 
    if int (coords .shape [0 ])!=int (B )or int (coords .shape [1 ])!=int (N )or int (coords .shape [2 ])!=2 :
        raise ValueError (f"coords must be (B,N,2), got {tuple(coords.shape)}")
    if tuple (valid_mask .shape )!=(int (B ),int (N )):
        raise ValueError (f"valid_mask must be (B,N), got {tuple(valid_mask.shape)}")

    idxs :List [torch .Tensor ]=[]
    score_valid =score .masked_fill (~valid_mask ,-1e9 )
    first =torch .argmax (score_valid ,dim =1 )
    idxs .append (first )

    sel =coords [torch .arange (B ,device =coords .device ),first ]
    min_d2 =((coords -sel [:,None ,:])**2 ).sum (dim =-1 )

    kk =int (max (int (k ),1 ))
    for _ in range (1 ,kk ):
        obj =score *torch .sqrt (min_d2 +1e-8 )
        obj =obj .masked_fill (~valid_mask ,-1e9 )
        nxt =torch .argmax (obj ,dim =1 )
        idxs .append (nxt )
        sel =coords [torch .arange (B ,device =coords .device ),nxt ]
        d2 =((coords -sel [:,None ,:])**2 ).sum (dim =-1 )
        min_d2 =torch .minimum (min_d2 ,d2 )

    return torch .stack (idxs ,dim =1 )


class SuperpointVoronoiRegions (nn .Module ):


    def __init__ (self ,*,num_pairs :int =4 ,add_center :bool =True ):
        super ().__init__ ()
        self .num_pairs =int (num_pairs )
        self .add_center =bool (add_center )

    def forward (
    self ,img_ref :torch .Tensor ,dyn_map :torch .Tensor ,out_hw :tuple [int ,int ]
    )->tuple [torch .Tensor ,List [Tuple [int ,int ]],Dict [str ,torch .Tensor ],Dict [str ,torch .Tensor ]]:
        if img_ref .ndim !=4 or int (img_ref .shape [1 ])!=3 :
            raise ValueError (f"img_ref must be (B,3,H,W), got {tuple(img_ref.shape)}")
        if dyn_map .ndim !=4 or int (dyn_map .shape [1 ])!=1 :
            raise ValueError (f"dyn_map must be (B,1,H,W), got {tuple(dyn_map.shape)}")

        B =int (img_ref .shape [0 ])
        h ,w =int (out_hw [0 ]),int (out_hw [1 ])
        if h <=0 or w <=0 :
            raise ValueError (f"out_hw must be positive, got {out_hw}")

        gray =(0.299 *img_ref [:,0 :1 ]+0.587 *img_ref [:,1 :2 ]+0.114 *img_ref [:,2 :3 ]).clamp (0 ,1 )
        gray =F .interpolate (gray ,size =(h ,w ),mode ="bilinear",align_corners =False )
        Hm =minmax_norm (sobel_grad_mag (gray ))

        Dm =F .interpolate (dyn_map .detach (),size =(h ,w ),mode ="bilinear",align_corners =False )
        Dm =minmax_norm (Dm )

        S =(Hm *Dm ).clamp (0 ,1 )
        Sflat =S .view (B ,-1 )

        grid =make_grid_norm (h ,w ,device =img_ref .device ,dtype =img_ref .dtype ).view (1 ,-1 ,2 ).repeat (B ,1 ,1 )
        valid =grid [...,0 ]<0 

        idx_left =weighted_farthest_points (Sflat ,grid ,self .num_pairs ,valid )
        pts_left =grid [torch .arange (B ,device =img_ref .device )[:,None ],idx_left ]

        pts_right =pts_left .clone ()
        pts_right [...,0 ]=-pts_right [...,0 ]

        pts =[pts_left ,pts_right ]
        if self .add_center :
            center_idx =torch .argmax (Sflat ,dim =1 )
            center_pt =grid [torch .arange (B ,device =img_ref .device ),center_idx ][:,None ,:]
            pts .append (center_pt )

        keypts =torch .cat (pts ,dim =1 )
        K =int (keypts .shape [1 ])

        dist2 =((grid [:,:,None ,:]-keypts [:,None ,:,:])**2 ).sum (dim =-1 )
        assign =torch .argmin (dist2 ,dim =-1 )
        onehot =F .one_hot (assign ,num_classes =K ).float ()
        wts =onehot .permute (0 ,2 ,1 ).reshape (B ,K ,h ,w )

        sym_pairs :List [Tuple [int ,int ]]=[(i ,i +self .num_pairs )for i in range (int (self .num_pairs ))]
        diag ={
        "sp_score_mean":Sflat .mean ().detach (),
        "sp_keypt_score_mean":torch .gather (Sflat ,1 ,idx_left ).mean ().detach (),
        "sp_regions":torch .tensor (K ,device =img_ref .device ),
        }
        return wts ,sym_pairs ,{},diag 


class WeakRuleLearnableSplits (nn .Module ):


    def __init__ (
    self ,
    *,
    delta_max :float =0.07 ,
    temp :float =0.02 ,
    x1_base :float =0.33 ,
    x2_base :float =0.66 ,
    y1_base :float =0.35 ,
    y2_base :float =0.65 ,
    ):
        super ().__init__ ()
        self .delta_max =float (delta_max )
        self .temp =float (temp )

        self .x1_base =float (x1_base )
        self .x2_base =float (x2_base )
        self .y1_base =float (y1_base )
        self .y2_base =float (y2_base )

        self .dx1 =nn .Parameter (torch .zeros ([]))
        self .dx2 =nn .Parameter (torch .zeros ([]))
        self .dy1 =nn .Parameter (torch .zeros ([]))
        self .dy2 =nn .Parameter (torch .zeros ([]))

    def _bound (self ,base :float ,d :torch .Tensor ,lo :float ,hi :float )->torch .Tensor :
        v =float (base )+float (self .delta_max )*torch .tanh (d )
        return torch .clamp (v ,float (lo ),float (hi ))

    def forward (
    self ,h :int ,w :int ,batch_size :int ,*,device =None ,dtype :torch .dtype =torch .float32 
    )->tuple [torch .Tensor ,List [Tuple [int ,int ]],Dict [str ,torch .Tensor ],Dict [str ,torch .Tensor ]]:
        h =int (h )
        w =int (w )
        x1 =self ._bound (self .x1_base ,self .dx1 ,0.15 ,0.49 )
        x2 =self ._bound (self .x2_base ,self .dx2 ,0.51 ,0.85 )
        y1 =self ._bound (self .y1_base ,self .dy1 ,0.15 ,0.49 )
        y2 =self ._bound (self .y2_base ,self .dy2 ,0.51 ,0.85 )

        ys =(torch .arange (h ,device =device ,dtype =dtype )+0.5 )/float (h )
        xs =(torch .arange (w ,device =device ,dtype =dtype )+0.5 )/float (w )
        yy ,xx =torch .meshgrid (ys ,xs ,indexing ="ij")
        xx =xx [None ,None ,:,:]
        yy =yy [None ,None ,:,:]

        def box (a :torch .Tensor ,b :torch .Tensor ,v :torch .Tensor )->torch .Tensor :
            return torch .sigmoid ((v -a )/float (self .temp ))*torch .sigmoid ((b -v )/float (self .temp ))

        x_edges =[torch .zeros ([],device =device ,dtype =dtype ),x1 ,x2 ,torch .ones ([],device =device ,dtype =dtype )]
        y_edges =[torch .zeros ([],device =device ,dtype =dtype ),y1 ,y2 ,torch .ones ([],device =device ,dtype =dtype )]

        masks =[]
        for ry in range (3 ):
            for rx in range (3 ):
                mx =box (x_edges [rx ],x_edges [rx +1 ],xx )
                my =box (y_edges [ry ],y_edges [ry +1 ],yy )
                masks .append ((mx *my ).squeeze (1 ))
        m =torch .stack (masks ,dim =1 ).repeat (int (batch_size ),1 ,1 ,1 )
        wts =m /(m .sum (dim =1 ,keepdim =True )+1e-6 )

        reg_delta =(self .dx1 .abs ()+self .dx2 .abs ()+self .dy1 .abs ()+self .dy2 .abs ())
        sym_pairs =[(0 ,2 ),(3 ,5 ),(6 ,8 )]
        diag ={"splits_x1":x1 .detach (),"splits_x2":x2 .detach (),"splits_y1":y1 .detach (),"splits_y2":y2 .detach ()}
        return wts ,sym_pairs ,{"delta":reg_delta },diag 


class RegionSelector (nn .Module ):
    def __init__ (self ,cfg :dict ):
        super ().__init__ ()
        self .mode =str (cfg .get ("mode","quad4")).strip ().lower ()

        self .is_dynamic =bool (self .mode in {"superpoint_voronoi"})
        self .num_regions =int (cfg .get ("num_regions",9 ))
        self .cov_w =float (cfg .get ("cov_w",0.01 ))
        self .sym_w =float (cfg .get ("sym_w",0.01 ))
        self .delta_w =float (cfg .get ("delta_w",0.01 ))

        self .soft :SoftGaussianRegions |None =None 
        self .superpt :SuperpointVoronoiRegions |None =None 
        self .splits :WeakRuleLearnableSplits |None =None 

        if self .mode =="soft_gaussian":
            self .soft =SoftGaussianRegions (num_regions =self .num_regions )
        elif self .mode =="superpoint_voronoi":
            num_pairs =int (cfg .get ("num_pairs",4 ))
            add_center =bool (cfg .get ("add_center",True ))
            self .superpt =SuperpointVoronoiRegions (num_pairs =num_pairs ,add_center =add_center )
        elif self .mode =="weak_splits_3x3":
            self .splits =WeakRuleLearnableSplits (
            delta_max =float (cfg .get ("delta_max",0.07 )),
            temp =float (cfg .get ("temp",0.02 )),
            x1_base =float (cfg .get ("x1_base",0.33 )),
            x2_base =float (cfg .get ("x2_base",0.66 )),
            y1_base =float (cfg .get ("y1_base",0.35 )),
            y2_base =float (cfg .get ("y2_base",0.65 )),
            )
        else :
            raise ValueError (f"Unknown region mode: {self.mode}")

    def forward (
    self ,
    feat_hw :tuple [int ,int ],
    batch_size :int ,
    *,
    img_ref :torch .Tensor |None =None ,
    dyn_map :torch .Tensor |None =None ,
    )->tuple [torch .Tensor ,List [Tuple [int ,int ]],torch .Tensor ,Dict [str ,torch .Tensor ]]:
        h ,w =int (feat_hw [0 ]),int (feat_hw [1 ])
        device =img_ref .device if img_ref is not None else None 
        dtype =img_ref .dtype if img_ref is not None else torch .float32 

        if self .mode =="soft_gaussian":
            if self .soft is None :
                raise RuntimeError ("soft_gaussian selected but module missing")
            wts ,reg ,diag =self .soft (h ,w ,int (batch_size ),device =device ,dtype =dtype )
            sym_pairs =[(0 ,2 ),(3 ,5 ),(6 ,8 )]if int (self .num_regions )==9 else []
            loss_reg =self .cov_w *reg ["cov_kl"]+self .sym_w *reg ["sym"]
            diag ={**diag ,"region_mode":torch .tensor (1 ,device =device )}
            return wts ,sym_pairs ,loss_reg ,diag 

        if self .mode =="superpoint_voronoi":
            if self .superpt is None :
                raise RuntimeError ("superpoint_voronoi selected but module missing")
            if img_ref is None or dyn_map is None :
                raise ValueError ("superpoint_voronoi requires img_ref and dyn_map")
            wts ,sym_pairs ,_reg ,diag =self .superpt (img_ref ,dyn_map ,(h ,w ))
            loss_reg =torch .zeros ([],device =img_ref .device ,dtype =img_ref .dtype )
            diag ={**diag ,"region_mode":torch .tensor (2 ,device =img_ref .device )}
            return wts ,sym_pairs ,loss_reg ,diag 

        if self .mode =="weak_splits_3x3":
            if self .splits is None :
                raise RuntimeError ("weak_splits_3x3 selected but module missing")
            wts ,sym_pairs ,reg ,diag =self .splits (h ,w ,int (batch_size ),device =device ,dtype =dtype )
            loss_reg =self .delta_w *reg ["delta"]
            diag ={**diag ,"region_mode":torch .tensor (3 ,device =device )}
            return wts ,sym_pairs ,loss_reg ,diag 

        raise ValueError (f"Unknown region mode: {self.mode}")
