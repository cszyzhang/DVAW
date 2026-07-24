from __future__ import annotations 

import re 
from dataclasses import dataclass 
from typing import Any ,Dict ,Optional ,Tuple 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 

from model .utils .flow_global import remove_affine ,remove_lowpass 
from model .utils .warp import affine_warp ,flow_valid_mask ,warp 

from model .reco_support .losses import info_nce ,masked_l1 ,rank_hinge ,simsiam_negcos 
from model .reco_support .region_partition import build_region_id_map ,region_mean 

from model .reliability .losses import ReCoAuxiliary 
from model .reliability .regions import region_pool as reco_region_pool 
from model .reliability .trust_propagation import propagate_trust_tokens 

from .gradient_scaled_classifier_head import GradientScaledClassifierHead 
from .dvaw_flow_predictor import DVAWFlowPredictor 
from .motion_classifier import MotionClassifier 


def reco_enabled_any (reco_cfg :Dict [str ,Any ]|None ,*,channel_swap_partition :str |None =None )->bool :

    sp =str (channel_swap_partition or "").strip ().lower ()
    if sp .startswith ("reco"):
        return True 

    if not isinstance (reco_cfg ,dict )or not reco_cfg :
        return False 

    region_cfg =reco_cfg .get ("region",{})or {}
    if isinstance (region_cfg ,dict ):
        region_mode =str (region_cfg .get ("mode","quad4")).strip ().lower ()
        if region_mode and region_mode !="quad4":
            return True 

    for k in ("cross_view_mask","multiframe","channel_swap","trusted_transport"):
        sub =reco_cfg .get (k ,None )
        if isinstance (sub ,dict ):
            if bool (sub .get ("enabled",False )):
                return True 
        elif isinstance (sub ,bool ):
            if bool (sub ):
                return True 

    return False 


def _alias_tensor (aux :Dict [str ,Any ],new_key :str ,old_key :str )->None :
    if new_key in aux :
        return 
    value =aux .get (old_key ,None )
    if torch .is_tensor (value ):
        aux [new_key ]=value 


def _alias_mean (aux :Dict [str ,Any ],new_key :str ,key_a :str ,key_b :str )->None :
    if new_key in aux :
        return 
    a =aux .get (key_a ,None )
    b =aux .get (key_b ,None )
    if torch .is_tensor (a )and torch .is_tensor (b )and tuple (a .shape )==tuple (b .shape ):
        aux [new_key ]=0.5 *(a +b )


def _apply_view_mode_pair (
view_mode :str ,
on :Optional [torch .Tensor ],
off :Optional [torch .Tensor ],
)->Tuple [Optional [torch .Tensor ],Optional [torch .Tensor ]]:
    if view_mode =="dual":
        return on ,off 
    if view_mode =="onset_only":
        if torch .is_tensor (off ):
            off =torch .zeros_like (off )
        return on ,off 
    if view_mode =="offset_only":
        if torch .is_tensor (on ):
            on =torch .zeros_like (on )
        return on ,off 
    raise ValueError (f"Unsupported view_mode: {view_mode!r}")


def _inject_paper_named_aliases (aux :Dict [str ,Any ])->None :



    _alias_tensor (aux ,"f_A_from_O","flow_on")
    _alias_tensor (aux ,"f_A_from_F","flow_off")
    _alias_tensor (aux ,"Ihat_A_from_O","a_hat_on")
    _alias_tensor (aux ,"Ihat_A_from_F","a_hat_off")


    _alias_tensor (aux ,"M_dyn_O","W_dyn_on")
    _alias_tensor (aux ,"M_dyn_F","W_dyn_off")
    _alias_tensor (aux ,"W_val_O","W_valid_on")
    _alias_tensor (aux ,"W_val_F","W_valid_off")
    _alias_tensor (aux ,"W_photo_O","C_photo_on")
    _alias_tensor (aux ,"W_photo_F","C_photo_off")
    _alias_tensor (aux ,"C_ph_O","C_photo_on")
    _alias_tensor (aux ,"C_ph_F","C_photo_off")
    _alias_tensor (aux ,"C_ph","C_photo")
    _alias_mean (aux ,"W_photo","W_photo_O","W_photo_F")


    _alias_tensor (aux ,"r_rt_O","ce_on")
    _alias_tensor (aux ,"r_rt_F","ce_off")
    _alias_tensor (aux ,"r_rt","r_cyc_map")
    _alias_tensor (aux ,"W_rt_O","W_valid_cyc_on")
    _alias_tensor (aux ,"W_rt_F","W_valid_cyc_off")


    _alias_tensor (aux ,"T_O","W_fb_on")
    _alias_tensor (aux ,"T_F","W_fb_off")
    _alias_mean (aux ,"T","T_O","T_F")
    _alias_tensor (aux ,"U_O","U_on")
    _alias_tensor (aux ,"U_F","U_off")


    _alias_tensor (aux ,"c_tok","C_tok_photo")
    _alias_tensor (aux ,"v_tok","validTok")
    _alias_tensor (aux ,"g_tok","gate_mask_tok")
    _alias_tensor (aux ,"t_tok","trustTok")
    _alias_tensor (aux ,"w_tok","weight_tok")
    _alias_tensor (aux ,"u_tok","U_tok")


    d_on =aux .get ("desc_on",None )
    d_off =aux .get ("desc_off",None )
    if torch .is_tensor (d_on ):
        aux .setdefault ("D_O",d_on )
    if torch .is_tensor (d_off ):
        aux .setdefault ("D_F",d_off )
    if torch .is_tensor (d_on )and torch .is_tensor (d_off ):
        d_cat =torch .cat ([d_on ,d_off ],dim =1 )
        aux .setdefault ("D",d_cat )
        aux .setdefault ("D_cls",d_cat )


    _alias_tensor (aux ,"L_tr_align","L_transport_align")
    _alias_tensor (aux ,"L_tr_rt","L_round_trip_consistency")
    _alias_tensor (aux ,"P_transport","transport_T_matrix")


    _alias_tensor (aux ,"D_cls_eff","x_cls")


class ChannelAdapter (nn .Module ):
    def __init__ (self ,in_channels :int ,out_channels :int ):
        super ().__init__ ()
        self .proj =nn .Conv2d (int (in_channels ),int (out_channels ),kernel_size =1 ,bias =True )

    def forward (self ,x :torch .Tensor )->torch .Tensor :
        return self .proj (x )


@dataclass 
class MotionWarpOutputs :
    logits :torch .Tensor 
    aux :Dict [str ,Any ]


class _GlobalThetaHead (nn .Module ):
    def __init__ (self ,*,in_channels :int ,base_channels :int =32 ,theta_scale :float =0.1 ):
        super ().__init__ ()
        b =int (base_channels )
        self .theta_scale =float (theta_scale )
        self .net =nn .Sequential (
        nn .Conv2d (int (in_channels ),b ,kernel_size =3 ,stride =2 ,padding =1 ,bias =False ),
        nn .GroupNorm (8 if (b %8 )==0 else 1 ,b ),
        nn .ReLU (inplace =True ),
        nn .Conv2d (b ,2 *b ,kernel_size =3 ,stride =2 ,padding =1 ,bias =False ),
        nn .GroupNorm (8 if ((2 *b )%8 )==0 else 1 ,2 *b ),
        nn .ReLU (inplace =True ),
        nn .Conv2d (2 *b ,4 *b ,kernel_size =3 ,stride =2 ,padding =1 ,bias =False ),
        nn .GroupNorm (8 if ((4 *b )%8 )==0 else 1 ,4 *b ),
        nn .ReLU (inplace =True ),
        nn .AdaptiveAvgPool2d ((1 ,1 )),
        )
        self .fc =nn .Linear (4 *b ,6 ,bias =True )
        nn .init .zeros_ (self .fc .weight )
        nn .init .zeros_ (self .fc .bias )

        self .register_buffer ("theta_id_6",torch .tensor ([1.0 ,0.0 ,0.0 ,0.0 ,1.0 ,0.0 ]).view (1 ,6 ),persistent =False )

    def forward (self ,pair :torch .Tensor )->torch .Tensor :
        if pair .ndim !=4 :
            raise ValueError (f"pair must be [B,C,H,W], got {tuple(pair.shape)}")
        bsz =int (pair .shape [0 ])
        x =self .net (pair ).view (bsz ,-1 )
        delta =self .fc (x )
        if self .theta_scale >0.0 :
            delta =torch .tanh (delta )*float (self .theta_scale )
        theta6 =self .theta_id_6 .to (device =pair .device ,dtype =delta .dtype ).expand (bsz ,-1 )+delta 
        return theta6 .view (bsz ,2 ,3 )


class DVAWReCoModel (nn .Module ):


    def __init__ (
    self ,
    *,
    image_channels :int ,
    num_classes :int ,
    classifier_input_channels :int =3 ,
    flow_downscale :int =4 ,
    classifier_in_channels :int =4 ,
    motion_base_channels :int =32 ,
    max_disp :float =3.0 ,
    compose_mode :str ="disp",
    motion_mode :str ="endpoints_dt",
    corr_radius :int =2 ,
    cls_global_removal :str ="none",
    cls_lowpass_kernel :int =31 ,
    cls_dual_stream_enabled :bool =False ,
    cls_dual_stream_fusion :str ="concat",
    cls_use_err :bool =False ,
    cls_err_norm :str ="mean",
    cls_motion_scale_enabled :bool =False ,
    cls_motion_scale_target :float =0.5 ,
    cls_motion_scale_clip :Tuple [float ,float ]=(0.5 ,3.0 ),
    cls_motion_scale_source :str ="raw",
    cls_motion_scale_stat :str ="mean_abs",
    cls_motion_scale_detach :bool =False ,
    cls_region_masks_enabled :bool =False ,
    cls_region_masks_k :int =3 ,
    cls_region_masks_include_base :bool =True ,
    cls_roi_enabled :bool =False ,
    cls_roi_source :str ="err",
    cls_roi_ratio :float =0.05 ,
    refine_base_channels :int =32 ,
    refine_enabled :bool =False ,
    refine_steps :int =1 ,
    refine_delta_scale :float =0.5 ,
    global_local_enabled :bool =False ,
    global_local_base_channels :int =32 ,
    global_local_theta_scale :float =0.1 ,
    cls_rep :str ="raw",
    deformation_descriptor_scales :Tuple [int ,...]=(1 ,),
    deformation_descriptor_ops :Tuple [str ,...]=("div","curl","shear1","shear2"),
    sc_enabled :bool =False ,
    sc_region_source :str ="fixed",
    sc_region_partition :str ="quad4",
    sc_allow_dynamic_regions :bool =False ,
    gradient_classifier_enabled :bool =False ,
    gradient_classifier_hidden_dim :int =192 ,
    gradient_classifier_dropout :float =0.1 ,
    gradient_classifier_num_layers :int =2 ,
    gradient_classifier_mode :str ="detach_backbone",
    gradient_classifier_grad_scale :float =0.1 ,
    cls_dual_branch_enabled :bool =False ,
    cls_dual_branch_branch_b :str ="D_flow",
    cls_dual_branch_fuse :str ="fixed_avg",
    cls_dual_branch_wA :float =0.5 ,
    cls_dual_branch_wB :float =0.5 ,
    basis_displacement_enabled :bool =False ,
    basis_displacement_K :int =8 ,
    basis_displacement_base_res :int =28 ,
    fusion_enabled :bool =False ,
    fusion_gate_hidden :int =32 ,
    predict_backward :bool =False ,
    region_reconstruction_enabled :bool =False ,
    region_reconstruction_mode :str ="fixed_boxes",
    region_reconstruction_apply_to_cls :bool =False ,
    texture_saliency_mask_enabled :bool =False ,
    texture_saliency_mask_grid :int =4 ,
    texture_saliency_mask_radius :int =9 ,
    texture_saliency_mask_score :str ="lap_var",
    texture_saliency_mask_apply_to_cls :bool =False ,
    midframe_enabled :bool =False ,
    round_trip_composition_enabled :bool =False ,
    dynamic_support_mask_enabled :bool =False ,
    dynamic_support_mask_apply_to_cls :bool =False ,
    superpoints_K :int =0 ,
    superpoints_apply_to_cls :bool =False ,
    descriptor_norm_topk_ratio :float =0.10 ,
    descriptor_detach_scale :bool =False ,
    stopgrad_cls_to_motion :bool =True ,
    classifier_saliency_mask_enabled :bool =False ,
    classifier_saliency_mask_ratio :float =0.03 ,
    classifier_saliency_mask_blur_k :int =5 ,
    classifier_saliency_mask_blur_sigma :float =1.0 ,
    cls_dual_view_enabled :bool =False ,
    cls_dual_view_merge :str ="avg_logits",
    cls_dual_view_w_unmasked :float =0.5 ,
    conf_dyn_enabled :bool =False ,
    conf_dyn_topk_ratio :float =0.15 ,
    conf_dyn_dilate_kernel :int =3 ,
    view_mode :str ="dual",
    routing_mode :str ="full",
    transport_mode :str ="symmetric",
    feat_loss_enabled :bool =False ,
    feat_embed_channels :int =16 ,
    two_stage_cls_enabled :bool =False ,
    two_stage_cls_fuse :str ="avg_logits",
    two_stage_aux_ce_weight :float =0.0 ,
    contrastive_enabled :bool =False ,
    contrastive_mode :str ="",
    contrastive_stop_grad_motion :bool =False ,
    contrastive_proj_dim :int =0 ,

    channel_swap_enabled :bool =False ,
    channel_swap_partition :str ="grid3x3",
    channel_swap_sampling_source :str ="uncertainty",
    channel_swap_mode :str ="infonce",
    channel_swap_proj_dim :int =128 ,
    channel_swap_pred_dim :int =256 ,
    channel_swap_k :int =1 ,
    cross_view_mask_enabled :bool =False ,
    cross_view_mask_partition :str ="grid3x3",
    cross_view_mask_mode :str ="on_from_off",
    cross_view_mask_hidden :int =64 ,
    cross_view_mask_k :int =1 ,
    multiframe_enabled :bool =False ,
    multiframe_variant :str ="extra_rec",
    multiframe_extra_rec_in_loss :bool =False ,
    fb_conf_enabled :bool =True ,
    fb_conf_tau :float =0.5 ,
    fb_conf_min :float =0.05 ,
    routing_token_hw :Tuple [int ,int ]=(14 ,14 ),
    routing_gate_thr :float =0.2 ,
    trust_propagation_enabled :bool =False ,
    trust_propagation_alpha :float =0.5 ,
    trust_propagation_steps :int =2 ,

    reco :Optional [Dict [str ,Any ]]=None ,
    eps :float =1e-6 ,
    ):
        super ().__init__ ()
        self .image_channels =int (image_channels )
        self .num_classes =int (num_classes )
        self .classifier_input_channels =int (classifier_input_channels )
        self .max_disp =float (max_disp )
        self .eps =float (eps )

        mm =str (motion_mode ).strip ().lower ()
        mm =mm .replace ("-","_")
        if mm in {"endpoints","endpoints_dt"}:
            mm ="endpoints_dt"
        if mm in {"apex_pair","apex_pairflow","apex_pair_flow"}:
            mm ="apex_pair_flow"
        if mm in {"apex_pair_flow_dt","apex_pairflow_dt"}:
            mm ="apex_pair_flow_dt"
        if mm in {"triplet_apex_cond","triplet_apex_cond_dt","triplet"}:
            mm ="triplet_apex_cond_dt"
        if mm in {"cost_volume_apex_pair_flow","cost_volume_pair_flow","cost_volume"}:
            mm ="cost_volume_apex_pair_flow"
        if mm in {"apex_neighborhood_flow","apex_nb_flow","neighborhood_flow"}:
            mm ="apex_neighborhood_flow"
        if mm in {"error_feedback_refine","error_feedback","refine"}:
            mm ="error_feedback_refine"
        if mm not in {
        "endpoints_dt",
        "apex_pair_flow",
        "apex_pair_flow_dt",
        "triplet_apex_cond_dt",
        "cost_volume_apex_pair_flow",
        "apex_neighborhood_flow",
        "error_feedback_refine",
        }:
            raise ValueError (f"Unsupported motion_mode: {motion_mode!r}")
        self .motion_mode =mm 

        cm =str (compose_mode ).strip ().lower ()
        if cm in {"displacement","disp"}:
            cm ="disp"
        else :
            raise ValueError (f"compose_mode must be 'disp', got {compose_mode!r}")
        self .classifier_in_channels =int (classifier_in_channels )

        rm =str (cls_global_removal ).strip ().lower ().replace ("-","_")
        if rm in {"none","off",""}:
            rm ="none"
        if rm in {"learnable","learnable_lp","learnable_lowpass"}:
            rm ="learnable_lowpass"
        if rm not in {"none","lowpass","affine","learnable_lowpass"}:
            raise ValueError (
            f"Unsupported cls_global_removal={cls_global_removal!r}; choose from none/lowpass/affine/learnable_lowpass"
            )
        self .cls_global_removal =rm 
        self .cls_lowpass_kernel =int (cls_lowpass_kernel )

        self .cls_dual_stream_enabled =bool (cls_dual_stream_enabled )
        self .cls_dual_stream_fusion =str (cls_dual_stream_fusion ).strip ().lower ()
        if self .cls_dual_stream_fusion in {"","cat"}:
            self .cls_dual_stream_fusion ="concat"
        if self .cls_dual_stream_fusion not in {"concat","gated"}:
            raise ValueError ("cls_dual_stream_fusion must be 'concat' or 'gated'")
        self .cls_use_err =bool (cls_use_err )
        self .cls_err_norm =str (cls_err_norm ).strip ().lower ()

        self .cls_motion_scale_enabled =bool (cls_motion_scale_enabled )
        self .cls_motion_scale_target =float (cls_motion_scale_target )
        self .cls_motion_scale_clip =(float (cls_motion_scale_clip [0 ]),float (cls_motion_scale_clip [1 ]))
        self .cls_motion_scale_source =str (cls_motion_scale_source ).strip ().lower ()
        if self .cls_motion_scale_source in {"x_cls","cls","desc","descriptor"}:
            self .cls_motion_scale_source ="x_cls"
        if self .cls_motion_scale_source not in {"raw","resid","x_cls"}:
            raise ValueError ("cls_motion_scale_source must be 'raw', 'resid', or 'x_cls'")

        self .cls_motion_scale_stat =str (cls_motion_scale_stat ).strip ().lower ().replace ("-","_")
        if self .cls_motion_scale_stat in {"","mean"}:
            self .cls_motion_scale_stat ="mean_abs"
        if self .cls_motion_scale_stat in {"median"}:
            self .cls_motion_scale_stat ="median_abs"
        if self .cls_motion_scale_stat in {"p75","q75","p75_abs","q75_abs"}:
            self .cls_motion_scale_stat ="p75_abs"
        if self .cls_motion_scale_stat not in {"mean_abs","median_abs","p75_abs"}:
            raise ValueError ("cls_motion_scale_stat must be 'mean_abs', 'median_abs', or 'p75_abs'")
        self .cls_motion_scale_detach =bool (cls_motion_scale_detach )

        self .cls_region_masks_enabled =bool (cls_region_masks_enabled )
        self .cls_region_masks_k =int (cls_region_masks_k )
        self .cls_region_masks_include_base =bool (cls_region_masks_include_base )
        if self .cls_region_masks_k <=0 :
            raise ValueError ("cls_region_masks_k must be > 0")

        self .cls_roi_enabled =bool (cls_roi_enabled )
        self .cls_roi_source =str (cls_roi_source ).strip ().lower ()
        if self .cls_roi_source not in {"err","resid"}:
            raise ValueError ("cls_roi_source must be 'err' or 'resid'")
        self .cls_roi_ratio =float (cls_roi_ratio )

        self .refine_enabled =bool (refine_enabled )
        self .refine_steps =max (int (refine_steps ),1 )
        self .refine_delta_scale =float (refine_delta_scale )

        self .global_local_enabled =bool (global_local_enabled )
        self .global_local_theta_scale =float (global_local_theta_scale )

        self .cls_rep =str (cls_rep ).strip ().lower ().replace ("-","_")
        if self .cls_rep in {"","raw","flow"}:
            self .cls_rep ="raw"
        if self .cls_rep in {"multiscale_deformation_descriptor","multiscale_deformation_descriptor","multiscale_deformation_descriptor"}:
            self .cls_rep ="multiscale_deformation_descriptor"
        if self .cls_rep in {"differential_deformation_descriptor","differential_deformation_descriptor","differential_deformation_descriptor"}:
            self .cls_rep ="differential_deformation_descriptor"
        if self .cls_rep not in {"raw","eight_channel_deformation_descriptor","multiscale_deformation_descriptor","differential_deformation_descriptor"}:
            raise ValueError ("cls_rep must be 'raw', 'eight_channel_deformation_descriptor', 'multiscale_deformation_descriptor', or 'differential_deformation_descriptor'")

        self .two_stage_cls_enabled =bool (two_stage_cls_enabled )
        self .two_stage_cls_fuse =str (two_stage_cls_fuse ).strip ().lower ().replace ("-","_")
        if self .two_stage_cls_fuse in {"avg","mean","avg_logits"}:
            self .two_stage_cls_fuse ="avg_logits"
        if self .two_stage_cls_fuse not in {"avg_logits"}:
            raise ValueError (f"Unsupported two_stage_cls_fuse: {two_stage_cls_fuse!r} (expected avg_logits)")
        self .two_stage_aux_ce_weight =float (two_stage_aux_ce_weight )

        self .contrastive_enabled =bool (contrastive_enabled )
        self .contrastive_mode =str (contrastive_mode ).strip ().lower ()
        self .contrastive_stop_grad_motion =bool (contrastive_stop_grad_motion )
        self .contrastive_proj_dim =int (contrastive_proj_dim )
        if self .contrastive_enabled and self .contrastive_mode not in {"a1","a2","a3","a4","a5"}:
            raise ValueError (f"contrastive_mode must be one of a1..a5 when contrastive_enabled. Got: {contrastive_mode!r}")
        if self .contrastive_enabled and not self .two_stage_cls_enabled :
            raise ValueError ("contrastive_enabled requires two_stage_cls_enabled (need on/off views).")
        if self .contrastive_enabled and self .contrastive_mode in {"a4","a5"}and self .contrastive_proj_dim <=0 :
            raise ValueError ("contrastive_proj_dim must be > 0 for contrastive modes a4/a5.")
        if self .two_stage_cls_enabled and self .cls_rep !="multiscale_deformation_descriptor":
            raise ValueError ("two_stage_cls_enabled is currently only supported with cls_rep='multiscale_deformation_descriptor'.")

        self .deformation_descriptor_scales =tuple (int (s )for s in deformation_descriptor_scales )
        self .deformation_descriptor_ops =tuple (str (op )for op in deformation_descriptor_ops )
        if not self .deformation_descriptor_scales :
            self .deformation_descriptor_scales =(1 ,)

        self .sc_enabled =bool (sc_enabled )
        self .sc_region_source =str (sc_region_source ).strip ().lower ()
        if self .sc_region_source in {"","fixed"}:
            self .sc_region_source ="fixed"
        if self .sc_region_source in {"reco","regions_reco"}:
            self .sc_region_source ="reco"
        if self .sc_region_source not in {"fixed","reco"}:
            raise ValueError (f"sc_region_source must be 'fixed' or 'reco', got {sc_region_source!r}")
        self .sc_region_partition =str (sc_region_partition ).strip ().lower ()
        self .sc_allow_dynamic_regions =bool (sc_allow_dynamic_regions )
        self .gradient_classifier_enabled =bool (gradient_classifier_enabled )
        self .gradient_classifier_hidden_dim =int (min (max (int (gradient_classifier_hidden_dim ),1 ),192 ))
        self .gradient_classifier_dropout =float (gradient_classifier_dropout )
        self .gradient_classifier_num_layers =int (max (int (gradient_classifier_num_layers ),1 ))
        bm =str (gradient_classifier_mode ).strip ().lower ().replace ("-","_")
        if bm in {"","detach","detachbackbone"}:
            bm ="detach_backbone"
        if bm in {"grad","gradscale","grad_scale_to_backbone"}:
            bm ="grad_scale"
        if bm not in {"detach_backbone","grad_scale"}:
            raise ValueError (f"gradient_classifier_mode must be detach_backbone/grad_scale, got {gradient_classifier_mode!r}")
        self .gradient_classifier_mode =bm 
        self .gradient_classifier_grad_scale =float (gradient_classifier_grad_scale )

        self .routing_token_hw =(int (routing_token_hw [0 ]),int (routing_token_hw [1 ]))
        self .routing_gate_thr =float (routing_gate_thr )
        self .trust_propagation_enabled =bool (trust_propagation_enabled )
        self .trust_propagation_alpha =float (trust_propagation_alpha )
        self .trust_propagation_steps =int (max (int (trust_propagation_steps ),0 ))


        self .channel_swap_enabled =bool (channel_swap_enabled )
        self .channel_swap_partition_raw =str (channel_swap_partition ).strip ().lower ()
        _channel_swap_parts =[p .strip ().lower ()for p in str (self .channel_swap_partition_raw ).split ("|")if p .strip ()]
        self .channel_swap_partition_base =_channel_swap_parts [0 ]if _channel_swap_parts else "grid3x3"
        self .channel_swap_partition_opts =set (_channel_swap_parts [1 :])

        self .channel_swap_partition =str (self .channel_swap_partition_base )
        self .channel_swap_mode =str (channel_swap_mode ).strip ().lower ().replace ("-","_")
        if self .channel_swap_mode in {"","info_nce","nce"}:
            self .channel_swap_mode ="infonce"
        if self .channel_swap_mode in {"simsiam","byol","siam"}:
            self .channel_swap_mode ="siam"
        if self .channel_swap_mode not in {"infonce","siam","region_align"}:
            raise ValueError (f"channel_swap_mode must be infonce/siam/region_align, got {channel_swap_mode!r}")
        self .channel_swap_sampling_source =str (channel_swap_sampling_source ).strip ().lower ().replace ("-","_")
        if self .channel_swap_sampling_source in {"","random"}:
            self .channel_swap_sampling_source ="uniform"
        if self .channel_swap_sampling_source not in {"uniform","uncertainty","hard_error","probe_error"}:
            raise ValueError ("channel_swap_sampling_source must be one of uniform/uncertainty/hard_error/probe_error")
        self .channel_swap_proj_dim =int (channel_swap_proj_dim )
        self .channel_swap_pred_dim =int (channel_swap_pred_dim )
        self .channel_swap_k =int (max (int (channel_swap_k ),1 ))

        self .cross_view_mask_enabled =bool (cross_view_mask_enabled )
        self .cross_view_mask_partition =str (cross_view_mask_partition ).strip ().lower ()
        self .cross_view_mask_mode =str (cross_view_mask_mode ).strip ().lower ().replace ("-","_")
        if self .cross_view_mask_mode in {"","on_from_off"}:
            self .cross_view_mask_mode ="on_from_off"
        if self .cross_view_mask_mode in {"bidirectional","bi"}:
            self .cross_view_mask_mode ="bi"
        if self .cross_view_mask_mode not in {"on_from_off","bi","hard"}:
            raise ValueError (f"cross_view_mask_mode must be on_from_off/bi/hard, got {cross_view_mask_mode!r}")
        self .cross_view_mask_hidden =int (cross_view_mask_hidden )
        self .cross_view_mask_k =int (max (int (cross_view_mask_k ),1 ))

        self .multiframe_enabled =bool (multiframe_enabled )
        self .multiframe_variant =str (multiframe_variant ).strip ().lower ()
        if not self .multiframe_variant :
            self .multiframe_variant ="extra_rec"
        vset_raw ={t .strip ().lower ().replace ("-","_")for t in re .split (r"[+|,]",str (self .multiframe_variant ))if t .strip ()}
        if not vset_raw :
            vset_raw ={"extra_rec"}
        vset :set [str ]=set ()
        unknown :set [str ]=set ()
        for tok in vset_raw :
            if tok in {"extra"}:
                tok ="extra_rec"
            if tok in {"flow"}:
                tok ="round_trip_composition"
            if tok in {"rank"}:
                tok ="rank_mag"
            if tok in {"extra_rec","round_trip_composition","rank_mag"}:
                vset .add (tok )
            else :
                unknown .add (tok )
        if unknown :
            raise ValueError (
            f"multiframe_variant contains unknown token(s): {sorted(unknown)}. "
            f"Allowed tokens: extra_rec, round_trip_composition, rank_mag. Got: {multiframe_variant!r}"
            )
        if not vset :
            vset ={"extra_rec"}

        self .multiframe_variant ="+".join (sorted (vset ))
        self .multiframe_extra_rec_in_loss =bool (multiframe_extra_rec_in_loss )
        self .fb_conf_enabled =bool (fb_conf_enabled )
        self .fb_conf_tau =float (max (float (fb_conf_tau ),1e-6 ))
        self .fb_conf_min =float (min (max (float (fb_conf_min ),0.0 ),1.0 ))

        if (self .channel_swap_enabled or self .cross_view_mask_enabled )and bool (self .two_stage_cls_enabled ):
            raise ValueError ("channel_swap/cross_view_mask reliability objectives require two_stage_cls_enabled=false (need paired desc input).")
        if self .channel_swap_enabled or self .cross_view_mask_enabled :
            if (int (self .classifier_in_channels )%2 )!=0 :
                raise ValueError ("channel_swap/cross_view_mask require classifier_in_channels to be even (paired halves).")



        self .channel_swap_proj :Optional [nn .Module ]=None 
        self .channel_swap_pred :Optional [nn .Module ]=None 
        self .region_proj :Optional [nn .Module ]=None 
        self .cross_view_mask_pred :Optional [nn .Module ]=None 



        self .reco_cfg =reco if isinstance (reco ,dict )else {}
        self .reco :Optional [ReCoAuxiliary ]=None 

        self .cls_dual_branch_enabled =bool (cls_dual_branch_enabled )
        bb =str (cls_dual_branch_branch_b ).strip ().lower ().replace ("-","_")
        if bb in {"d_flow","d"}:
            bb ="d_flow"
        if bb not in {"d_flow"}:
            raise ValueError ("cls_dual_branch_branch_b must be 'D_flow' (current supported option)")
        self .cls_dual_branch_branch_b =bb 
        ff =str (cls_dual_branch_fuse ).strip ().lower ().replace ("-","_")
        if ff in {"avg","average","fixed","fixedavg"}:
            ff ="fixed_avg"
        if ff not in {"fixed_avg"}:
            raise ValueError ("cls_dual_branch_fuse must be 'fixed_avg'")
        self .cls_dual_branch_fuse =ff 
        self .cls_dual_branch_wA =float (cls_dual_branch_wA )
        self .cls_dual_branch_wB =float (cls_dual_branch_wB )

        self .basis_displacement_enabled =bool (basis_displacement_enabled )
        self .basis_displacement_K =int (basis_displacement_K )
        self .basis_displacement_base_res =int (basis_displacement_base_res )

        self .fusion_enabled =bool (fusion_enabled )

        self .predict_backward =bool (predict_backward )

        self .region_reconstruction_enabled =bool (region_reconstruction_enabled )
        self .region_reconstruction_mode =str (region_reconstruction_mode ).strip ().lower ().replace ("-","_")
        if self .region_reconstruction_mode in {"","fixed","boxes","fixed_boxes"}:
            self .region_reconstruction_mode ="fixed_boxes"
        if self .region_reconstruction_mode not in {"fixed_boxes"}:
            raise ValueError ("region_reconstruction_mode must be 'fixed_boxes'")
        self .region_reconstruction_apply_to_cls =bool (region_reconstruction_apply_to_cls )

        self .texture_saliency_mask_enabled =bool (texture_saliency_mask_enabled )
        self .texture_saliency_mask_grid =int (texture_saliency_mask_grid )
        self .texture_saliency_mask_radius =int (texture_saliency_mask_radius )
        self .texture_saliency_mask_score =str (texture_saliency_mask_score ).strip ().lower ().replace ("-","_")
        if self .texture_saliency_mask_score in {"lap","lap_var","laplacian_var","laplacian_variance"}:
            self .texture_saliency_mask_score ="lap_var"
        if self .texture_saliency_mask_score not in {"lap_var"}:
            raise ValueError ("texture_saliency_mask_score must be 'lap_var'")
        if self .texture_saliency_mask_grid <=0 :
            raise ValueError ("texture_saliency_mask_grid must be > 0")
        if self .texture_saliency_mask_radius <=0 :
            raise ValueError ("texture_saliency_mask_radius must be > 0")
        self .texture_saliency_mask_apply_to_cls =bool (texture_saliency_mask_apply_to_cls )

        self .midframe_enabled =bool (midframe_enabled )
        self .round_trip_composition_enabled =bool (round_trip_composition_enabled )

        self .dynamic_support_mask_enabled =bool (dynamic_support_mask_enabled )
        self .dynamic_support_mask_apply_to_cls =bool (dynamic_support_mask_apply_to_cls )

        self .superpoints_K =int (superpoints_K )
        if self .superpoints_K <0 :
            raise ValueError ("superpoints_K must be >= 0")
        self .superpoints_apply_to_cls =bool (superpoints_apply_to_cls )

        self .descriptor_norm_topk_ratio =float (descriptor_norm_topk_ratio )
        if not (0.0 <self .descriptor_norm_topk_ratio <1.0 ):
            raise ValueError ("descriptor_norm_topk_ratio must be in (0,1)")
        self .descriptor_detach_scale =bool (descriptor_detach_scale )
        self .stopgrad_cls_to_motion =bool (stopgrad_cls_to_motion )
        self .classifier_saliency_mask_enabled =bool (classifier_saliency_mask_enabled )
        self .classifier_saliency_mask_ratio =float (classifier_saliency_mask_ratio )
        if not (0.0 <self .classifier_saliency_mask_ratio <1.0 ):
            raise ValueError ("classifier_saliency_mask_ratio must be in (0,1)")
        self .classifier_saliency_mask_blur_k =int (classifier_saliency_mask_blur_k )
        if self .classifier_saliency_mask_blur_k <0 :
            raise ValueError ("classifier_saliency_mask_blur_k must be >= 0")
        if self .classifier_saliency_mask_blur_k >0 and (self .classifier_saliency_mask_blur_k %2 )==0 :
            raise ValueError ("classifier_saliency_mask_blur_k must be odd when > 0")
        self .classifier_saliency_mask_blur_sigma =float (classifier_saliency_mask_blur_sigma )
        if self .classifier_saliency_mask_blur_sigma <=0.0 :
            raise ValueError ("classifier_saliency_mask_blur_sigma must be > 0")
        self .cls_dual_view_enabled =bool (cls_dual_view_enabled )
        cm =str (cls_dual_view_merge ).strip ().lower ().replace ("-","_")
        if cm in {"avg","mean"}:
            cm ="avg_logits"
        if cm not in {"avg_logits"}:
            raise ValueError ("cls_dual_view_merge must be 'avg_logits'")
        self .cls_dual_view_merge =cm 
        self .cls_dual_view_w_unmasked =float (cls_dual_view_w_unmasked )
        if not (0.0 <=self .cls_dual_view_w_unmasked <=1.0 ):
            raise ValueError ("cls_dual_view_w_unmasked must be in [0,1]")
        self .conf_dyn_enabled =bool (conf_dyn_enabled )
        self .conf_dyn_topk_ratio =float (conf_dyn_topk_ratio )
        if not (0.0 <self .conf_dyn_topk_ratio <1.0 ):
            raise ValueError ("conf_dyn_topk_ratio must be in (0,1)")
        self .conf_dyn_dilate_kernel =int (conf_dyn_dilate_kernel )
        if self .conf_dyn_dilate_kernel <1 :
            raise ValueError ("conf_dyn_dilate_kernel must be >= 1")

        vm =str (view_mode ).strip ().lower ().replace ("-","_")
        if vm in {"","dual","full","dual_view"}:
            vm ="dual"
        elif vm in {"on","onset","onset_only"}:
            vm ="onset_only"
        elif vm in {"off","offset","offset_only"}:
            vm ="offset_only"
        else :
            raise ValueError ("view_mode must be one of dual/onset_only/offset_only")
        self .view_mode =vm 

        ba =str (routing_mode ).strip ().lower ().replace ("-","_")
        if ba in {"","full","default"}:
            ba ="full"
        if ba not in {"full","uniform_valid","support_only","trust_only"}:
            raise ValueError ("routing_mode must be one of full/uniform_valid/support_only/trust_only")
        self .routing_mode =ba 

        tm =str (transport_mode ).strip ().lower ().replace ("-","_")
        if tm in {"","symmetric","dual"}:
            tm ="symmetric"
        elif tm in {"on","onset","onset_only"}:
            tm ="onset_only"
        elif tm in {"off","offset","offset_only"}:
            tm ="offset_only"
        else :
            raise ValueError ("transport_mode must be one of symmetric/onset_only/offset_only")
        self .transport_mode =tm 

        mask_modes =int (self .region_reconstruction_enabled )+int (self .texture_saliency_mask_enabled )+int (self .dynamic_support_mask_enabled )+int (self .superpoints_K >0 )
        if mask_modes >1 :
            raise ValueError ("At most one of region_reconstruction/texture_saliency_mask/dynamic_support_mask/superpoints can be enabled (keep optional objectives isolated).")
        if mask_modes and (self .cls_dual_stream_enabled or self .cls_roi_enabled or self .cls_region_masks_enabled or self .cls_use_err or self .fusion_enabled ):
            raise ValueError ("New mask-style optional objectives are not supported with dual-stream/cls_roi/region-masks/err/fusion (keep optional objectives isolated).")
        if (self .midframe_enabled or self .round_trip_composition_enabled or self .predict_backward )and self .motion_mode not in {"apex_pair_flow","apex_pair_flow_dt"}:
            raise ValueError ("predict_backward/midframe/round_trip_composition are only supported for apex_pair_flow(_dt).")
        if (self .dynamic_support_mask_enabled or (self .superpoints_K >0 ))and self .motion_mode not in {"apex_pair_flow","apex_pair_flow_dt"}:
            raise ValueError ("dynamic_support_mask/superpoints are only supported for apex_pair_flow(_dt).")

        self .feat_loss_enabled =bool (feat_loss_enabled )
        self .feat_embed_channels =int (feat_embed_channels )

        if self .global_local_enabled and self .motion_mode not in {"apex_pair_flow","apex_pair_flow_dt"}:
            raise ValueError ("global_local_enabled is only supported for motion_mode=apex_pair_flow(_dt)")
        if self .global_local_enabled and (self .motion_mode =="error_feedback_refine"or self .refine_enabled ):
            raise ValueError ("global_local_enabled is not supported together with refine modes (keep optional objectives isolated).")
        if self .refine_enabled and self .motion_mode not in {"apex_pair_flow","apex_pair_flow_dt"}:
            raise ValueError ("refine_enabled is only supported for motion_mode=apex_pair_flow(_dt)")
        if self .refine_enabled and self .motion_mode =="error_feedback_refine":
            raise ValueError ("refine_enabled should use motion_mode=apex_pair_flow(_dt), not error_feedback_refine")
        if self .fusion_enabled and self .cls_global_removal =="none":
            raise ValueError ("fusion_enabled requires cls_global_removal != 'none' to produce a residual stream.")
        if self .fusion_enabled :
            if self .classifier_in_channels !=4 :
                raise ValueError ("fusion_enabled requires classifier_in_channels=4 (raw/residual streams).")
            if self .cls_rep !="raw":
                raise ValueError ("fusion_enabled requires cls_rep='raw'")
            if self .cls_dual_stream_enabled or self .cls_roi_enabled or self .cls_region_masks_enabled or self .cls_use_err or self .cls_motion_scale_enabled :
                raise ValueError ("fusion_enabled is not supported with dual-stream/roi/region-masks/err/motion-scale (keep optional objectives isolated).")
        if self .sc_enabled and self .fusion_enabled :
            raise ValueError ("sc_enabled is not supported together with fusion_enabled (keep optional objectives isolated).")
        if self .cls_dual_branch_enabled :
            if self .motion_mode !="apex_pair_flow":
                raise ValueError ("cls_dual_branch_enabled requires motion_mode='apex_pair_flow'")
            if self .fusion_enabled :
                raise ValueError ("cls_dual_branch_enabled is not supported together with fusion_enabled (keep optional objectives isolated).")
            if self .sc_enabled or self .gradient_classifier_enabled :
                raise ValueError ("cls_dual_branch_enabled is not supported together with sc_enabled/gradient_classifier_enabled (keep optional objectives isolated).")
            if self .cls_dual_stream_enabled or self .cls_roi_enabled or self .cls_region_masks_enabled or self .cls_use_err or self .cls_motion_scale_enabled :
                raise ValueError ("cls_dual_branch_enabled is not supported with dual-stream/roi/region-masks/err/motion-scale (keep optional objectives isolated).")
        if self .classifier_saliency_mask_enabled and self .cls_rep !="differential_deformation_descriptor":
            raise ValueError ("classifier_saliency_mask_enabled=true currently requires cls_rep='differential_deformation_descriptor'.")
        if self .cls_dual_view_enabled :
            if not self .classifier_saliency_mask_enabled :
                raise ValueError ("cls_dual_view_enabled=true requires classifier_saliency_mask_enabled=true.")
            if self .two_stage_cls_enabled :
                raise ValueError ("cls_dual_view_enabled is not supported with two_stage_cls_enabled.")
            if self .cls_dual_branch_enabled or self .fusion_enabled :
                raise ValueError ("cls_dual_view_enabled is not supported with cls_dual_branch_enabled/fusion_enabled.")
            if self .sc_enabled :
                raise ValueError ("cls_dual_view_enabled is not supported with sc_enabled.")
            if self .contrastive_enabled :
                raise ValueError ("cls_dual_view_enabled is not supported with contrastive_enabled.")

        self .motion :MotionNet |None =None 
        self .pair_flow :DVAWFlowPredictor |None =None 
        self .triplet_flow :TripletFlowNet |None =None 
        self .cost_volume_pair_flow :CostVolumePairFlowNet |None =None 
        self .refine_head :FlowRefineHead |None =None 
        if self .motion_mode =="endpoints_dt":
            from .motion_net import MotionNet 
            self .motion =MotionNet (
            image_channels =self .image_channels ,
            base_channels =int (motion_base_channels ),
            flow_downscale =int (flow_downscale ),
            max_disp =float (max_disp ),
            )
        elif self .motion_mode in {"apex_pair_flow","apex_pair_flow_dt","apex_neighborhood_flow","error_feedback_refine"}:
            self .pair_flow =DVAWFlowPredictor (
            image_channels =self .image_channels ,
            base_channels =int (motion_base_channels ),
            flow_downscale =int (flow_downscale ),
            max_disp =float (max_disp ),
            basis_displacement_enabled =bool (self .basis_displacement_enabled ),
            basis_displacement_K =int (self .basis_displacement_K ),
            basis_displacement_base_res =int (self .basis_displacement_base_res ),
            dynamic_support_mask_enabled =bool (self .dynamic_support_mask_enabled ),
            superpoints_K =int (self .superpoints_K ),
            )
        elif self .motion_mode =="triplet_apex_cond_dt":
            from .triplet_flow_net import TripletFlowNet 
            self .triplet_flow =TripletFlowNet (
            image_channels =self .image_channels ,
            base_channels =int (motion_base_channels ),
            flow_downscale =int (flow_downscale ),
            max_disp =float (max_disp ),
            )
        else :
            from .cost_volume_pair_flow_net import CostVolumePairFlowNet 
            self .cost_volume_pair_flow =CostVolumePairFlowNet (
            image_channels =self .image_channels ,
            base_channels =int (motion_base_channels ),
            corr_radius =int (corr_radius ),
            max_disp =float (max_disp ),
            )

        if self .motion_mode =="error_feedback_refine"or self .refine_enabled :
            from .refine_head import FlowRefineHead 
            self .refine_head =FlowRefineHead (image_channels =self .image_channels ,base_channels =int (refine_base_channels ))

        self .global_theta_head :nn .Module |None =None 
        if self .global_local_enabled :
            self .global_theta_head =_GlobalThetaHead (
            in_channels =2 *self .image_channels ,
            base_channels =int (global_local_base_channels ),
            theta_scale =float (self .global_local_theta_scale ),
            )

        self .fusion_gate :nn .Module |None =None 
        if self .fusion_enabled :
            h =int (fusion_gate_hidden )
            self .fusion_gate =nn .Sequential (
            nn .Linear (3 ,h ,bias =True ),
            nn .ReLU (inplace =True ),
            nn .Linear (h ,1 ,bias =True ),
            )

        self .feat_encoder :nn .Module |None =None 
        if self .feat_loss_enabled :
            from .embed_diff_model import FrameEncoder 
            self .feat_encoder =FrameEncoder (in_channels =self .image_channels ,embed_channels =int (self .feat_embed_channels ))

        self .learnable_lowpass :nn .Conv2d |None =None 
        if self .cls_global_removal =="learnable_lowpass":
            k =int (self .cls_lowpass_kernel )
            if k <=1 or (k %2 )==0 :
                raise ValueError ("cls_lowpass_kernel must be an odd integer >= 3 for learnable_lowpass")
            pad =k //2 
            self .learnable_lowpass =nn .Conv2d (2 ,2 ,kernel_size =k ,padding =pad ,groups =2 ,bias =False )
            with torch .no_grad ():
                self .learnable_lowpass .weight .fill_ (1.0 /float (k *k ))

        self .dual_stream_gate :nn .Conv2d |None =None 
        if self .cls_dual_stream_enabled and self .cls_dual_stream_fusion =="gated":

            self .dual_stream_gate =nn .Conv2d (8 ,4 ,kernel_size =1 ,bias =True )

        self .region_mask_head :nn .Module |None =None 
        if self .cls_region_masks_enabled :

            self .region_mask_head =nn .Sequential (
            nn .Conv2d (4 ,16 ,kernel_size =3 ,padding =1 ,bias =True ),
            nn .ReLU (inplace =True ),
            nn .Conv2d (16 ,int (self .cls_region_masks_k ),kernel_size =1 ,bias =True ),
            )

        self .adapter =ChannelAdapter (in_channels =self .classifier_in_channels ,out_channels =self .classifier_input_channels )
        self .classifier =MotionClassifier (in_channels =self .classifier_input_channels ,out_channels =self .num_classes )

        self .contrastive_proj :nn .Module |None =None 
        if self .contrastive_enabled and self .contrastive_proj_dim >0 :
            in_dim =int (getattr (getattr (self .classifier ,"fc",None ),"in_features",16 ))
            self .contrastive_proj =nn .Linear (in_dim ,int (self .contrastive_proj_dim ),bias =True )

        self .gradient_classifier_head :nn .Module |None =None 
        self .gradient_classifier_param_count :int =0 
        if self .gradient_classifier_enabled :
            in_dim =int (getattr (getattr (self .classifier ,"fc",None ),"in_features",16 ))
            self .gradient_classifier_head =GradientScaledClassifierHead (
            in_dim =int (in_dim ),
            num_classes =int (self .num_classes ),
            hidden_dim =int (self .gradient_classifier_hidden_dim ),
            num_layers =int (self .gradient_classifier_num_layers ),
            dropout =float (self .gradient_classifier_dropout ),
            )
            self .gradient_classifier_param_count =int (sum (int (p .numel ())for p in self .gradient_classifier_head .parameters ()))

        self .adapter_a :ChannelAdapter |None =None 
        self .adapter_b :ChannelAdapter |None =None 
        self .classifier_a :nn .Module |None =None 
        self .classifier_b :nn .Module |None =None 
        if self .cls_dual_branch_enabled :
            self .adapter_a =ChannelAdapter (in_channels =4 ,out_channels =self .classifier_input_channels )
            self .adapter_b =ChannelAdapter (in_channels =2 ,out_channels =self .classifier_input_channels )
            self .classifier_a =MotionClassifier (in_channels =self .classifier_input_channels ,out_channels =self .num_classes )
            self .classifier_b =MotionClassifier (in_channels =self .classifier_input_channels ,out_channels =self .num_classes )


        if self .channel_swap_enabled :
            c_in =int (self .classifier_in_channels )
            self .channel_swap_proj =nn .Sequential (
            nn .AdaptiveAvgPool2d (1 ),
            nn .Flatten (),
            nn .Linear (c_in ,int (self .channel_swap_proj_dim )),
            )
            if self .channel_swap_mode =="siam":
                self .channel_swap_pred =nn .Sequential (
                nn .Linear (int (self .channel_swap_proj_dim ),int (self .channel_swap_pred_dim )),
                nn .ReLU (inplace =True ),
                nn .Linear (int (self .channel_swap_pred_dim ),int (self .channel_swap_proj_dim )),
                )
            if self .channel_swap_mode =="region_align":
                half_c =int (self .classifier_in_channels )//2 
                self .region_proj =nn .Linear (int (half_c ),int (self .channel_swap_proj_dim ))

        if self .cross_view_mask_enabled :
            half_c =int (self .classifier_in_channels )//2 
            self .cross_view_mask_pred =nn .Sequential (
            nn .Conv2d (int (half_c )*2 ,int (self .cross_view_mask_hidden ),kernel_size =3 ,padding =1 ),
            nn .ReLU (inplace =True ),
            nn .Conv2d (int (self .cross_view_mask_hidden ),int (self .cross_view_mask_hidden ),kernel_size =3 ,padding =1 ),
            nn .ReLU (inplace =True ),
            nn .Conv2d (int (self .cross_view_mask_hidden ),int (half_c ),kernel_size =3 ,padding =1 ),
            )



        reco_cfg =self .reco_cfg if isinstance (self .reco_cfg ,dict )else {}
        region_cfg =(reco_cfg .get ("region",{})or {})if isinstance (reco_cfg ,dict )else {}
        region_mode =str (region_cfg .get ("mode","quad4")).strip ().lower ()

        reco_channel_swap_cfg =(reco_cfg .get ("channel_swap",{})or {})if isinstance (reco_cfg ,dict )else {}
        reco_cross_view_mask_cfg =(reco_cfg .get ("cross_view_mask",{})or {})if isinstance (reco_cfg ,dict )else {}
        reco_mf_cfg =(reco_cfg .get ("multiframe",{})or {})if isinstance (reco_cfg ,dict )else {}
        reco_transport_cfg =(reco_cfg .get ("trusted_transport",{})or {})if isinstance (reco_cfg ,dict )else {}

        reco_channel_swap_enabled =bool (reco_channel_swap_cfg .get ("enabled",False ))
        reco_cross_view_mask_enabled =bool (reco_cross_view_mask_cfg .get ("enabled",False ))
        reco_mf_enabled =bool (reco_mf_cfg .get ("enabled",False ))
        reco_transport_enabled =bool (reco_transport_cfg .get ("enabled",False ))
        trust_token_source =str (reco_transport_cfg .get ("conf_tok_source","photo")).strip ().lower ().replace ("-","_")
        if trust_token_source in {"","photo"}:
            trust_token_source ="photo"
        if trust_token_source not in {"photo","reliability"}:
            raise ValueError (
            "reco.trusted_transport.conf_tok_source must be 'photo' or 'reliability'. "
            f"Got: {trust_token_source!r}"
            )
        self .trust_token_source =trust_token_source 

        reco_cross_view_mask_region_source =str (reco_cross_view_mask_cfg .get ("region_source","reco")).strip ().lower ()
        if reco_cross_view_mask_region_source in {"","reco"}:
            reco_cross_view_mask_region_source ="reco"
        if reco_cross_view_mask_region_source not in {"reco"}:
            raise ValueError (f"reco.cross_view_mask.region_source must be 'reco'. Got: {reco_cross_view_mask_region_source!r}")
        self .reco_cross_view_mask_region_source =reco_cross_view_mask_region_source 



        need_reco_regions =bool (
        (self .sc_enabled and self .sc_region_source =="reco")
        or (self .channel_swap_enabled and str (self .channel_swap_partition_base )=="reco")
        or (reco_channel_swap_enabled )
        or (reco_cross_view_mask_enabled and reco_cross_view_mask_region_source =="reco")
        or (reco_mf_enabled and str (reco_mf_cfg .get ("variant","")).strip ().lower ()=="multidelta_supcon")
        )
        self .need_reco_regions =bool (need_reco_regions )


        reco_any_module_enabled =bool (reco_channel_swap_enabled or reco_cross_view_mask_enabled or reco_mf_enabled or reco_transport_enabled )
        if self .need_reco_regions and region_mode in {"","quad4"}:
            raise ValueError (
            "need_reco_regions=true but reco.region.mode is 'quad4'. "
            "Set reco.region.mode to a non-quad4 provider (e.g. weak_splits_3x3 or superpoint_voronoi)."
            )
        if reco_any_module_enabled or self .need_reco_regions :
            self .reco =ReCoAuxiliary (self .reco_cfg ,enable_regions =bool (self .need_reco_regions ))

        self ._fixed_roi_mask_base :torch .Tensor |None =None 
        self ._entropy_gaussian_kernel_base :torch .Tensor |None =None 
        self ._entropy_gaussian_radius_cached :int |None =None 

    def _apply_global_removal (self ,flow :torch .Tensor )->Tuple [torch .Tensor ,torch .Tensor ]:
        m =self .cls_global_removal 
        if m =="none":
            z =torch .zeros_like (flow )
            return flow ,z 
        if m =="lowpass":
            return remove_lowpass (flow ,kernel_size =int (self .cls_lowpass_kernel ))
        if m =="affine":
            return remove_affine (flow )
        if m =="learnable_lowpass":
            if self .learnable_lowpass is None :
                raise RuntimeError ("learnable_lowpass module is not initialized")
            glob =self .learnable_lowpass (flow )
            return flow -glob ,glob 
        raise RuntimeError (f"Unexpected cls_global_removal: {m}")

    def _apply_global_removal_to_xcls (self ,x_cls :torch .Tensor )->Tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:
        if self .cls_global_removal =="none":
            return x_cls ,{}
        if x_cls .ndim !=4 or (int (x_cls .shape [1 ])%2 )!=0 :
            raise ValueError (f"x_cls must be [B,C,H,W] with even C for global removal, got {tuple(x_cls.shape)}")
        parts =[]
        aux :Dict [str ,torch .Tensor ]={}
        for i in range (0 ,int (x_cls .shape [1 ]),2 ):
            f =x_cls [:,i :i +2 ]
            resid ,glob =self ._apply_global_removal (f )
            parts .append (resid )
            aux [f"global_{i//2}"]=glob 
            aux [f"residual_{i//2}"]=resid 
        return torch .cat (parts ,dim =1 ),aux 

    @staticmethod 
    def _err_map (pred :torch .Tensor ,target :torch .Tensor )->torch .Tensor :

        return (pred -target ).abs ().mean (dim =1 ,keepdim =True )

    @staticmethod 
    def _dyn_saliency_mask (
    apex :torch .Tensor ,
    source :torch .Tensor ,
    *,
    topk_ratio :float ,
    dilate_kernel :int ,
    )->torch .Tensor :

        if apex .ndim !=4 or source .ndim !=4 or tuple (apex .shape )!=tuple (source .shape ):
            raise ValueError (f"apex/source shape mismatch: apex={tuple(apex.shape)} source={tuple(source.shape)}")
        b =int (apex .shape [0 ])
        err =(apex -source ).abs ().mean (dim =1 ,keepdim =True )
        flat =err .view (b ,-1 )
        n =int (flat .shape [1 ])
        k =max (int (round (float (topk_ratio )*n )),1 )
        top ,_ =torch .topk (flat ,k =k ,dim =1 ,largest =True ,sorted =False )
        thr =top .min (dim =1 ).values .view (b ,1 ,1 ,1 )
        mask =(err >=thr ).to (dtype =err .dtype )
        dk =int (max (int (dilate_kernel ),1 ))
        if dk >1 :
            if (dk %2 )==0 :
                dk +=1 
            mask =F .max_pool2d (mask ,kernel_size =dk ,stride =1 ,padding =dk //2 )
            mask =(mask >0.5 ).to (dtype =err .dtype )
        return mask 

    @staticmethod 
    def _oob_mask (flow :torch .Tensor ,*,align_corners :bool =True )->torch .Tensor :
        if flow .ndim !=4 or flow .shape [1 ]!=2 :
            raise ValueError (f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
        valid =flow_valid_mask (flow ,align_corners =align_corners ,eps =0.0 )
        return (1.0 -valid ).clamp (0.0 ,1.0 )

    @staticmethod 
    def _fb_cycle_residual_map (flow_fwd :torch .Tensor ,flow_bwd :torch .Tensor )->torch .Tensor :

        if flow_fwd .ndim !=4 or flow_fwd .shape [1 ]!=2 :
            raise ValueError (f"flow_fwd must be [B,2,H,W], got {tuple(flow_fwd.shape)}")
        if flow_bwd .ndim !=4 or flow_bwd .shape [1 ]!=2 :
            raise ValueError (f"flow_bwd must be [B,2,H,W], got {tuple(flow_bwd.shape)}")
        if flow_fwd .shape !=flow_bwd .shape :
            raise ValueError (f"flow_fwd/flow_bwd shape mismatch: {tuple(flow_fwd.shape)} vs {tuple(flow_bwd.shape)}")
        cyc =flow_bwd +warp (flow_fwd ,flow_bwd )
        return torch .sqrt ((cyc *cyc ).sum (dim =1 ,keepdim =True )+1e-12 )

    @staticmethod 
    def _flow_to_deformation_descriptor (flow :torch .Tensor )->torch .Tensor :

        if flow .ndim !=4 or int (flow .shape [1 ])!=2 :
            raise ValueError (f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
        u =flow [:,0 :1 ]
        v =flow [:,1 :2 ]
        device =flow .device 
        dtype =flow .dtype 

        kx =torch .tensor ([[-1.0 ,0.0 ,1.0 ],[-2.0 ,0.0 ,2.0 ],[-1.0 ,0.0 ,1.0 ]],device =device ,dtype =dtype ).view (1 ,1 ,3 ,3 )
        ky =torch .tensor ([[-1.0 ,-2.0 ,-1.0 ],[0.0 ,0.0 ,0.0 ],[1.0 ,2.0 ,1.0 ]],device =device ,dtype =dtype ).view (1 ,1 ,3 ,3 )

        du_dx =F .conv2d (u ,kx ,padding =1 )
        du_dy =F .conv2d (u ,ky ,padding =1 )
        dv_dx =F .conv2d (v ,kx ,padding =1 )
        dv_dy =F .conv2d (v ,ky ,padding =1 )

        div =du_dx +dv_dy 
        curl =dv_dx -du_dy 
        shear1 =du_dx -dv_dy 
        shear2 =du_dy +dv_dx 
        return torch .cat ([div ,curl ,shear1 ,shear2 ],dim =1 )

    @staticmethod 
    def _normalize_deform_op (op :str )->str :
        o =str (op ).strip ().lower ().replace ("-","_")
        if o in {"divergence"}:
            o ="div"
        if o in {"rot","rotation"}:
            o ="curl"
        if o in {"s1","shear_1"}:
            o ="shear1"
        if o in {"s2","shear_2"}:
            o ="shear2"
        if o not in {"div","curl","shear1","shear2"}:
            raise ValueError (f"Unknown deform op: {op!r}. Choose from div/curl/shear1/shear2.")
        return o 

    @classmethod 
    def _flow_to_deformation_descriptor_ops (cls ,flow :torch .Tensor ,ops :Tuple [str ,...])->torch .Tensor :

        if flow .ndim !=4 or int (flow .shape [1 ])!=2 :
            raise ValueError (f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
        ops_n =[cls ._normalize_deform_op (o )for o in ops ]
        u =flow [:,0 :1 ]
        v =flow [:,1 :2 ]
        device =flow .device 
        dtype =flow .dtype 

        kx =torch .tensor ([[-1.0 ,0.0 ,1.0 ],[-2.0 ,0.0 ,2.0 ],[-1.0 ,0.0 ,1.0 ]],device =device ,dtype =dtype ).view (1 ,1 ,3 ,3 )
        ky =torch .tensor ([[-1.0 ,-2.0 ,-1.0 ],[0.0 ,0.0 ,0.0 ],[1.0 ,2.0 ,1.0 ]],device =device ,dtype =dtype ).view (1 ,1 ,3 ,3 )

        du_dx =F .conv2d (u ,kx ,padding =1 )
        du_dy =F .conv2d (u ,ky ,padding =1 )
        dv_dx =F .conv2d (v ,kx ,padding =1 )
        dv_dy =F .conv2d (v ,ky ,padding =1 )

        by ={
        "div":du_dx +dv_dy ,
        "curl":dv_dx -du_dy ,
        "shear1":du_dx -dv_dy ,
        "shear2":du_dy +dv_dx ,
        }
        return torch .cat ([by [o ]for o in ops_n ],dim =1 )

    @staticmethod 
    def _lowpass_ms (flow :torch .Tensor ,*,scale :int )->torch .Tensor :

        if flow .ndim !=4 :
            raise ValueError (f"flow must be [B,C,H,W], got {tuple(flow.shape)}")
        s =int (scale )
        if s <=1 :
            return flow 
        h =int (flow .shape [-2 ])
        w =int (flow .shape [-1 ])
        x =F .avg_pool2d (flow ,kernel_size =s ,stride =s )
        return F .interpolate (x ,size =(h ,w ),mode ="bilinear",align_corners =True )

    @classmethod 
    def _flow_to_multiscale_deformation_descriptor (
    cls ,
    flow :torch .Tensor ,
    *,
    scales :Tuple [int ,...],
    ops :Tuple [str ,...],
    )->Tuple [torch .Tensor ,Dict [int ,torch .Tensor ]]:
        if not scales :
            scales =(1 ,)
        desc_by_scale :Dict [int ,torch .Tensor ]={}
        parts =[]
        for s in scales :
            ss =int (s )
            f =cls ._lowpass_ms (flow ,scale =ss )
            d =cls ._flow_to_deformation_descriptor_ops (f ,ops )
            desc_by_scale [ss ]=d 
            parts .append (d )
        return torch .cat (parts ,dim =1 ),desc_by_scale 

    def _normalize_err (self ,err :torch .Tensor )->torch .Tensor :
        mode =str (self .cls_err_norm ).strip ().lower ()
        if mode in {"","none","off"}:
            return err 
        if mode in {"mean","avg"}:
            m =err .mean (dim =(1 ,2 ,3 ),keepdim =True )
            return err /(m +float (self .eps ))
        if mode in {"minmax","min_max"}:
            b =int (err .shape [0 ])
            flat =err .view (b ,-1 )
            mn =flat .min (dim =1 ).values .view (b ,1 ,1 ,1 )
            mx =flat .max (dim =1 ).values .view (b ,1 ,1 ,1 )
            return (err -mn )/(mx -mn +float (self .eps ))
        raise ValueError (f"Unsupported cls_err_norm: {self.cls_err_norm!r}")

    @staticmethod 
    def _saliency_mask (saliency :torch .Tensor ,*,ratio :float )->torch .Tensor :
        if saliency .ndim !=4 or saliency .shape [1 ]!=1 :
            raise ValueError (f"saliency must be [B,1,H,W], got {tuple(saliency.shape)}")
        r =float (ratio )
        if not (0.0 <r <1.0 ):
            raise ValueError (f"ratio must be in (0,1), got {ratio}")
        b =int (saliency .shape [0 ])
        flat =saliency .view (b ,-1 )
        n =int (flat .shape [1 ])
        k =max (int (round (r *n )),1 )
        topk ,_ =torch .topk (flat ,k =k ,dim =1 ,largest =True ,sorted =False )
        thr =topk .min (dim =1 ).values .view (b ,1 ,1 ,1 )
        return (saliency >=thr ).to (dtype =saliency .dtype )

    @staticmethod 
    def _gaussian_blur_2d (x :torch .Tensor ,*,k :int ,sigma :float )->torch .Tensor :
        if int (k )<=1 :
            return x 
        if (int (k )%2 )==0 :
            raise ValueError ("Gaussian blur kernel size k must be odd.")
        pad =int (k )//2 
        dev =x .device 
        dt =x .dtype 
        t =torch .arange (int (k ),device =dev ,dtype =dt )-float (pad )
        g =torch .exp (-(t *t )/(2.0 *float (sigma )*float (sigma )))
        g =g /g .sum ().clamp_min (1e-12 )
        k2d =(g [:,None ]*g [None ,:]).view (1 ,1 ,int (k ),int (k ))
        return F .conv2d (x ,k2d ,padding =pad )

    def _build_saliency_mask_from_xcls (self ,x_cls :torch .Tensor )->torch .Tensor :
        if x_cls .ndim !=4 :
            raise ValueError (f"x_cls must be [B,C,H,W], got {tuple(x_cls.shape)}")
        c =int (x_cls .shape [1 ])
        if c !=4 :
            raise ValueError (f"classifier_saliency_mask expects differential deformation descriptor 4-channel input, got C={c}")
        div_on =x_cls [:,0 :1 ]
        curl_on =x_cls [:,1 :2 ]
        div_off =x_cls [:,2 :3 ]
        curl_off =x_cls [:,3 :4 ]
        e_on =torch .sqrt (div_on *div_on +curl_on *curl_on +float (self .eps ))
        e_off =torch .sqrt (div_off *div_off +curl_off *curl_off +float (self .eps ))
        energy =0.5 *(e_on +e_off )
        energy =self ._gaussian_blur_2d (
        energy ,
        k =int (self .classifier_saliency_mask_blur_k ),
        sigma =float (self .classifier_saliency_mask_blur_sigma ),
        )
        return self ._saliency_mask (energy ,ratio =float (self .classifier_saliency_mask_ratio ))

    def _get_fixed_roi_mask (self ,*,batch_size :int ,height :int ,width :int ,device :torch .device ,dtype :torch .dtype )->torch .Tensor :
        h =int (height )
        w =int (width )
        if self ._fixed_roi_mask_base is None or tuple (self ._fixed_roi_mask_base .shape [-2 :])!=(h ,w ):
            base =torch .zeros ((1 ,1 ,h ,w ),dtype =torch .float32 ,device =torch .device ("cpu"))


            y0_eye =int (round (0.24 *h ))
            y1_eye =int (round (0.45 *h ))
            x0_le =int (round (0.18 *w ))
            x1_le =int (round (0.45 *w ))
            x0_re =int (round (0.55 *w ))
            x1_re =int (round (0.82 *w ))

            y0_m =int (round (0.55 *h ))
            y1_m =int (round (0.82 *h ))
            x0_m =int (round (0.25 *w ))
            x1_m =int (round (0.75 *w ))

            base [:,:,y0_eye :y1_eye ,x0_le :x1_le ]=1.0 
            base [:,:,y0_eye :y1_eye ,x0_re :x1_re ]=1.0 
            base [:,:,y0_m :y1_m ,x0_m :x1_m ]=1.0 
            self ._fixed_roi_mask_base =base 

        m =self ._fixed_roi_mask_base .to (device =device ,dtype =dtype )
        return m .expand (int (batch_size ),-1 ,-1 ,-1 )

    def _get_entropy_gaussian_kernel (self ,*,radius :int ,device :torch .device ,dtype :torch .dtype )->torch .Tensor :
        r =int (radius )
        if r <=0 :
            raise ValueError ("radius must be > 0")
        if self ._entropy_gaussian_kernel_base is None or self ._entropy_gaussian_radius_cached !=r :
            k =2 *r +1 
            yy ,xx =torch .meshgrid (torch .arange (k ),torch .arange (k ),indexing ="ij")
            yy =yy .to (dtype =torch .float32 )-float (r )
            xx =xx .to (dtype =torch .float32 )-float (r )
            sigma =float (max (r ,1 ))/2.0 
            g =torch .exp (-(xx *xx +yy *yy )/(2.0 *sigma *sigma ))
            g =(g /g .sum ().clamp_min (1e-12 )).view (1 ,1 ,k ,k )
            self ._entropy_gaussian_kernel_base =g 
            self ._entropy_gaussian_radius_cached =r 
        return self ._entropy_gaussian_kernel_base .to (device =device ,dtype =dtype )

    def _texture_saliency_mask_from_img (self ,img :torch .Tensor )->torch .Tensor :

        if img .ndim !=4 :
            raise ValueError (f"img must be [B,C,H,W], got {tuple(img.shape)}")
        bsz ,_c ,h ,w =img .shape 
        device =img .device 
        dtype =img .dtype 


        gray =img .mean (dim =1 ,keepdim =True )
        k =torch .tensor ([[0.0 ,1.0 ,0.0 ],[1.0 ,-4.0 ,1.0 ],[0.0 ,1.0 ,0.0 ]],device =device ,dtype =dtype ).view (1 ,1 ,3 ,3 )
        lap =F .conv2d (gray ,k ,padding =1 )
        win =7 
        pad =win //2 
        mu =F .avg_pool2d (lap ,kernel_size =win ,stride =1 ,padding =pad )
        mu2 =F .avg_pool2d (lap *lap ,kernel_size =win ,stride =1 ,padding =pad )
        score =(mu2 -mu *mu ).clamp_min (0.0 )

        g =int (self .texture_saliency_mask_grid )
        if (h %g )!=0 or (w %g )!=0 :
            raise ValueError (f"texture_saliency_mask_grid={g} must divide H,W. Got H={h} W={w}")
        ch =h //g 
        cw =w //g 


        s =score .view (int (bsz ),g ,ch ,g ,cw ).permute (0 ,1 ,3 ,2 ,4 ).contiguous ()
        flat =s .view (int (bsz ),g ,g ,ch *cw )
        idx =flat .argmax (dim =3 )
        iy =(idx //cw ).to (dtype =torch .int64 )
        ix =(idx %cw ).to (dtype =torch .int64 )
        gy =torch .arange (g ,device =device ,dtype =torch .int64 ).view (1 ,g ,1 ).expand (int (bsz ),g ,g )
        gx =torch .arange (g ,device =device ,dtype =torch .int64 ).view (1 ,1 ,g ).expand (int (bsz ),g ,g )
        y =gy *int (ch )+iy 
        x =gx *int (cw )+ix 
        lin =(y *int (w )+x ).view (int (bsz ),-1 )

        seeds =torch .zeros ((int (bsz ),int (h )*int (w )),device =device ,dtype =dtype )
        seeds .scatter_ (dim =1 ,index =lin ,src =torch .ones_like (lin ,dtype =dtype ))
        seeds =seeds .view (int (bsz ),1 ,int (h ),int (w ))

        kernel =self ._get_entropy_gaussian_kernel (radius =int (self .texture_saliency_mask_radius ),device =device ,dtype =dtype )
        mask =F .conv2d (seeds ,kernel ,padding =int (self .texture_saliency_mask_radius ))
        return mask .clamp (0.0 ,1.0 )

    def _normalize_flow_topk_mean (
    self ,
    flow :torch .Tensor ,
    *,
    topk_ratio :float ,
    weights :Optional [torch .Tensor ]=None ,
    detach_scale :Optional [bool ]=None ,
    weighted_reduce :bool =False ,
    )->tuple [torch .Tensor ,torch .Tensor ]:
        if flow .ndim !=4 or int (flow .shape [1 ])!=2 :
            raise ValueError (f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
        b =int (flow .shape [0 ])
        mag =torch .sqrt ((flow *flow ).sum (dim =1 ,keepdim =True )+float (self .eps ))
        mag_flat =mag .view (b ,-1 )
        n =int (mag_flat .shape [1 ])
        score_flat =mag_flat 
        if weights is not None :
            if weights .ndim !=4 or int (weights .shape [1 ])!=1 or tuple (weights .shape [-2 :])!=tuple (flow .shape [-2 :]):
                raise ValueError (f"weights must be [B,1,H,W] matching flow spatial. Got {tuple(weights.shape)} vs {tuple(flow.shape)}")
            w =weights .to (device =flow .device ,dtype =flow .dtype ).clamp_min (0.0 )
            w_flat =w .view (b ,-1 )
            score_flat =mag_flat *w_flat 

            score_sum =score_flat .sum (dim =1 ,keepdim =True )
            score_flat =torch .where (score_sum >float (self .eps ),score_flat ,mag_flat )
        k =max (int (round (float (topk_ratio )*n )),1 )
        _ ,top_idx =torch .topk (score_flat ,k =k ,dim =1 ,largest =True ,sorted =False )
        mag_top =torch .gather (mag_flat ,dim =1 ,index =top_idx )
        if weights is not None :
            w_top =torch .gather (w_flat ,dim =1 ,index =top_idx )
            den =w_top .sum (dim =1 )
            m_weighted =(mag_top *w_top ).sum (dim =1 )/den .clamp_min (float (self .eps ))
            m_plain =mag_top .mean (dim =1 )
            if bool (weighted_reduce ):
                m =torch .where (den >float (self .eps ),m_weighted ,m_plain )
            else :
                m =m_plain 
        else :
            m =mag_top .mean (dim =1 )
        do_detach =bool (self .descriptor_detach_scale )if detach_scale is None else bool (detach_scale )
        m_norm =m .detach ()if do_detach else m 
        flow_n =flow /(m_norm .view (b ,1 ,1 ,1 )+float (self .eps ))
        return flow_n ,m 

    @staticmethod 
    def _quadrant_pool (feat :torch .Tensor )->torch .Tensor :

        if feat .ndim !=4 :
            raise ValueError (f"feat must be [B,C,H,W], got {tuple(feat.shape)}")
        b ,c ,h ,w =feat .shape 
        h2 =int (h //2 )
        w2 =int (w //2 )

        def _pool (x :torch .Tensor )->torch .Tensor :
            if int (x .shape [-2 ])==0 or int (x .shape [-1 ])==0 :
                return feat .mean (dim =(2 ,3 ))
            return x .mean (dim =(2 ,3 ))

        ul =_pool (feat [:,:,:h2 ,:w2 ])
        ur =_pool (feat [:,:,:h2 ,w2 :])
        bl =_pool (feat [:,:,h2 :,:w2 ])
        br =_pool (feat [:,:,h2 :,w2 :])
        z =torch .stack ([ul ,ur ,bl ,br ],dim =1 )
        return F .normalize (z ,dim =2 )

    def _region_pool (self ,feat :torch .Tensor )->torch .Tensor :

        if feat .ndim !=4 :
            raise ValueError (f"feat must be [B,C,H,W], got {tuple(feat.shape)}")
        _b ,_c ,h ,w =feat .shape 
        rid ,nreg ,_ =build_region_id_map (self .sc_region_partition ,int (h ),int (w ),device =feat .device )
        z ,_ =region_mean (feat ,rid ,int (nreg ))
        return F .normalize (z ,dim =2 )

    def forward (
    self ,
    onset :torch .Tensor ,
    apex :torch .Tensor ,
    offset :torch .Tensor ,
    t_on :torch .Tensor |int |list ,
    t_ap :torch .Tensor |int |list ,
    t_off :torch .Tensor |int |list ,
    *,
    rng_seeds :Optional [Dict [str ,int ]]=None ,
    apex_m1 :Optional [torch .Tensor ]=None ,
    apex_p1 :Optional [torch .Tensor ]=None ,
    nb_valid :Optional [torch .Tensor ]=None ,
    Fi :Optional [torch .Tensor ]=None ,
    Fj :Optional [torch .Tensor ]=None ,
    t_i :Optional [torch .Tensor |int |list ]=None ,
    t_j :Optional [torch .Tensor |int |list ]=None ,
    sampled_pair_valid :Optional [torch .Tensor ]=None ,
    mid_pre :Optional [torch .Tensor ]=None ,
    mid_post :Optional [torch .Tensor ]=None ,
    neg_targets :Optional [torch .Tensor ]=None ,
    extra_pre :Optional [torch .Tensor ]=None ,
    extra_post :Optional [torch .Tensor ]=None ,
    dt_pre :Optional [torch .Tensor ]=None ,
    dt_post :Optional [torch .Tensor ]=None ,
    trust_propagation_runtime :Optional [Dict [str ,Any ]]=None ,
    )->MotionWarpOutputs :
        eps =self .eps 
        device =onset .device 
        batch_size =int (onset .shape [0 ])

        def _local_gen (tag :str )->Optional [torch .Generator ]:
            if rng_seeds is None :


                if self .training :
                    return None 
                fallback ={"channel_swap":1001 ,"cross_view_mask":1002 ,"transport_on":1003 ,"transport_off":1004 }
                s =fallback .get (str (tag ),None )
                if s is None :
                    return None 
                g =torch .Generator (device =device )
                g .manual_seed (int (s ))
                return g 
            s =rng_seeds .get (tag ,None )if isinstance (rng_seeds ,dict )else None 
            if s is None :
                return None 
            g =torch .Generator (device =device )
            g .manual_seed (int (s ))
            return g 

        def _as_time_tensor (t :torch .Tensor |int |list )->torch .Tensor :
            if isinstance (t ,torch .Tensor ):
                tt =t .to (device =device )
            else :
                tt =torch .tensor (t ,dtype =torch .float32 ,device =device )

            if tt .ndim ==0 :
                tt =tt .repeat (batch_size )
            elif tt .ndim ==2 and tt .shape [1 ]==1 :
                tt =tt .squeeze (1 )

            if tt .ndim !=1 or tt .shape [0 ]!=batch_size :
                raise ValueError (f"time tensor must be shape [B] or [B,1], got {tuple(tt.shape)}")

            if not torch .is_floating_point (tt ):
                tt =tt .float ()
            else :
                tt =tt .to (dtype =torch .float32 )
            return tt 

        t_on_t =_as_time_tensor (t_on )
        t_ap_t =_as_time_tensor (t_ap )
        t_off_t =_as_time_tensor (t_off )

        tau_on =(t_ap_t -t_on_t ).clamp_min (1.0 )
        tau_off =(t_off_t -t_ap_t ).clamp_min (1.0 )
        tau =(t_off_t -t_on_t ).clamp_min (1.0 )

        alpha =(tau_on /(tau +eps )).clamp (0.0 ,1.0 )
        alpha_map =alpha .view (batch_size ,1 ,1 ,1 ).expand (batch_size ,1 ,onset .shape [-2 ],onset .shape [-1 ])


        src_on_for_warp =onset 
        src_off_for_warp =offset 
        src_on_for_flow =onset 
        src_off_for_flow =offset 
        aux_extra :Dict [str ,torch .Tensor ]={}
        trust_propagation_runtime_enabled =bool (self .trust_propagation_enabled )
        trust_propagation_active_min =4 
        if isinstance (trust_propagation_runtime ,dict ):
            trust_propagation_runtime_enabled =bool (trust_propagation_runtime .get ("enabled_runtime",trust_propagation_runtime_enabled ))
            trust_propagation_active_min =int (trust_propagation_runtime .get ("active_min_cnt",trust_propagation_active_min ))
        aux_extra ["trust_propagation_enabled_runtime"]=torch .tensor (int (trust_propagation_runtime_enabled ),device =device ,dtype =torch .int64 )
        aux_extra ["trust_propagation_applied"]=torch .tensor (0 ,device =device ,dtype =torch .int64 )
        need_reco_regions =bool (
        (self .sc_enabled and self .sc_region_source =="reco")
        or (self .channel_swap_enabled and str (self .channel_swap_partition_base )=="reco")
        or (self .reco is not None and bool (self .reco .channel_swap_enabled ))
        or (self .reco is not None and bool (self .reco .cross_view_mask_enabled )and str (self .reco_cross_view_mask_region_source )=="reco")
        or (
        self .reco is not None 
        and bool (self .reco .mf_enabled )
        and str (getattr (self .reco ,"mf_variant","")).strip ().lower ()=="multidelta_supcon"
        )
        )
        aux_extra ["need_reco_regions"]=torch .tensor (int (need_reco_regions ),device =device ,dtype =torch .int64 )
        mask_dyn_on =mask_dyn_off =None 
        sp_mask_on =sp_mask_off =None 
        sp_heatmaps_on =sp_heatmaps_off =None 
        flow_on_to_ap =flow_off_to_ap =None 
        flow_on_to_off =None 
        ce_on =ce_off =r_cyc_map =None 
        w_valid_cyc_on =w_valid_cyc_off =None 
        w_fb_on =w_fb_off =None 
        w_dyn_on =w_dyn_off =None 
        w_valid_on =w_valid_off =None 
        c_on =c_off =c_map =None 
        c_reliability_on =c_reliability_off =c_reliability_map =None 
        u_on =u_off =u_map =None 

        if self .motion_mode =="endpoints_dt":
            if self .motion is None :
                raise RuntimeError ("motion is not initialized for motion_mode='endpoints_dt'")
            motion_out =self .motion (onset ,offset ,alpha_map )
            D_flow =motion_out .D_flow 
            T_flow =motion_out .T_flow 
            flow_ap_to_on =alpha_map *D_flow +T_flow 
            flow_ap_to_off =-(1.0 -alpha_map )*D_flow +T_flow 
            x_cls_base =torch .cat ([T_flow ,D_flow ],dim =1 )
            aux_refine :Dict [str ,torch .Tensor ]={}
        elif self .motion_mode in {"apex_pair_flow","apex_pair_flow_dt","apex_neighborhood_flow","error_feedback_refine"}:
            if self .pair_flow is None :
                raise RuntimeError ("pair_flow is not initialized for pair-flow modes")
            src_on_for_flow =onset 
            src_off_for_flow =offset 
            if self .global_local_enabled :
                if self .global_theta_head is None :
                    raise RuntimeError ("global_theta_head is not initialized for global_local_enabled")
                pair_on =torch .cat ([onset ,apex ],dim =1 )
                pair_off =torch .cat ([offset ,apex ],dim =1 )
                theta_on =self .global_theta_head (pair_on )
                theta_off =self .global_theta_head (pair_off )
                warp_on_global =affine_warp (onset ,theta_on )
                warp_off_global =affine_warp (offset ,theta_off )
                src_on_for_flow =warp_on_global 
                src_off_for_flow =warp_off_global 
                src_on_for_warp =warp_on_global 
                src_off_for_warp =warp_off_global 
                aux_extra .update (
                {
                "theta_on":theta_on ,
                "theta_off":theta_off ,
                "global_theta":torch .stack ([theta_on ,theta_off ],dim =1 ),
                "warp_on_global":warp_on_global ,
                "warp_off_global":warp_off_global ,
                }
                )

            flow_ap_to_on0 ,aux_on0 =self .pair_flow .forward_with_aux (src_on_for_flow ,apex )
            flow_ap_to_off0 ,aux_off0 =self .pair_flow .forward_with_aux (src_off_for_flow ,apex )
            if "basis_coeff"in aux_on0 :
                aux_extra ["basis_coeff_on"]=aux_on0 ["basis_coeff"]
                aux_extra ["coeff_on"]=aux_on0 ["basis_coeff"]
            if "basis_coeff"in aux_off0 :
                aux_extra ["basis_coeff_off"]=aux_off0 ["basis_coeff"]
                aux_extra ["coeff_off"]=aux_off0 ["basis_coeff"]

            if self .dynamic_support_mask_enabled :
                mask_dyn_on =aux_on0 .get ("mask_dyn",None )
                mask_dyn_off =aux_off0 .get ("mask_dyn",None )
                if mask_dyn_on is None or mask_dyn_off is None :
                    raise RuntimeError ("dynamic_support_mask_enabled but DVAWFlowPredictor did not return mask_dyn")
                aux_extra ["dyn_mask_on"]=mask_dyn_on 
                aux_extra ["dyn_mask_off"]=mask_dyn_off 

            if self .superpoints_K >0 :
                sp_mask_on =aux_on0 .get ("sp_mask",None )
                sp_mask_off =aux_off0 .get ("sp_mask",None )
                sp_heatmaps_on =aux_on0 .get ("sp_heatmaps",None )
                sp_heatmaps_off =aux_off0 .get ("sp_heatmaps",None )
                if sp_mask_on is None or sp_mask_off is None or sp_heatmaps_on is None or sp_heatmaps_off is None :
                    raise RuntimeError ("superpoints_K > 0 but DVAWFlowPredictor did not return sp_mask/sp_heatmaps")
                aux_extra ["sp_mask_on"]=sp_mask_on 
                aux_extra ["sp_mask_off"]=sp_mask_off 
                aux_extra ["sp_heatmaps_on"]=sp_heatmaps_on 
                aux_extra ["sp_heatmaps_off"]=sp_heatmaps_off 

            if self .predict_backward or self .round_trip_composition_enabled :
                flow_on_to_ap =self .pair_flow (apex ,src_on_for_flow )
                aux_extra ["flow_on_to_ap"]=flow_on_to_ap 
            if self .predict_backward :
                flow_off_to_ap =self .pair_flow (apex ,src_off_for_flow )
                aux_extra ["flow_off_to_ap"]=flow_off_to_ap 
            if self .round_trip_composition_enabled :
                flow_on_to_off =self .pair_flow (src_off_for_flow ,src_on_for_flow )
                aux_extra ["flow_on_to_off"]=flow_on_to_off 

            if self .basis_displacement_enabled :
                b =getattr (self .pair_flow ,"basis",None )
                if b is not None and torch .is_tensor (b )and b .ndim ==4 and int (b .shape [0 ])>=2 :
                    k =int (b .shape [0 ])
                    flat =b .view (k ,-1 )
                    n =torch .linalg .norm (flat ,dim =1 ,keepdim =True ).clamp_min (1e-12 )
                    u =flat /n 
                    sim =u @u .T 
                    off =sim -torch .eye (k ,device =sim .device ,dtype =sim .dtype )
                    aux_extra ["basis_div_offdiag"]=(off *off ).sum ()/float (k *(k -1 ))

            if self .motion_mode =="error_feedback_refine":
                if self .refine_head is None :
                    raise RuntimeError ("refine_head is not initialized for motion_mode='error_feedback_refine'")
                a_hat_on0 =warp (src_on_for_warp ,flow_ap_to_on0 )
                a_hat_off0 =warp (src_off_for_warp ,flow_ap_to_off0 )
                err_on0 =self ._err_map (a_hat_on0 ,apex )
                err_off0 =self ._err_map (a_hat_off0 ,apex )
                d_on =self .refine_head (src_on_for_warp ,apex ,a_hat_on0 ,err_on0 )
                d_off =self .refine_head (src_off_for_warp ,apex ,a_hat_off0 ,err_off0 )
                flow_ap_to_on =flow_ap_to_on0 +d_on 
                flow_ap_to_off =flow_ap_to_off0 +d_off 
                aux_refine ={
                "flow_on_init":flow_ap_to_on0 ,
                "flow_off_init":flow_ap_to_off0 ,
                "a_hat_on_init":a_hat_on0 ,
                "a_hat_off_init":a_hat_off0 ,
                "err_on_init":err_on0 ,
                "err_off_init":err_off0 ,
                "delta_flow_on":d_on ,
                "delta_flow_off":d_off ,
                }
            else :
                flow_ap_to_on =flow_ap_to_on0 
                flow_ap_to_off =flow_ap_to_off0 
                aux_refine ={}
                if self .refine_enabled :
                    if self .refine_head is None :
                        raise RuntimeError ("refine_head is not initialized for refine_enabled")
                    if not self .training :
                        aux_refine .update ({"flow_on_init":flow_ap_to_on0 .detach (),"flow_off_init":flow_ap_to_off0 .detach ()})
                    delta_clip =float (self .refine_delta_scale )*float (self .max_disp )*0.5 
                    for step in range (1 ,int (self .refine_steps )+1 ):
                        a_hat_on_s =warp (src_on_for_warp ,flow_ap_to_on )
                        a_hat_off_s =warp (src_off_for_warp ,flow_ap_to_off )
                        e_on_s =self ._err_map (a_hat_on_s ,apex )
                        e_off_s =self ._err_map (a_hat_off_s ,apex )
                        d_on_s =self .refine_head (src_on_for_warp ,apex ,a_hat_on_s ,e_on_s )
                        d_off_s =self .refine_head (src_off_for_warp ,apex ,a_hat_off_s ,e_off_s )
                        if delta_clip >0.0 :
                            d_on_s =torch .tanh (d_on_s )*float (delta_clip )
                            d_off_s =torch .tanh (d_off_s )*float (delta_clip )
                        flow_ap_to_on =flow_ap_to_on +d_on_s 
                        flow_ap_to_off =flow_ap_to_off +d_off_s 
                        if not self .training :
                            aux_refine [f"flow_on_step{step}"]=flow_ap_to_on .detach ()
                            aux_refine [f"flow_off_step{step}"]=flow_ap_to_off .detach ()

            T_lin =0.5 *(flow_ap_to_on +flow_ap_to_off )
            D_lin =0.5 *(flow_ap_to_on -flow_ap_to_off )
            T_flow =T_lin 
            D_flow =D_lin 

            if self .motion_mode =="apex_pair_flow":
                x_cls_base =torch .cat ([flow_ap_to_on ,flow_ap_to_off ],dim =1 )
            elif self .motion_mode =="apex_pair_flow_dt":
                x_cls_base =torch .cat ([T_lin ,D_lin ],dim =1 )
            elif self .motion_mode =="apex_neighborhood_flow":
                if apex_m1 is None or apex_p1 is None :
                    raise ValueError ("motion_mode='apex_neighborhood_flow' requires apex_m1 and apex_p1 tensors.")
                flow_m1 =self .pair_flow (apex_m1 ,apex )
                flow_p1 =self .pair_flow (apex_p1 ,apex )
                x_cls_base =torch .cat ([flow_m1 ,flow_p1 ],dim =1 )
            else :
                x_cls_base =torch .cat ([flow_ap_to_on ,flow_ap_to_off ],dim =1 )

        elif self .motion_mode =="triplet_apex_cond_dt":
            if self .triplet_flow is None :
                raise RuntimeError ("triplet_flow is not initialized for motion_mode='triplet_apex_cond_dt'")
            flow_cat =self .triplet_flow (onset ,apex ,offset ,alpha_map )
            flow_ap_to_on =flow_cat [:,0 :2 ]
            flow_ap_to_off =flow_cat [:,2 :4 ]
            T_lin =0.5 *(flow_ap_to_on +flow_ap_to_off )
            D_lin =0.5 *(flow_ap_to_on -flow_ap_to_off )
            T_flow =T_lin 
            D_flow =D_lin 
            x_cls_base =torch .cat ([T_lin ,D_lin ],dim =1 )
            aux_refine ={}
        else :
            if self .cost_volume_pair_flow is None :
                raise RuntimeError ("cost_volume_pair_flow is not initialized for motion_mode='cost_volume_apex_pair_flow'")
            flow_ap_to_on =self .cost_volume_pair_flow (onset ,apex )
            flow_ap_to_off =self .cost_volume_pair_flow (offset ,apex )
            T_lin =0.5 *(flow_ap_to_on +flow_ap_to_off )
            D_lin =0.5 *(flow_ap_to_on -flow_ap_to_off )
            T_flow =T_lin 
            D_flow =D_lin 
            x_cls_base =torch .cat ([flow_ap_to_on ,flow_ap_to_off ],dim =1 )
            aux_refine ={}

        recon_mask_on =recon_mask_off =None 
        if self .region_reconstruction_enabled :
            recon_mask_on =self ._get_fixed_roi_mask (
            batch_size =batch_size ,
            height =int (onset .shape [-2 ]),
            width =int (onset .shape [-1 ]),
            device =device ,
            dtype =onset .dtype ,
            )
            recon_mask_off =recon_mask_on 
            aux_extra ["roi_fixed_mask"]=recon_mask_on 
        elif self .texture_saliency_mask_enabled :
            recon_mask_on =self ._texture_saliency_mask_from_img (src_on_for_warp )
            recon_mask_off =self ._texture_saliency_mask_from_img (src_off_for_warp )
            aux_extra ["texture_saliency_mask_on"]=recon_mask_on 
            aux_extra ["texture_saliency_mask_off"]=recon_mask_off 
        elif self .dynamic_support_mask_enabled :
            if mask_dyn_on is None or mask_dyn_off is None :
                raise RuntimeError ("dynamic_support_mask_enabled but dyn masks are missing")
            recon_mask_on =mask_dyn_on 
            recon_mask_off =mask_dyn_off 
        elif self .superpoints_K >0 :
            if sp_mask_on is None or sp_mask_off is None :
                raise RuntimeError ("superpoints_K > 0 but sp masks are missing")
            recon_mask_on =sp_mask_on 
            recon_mask_off =sp_mask_off 

        apply_mask_to_cls =False 
        if recon_mask_on is not None and recon_mask_off is not None :
            aux_extra ["recon_mask_on"]=recon_mask_on 
            aux_extra ["recon_mask_off"]=recon_mask_off 
            if self .region_reconstruction_enabled and self .region_reconstruction_apply_to_cls :
                apply_mask_to_cls =True 
            if self .texture_saliency_mask_enabled and self .texture_saliency_mask_apply_to_cls :
                apply_mask_to_cls =True 
            if self .dynamic_support_mask_enabled and self .dynamic_support_mask_apply_to_cls :
                apply_mask_to_cls =True 
            if (self .superpoints_K >0 )and self .superpoints_apply_to_cls :
                apply_mask_to_cls =True 

        if apply_mask_to_cls and self .motion_mode =="apex_pair_flow":
            if recon_mask_on is None or recon_mask_off is None :
                raise RuntimeError ("apply_mask_to_cls=true but recon masks are missing")
            x_cls_base =torch .cat ([flow_ap_to_on *recon_mask_on ,flow_ap_to_off *recon_mask_off ],dim =1 )

        if self .dynamic_support_mask_enabled :
            if recon_mask_on is None or recon_mask_off is None :
                raise RuntimeError ("dynamic_support_mask_enabled but recon masks are missing")
            warp_on =warp (src_on_for_warp ,flow_ap_to_on )
            warp_off =warp (src_off_for_warp ,flow_ap_to_off )
            a_hat_on =recon_mask_on *warp_on +(1.0 -recon_mask_on )*src_on_for_warp 
            a_hat_off =recon_mask_off *warp_off +(1.0 -recon_mask_off )*src_off_for_warp 
            d_on =(apex -src_on_for_warp ).abs ().mean (dim =1 ,keepdim =True )
            d_off =(apex -src_off_for_warp ).abs ().mean (dim =1 ,keepdim =True )
            d_on =(d_on /(d_on .mean (dim =(1 ,2 ,3 ),keepdim =True )+float (self .eps ))).clamp (0.0 ,1.0 ).detach ()
            d_off =(d_off /(d_off .mean (dim =(1 ,2 ,3 ),keepdim =True )+float (self .eps ))).clamp (0.0 ,1.0 ).detach ()
            aux_extra ["mask_align_target_on"]=d_on 
            aux_extra ["mask_align_target_off"]=d_off 
            if not self .training :
                aux_extra ["warp_on_raw"]=warp_on 
                aux_extra ["warp_off_raw"]=warp_off 
        else :
            a_hat_on =warp (src_on_for_warp ,flow_ap_to_on )
            a_hat_off =warp (src_off_for_warp ,flow_ap_to_off )


        if flow_on_to_ap is not None and flow_off_to_ap is not None :
            ce_on =self ._fb_cycle_residual_map (flow_on_to_ap ,flow_ap_to_on )
            ce_off =self ._fb_cycle_residual_map (flow_off_to_ap ,flow_ap_to_off )
            r_cyc_map =0.5 *(ce_on +ce_off )

            w_valid_cyc_on =flow_valid_mask (flow_ap_to_on ,align_corners =True ,eps =0.0 ).detach ()
            w_valid_cyc_off =flow_valid_mask (flow_ap_to_off ,align_corners =True ,eps =0.0 ).detach ()

            if self .fb_conf_enabled :
                tau_fb =float (self .fb_conf_tau )
                w_fb_on_raw =1.0 /(1.0 +(ce_on /tau_fb ))
                w_fb_off_raw =1.0 /(1.0 +(ce_off /tau_fb ))
                w_fb_on =w_fb_on_raw .clamp (min =float (self .fb_conf_min ),max =1.0 )*w_valid_cyc_on 
                w_fb_off =w_fb_off_raw .clamp (min =float (self .fb_conf_min ),max =1.0 )*w_valid_cyc_off 
            else :
                w_fb_on =w_valid_cyc_on 
                w_fb_off =w_valid_cyc_off 

            w_fb_on =w_fb_on .detach ()
            w_fb_off =w_fb_off .detach ()


        if self .conf_dyn_enabled :
            w_dyn_on =self ._dyn_saliency_mask (
            apex ,
            src_on_for_warp ,
            topk_ratio =float (self .conf_dyn_topk_ratio ),
            dilate_kernel =int (self .conf_dyn_dilate_kernel ),
            ).detach ()
            w_dyn_off =self ._dyn_saliency_mask (
            apex ,
            src_off_for_warp ,
            topk_ratio =float (self .conf_dyn_topk_ratio ),
            dilate_kernel =int (self .conf_dyn_dilate_kernel ),
            ).detach ()
        else :
            w_dyn_on =torch .ones ((batch_size ,1 ,int (apex .shape [-2 ]),int (apex .shape [-1 ])),device =device ,dtype =apex .dtype )
            w_dyn_off =torch .ones ((batch_size ,1 ,int (apex .shape [-2 ]),int (apex .shape [-1 ])),device =device ,dtype =apex .dtype )

        w_valid_on =flow_valid_mask (flow_ap_to_on ,align_corners =True ,eps =0.0 ).detach ()
        w_valid_off =flow_valid_mask (flow_ap_to_off ,align_corners =True ,eps =0.0 ).detach ()

        if w_fb_on is None or w_fb_off is None :
            w_fb_on_eff =torch .ones_like (w_valid_on )
            w_fb_off_eff =torch .ones_like (w_valid_off )
        else :
            w_fb_on_eff =w_fb_on 
            w_fb_off_eff =w_fb_off 




        c_on =(w_dyn_on *w_valid_on ).detach ()
        c_off =(w_dyn_off *w_valid_off ).detach ()
        c_map =(0.5 *(c_on +c_off )).detach ()
        c_reliability_on =(c_on *w_fb_on_eff ).detach ()
        c_reliability_off =(c_off *w_fb_off_eff ).detach ()
        c_reliability_map =(0.5 *(c_reliability_on +c_reliability_off )).detach ()

        route_c_on =c_on 
        route_c_off =c_off 
        route_trust_on =w_fb_on_eff .clamp (0.0 ,1.0 ).detach ()
        route_trust_off =w_fb_off_eff .clamp (0.0 ,1.0 ).detach ()
        if self .routing_mode =="uniform_valid":
            route_c_on =w_valid_on 
            route_c_off =w_valid_off 
            route_trust_on =w_valid_on 
            route_trust_off =w_valid_off 
        elif self .routing_mode =="support_only":
            route_trust_on =w_valid_on 
            route_trust_off =w_valid_off 
        elif self .routing_mode =="trust_only":
            route_c_on =w_valid_on 
            route_c_off =w_valid_off 

        route_c_map =(0.5 *(route_c_on +route_c_off )).detach ()
        route_c_reliability_on =(route_c_on *route_trust_on ).detach ()
        route_c_reliability_off =(route_c_off *route_trust_off ).detach ()
        route_c_reliability_map =(0.5 *(route_c_reliability_on +route_c_reliability_off )).detach ()

        c_tok =None 
        c_tok_photo =None 
        c_tok_reliability =None 
        trust_tok =None 
        uncertainty_tok =None 
        weight_tok =None 
        u_tok =None 
        gate_mask_tok =None 
        valid_tok =None 
        active_tok_cnt =None 
        outside_gate_weight_rate =torch .tensor (0.0 ,device =device ,dtype =apex .dtype )
        token_hw =tuple (self .routing_token_hw )
        if token_hw is not None :
            c_tok_photo =F .adaptive_avg_pool2d (route_c_map ,output_size =token_hw ).detach ()
            c_tok_reliability =F .adaptive_avg_pool2d (route_c_reliability_map ,output_size =token_hw ).detach ()
            valid_map =(0.5 *(w_valid_on +w_valid_off )).detach ()
            valid_tok =(F .adaptive_avg_pool2d (valid_map ,output_size =token_hw )>0.99 ).detach ()
            conf_tok_photo =c_tok_photo 
            gate_mask_tok =((conf_tok_photo >float (self .routing_gate_thr ))&valid_tok ).detach ()
            active_tok_cnt =gate_mask_tok .view (int (batch_size ),-1 ).sum (dim =1 ).to (dtype =apex .dtype )

            trust_map =(0.5 *(route_trust_on +route_trust_off )).clamp (0.0 ,1.0 ).detach ()
            trust_tok =F .adaptive_avg_pool2d (trust_map ,output_size =token_hw ).detach ().clamp (0.0 ,1.0 )
            uncertainty_tok =(1.0 -trust_tok ).detach ()
            eps_tok =1e-3 
            weight_tok_raw =(conf_tok_photo *trust_tok ).detach ()
            u_tok_raw =(conf_tok_photo *uncertainty_tok ).detach ()
            z =torch .zeros_like (weight_tok_raw )
            weight_tok =torch .where (gate_mask_tok ,weight_tok_raw .clamp (min =eps_tok ,max =1.0 -eps_tok ),z )
            u_tok =torch .where (gate_mask_tok ,u_tok_raw .clamp (min =eps_tok ,max =1.0 -eps_tok ),z )
            outside =(~gate_mask_tok )
            if bool (outside .any ()):
                outside_sum =(weight_tok [outside ].abs ().sum ()+u_tok [outside ].abs ().sum ()).detach ()
                total_sum =(weight_tok .abs ().sum ()+u_tok .abs ().sum ()).detach ().clamp_min (1e-12 )
                outside_gate_weight_rate =(outside_sum /total_sum ).detach ()
                if self .training and float (outside_sum .item ())>0.0 :
                    raise AssertionError ("weight_tok/U_tok has nonzero values outside gate_mask_tok")

        if str (getattr (self ,"trust_token_source","photo"))=="photo":
            c_tok =c_tok_photo 
            transport_conf_map_on =route_c_on 
            transport_conf_map_off =route_c_off 
        else :
            c_tok =c_tok_reliability 
            transport_conf_map_on =route_c_reliability_on 
            transport_conf_map_off =route_c_reliability_off 

        u_on =(route_c_on *(1.0 -route_trust_on .clamp (0.0 ,1.0 ))).detach ()
        u_off =(route_c_off *(1.0 -route_trust_off .clamp (0.0 ,1.0 ))).detach ()
        u_map =torch .maximum (u_on ,u_off ).detach ()
        if u_tok is not None :
            u_map =F .interpolate (u_tok ,size =(int (apex .shape [-2 ]),int (apex .shape [-1 ])),mode ="nearest")

        if self .midframe_enabled :
            if self .pair_flow is None :
                raise RuntimeError ("midframe_enabled requires pair_flow")
            if mid_pre is None or mid_post is None :
                raise ValueError ("midframe_enabled requires mid_pre and mid_post tensors")
            flow_ap_to_mid_pre =self .pair_flow (mid_pre ,apex )
            flow_ap_to_mid_post =self .pair_flow (mid_post ,apex )
            a_hat_mid_pre =warp (mid_pre ,flow_ap_to_mid_pre )
            a_hat_mid_post =warp (mid_post ,flow_ap_to_mid_post )
            aux_extra .update (
            {
            "flow_mid_pre":flow_ap_to_mid_pre ,
            "flow_mid_post":flow_ap_to_mid_post ,
            "a_hat_mid_pre":a_hat_mid_pre ,
            "a_hat_mid_post":a_hat_mid_post ,
            }
            )

        if self .multiframe_enabled :
            if self .pair_flow is None :
                raise RuntimeError ("multiframe_enabled requires pair_flow")
            if extra_pre is None or extra_post is None :
                raise ValueError ("multiframe_enabled requires extra_pre and extra_post tensors")
            if not (torch .is_tensor (extra_pre )and torch .is_tensor (extra_post )):
                raise ValueError ("extra_pre/extra_post must be torch tensors")
            if extra_pre .ndim !=5 or extra_post .ndim !=5 :
                raise ValueError (
                f"extra_pre/extra_post must be [B,K,C,H,W]. Got extra_pre={tuple(extra_pre.shape)} extra_post={tuple(extra_post.shape)}"
                )
            if int (extra_pre .shape [0 ])!=batch_size or int (extra_post .shape [0 ])!=batch_size :
                raise ValueError ("extra_pre/extra_post batch size mismatch")
            if tuple (extra_pre .shape [-2 :])!=tuple (apex .shape [-2 :])or tuple (extra_post .shape [-2 :])!=tuple (apex .shape [-2 :]):
                raise ValueError ("extra_pre/extra_post spatial must match apex")

            mf_extra_rec =torch .tensor (0.0 ,device =device )
            mf_round_trip_composition =torch .tensor (0.0 ,device =device )
            mf_hat_pre_list :list [torch .Tensor ]=[]
            mf_hat_post_list :list [torch .Tensor ]=[]

            vset ={t .strip ().lower ().replace ("-","_")for t in re .split (r"[+|,]",str (self .multiframe_variant ))if t .strip ()}
            if not vset :
                vset ={"extra_rec"}
            if "extra"in vset :
                vset .discard ("extra")
                vset .add ("extra_rec")
            if "flow"in vset :
                vset .discard ("flow")
                vset .add ("round_trip_composition")
            if "rank"in vset :
                vset .discard ("rank")
                vset .add ("rank_mag")

            do_extra_rec ="extra_rec"in vset 
            do_round_trip_composition ="round_trip_composition"in vset 
            do_rank_mag ="rank_mag"in vset 

            f_pre_list :list [torch .Tensor ]=[]
            f_post_list :list [torch .Tensor ]=[]

            if do_extra_rec or do_rank_mag :

                if int (extra_pre .shape [1 ])<1 or int (extra_post .shape [1 ])<1 :
                    raise ValueError ("multiframe requires extra_pre/post with at least 1 frame each")

                errs :list [torch .Tensor ]=[]
                for k in range (int (extra_pre .shape [1 ])):
                    src =extra_pre [:,k ].to (device =device )
                    f =self .pair_flow (src ,apex )
                    f_pre_list .append (f )
                    if do_extra_rec and (not self .multiframe_extra_rec_in_loss ):
                        hat =warp (src ,f )
                        e_map =self ._err_map (hat ,apex )
                        if c_on is not None :
                            w =c_on .to (device =device ,dtype =e_map .dtype )
                            errs .append ((e_map *w ).sum (dim =(1 ,2 ,3 ))/(w .sum (dim =(1 ,2 ,3 )).clamp_min (float (self .eps ))))
                        else :
                            errs .append (e_map .mean (dim =(1 ,2 ,3 )))
                        if not self .training :
                            mf_hat_pre_list .append (hat .detach ())
                for k in range (int (extra_post .shape [1 ])):
                    src =extra_post [:,k ].to (device =device )
                    f =self .pair_flow (src ,apex )
                    f_post_list .append (f )
                    if do_extra_rec and (not self .multiframe_extra_rec_in_loss ):
                        hat =warp (src ,f )
                        e_map =self ._err_map (hat ,apex )
                        if c_off is not None :
                            w =c_off .to (device =device ,dtype =e_map .dtype )
                            errs .append ((e_map *w ).sum (dim =(1 ,2 ,3 ))/(w .sum (dim =(1 ,2 ,3 )).clamp_min (float (self .eps ))))
                        else :
                            errs .append (e_map .mean (dim =(1 ,2 ,3 )))
                        if not self .training :
                            mf_hat_post_list .append (hat .detach ())

                if do_extra_rec and errs :
                    mf_extra_rec =torch .stack (errs ,dim =0 ).mean ()
                if do_extra_rec and self .multiframe_extra_rec_in_loss :

                    if f_pre_list :
                        aux_extra ["mf_flows_pre"]=torch .stack (f_pre_list ,dim =1 )
                    if f_post_list :
                        aux_extra ["mf_flows_post"]=torch .stack (f_post_list ,dim =1 )

            if do_rank_mag :

                if int (extra_pre .shape [1 ])<2 or int (extra_post .shape [1 ])<2 :
                    raise ValueError ("multiframe rank_mag requires extra_pre/post with at least 2 frames each (far/near)")
                if len (f_pre_list )<2 :
                    f_pre_list =[
                    self .pair_flow (extra_pre [:,0 ].to (device =device ),apex ),
                    self .pair_flow (extra_pre [:,1 ].to (device =device ),apex ),
                    ]
                if len (f_post_list )<2 :
                    f_post_list =[
                    self .pair_flow (extra_post [:,0 ].to (device =device ),apex ),
                    self .pair_flow (extra_post [:,1 ].to (device =device ),apex ),
                    ]

                f_pre_far =f_pre_list [0 ]
                f_pre_near =f_pre_list [1 ]
                f_pre_far_n ,m_pre_far =self ._normalize_flow_topk_mean (
                f_pre_far ,
                topk_ratio =float (self .descriptor_norm_topk_ratio ),
                weights =c_on ,
                detach_scale =False ,
                weighted_reduce =True ,
                )
                f_pre_near_n ,m_pre_near =self ._normalize_flow_topk_mean (
                f_pre_near ,
                topk_ratio =float (self .descriptor_norm_topk_ratio ),
                weights =c_on ,
                detach_scale =False ,
                weighted_reduce =True ,
                )

                f_post_far =f_post_list [0 ]
                f_post_near =f_post_list [1 ]
                f_post_far_n ,m_post_far =self ._normalize_flow_topk_mean (
                f_post_far ,
                topk_ratio =float (self .descriptor_norm_topk_ratio ),
                weights =c_off ,
                detach_scale =False ,
                weighted_reduce =True ,
                )
                f_post_near_n ,m_post_near =self ._normalize_flow_topk_mean (
                f_post_near ,
                topk_ratio =float (self .descriptor_norm_topk_ratio ),
                weights =c_off ,
                detach_scale =False ,
                weighted_reduce =True ,
                )

                _ ,m_pre_far_norm =self ._normalize_flow_topk_mean (
                f_pre_far_n .detach (),
                topk_ratio =float (self .descriptor_norm_topk_ratio ),
                weights =c_on ,
                detach_scale =False ,
                weighted_reduce =True ,
                )
                _ ,m_pre_near_norm =self ._normalize_flow_topk_mean (
                f_pre_near_n .detach (),
                topk_ratio =float (self .descriptor_norm_topk_ratio ),
                weights =c_on ,
                detach_scale =False ,
                weighted_reduce =True ,
                )
                _ ,m_post_far_norm =self ._normalize_flow_topk_mean (
                f_post_far_n .detach (),
                topk_ratio =float (self .descriptor_norm_topk_ratio ),
                weights =c_off ,
                detach_scale =False ,
                weighted_reduce =True ,
                )
                _ ,m_post_near_norm =self ._normalize_flow_topk_mean (
                f_post_near_n .detach (),
                topk_ratio =float (self .descriptor_norm_topk_ratio ),
                weights =c_off ,
                detach_scale =False ,
                weighted_reduce =True ,
                )


                aux_extra ["mf_mag_pre_far"]=m_pre_far 
                aux_extra ["mf_mag_pre_near"]=m_pre_near 
                aux_extra ["mf_mag_post_far"]=m_post_far 
                aux_extra ["mf_mag_post_near"]=m_post_near 
                aux_extra ["mf_mag_pre_far_raw"]=m_pre_far .detach ()
                aux_extra ["mf_mag_pre_near_raw"]=m_pre_near .detach ()
                aux_extra ["mf_mag_post_far_raw"]=m_post_far .detach ()
                aux_extra ["mf_mag_post_near_raw"]=m_post_near .detach ()
                aux_extra ["mf_mag_pre_far_norm"]=m_pre_far_norm .detach ()
                aux_extra ["mf_mag_pre_near_norm"]=m_pre_near_norm .detach ()
                aux_extra ["mf_mag_post_far_norm"]=m_post_far_norm .detach ()
                aux_extra ["mf_mag_post_near_norm"]=m_post_near_norm .detach ()
                if torch .is_tensor (dt_pre )and dt_pre .ndim ==2 and int (dt_pre .shape [1 ])>=2 :
                    aux_extra ["mf_dt_pre_far"]=dt_pre [:,0 ].to (device =device ,dtype =torch .float32 ).detach ()
                    aux_extra ["mf_dt_pre_near"]=dt_pre [:,1 ].to (device =device ,dtype =torch .float32 ).detach ()
                if torch .is_tensor (dt_post )and dt_post .ndim ==2 and int (dt_post .shape [1 ])>=2 :
                    aux_extra ["mf_dt_post_far"]=dt_post [:,0 ].to (device =device ,dtype =torch .float32 ).detach ()
                    aux_extra ["mf_dt_post_near"]=dt_post [:,1 ].to (device =device ,dtype =torch .float32 ).detach ()

            if do_round_trip_composition :
                if int (extra_pre .shape [1 ])<1 or int (extra_post .shape [1 ])<1 :
                    raise ValueError ("multiframe round_trip_composition requires extra_pre/post with at least 1 frame each")
                midp =extra_pre [:,0 ].to (device =device )
                midq =extra_post [:,0 ].to (device =device )

                f_on_mid =self .pair_flow (midp ,onset )
                f_mid_ap =self .pair_flow (apex ,midp )
                f_on_ap =self .pair_flow (apex ,onset )
                comp_on =f_on_mid +warp (f_mid_ap ,f_on_mid )
                l_on =(comp_on -f_on_ap ).abs ().mean ()

                f_off_mid =self .pair_flow (midq ,offset )
                f_mid_ap2 =self .pair_flow (apex ,midq )
                f_off_ap =self .pair_flow (apex ,offset )
                comp_off =f_off_mid +warp (f_mid_ap2 ,f_off_mid )
                l_off =(comp_off -f_off_ap ).abs ().mean ()
                mf_round_trip_composition =0.5 *(l_on +l_off )

            aux_extra ["mf_extra_rec"]=mf_extra_rec 
            aux_extra ["mf_round_trip_composition"]=mf_round_trip_composition 
            if not self .training :
                if mf_hat_pre_list :
                    aux_extra ["a_hat_extra_pre"]=torch .stack (mf_hat_pre_list ,dim =1 )
                if mf_hat_post_list :
                    aux_extra ["a_hat_extra_post"]=torch .stack (mf_hat_post_list ,dim =1 )

        if neg_targets is not None :
            if self .pair_flow is None :
                raise RuntimeError ("neg_targets is only supported for pair-flow modes")
            if not (torch .is_tensor (neg_targets )and neg_targets .ndim ==5 ):
                raise ValueError (f"neg_targets must be [B,K,C,H,W], got {type(neg_targets).__name__} {getattr(neg_targets, 'shape', None)}")
            if int (neg_targets .shape [0 ])!=batch_size :
                raise ValueError (f"neg_targets batch mismatch: neg_targets.shape[0]={int(neg_targets.shape[0])} batch_size={batch_size}")
            k_negs =int (neg_targets .shape [1 ])
            if k_negs <=0 :
                raise ValueError ("neg_targets must have K>=1")
            e_pos_on =self ._err_map (a_hat_on ,apex ).mean (dim =(1 ,2 ,3 ))
            e_pos_off =self ._err_map (a_hat_off ,apex ).mean (dim =(1 ,2 ,3 ))
            e_negs_on =[]
            e_negs_off =[]
            for kk in range (k_negs ):
                tgt =neg_targets [:,kk ].to (device =device )
                f_tgt_on =self .pair_flow (src_on_for_flow ,tgt )
                f_tgt_off =self .pair_flow (src_off_for_flow ,tgt )
                hat_tgt_on =warp (src_on_for_warp ,f_tgt_on )
                hat_tgt_off =warp (src_off_for_warp ,f_tgt_off )
                e_negs_on .append (self ._err_map (hat_tgt_on ,tgt ).mean (dim =(1 ,2 ,3 )))
                e_negs_off .append (self ._err_map (hat_tgt_off ,tgt ).mean (dim =(1 ,2 ,3 )))
            E_neg_on =torch .stack (e_negs_on ,dim =1 )
            E_neg_off =torch .stack (e_negs_off ,dim =1 )
            aux_extra ["tc_E_pos_on"]=e_pos_on 
            aux_extra ["tc_E_pos_off"]=e_pos_off 
            aux_extra ["tc_E_neg_on"]=E_neg_on 
            aux_extra ["tc_E_neg_off"]=E_neg_off 
            aux_extra ["tc_rank_on"]=(e_pos_on <E_neg_on .min (dim =1 ).values ).to (dtype =torch .float32 )
            aux_extra ["tc_rank_off"]=(e_pos_off <E_neg_off .min (dim =1 ).values ).to (dtype =torch .float32 )
            aux_extra ["tc_gap_on"]=(E_neg_on .min (dim =1 ).values -e_pos_on )
            aux_extra ["tc_gap_off"]=(E_neg_off .min (dim =1 ).values -e_pos_off )

        flow_ap_to_on_cls =flow_ap_to_on .detach ()if self .stopgrad_cls_to_motion else flow_ap_to_on 
        flow_ap_to_off_cls =flow_ap_to_off .detach ()if self .stopgrad_cls_to_motion else flow_ap_to_off 
        x_cls_base_cls =x_cls_base .detach ()if self .stopgrad_cls_to_motion else x_cls_base 
        D_flow_cls =D_flow .detach ()if self .stopgrad_cls_to_motion else D_flow 

        x_flow_raw =x_cls_base_cls 
        x_flow_resid =x_flow_raw 
        aux_global :Dict [str ,torch .Tensor ]={}
        aux_rep :Dict [str ,torch .Tensor ]={}
        desc_on =desc_off =None 
        if self .cls_rep =="eight_channel_deformation_descriptor":
            if self .motion_mode !="apex_pair_flow":
                raise ValueError ("cls_rep='eight_channel_deformation_descriptor' requires motion_mode='apex_pair_flow'")
            if self .cls_global_removal !="none":
                raise ValueError ("cls_rep='eight_channel_deformation_descriptor' does not support cls_global_removal (keep optional objectives isolated).")
            if self .cls_dual_stream_enabled or self .cls_roi_enabled or self .cls_region_masks_enabled or self .cls_use_err or self .fusion_enabled :
                raise ValueError ("cls_rep='eight_channel_deformation_descriptor' does not support dual-stream/roi/region-masks/err/fusion (keep optional objectives isolated).")
            desc_on =self ._flow_to_deformation_descriptor (flow_ap_to_on_cls )
            desc_off =self ._flow_to_deformation_descriptor (flow_ap_to_off_cls )
            desc_on ,desc_off =_apply_view_mode_pair (self .view_mode ,desc_on ,desc_off )
            x_flow_feat =torch .cat ([desc_on ,desc_off ],dim =1 )
            aux_rep .update ({"desc_on":desc_on ,"desc_off":desc_off })
        elif self .cls_rep =="multiscale_deformation_descriptor":
            if self .motion_mode !="apex_pair_flow":
                raise ValueError ("cls_rep='multiscale_deformation_descriptor' requires motion_mode='apex_pair_flow'")
            if self .cls_global_removal !="none":
                raise ValueError ("cls_rep='multiscale_deformation_descriptor' does not support cls_global_removal (keep optional objectives isolated).")
            if self .cls_dual_stream_enabled or self .cls_roi_enabled or self .cls_region_masks_enabled or self .cls_use_err or self .fusion_enabled :
                raise ValueError ("cls_rep='multiscale_deformation_descriptor' does not support dual-stream/roi/region-masks/err/fusion (keep optional objectives isolated).")

            desc_on ,desc_on_by_s =self ._flow_to_multiscale_deformation_descriptor (
            flow_ap_to_on_cls ,
            scales =self .deformation_descriptor_scales ,
            ops =self .deformation_descriptor_ops ,
            )
            desc_off ,desc_off_by_s =self ._flow_to_multiscale_deformation_descriptor (
            flow_ap_to_off_cls ,
            scales =self .deformation_descriptor_scales ,
            ops =self .deformation_descriptor_ops ,
            )
            desc_on ,desc_off =_apply_view_mode_pair (self .view_mode ,desc_on ,desc_off )
            if self .two_stage_cls_enabled :

                x_flow_feat =desc_on 
                aux_rep ["x_cls_pair"]=torch .cat ([desc_on ,desc_off ],dim =1 )
            else :
                x_flow_feat =torch .cat ([desc_on ,desc_off ],dim =1 )

            aux_rep ["desc_on_ms"]=desc_on 
            aux_rep ["desc_off_ms"]=desc_off 
            for s in self .deformation_descriptor_scales :
                ss =int (s )
                if ss in desc_on_by_s and ss in desc_off_by_s :
                    aux_rep [f"desc_ms_s{ss}"]=torch .cat ([desc_on_by_s [ss ],desc_off_by_s [ss ]],dim =1 )
        elif self .cls_rep =="differential_deformation_descriptor":
            if self .motion_mode !="apex_pair_flow":
                raise ValueError ("cls_rep='differential_deformation_descriptor' requires motion_mode='apex_pair_flow'")
            if self .cls_global_removal !="none":
                raise ValueError ("cls_rep='differential_deformation_descriptor' does not support cls_global_removal (keep optional objectives isolated).")
            if self .cls_dual_stream_enabled or self .cls_roi_enabled or self .cls_region_masks_enabled or self .cls_use_err or self .fusion_enabled :
                raise ValueError ("cls_rep='differential_deformation_descriptor' does not support dual-stream/roi/region-masks/err/fusion (keep optional objectives isolated).")

            f_on_n ,scale_on =self ._normalize_flow_topk_mean (
            flow_ap_to_on_cls ,
            topk_ratio =float (self .descriptor_norm_topk_ratio ),
            weights =c_on ,
            detach_scale =bool (self .descriptor_detach_scale ),
            )
            f_off_n ,scale_off =self ._normalize_flow_topk_mean (
            flow_ap_to_off_cls ,
            topk_ratio =float (self .descriptor_norm_topk_ratio ),
            weights =c_off ,
            detach_scale =bool (self .descriptor_detach_scale ),
            )
            desc_on =self ._flow_to_deformation_descriptor_ops (f_on_n ,ops =("div","curl"))
            desc_off =self ._flow_to_deformation_descriptor_ops (f_off_n ,ops =("div","curl"))
            desc_on ,desc_off =_apply_view_mode_pair (self .view_mode ,desc_on ,desc_off )
            x_flow_feat =torch .cat ([desc_on ,desc_off ],dim =1 )
            aux_rep .update (
            {
            "desc_on":desc_on ,
            "desc_off":desc_off ,
            "scale_topk_mean_on":scale_on ,
            "scale_topk_mean_off":scale_off ,
            }
            )
        else :
            if self .cls_global_removal !="none":
                x_flow_resid ,aux_global =self ._apply_global_removal_to_xcls (x_flow_raw )

            if self .cls_dual_stream_enabled and self .cls_global_removal =="none":
                raise ValueError ("cls_dual_stream_enabled requires cls_global_removal != 'none' to produce residual flow features.")
            if self .cls_dual_stream_enabled :
                if self .cls_dual_stream_fusion =="concat":
                    x_flow_feat =torch .cat ([x_flow_raw ,x_flow_resid ],dim =1 )
                else :
                    if self .dual_stream_gate is None :
                        raise RuntimeError ("dual_stream_gate is not initialized for fusion='gated'")
                    gate_in =torch .cat ([x_flow_raw ,x_flow_resid ],dim =1 )
                    if int (gate_in .shape [1 ])!=8 :
                        raise ValueError (f"dual_stream gated fusion expects 8ch input, got {tuple(gate_in.shape)}")
                    g =torch .sigmoid (self .dual_stream_gate (gate_in ))
                    x_flow_feat =torch .cat ([x_flow_raw ,g *x_flow_resid ],dim =1 )
            else :
                x_flow_feat =x_flow_resid if self .cls_global_removal !="none"else x_flow_raw 


        err_on =self ._normalize_err (self ._err_map (a_hat_on ,apex ))
        err_off =self ._normalize_err (self ._err_map (a_hat_off ,apex ))




        reco_region_wts =None 
        reco_sym_pairs :list [tuple [int ,int ]]=[]
        if self .reco is not None :
            reco_loss =torch .zeros ([],device =device ,dtype =err_on .dtype )
            reco_diag :Dict [str ,Any ]={}

            dyn_map =c_map if (c_map is not None and torch .is_tensor (c_map ))else (err_on +err_off ).detach ()

            need_motion_maps =bool (need_reco_regions or self .reco .channel_swap_enabled or self .reco .cross_view_mask_enabled or self .reco .mf_enabled )
            M_OA =M_AF =None 
            if need_motion_maps :
                if desc_on is None or desc_off is None or (not torch .is_tensor (desc_on ))or (not torch .is_tensor (desc_off )):
                    raise RuntimeError (
                    "reco requires motion descriptor maps (desc_on/desc_off). "
                    "Use cls_rep='multiscale_deformation_descriptor' or cls_rep='differential_deformation_descriptor'."
                    )
                M_OA =desc_on .abs ()
                M_AF =desc_off .abs ()


            if need_reco_regions :
                if self .reco .region is None :
                    raise RuntimeError ("need_reco_regions=true but reco.region is not initialized (set reco.region.mode != 'quad4').")
                if M_OA is None or M_AF is None :
                    raise RuntimeError ("need_reco_regions=true requires motion maps (M_OA/M_AF).")
                reco_region_wts ,reco_sym_pairs ,L_reg ,d_reg =self .reco .build_regions (
                feat_hw =(int (M_OA .shape [-2 ]),int (M_OA .shape [-1 ])),
                batch_size =int (M_OA .shape [0 ]),
                img_ref =apex ,
                dyn_map =dyn_map ,
                )
                reco_loss =reco_loss +L_reg 
                reco_diag .update (d_reg )


            if self .reco .channel_swap_enabled :
                if M_OA is None or M_AF is None or reco_region_wts is None :
                    raise RuntimeError ("channel_swap_enabled requires region_wts and motion maps (M_OA/M_AF).")
                L_channel_swap ,d_channel_swap =self .reco .channel_swap_losses (M_OA ,M_AF ,reco_region_wts )
                reco_loss =reco_loss +L_channel_swap 
                reco_diag .update (d_channel_swap )


            if self .reco .cross_view_mask_enabled :
                if M_OA is None or M_AF is None :
                    raise RuntimeError ("cross_view_mask_enabled requires motion maps (M_OA/M_AF).")


                if self .pair_flow is None :
                    raise RuntimeError ("cross_view_mask_enabled requires pair_flow (motion_mode='apex_pair_flow').")
                flow_on_to_off_eff =flow_on_to_off 
                if flow_on_to_off_eff is None :
                    flow_on_to_off_eff =self .pair_flow (src_off_for_flow ,src_on_for_flow )
                flow_onoff_in_ap =warp (flow_on_to_off_eff ,flow_ap_to_on )
                if self .cls_rep =="multiscale_deformation_descriptor":
                    M_OF ,_by_s =self ._flow_to_multiscale_deformation_descriptor (
                    flow_onoff_in_ap ,
                    scales =self .deformation_descriptor_scales ,
                    ops =self .deformation_descriptor_ops ,
                    )
                    M_OF =M_OF .abs ()
                elif self .cls_rep =="differential_deformation_descriptor":
                    f_of_n ,_scale_of =self ._normalize_flow_topk_mean (flow_onoff_in_ap ,topk_ratio =float (self .descriptor_norm_topk_ratio ))
                    M_OF =self ._flow_to_deformation_descriptor_ops (f_of_n ,ops =("div","curl")).abs ()
                else :
                    raise ValueError (
                    "reco.cross_view_mask requires cls_rep='multiscale_deformation_descriptor' or cls_rep='differential_deformation_descriptor' "
                    f"(got cls_rep={self.cls_rep!r})"
                    )

                L_cross_view_mask ,d_cross_view_mask =self .reco .cross_view_mask_losses (
                M_OA ,
                M_AF ,
                M_OF ,
                img_ref =apex ,
                dyn_map =dyn_map ,
                region_wts =reco_region_wts ,
                rng =_local_gen ("cross_view_mask"),
                )
                reco_loss =reco_loss +L_cross_view_mask 
                reco_diag .update (d_cross_view_mask )


            if self .reco .mf_enabled :
                flows_dict :Dict [str ,Any ]={}

                if self .reco .mf_variant .startswith ("v_scalars"):
                    flows_dict ["f_onA"]=flow_ap_to_on 

                    pre_list =[]
                    post_list =[]
                    if extra_pre is not None :
                        if dt_pre is None :
                            raise ValueError ("multiframe.v_scalars requires dt_pre (B,n_pre) from the dataloader.")
                        if extra_pre .ndim !=5 or dt_pre .ndim !=2 :
                            raise ValueError (f"expected extra_pre (B,n_pre,3,H,W) and dt_pre (B,n_pre), got {tuple(extra_pre.shape)} {tuple(dt_pre.shape)}")
                        n_pre =int (extra_pre .shape [1 ])
                        for kk in range (n_pre ):
                            f =self .pair_flow (extra_pre [:,kk ].to (device =device ),apex )
                            pre_list .append ((f ,dt_pre [:,kk ].to (device =device ,dtype =torch .float32 )))
                    if extra_post is not None :
                        if dt_post is None :
                            raise ValueError ("multiframe.v_scalars requires dt_post (B,n_post) from the dataloader.")
                        if extra_post .ndim !=5 or dt_post .ndim !=2 :
                            raise ValueError (
                            f"expected extra_post (B,n_post,3,H,W) and dt_post (B,n_post), got {tuple(extra_post.shape)} {tuple(dt_post.shape)}"
                            )
                        n_post =int (extra_post .shape [1 ])
                        for kk in range (n_post ):
                            f =self .pair_flow (extra_post [:,kk ].to (device =device ),apex )
                            post_list .append ((f ,dt_post [:,kk ].to (device =device ,dtype =torch .float32 )))

                    flows_dict ["pre_list"]=pre_list 
                    flows_dict ["post_list"]=post_list 

                elif self .reco .mf_variant =="round_trip_composition":
                    if flow_on_to_ap is None or flow_on_to_off is None :
                        raise RuntimeError ("multiframe.round_trip_composition requires round_trip_composition_enabled (need flow_on_to_ap and flow_on_to_off).")
                    flows_dict ["f_onA"]=flow_on_to_ap 
                    flows_dict ["f_Aoff"]=flow_ap_to_off 
                    flows_dict ["f_onoff"]=flow_on_to_off 


                L_mf ,d_mf =self .reco .multiframe_losses (flows_dict ,tau_on =tau_on ,tau_off =tau_off )
                reco_loss =reco_loss +L_mf 
                reco_diag .update (d_mf )

                if self .reco .mf_variant =="multidelta_supcon":
                    if reco_region_wts is None :
                        raise RuntimeError ("multidelta_supcon requires region_wts (enable reco.region).")
                    if self .pair_flow is None :
                        raise RuntimeError ("multidelta_supcon requires pair_flow.")


                    views :List [torch .Tensor ]=[M_OA ,M_AF ]if (M_OA is not None and M_AF is not None )else []

                    if extra_pre is not None and extra_pre .ndim ==5 :
                        for kk in range (int (extra_pre .shape [1 ])):
                            f =self .pair_flow (extra_pre [:,kk ].to (device =device ),apex )
                            d ,_ =self ._flow_to_multiscale_deformation_descriptor (f ,scales =self .deformation_descriptor_scales ,ops =self .deformation_descriptor_ops )
                            views .append (d .abs ())
                    if extra_post is not None and extra_post .ndim ==5 :
                        for kk in range (int (extra_post .shape [1 ])):
                            f =self .pair_flow (extra_post [:,kk ].to (device =device ),apex )
                            d ,_ =self ._flow_to_multiscale_deformation_descriptor (f ,scales =self .deformation_descriptor_scales ,ops =self .deformation_descriptor_ops )
                            views .append (d .abs ())

                    if len (views )>=2 :
                        z_list =[]
                        sid_list =[]
                        rid_list =[]
                        B0 =int (views [0 ].shape [0 ])
                        R0 =int (reco_region_wts .shape [1 ])
                        sid =torch .arange (B0 ,device =device ,dtype =torch .long ).view (B0 ,1 ).expand (B0 ,R0 ).reshape (-1 )
                        rid =torch .arange (R0 ,device =device ,dtype =torch .long ).view (1 ,R0 ).expand (B0 ,R0 ).reshape (-1 )
                        for vv in views :
                            zr =reco_region_pool (vv ,reco_region_wts ).reshape (B0 *R0 ,-1 )
                            z_list .append (zr )
                            sid_list .append (sid )
                            rid_list .append (rid )
                        embeds =torch .cat (z_list ,dim =0 )
                        sample_ids =torch .cat (sid_list ,dim =0 )
                        region_ids =torch .cat (rid_list ,dim =0 )
                        L_mfs ,d_mfs =self .reco .multidelta_supcon_losses (
                        embeds =embeds ,sample_ids =sample_ids ,region_ids =region_ids 
                        )
                        reco_loss =reco_loss +L_mfs 
                        reco_diag .update (d_mfs )


            if self .reco .transport_enabled :
                transport_weight_map_on =None 
                transport_weight_map_off =None 
                if weight_tok is not None :
                    transport_weight_map_on =F .interpolate (
                    weight_tok ,
                    size =(int (onset .shape [-2 ]),int (onset .shape [-1 ])),
                    mode ="nearest",
                    )
                    transport_weight_map_off =F .interpolate (
                    weight_tok ,
                    size =(int (offset .shape [-2 ]),int (offset .shape [-1 ])),
                    mode ="nearest",
                    )

                def _call_transport_on ():
                    return self .reco .trusted_transport_losses (
                    apex ,
                    onset ,
                    flow_ap_to_on ,
                    full_hw_px =(int (onset .shape [-2 ]),int (onset .shape [-1 ])),
                    conf_map =transport_conf_map_on ,
                    weight_map =transport_weight_map_on ,
                    valid_map =w_valid_on ,
                    gate_thr =float (self .routing_gate_thr ),
                    rng =_local_gen ("transport_on"),
                    )

                def _call_transport_off ():
                    return self .reco .trusted_transport_losses (
                    apex ,
                    offset ,
                    flow_ap_to_off ,
                    full_hw_px =(int (offset .shape [-2 ]),int (offset .shape [-1 ])),
                    conf_map =transport_conf_map_off ,
                    weight_map =transport_weight_map_off ,
                    valid_map =w_valid_off ,
                    gate_thr =float (self .routing_gate_thr ),
                    rng =_local_gen ("transport_off"),
                    )

                transport_mode =str (getattr (self ,"transport_mode","symmetric"))
                d_transport :Dict [str ,Any ]={}
                if transport_mode =="symmetric":
                    L_transport_on ,d_transport_on =_call_transport_on ()
                    L_transport_off ,d_transport_off =_call_transport_off ()
                    L_transport =0.5 *(L_transport_on +L_transport_off )

                    diag_keys =set (d_transport_on .keys ())|set (d_transport_off .keys ())
                    for key in sorted (diag_keys ):
                        v_on =d_transport_on .get (key ,None )
                        v_off =d_transport_off .get (key ,None )
                        if torch .is_tensor (v_on )and torch .is_tensor (v_off ):
                            if v_on .shape ==v_off .shape :
                                d_transport [key ]=0.5 *(v_on +v_off )
                                if key !="transport_T_matrix"and int (v_on .ndim )==0 :
                                    d_transport [f"{key}_on"]=v_on .detach ()
                                    d_transport [f"{key}_off"]=v_off .detach ()
                            else :
                                d_transport [key ]=v_on .detach ()
                        elif torch .is_tensor (v_on ):
                            d_transport [key ]=v_on .detach ()
                        elif torch .is_tensor (v_off ):
                            d_transport [key ]=v_off .detach ()
                elif transport_mode =="onset_only":
                    L_transport_on ,d_transport_on =_call_transport_on ()
                    L_transport =L_transport_on 
                    for key ,value in d_transport_on .items ():
                        if torch .is_tensor (value ):
                            d_transport [key ]=value .detach ()
                            if key !="transport_T_matrix"and int (value .ndim )==0 :
                                d_transport [f"{key}_on"]=value .detach ()
                                d_transport [f"{key}_off"]=torch .zeros_like (value .detach ())
                else :
                    L_transport_off ,d_transport_off =_call_transport_off ()
                    L_transport =L_transport_off 
                    for key ,value in d_transport_off .items ():
                        if torch .is_tensor (value ):
                            d_transport [key ]=value .detach ()
                            if key !="transport_T_matrix"and int (value .ndim )==0 :
                                d_transport [f"{key}_on"]=torch .zeros_like (value .detach ())
                                d_transport [f"{key}_off"]=value .detach ()

                reco_loss =reco_loss +L_transport 
                reco_diag .update (d_transport )

                if "diag_mass_T_mean"in d_transport :
                    aux_extra ["diag_mass_T_mean"]=d_transport ["diag_mass_T_mean"].detach ()
                if "entropy_T_mean"in d_transport :
                    aux_extra ["entropy_T_mean"]=d_transport ["entropy_T_mean"].detach ()
                if "active_tok_cnt_mean"in d_transport :
                    aux_extra ["active_tok_cnt_mean"]=d_transport ["active_tok_cnt_mean"].detach ()

                if (
                bool (trust_propagation_runtime_enabled )
                and self .trust_propagation_enabled 
                and torch .is_tensor (d_transport .get ("transport_T_matrix",None ))
                and (trust_tok is not None )
                and (gate_mask_tok is not None )
                ):
                    active_now =gate_mask_tok .view (int (batch_size ),-1 ).sum (dim =1 ).float ().mean ()
                    if float (active_now .detach ().cpu ().item ())>=float (max (int (trust_propagation_active_min ),1 )):
                        trust_tok =propagate_trust_tokens (
                        trust_tok .detach (),
                        d_transport ["transport_T_matrix"].detach (),
                        gate_mask_tok ,
                        alpha =float (self .trust_propagation_alpha ),
                        K =int (self .trust_propagation_steps ),
                        ).detach ().clamp (0.0 ,1.0 )
                        uncertainty_tok =(1.0 -trust_tok ).detach ()
                        if c_tok_photo is not None and weight_tok is not None and u_tok is not None :
                            eps_tok =1e-3 
                            weight_tok_raw =(c_tok_photo *trust_tok ).detach ()
                            u_tok_raw =(c_tok_photo *uncertainty_tok ).detach ()
                            z_tok =torch .zeros_like (weight_tok_raw )
                            weight_tok =torch .where (gate_mask_tok ,weight_tok_raw .clamp (min =eps_tok ,max =1.0 -eps_tok ),z_tok )
                            u_tok =torch .where (gate_mask_tok ,u_tok_raw .clamp (min =eps_tok ,max =1.0 -eps_tok ),z_tok )
                            u_map =F .interpolate (u_tok ,size =(int (apex .shape [-2 ]),int (apex .shape [-1 ])),mode ="nearest")
                        aux_extra ["trust_propagation_applied"]=torch .tensor (1 ,device =device ,dtype =torch .int64 )

            aux_extra ["reco_loss"]=reco_loss 
            aux_extra ["reco_diag"]=reco_diag 

        aux_extra ["reco_regions_built"]=torch .tensor (int (reco_region_wts is not None ),device =device ,dtype =torch .int64 )
        if reco_region_wts is not None :
            aux_extra ["reco_num_regions"]=torch .tensor (int (reco_region_wts .shape [1 ]),device =device ,dtype =torch .int64 )
            aux_extra ["reco_sym_pairs_cnt"]=torch .tensor (int (len (reco_sym_pairs )),device =device ,dtype =torch .int64 )
            if self .reco is not None and isinstance (aux_extra .get ("reco_diag",None ),dict ):
                rm =aux_extra ["reco_diag"].get ("region_mode",None )
                if torch .is_tensor (rm )and int (rm .numel ())==1 :
                    aux_extra ["reco_region_mode"]=rm .to (device =device )

        roi_mask =None 
        x_roi_local =None 
        if self .cls_roi_enabled :
            if self .cls_dual_stream_enabled :
                raise ValueError ("cls_roi_enabled is not supported together with cls_dual_stream_enabled (keep optional objectives isolated).")
            if self .cls_region_masks_enabled :
                raise ValueError ("cls_roi_enabled is not supported together with cls_region_masks_enabled (keep optional objectives isolated).")
            if self .cls_roi_source =="err":
                if err_on is None or err_off is None :
                    raise RuntimeError ("roi_source='err' requires err maps but they were not computed")
                sal =err_on +err_off 
            else :

                sal =torch .sqrt ((x_flow_resid *x_flow_resid ).sum (dim =1 ,keepdim =True )+float (self .eps ))
            roi_mask =self ._saliency_mask (sal ,ratio =float (self .cls_roi_ratio ))
            x_local =x_flow_raw *roi_mask 
            x_roi_local =x_local 
            x_flow_feat =torch .cat ([x_flow_raw ,x_local ],dim =1 )

        if self .cls_region_masks_enabled :
            if self .cls_dual_stream_enabled :
                raise ValueError ("cls_region_masks_enabled is not supported together with cls_dual_stream_enabled (keep optional objectives isolated).")
            if int (x_flow_feat .shape [1 ])!=4 :
                raise ValueError (f"cls_region_masks expects 4ch flow features, got {tuple(x_flow_feat.shape)}")
            if self .region_mask_head is None :
                raise RuntimeError ("region_mask_head is not initialized")
            logits_m =self .region_mask_head (x_flow_feat )
            masks =F .softmax (logits_m ,dim =1 )
            masked_feats =[x_flow_feat *masks [:,k :k +1 ]for k in range (int (masks .shape [1 ]))]
            x_flow_feat =torch .cat (([x_flow_feat ]if self .cls_region_masks_include_base else [])+masked_feats ,dim =1 )

        x_cls_eff =x_flow_feat 
        if self .cls_use_err :
            if err_on is None or err_off is None :
                raise RuntimeError ("cls_use_err=true but err maps were not computed")
            x_cls_eff =torch .cat ([x_cls_eff ,err_on ,err_off ],dim =1 )

        cls_scale =None 
        if self .cls_motion_scale_enabled :
            if self .cls_motion_scale_source =="raw":
                src =x_flow_raw 
            elif self .cls_motion_scale_source =="resid":
                src =x_flow_resid 
            else :
                src =x_cls_eff 
            abs_src =src .abs ()
            if self .cls_motion_scale_stat =="mean_abs":
                m =abs_src .mean (dim =(1 ,2 ,3 ))
            elif self .cls_motion_scale_stat =="median_abs":
                m =abs_src .flatten (1 ).median (dim =1 ).values 
            else :
                m =torch .quantile (abs_src .flatten (1 ),0.75 ,dim =1 )
            scale =(float (self .cls_motion_scale_target )/(m +float (self .eps ))).clamp (
            min =float (self .cls_motion_scale_clip [0 ]),max =float (self .cls_motion_scale_clip [1 ])
            )
            if self .cls_motion_scale_detach :
                scale =scale .detach ()
            cls_scale =scale 
            x_cls_eff =x_cls_eff *scale .view (-1 ,1 ,1 ,1 )

        x_cls_eff_unmasked =x_cls_eff 
        x_cls_eff_masked =x_cls_eff 
        classifier_saliency_mask =None 
        if self .classifier_saliency_mask_enabled :
            classifier_saliency_mask =self ._build_saliency_mask_from_xcls (x_cls_eff .detach ())
            x_cls_eff_masked =x_cls_eff *classifier_saliency_mask 
            aux_extra ["classifier_saliency_mask_ratio"]=classifier_saliency_mask .mean ().detach ()
            if not self .training :
                aux_extra ["classifier_saliency_mask"]=classifier_saliency_mask .detach ()



        if self .cls_dual_view_enabled :
            x_cls_eff =x_cls_eff_unmasked 
        else :
            x_cls_eff =x_cls_eff_masked 

        if int (x_cls_eff .shape [1 ])!=int (self .classifier_in_channels ):
            raise ValueError (
            f"classifier_in_channels mismatch: cfg.motion.classifier_in_channels={self.classifier_in_channels} "
            f"but x_cls has {int(x_cls_eff.shape[1])} channels"
            )
        x_cls_eff_disc =x_cls_eff .detach ()if self .stopgrad_cls_to_motion else x_cls_eff 
        x_cls_eff_masked_disc =x_cls_eff_masked .detach ()if self .stopgrad_cls_to_motion else x_cls_eff_masked 
        x_flow_raw_disc =x_flow_raw .detach ()if self .stopgrad_cls_to_motion else x_flow_raw 
        x_flow_resid_disc =x_flow_resid .detach ()if self .stopgrad_cls_to_motion else x_flow_resid 
        desc_on_disc =desc_on .detach ()if (self .stopgrad_cls_to_motion and torch .is_tensor (desc_on ))else desc_on 
        desc_off_disc =desc_off .detach ()if (self .stopgrad_cls_to_motion and torch .is_tensor (desc_off ))else desc_off 


        if self .channel_swap_enabled :
            if self .channel_swap_proj is None :
                raise RuntimeError ("channel_swap_enabled but channel_swap_proj is not initialized")
            B ,C ,H ,W =x_cls_eff_disc .shape 
            half =int (C )//2 
            desc_on =x_cls_eff_disc [:,:half ]
            desc_off =x_cls_eff_disc [:,half :]
            g_channel_swap =_local_gen ("channel_swap")

            is_reco =str (self .channel_swap_partition_base )=="reco"
            rid_map_t315 =None 
            w_reco =None 

            if is_reco :
                if reco_region_wts is None or (not torch .is_tensor (reco_region_wts )):
                    raise RuntimeError ("channel_swap_partition='reco' requires reco.region to be enabled (need reco_region_wts).")
                w_reco =F .interpolate (reco_region_wts ,size =(int (H ),int (W )),mode ="bilinear",align_corners =False )
                nreg =int (w_reco .shape [1 ])
                rid_map_t315 =torch .argmax (w_reco ,dim =1 ).to (device =x_cls_eff .device ,dtype =torch .long )
            else :
                rid ,nreg ,_ =build_region_id_map (self .channel_swap_partition_base ,int (H ),int (W ),device =x_cls_eff .device )

            opts =set (getattr (self ,"channel_swap_partition_opts",set ())or set ())
            aux_extra ["channel_swap_partition_base"]=str (self .channel_swap_partition_base )
            aux_extra ["channel_swap_partition_opts"]="|".join (sorted (opts ))if opts else ""
            aux_extra ["channel_swap_region_uncertainty_mean"]=torch .tensor (0.0 ,device =x_cls_eff_disc .device ,dtype =x_cls_eff_disc .dtype )


            k =int (self .channel_swap_k )
            k_eff =int (min (int (k ),int (nreg )))if int (nreg )>0 else int (k )

            sampling_source =str (getattr (self ,"channel_swap_sampling_source","uncertainty")).strip ().lower ()
            if sampling_source in {"","random"}:
                sampling_source ="uniform"
            use_hard_error =bool (sampling_source =="hard_error")or ("hard_error"in opts and sampling_source =="uniform")
            use_probe_error =bool (sampling_source =="probe_error")or ("probe_error"in opts and sampling_source =="uniform")
            no_replacement ="no_replacementacement"in opts 
            if use_hard_error and use_probe_error :
                raise ValueError ("channel_swap_partition opts cannot include both 'hard_error' and 'probe_error'.")
            aux_extra ["channel_swap_sampling_source"]=sampling_source 

            if sampling_source =="uncertainty":
                if use_hard_error or use_probe_error :
                    raise ValueError ("channel_swap_sampling_source='uncertainty' cannot be combined with hard_error/probe_error opts.")
                if u_map is None :
                    raise RuntimeError ("channel_swap_sampling_source='uncertainty' requires U map but it is missing.")
                uncertainty_sal =u_map .detach ()
                if tuple (uncertainty_sal .shape [-2 :])!=(int (H ),int (W )):
                    uncertainty_sal =F .interpolate (uncertainty_sal ,size =(int (H ),int (W )),mode ="bilinear",align_corners =False )
                if is_reco :
                    if w_reco is None :
                        raise RuntimeError ("internal error: is_reco but w_reco is None")
                    num =(uncertainty_sal *w_reco ).sum (dim =(2 ,3 ))
                    den =w_reco .sum (dim =(2 ,3 )).clamp_min (1e-6 )
                    w_reg =(num /den ).clamp_min (0.0 )
                else :
                    u_reg ,_ =region_mean (uncertainty_sal ,rid ,int (nreg ))
                    w_reg =u_reg .squeeze (-1 ).clamp_min (0.0 )
                weights =w_reg +1e-6 
                if int (k_eff )<=0 :
                    k_eff =1 
                if no_replacement and int (k_eff )>int (nreg ):
                    k_eff =int (nreg )
                if int (nreg )<=1 :
                    choices =torch .zeros ((int (B ),1 ),device =x_cls_eff_disc .device ,dtype =torch .long )
                else :
                    if g_channel_swap is None :
                        choices =torch .multinomial (weights ,num_samples =int (k_eff ),replacement =not bool (no_replacement ))
                    else :
                        choices =torch .multinomial (weights ,num_samples =int (k_eff ),replacement =not bool (no_replacement ),generator =g_channel_swap )
                aux_extra ["channel_swap_region_uncertainty_mean"]=torch .gather (w_reg ,dim =1 ,index =choices ).mean ().detach ()
            elif use_hard_error or use_probe_error :
                if err_on is None or err_off is None :
                    raise RuntimeError ("channel_swap_partition opts hard_error/probe_error require err_on/err_off but they are missing.")
                err_sal =(err_on +err_off ).detach ()
                if tuple (err_sal .shape [-2 :])!=(int (H ),int (W )):
                    err_sal =F .interpolate (err_sal ,size =(int (H ),int (W )),mode ="bilinear",align_corners =False )
                if is_reco :
                    if w_reco is None :
                        raise RuntimeError ("internal error: is_reco but w_reco is None")

                    num =(err_sal *w_reco ).sum (dim =(2 ,3 ))
                    den =w_reco .sum (dim =(2 ,3 )).clamp_min (1e-6 )
                    w_reg =(num /den ).clamp_min (0.0 )
                else :
                    e_reg ,_ =region_mean (err_sal ,rid ,int (nreg ))
                    w_reg =e_reg .squeeze (-1 ).clamp_min (0.0 )
                if use_hard_error :
                    if int (k_eff )<=1 :
                        choices =w_reg .argmax (dim =1 ,keepdim =True )
                    else :
                        choices =torch .topk (w_reg ,k =int (k_eff ),dim =1 ,largest =True ,sorted =False ).indices 
                else :

                    weights =w_reg +1e-6 
                    if int (k_eff )<=0 :
                        k_eff =1 
                    if no_replacement and int (k_eff )>int (nreg ):
                        k_eff =int (nreg )
                    if int (nreg )<=1 :
                        choices =torch .zeros ((int (B ),1 ),device =x_cls_eff_disc .device ,dtype =torch .long )
                    else :
                        if g_channel_swap is None :
                            choices =torch .multinomial (weights ,num_samples =int (k_eff ),replacement =not bool (no_replacement ))
                        else :
                            choices =torch .multinomial (weights ,num_samples =int (k_eff ),replacement =not bool (no_replacement ),generator =g_channel_swap )
            else :
                if int (nreg )<=1 :
                    choices =torch .zeros ((int (B ),1 ),device =x_cls_eff_disc .device ,dtype =torch .long )
                elif is_reco :
                    if w_reco is None :
                        raise RuntimeError ("internal error: is_reco but w_reco is None")

                    weights =w_reco .mean (dim =(2 ,3 )).clamp_min (1e-6 )
                    if int (k_eff )<=0 :
                        k_eff =1 
                    if no_replacement and int (k_eff )>int (nreg ):
                        k_eff =int (nreg )
                    if g_channel_swap is None :
                        choices =torch .multinomial (weights ,num_samples =int (k_eff ),replacement =not bool (no_replacement ))
                    else :
                        choices =torch .multinomial (weights ,num_samples =int (k_eff ),replacement =not bool (no_replacement ),generator =g_channel_swap )
                elif no_replacement :
                    if int (k_eff )<=0 :
                        k_eff =1 
                    ones =torch .ones ((int (B ),int (nreg )),device =x_cls_eff_disc .device )
                    if g_channel_swap is None :
                        choices =torch .multinomial (ones ,num_samples =int (k_eff ),replacement =False )
                    else :
                        choices =torch .multinomial (ones ,num_samples =int (k_eff ),replacement =False ,generator =g_channel_swap )
                else :
                    if g_channel_swap is None :
                        choices =torch .randint (low =0 ,high =int (nreg ),size =(int (B ),int (k_eff )),device =x_cls_eff_disc .device )
                    else :
                        choices =torch .randint (low =0 ,high =int (nreg ),size =(int (B ),int (k_eff )),device =x_cls_eff_disc .device ,generator =g_channel_swap )

            mask =torch .zeros ((int (B ),1 ,int (H ),int (W )),device =x_cls_eff_disc .device ,dtype =x_cls_eff_disc .dtype )
            for kk in range (int (choices .shape [1 ])):
                mask =torch .maximum (
                mask ,
                (
                (rid_map_t315 ==choices [:,kk ].view (int (B ),1 ,1 ))
                if is_reco 
                else (rid .view (1 ,int (H ),int (W ))==choices [:,kk ].view (int (B ),1 ,1 ))
                )
                .to (dtype =x_cls_eff_disc .dtype )
                .unsqueeze (1 ),
                )
            on_sw =desc_on *(1.0 -mask )+desc_off *mask 
            off_sw =desc_off *(1.0 -mask )+desc_on *mask 
            desc_sw =torch .cat ([on_sw ,off_sw ],dim =1 )

            z1 =self .channel_swap_proj (x_cls_eff_disc )
            z2 =self .channel_swap_proj (desc_sw )
            aux_extra ["channel_swap_z1"]=z1 
            aux_extra ["channel_swap_z2"]=z2 
            aux_extra ["channel_swap_mask_ratio"]=mask .mean ()
            if token_hw is not None :
                channel_swap_mask_tok =(F .adaptive_avg_pool2d (mask ,output_size =token_hw )>0.0 ).to (dtype =mask .dtype )
                aux_extra ["channel_swap_mask_tok"]=channel_swap_mask_tok 
                aux_extra ["channel_swap_mask_ratio_tok"]=channel_swap_mask_tok .mean ()

            if self .channel_swap_mode =="siam":
                if self .channel_swap_pred is None :
                    raise RuntimeError ("channel_swap_mode='siam' but channel_swap_pred is not initialized")
                aux_extra ["channel_swap_p1"]=self .channel_swap_pred (z1 )
                aux_extra ["channel_swap_p2"]=self .channel_swap_pred (z2 )
            elif self .channel_swap_mode =="region_align":
                if self .region_proj is None :
                    raise RuntimeError ("channel_swap_mode='region_align' but region_proj is not initialized")
                denom =mask .sum (dim =(2 ,3 )).clamp_min (1e-6 )
                on_r =(desc_on *mask ).sum (dim =(2 ,3 ))/denom 
                off_r =(desc_off *mask ).sum (dim =(2 ,3 ))/denom 
                aux_extra ["channel_swap_e1"]=F .normalize (self .region_proj (on_r ),dim =1 )
                aux_extra ["channel_swap_e2"]=F .normalize (self .region_proj (off_r ),dim =1 )

            if not self .training :
                aux_extra ["channel_swap_mask"]=mask .detach ()
                aux_extra ["channel_swap_desc_sw"]=desc_sw .detach ()
        else :
            if token_hw is not None :
                z_tok =torch .zeros ((int (batch_size ),1 ,int (token_hw [0 ]),int (token_hw [1 ])),device =device ,dtype =apex .dtype )
                aux_extra ["channel_swap_mask_tok"]=z_tok 
                aux_extra ["channel_swap_mask_ratio_tok"]=z_tok .mean ()


        if self .cross_view_mask_enabled :
            if self .cross_view_mask_pred is None :
                raise RuntimeError ("cross_view_mask_enabled but cross_view_mask_pred is not initialized")
            B ,C ,H ,W =x_cls_eff_disc .shape 
            half =int (C )//2 
            desc_on =x_cls_eff_disc [:,:half ]
            desc_off =x_cls_eff_disc [:,half :]

            rid ,nreg ,_ =build_region_id_map (self .cross_view_mask_partition ,int (H ),int (W ),device =x_cls_eff .device )
            g_cross_view_mask =_local_gen ("cross_view_mask")
            if self .cross_view_mask_mode =="hard":
                hard_map =err_on .detach ()if err_on is not None else self ._err_map (a_hat_on ,apex ).detach ()
                e_reg ,_ =region_mean (hard_map ,rid ,int (nreg ))
                if int (self .cross_view_mask_k )==1 :
                    choices =e_reg .squeeze (-1 ).argmax (dim =1 ,keepdim =True )
                else :
                    choices =torch .topk (e_reg .squeeze (-1 ),k =int (self .cross_view_mask_k ),dim =1 ,largest =True ,sorted =False ).indices 
            else :
                if g_cross_view_mask is None :
                    choices =torch .randint (low =0 ,high =int (nreg ),size =(int (B ),int (self .cross_view_mask_k )),device =x_cls_eff_disc .device )
                else :
                    choices =torch .randint (low =0 ,high =int (nreg ),size =(int (B ),int (self .cross_view_mask_k )),device =x_cls_eff_disc .device ,generator =g_cross_view_mask )

            mask =torch .zeros ((int (B ),1 ,int (H ),int (W )),device =x_cls_eff_disc .device ,dtype =x_cls_eff_disc .dtype )
            for kk in range (int (choices .shape [1 ])):
                mask =torch .maximum (
                mask ,
                (rid .view (1 ,int (H ),int (W ))==choices [:,kk ].view (int (B ),1 ,1 ))
                .to (dtype =x_cls_eff_disc .dtype )
                .unsqueeze (1 ),
                )

            on_in =desc_on *(1.0 -mask )
            pred_on =self .cross_view_mask_pred (torch .cat ([on_in ,desc_off ],dim =1 ))
            l_on =masked_l1 (pred_on ,desc_on .detach (),mask )

            pred_off =None 
            if self .cross_view_mask_mode =="bi":
                off_in =desc_off *(1.0 -mask )
                pred_off =self .cross_view_mask_pred (torch .cat ([off_in ,desc_on ],dim =1 ))
                l_off =masked_l1 (pred_off ,desc_off .detach (),mask )
                L_cross_view_mask =0.5 *(l_on +l_off )
            else :
                L_cross_view_mask =l_on 

            aux_extra ["L_cross_view_mask"]=L_cross_view_mask 
            aux_extra ["cross_view_mask_ratio"]=mask .mean ()
            aux_extra ["cross_view_mask_recon_l1"]=L_cross_view_mask .detach ()
            if not self .training :
                aux_extra ["cross_view_mask"]=mask .detach ()
                aux_extra ["cross_view_mask_pred_on"]=pred_on .detach ()
                if pred_off is not None :
                    aux_extra ["cross_view_mask_pred_off"]=pred_off .detach ()

        logits_a =logits_b =gate_w =None 
        sc_z =None 
        feat_norm =None 
        logits_on =logits_off =None 
        feat_on_map =feat_off_map =None 
        feat_on_vec =feat_off_vec =None 
        contrast_feat_on_map =contrast_feat_off_map =None 
        contrast_z_on =contrast_z_off =contrast_z_diff =None 

        def _gradient_classifier_input (feat_vec_in :torch .Tensor )->torch .Tensor :
            if str (self .gradient_classifier_mode )=="detach_backbone":
                return feat_vec_in .detach ()
            if str (self .gradient_classifier_mode )=="grad_scale":
                s =float (max (float (self .gradient_classifier_grad_scale ),0.0 ))
                fd =feat_vec_in .detach ()
                return fd +s *(feat_vec_in -fd )
            return feat_vec_in 

        if self .cls_dual_branch_enabled :
            if self .adapter_a is None or self .adapter_b is None or self .classifier_a is None or self .classifier_b is None :
                raise RuntimeError ("cls_dual_branch_enabled but branch modules are not initialized")
            xA =x_flow_raw_disc 
            if self .cls_dual_branch_branch_b =="d_flow":
                xB =D_flow_cls 
            else :
                raise RuntimeError (f"Unsupported cls_dual_branch_branch_b: {self.cls_dual_branch_branch_b}")
            logits_a =self .classifier_a (self .adapter_a (xA ))
            logits_b =self .classifier_b (self .adapter_b (xB ))
            logits =float (self .cls_dual_branch_wA )*logits_a +float (self .cls_dual_branch_wB )*logits_b 
        elif self .fusion_enabled :
            if self .fusion_gate is None :
                raise RuntimeError ("fusion_gate is not initialized for fusion_enabled")
            if self .cls_rep !="raw":
                raise ValueError ("fusion_enabled requires cls_rep='raw'")
            logits_a =self .classifier (self .adapter (x_flow_raw_disc ))
            logits_b =self .classifier (self .adapter (x_flow_resid_disc ))
            mag_raw =x_flow_raw_disc .abs ().mean (dim =(1 ,2 ,3 ))
            mag_resid =x_flow_resid_disc .abs ().mean (dim =(1 ,2 ,3 ))
            ratio =mag_resid /(mag_raw +float (self .eps ))
            stats =torch .stack ([mag_raw ,mag_resid ,ratio ],dim =1 )
            gate_w =torch .sigmoid (self .fusion_gate (stats )).view (-1 )
            logits =gate_w .view (-1 ,1 )*logits_a +(1.0 -gate_w ).view (-1 ,1 )*logits_b 
        else :
            if self .two_stage_cls_enabled :
                if desc_on_disc is None or desc_off_disc is None or not torch .is_tensor (desc_on_disc )or not torch .is_tensor (desc_off_disc ):
                    raise RuntimeError ("two_stage_cls_enabled but desc_on/desc_off are missing (cls_rep must provide them).")
                if int (desc_on_disc .shape [1 ])!=int (self .classifier_in_channels )or int (desc_off_disc .shape [1 ])!=int (self .classifier_in_channels ):
                    raise ValueError (
                    "two-stage classifier_in_channels mismatch: "
                    f"classifier_in_channels={int(self.classifier_in_channels)} "
                    f"desc_on={tuple(desc_on_disc.shape)} desc_off={tuple(desc_off_disc.shape)}"
                    )

                x_on =self .adapter (desc_on_disc )
                x_off =self .adapter (desc_off_disc )
                logits_on_base ,feat_on_map ,feat_on_vec =self .classifier (x_on ,return_features =True )
                logits_off_base ,feat_off_map ,feat_off_vec =self .classifier (x_off ,return_features =True )

                if self .gradient_classifier_enabled :
                    if self .gradient_classifier_head is None :
                        raise RuntimeError ("gradient_classifier_enabled but gradient_classifier_head is not initialized")
                    logits_on =self .gradient_classifier_head (_gradient_classifier_input (feat_on_vec ))
                    logits_off =self .gradient_classifier_head (_gradient_classifier_input (feat_off_vec ))
                    feat_norm =0.5 *(feat_on_vec .norm (dim =1 )+feat_off_vec .norm (dim =1 ))
                else :
                    logits_on =logits_on_base 
                    logits_off =logits_off_base 

                logits =0.5 *(logits_on +logits_off )

                if self .sc_enabled :
                    feat_sc =0.5 *(feat_on_map +feat_off_map )
                    if self .sc_region_source =="fixed":
                        sc_z =self ._region_pool (feat_sc )
                    else :
                        if reco_region_wts is None or (not torch .is_tensor (reco_region_wts )):
                            raise RuntimeError ("sc.region_source='reco' requires reco_region_wts (enable need_reco_regions).")
                        if self .reco is None or self .reco .region is None :
                            raise RuntimeError ("sc.region_source='reco' requires reco.region provider.")
                        if bool (getattr (self .reco .region ,"is_dynamic",False ))and (not bool (self .sc_allow_dynamic_regions )):
                            raise AssertionError ("sc.allow_dynamic_regions=false but the selected reco region provider is dynamic (per-sample). Set sc.allow_dynamic_regions=true to proceed.")

                        w_ds =F .interpolate (reco_region_wts ,size =tuple (feat_sc .shape [-2 :]),mode ="bilinear",align_corners =False )
                        w_ds =w_ds /(w_ds .sum (dim =1 ,keepdim =True )+float (self .eps ))
                        sc_z =F .normalize (reco_region_pool (feat_sc ,w_ds ),dim =2 )
                        aux_extra ["sc_sym_pairs"]=list (reco_sym_pairs )

                if self .contrastive_enabled :
                    if self .contrastive_mode in {"a1","a2","a3"}:
                        if bool (self .contrastive_stop_grad_motion ):
                            _ ,m_on ,_ =self .classifier (self .adapter (desc_on_disc .detach ()),return_features =True )
                            _ ,m_off ,_ =self .classifier (self .adapter (desc_off_disc .detach ()),return_features =True )
                            contrast_feat_on_map =m_on 
                            contrast_feat_off_map =m_off 
                        else :
                            contrast_feat_on_map =feat_on_map 
                            contrast_feat_off_map =feat_off_map 
                    elif self .contrastive_mode in {"a4","a5"}:
                        if self .contrastive_proj is None :
                            raise RuntimeError ("contrastive_enabled but contrastive_proj is not initialized")

                        if bool (self .contrastive_stop_grad_motion ):
                            _ ,_m_on ,v_on =self .classifier (self .adapter (desc_on_disc .detach ()),return_features =True )
                            _ ,_m_off ,v_off =self .classifier (self .adapter (desc_off_disc .detach ()),return_features =True )
                            vec_on =v_on 
                            vec_off =v_off 
                        else :
                            vec_on =feat_on_vec 
                            vec_off =feat_off_vec 

                        contrast_z_on =F .normalize (self .contrastive_proj (vec_on ),dim =1 )
                        contrast_z_off =F .normalize (self .contrastive_proj (vec_off ),dim =1 )

                        if self .contrastive_mode =="a5":
                            desc_diff =desc_on_disc -desc_off_disc 
                            if bool (self .contrastive_stop_grad_motion ):
                                desc_diff =desc_diff .detach ()
                            _ ,_m_d ,v_d =self .classifier (self .adapter (desc_diff ),return_features =True )
                            contrast_z_diff =F .normalize (self .contrastive_proj (v_d ),dim =1 )
                    else :
                        raise RuntimeError (f"Unsupported contrastive_mode: {self.contrastive_mode}")
            else :
                want_features =self .sc_enabled or self .gradient_classifier_enabled 

                def _classify_single (x_cls_in :torch .Tensor )->tuple [torch .Tensor ,Optional [torch .Tensor ],Optional [torch .Tensor ]]:
                    x_in =self .adapter (x_cls_in )
                    if want_features :
                        logits_base_i ,feat_map_i ,feat_vec_i =self .classifier (x_in ,return_features =True )
                        if self .gradient_classifier_enabled :
                            if self .gradient_classifier_head is None :
                                raise RuntimeError ("gradient_classifier_enabled but gradient_classifier_head is not initialized")
                            logits_i =self .gradient_classifier_head (_gradient_classifier_input (feat_vec_i ))
                        else :
                            logits_i =logits_base_i 
                        return logits_i ,feat_map_i ,feat_vec_i 
                    logits_i =self .classifier (x_in )
                    return logits_i ,None ,None 

                if self .cls_dual_view_enabled :
                    logits_unmasked ,feat_map_unmasked ,feat_vec_unmasked =_classify_single (x_cls_eff_disc )
                    logits_masked ,_feat_map_masked ,feat_vec_masked =_classify_single (x_cls_eff_masked_disc )
                    w_unmasked =float (self .cls_dual_view_w_unmasked )
                    logits =w_unmasked *logits_unmasked +(1.0 -w_unmasked )*logits_masked 
                    aux_extra ["cls_dual_view_w_unmasked"]=torch .tensor (w_unmasked ,device =logits .device ,dtype =logits .dtype )
                    aux_extra ["logits_unmasked"]=logits_unmasked .detach ()
                    aux_extra ["logits_masked"]=logits_masked .detach ()
                    if feat_vec_unmasked is not None and feat_vec_masked is not None :
                        feat_norm =w_unmasked *feat_vec_unmasked .norm (dim =1 )+(1.0 -w_unmasked )*feat_vec_masked .norm (dim =1 )
                    feat_map =feat_map_unmasked 
                else :
                    logits ,feat_map ,feat_vec =_classify_single (x_cls_eff_disc )
                    if feat_vec is not None :
                        feat_norm =feat_vec .norm (dim =1 )

                if self .sc_enabled :
                    if feat_map is None :
                        raise RuntimeError ("sc_enabled requires feature maps but got None.")
                    if self .sc_region_source =="fixed":
                        sc_z =self ._region_pool (feat_map )
                    else :
                        if reco_region_wts is None or (not torch .is_tensor (reco_region_wts )):
                            raise RuntimeError ("sc.region_source='reco' requires reco_region_wts (enable need_reco_regions).")
                        if self .reco is None or self .reco .region is None :
                            raise RuntimeError ("sc.region_source='reco' requires reco.region provider.")
                        if bool (getattr (self .reco .region ,"is_dynamic",False ))and (not bool (self .sc_allow_dynamic_regions )):
                            raise AssertionError (
                            "sc.allow_dynamic_regions=false but the selected reco region provider is dynamic (per-sample). "
                            "Set sc.allow_dynamic_regions=true to proceed."
                            )

                        w_ds =F .interpolate (reco_region_wts ,size =tuple (feat_map .shape [-2 :]),mode ="bilinear",align_corners =False )
                        w_ds =w_ds /(w_ds .sum (dim =1 ,keepdim =True )+float (self .eps ))
                        sc_z =F .normalize (reco_region_pool (feat_map ,w_ds ),dim =2 )
                        aux_extra ["sc_sym_pairs"]=list (reco_sym_pairs )

        OOB_on =self ._oob_mask (flow_ap_to_on )
        OOB_off =self ._oob_mask (flow_ap_to_off )

        aux :Dict [str ,torch .Tensor ]={
        "alpha":alpha ,
        "tau_on":tau_on ,
        "tau_off":tau_off ,
        "tau":tau ,
        "tau_on_D":tau_on ,
        "tau_off_D":tau_off ,
        "D_flow":D_flow ,
        "T_flow":T_flow ,
        "flow_on":flow_ap_to_on ,
        "flow_off":flow_ap_to_off ,
        "a_hat_on":a_hat_on ,
        "a_hat_off":a_hat_off ,
        "warp_on":a_hat_on ,
        "warp_off":a_hat_off ,
        "x_cls":x_cls_eff_disc ,
        "x_cls_raw":x_flow_raw ,
        "OOB_on":OOB_on ,
        "OOB_off":OOB_off ,
        }
        if self .classifier_saliency_mask_enabled :
            aux ["x_cls_masked"]=x_cls_eff_masked_disc 
        if self .cls_dual_view_enabled :
            aux ["x_cls_unmasked"]=x_cls_eff_disc 
        if ce_on is not None and ce_off is not None and r_cyc_map is not None :
            aux ["ce_on"]=ce_on 
            aux ["ce_off"]=ce_off 
            aux ["r_cyc_map"]=r_cyc_map 
        if w_dyn_on is not None and w_dyn_off is not None :
            aux ["W_dyn_on"]=w_dyn_on 
            aux ["W_dyn_off"]=w_dyn_off 
        if w_valid_on is not None and w_valid_off is not None :
            aux ["W_valid_on"]=w_valid_on 
            aux ["W_valid_off"]=w_valid_off 
        if w_valid_cyc_on is not None and w_valid_cyc_off is not None :
            aux ["W_valid_cyc_on"]=w_valid_cyc_on 
            aux ["W_valid_cyc_off"]=w_valid_cyc_off 
        if route_trust_on is not None and route_trust_off is not None :
            aux ["W_fb_on"]=route_trust_on 
            aux ["W_fb_off"]=route_trust_off 
        if route_c_on is not None and route_c_off is not None and route_c_map is not None :
            aux ["C_on"]=route_c_on 
            aux ["C_off"]=route_c_off 
            aux ["C"]=route_c_map 
            aux ["C_photo_on"]=route_c_on 
            aux ["C_photo_off"]=route_c_off 
            aux ["C_photo"]=route_c_map 
            if route_c_reliability_on is not None and route_c_reliability_off is not None and route_c_reliability_map is not None :
                aux ["C_reliability_on"]=route_c_reliability_on 
                aux ["C_reliability_off"]=route_c_reliability_off 
                aux ["C_reliability"]=route_c_reliability_map 
            if c_tok is not None :
                aux ["C_tok"]=c_tok 
            if c_tok_photo is not None :
                aux ["C_tok_photo"]=c_tok_photo 
                aux ["confTokPhoto"]=c_tok_photo 
            if c_tok_reliability is not None :
                aux ["C_tok_reliability"]=c_tok_reliability 
        if valid_tok is not None :
            aux ["validTok"]=valid_tok .to (dtype =apex .dtype )
        if gate_mask_tok is not None :
            aux ["gate_mask_tok"]=gate_mask_tok .to (dtype =apex .dtype )
        if active_tok_cnt is not None :
            aux ["active_tok_cnt"]=active_tok_cnt 
        if trust_tok is not None :
            aux ["trustTok"]=trust_tok 
        if uncertainty_tok is not None :
            aux ["uncertaintyTok"]=uncertainty_tok 
        if weight_tok is not None :
            aux ["weight_tok"]=weight_tok 
        if u_tok is not None :
            aux ["U_tok"]=u_tok 
        aux ["outside_gate_weight_rate"]=outside_gate_weight_rate 
        if u_on is not None and u_off is not None and u_map is not None :
            aux ["U_on"]=u_on 
            aux ["U_off"]=u_off 
            aux ["U"]=u_map 
            aux ["U_reliability_on"]=u_on 
            aux ["U_reliability_off"]=u_off 
            aux ["U_reliability"]=u_map 
        if logits_on is not None and logits_off is not None :
            aux ["logits_on"]=logits_on 
            aux ["logits_off"]=logits_off 
        if feat_on_map is not None and feat_off_map is not None :
            aux ["feat_on_map"]=feat_on_map 
            aux ["feat_off_map"]=feat_off_map 
        if feat_on_vec is not None and feat_off_vec is not None :
            aux ["feat_on_vec"]=feat_on_vec 
            aux ["feat_off_vec"]=feat_off_vec 
        if contrast_feat_on_map is not None and contrast_feat_off_map is not None :
            aux ["contrast_feat_on_map"]=contrast_feat_on_map 
            aux ["contrast_feat_off_map"]=contrast_feat_off_map 
        if contrast_z_on is not None and contrast_z_off is not None :
            aux ["contrast_z_on"]=contrast_z_on 
            aux ["contrast_z_off"]=contrast_z_off 
        if contrast_z_diff is not None :
            aux ["contrast_z_diff"]=contrast_z_diff 
        if sc_z is not None :
            aux ["sc_z"]=sc_z 
        if feat_norm is not None :
            aux ["feat_norm"]=feat_norm 
        aux ["gradient_classifier_param_count"]=torch .tensor (int (self .gradient_classifier_param_count ),device =device ,dtype =torch .int64 )
        aux ["gradient_classifier_mode_id"]=torch .tensor (0 if str (self .gradient_classifier_mode )=="detach_backbone"else 1 ,device =device ,dtype =torch .int64 )
        if logits_a is not None and logits_b is not None :
            aux ["logits_a"]=logits_a 
            aux ["logits_b"]=logits_b 
        if gate_w is not None :
            aux ["gate_w"]=gate_w 
        if self .cls_global_removal !="none":
            aux ["x_cls_resid"]=x_flow_resid 
        aux ["err_on"]=err_on 
        aux ["err_off"]=err_off 
        if roi_mask is not None :
            aux ["roi_mask"]=roi_mask 
        if x_roi_local is not None :
            aux ["x_roi_local"]=x_roi_local 
        if cls_scale is not None :
            aux ["cls_scale"]=cls_scale 
        if self .cls_region_masks_enabled :
            if "masks"not in locals ():
                raise RuntimeError ("Expected masks in locals for cls_region_masks_enabled")
            aux ["region_masks"]=masks 

            ent =-(masks .clamp_min (float (self .eps ))*masks .clamp_min (float (self .eps )).log ()).sum (dim =1 )
            aux ["mask_entropy"]=ent .mean (dim =(1 ,2 ))
            aux ["mask_usage"]=masks .mean (dim =(2 ,3 ))
            for k in range (int (masks .shape [1 ])):
                aux [f"masked_{k}"]=x_flow_raw *masks [:,k :k +1 ]
        aux .update (aux_global )
        aux .update (aux_extra )
        aux .update (aux_rep )
        aux .update (aux_refine )
        _inject_paper_named_aliases (aux )

        if self .feat_loss_enabled :
            if self .feat_encoder is None :
                raise RuntimeError ("feat_loss_enabled but feat_encoder is not initialized")
            aux ["emb_apex"]=self .feat_encoder (apex )
            aux ["emb_warp_on"]=self .feat_encoder (a_hat_on )
            aux ["emb_warp_off"]=self .feat_encoder (a_hat_off )

        if self .motion_mode =="apex_neighborhood_flow":
            if self .pair_flow is None :
                raise RuntimeError ("pair_flow is not initialized for motion_mode='apex_neighborhood_flow'")
            if apex_m1 is None or apex_p1 is None :
                raise RuntimeError ("motion_mode='apex_neighborhood_flow' requires apex_m1/apex_p1.")
            flow_m1 =self .pair_flow (apex_m1 ,apex )
            flow_p1 =self .pair_flow (apex_p1 ,apex )
            a_hat_m1 =warp (apex_m1 ,flow_m1 )
            a_hat_p1 =warp (apex_p1 ,flow_p1 )
            aux ["flow_m1"]=flow_m1 
            aux ["flow_p1"]=flow_p1 
            aux ["a_hat_m1"]=a_hat_m1 
            aux ["a_hat_p1"]=a_hat_p1 
            if nb_valid is not None :
                aux ["nb_valid"]=nb_valid .to (device =device ,dtype =torch .float32 ).view (-1 )


        if Fi is not None and Fj is not None and t_i is not None and t_j is not None :
            t_i_t =_as_time_tensor (t_i )
            t_j_t =_as_time_tensor (t_j )
            denom =(t_j_t -t_i_t ).clamp_min (1.0 )
            alpha_ij =((t_ap_t -t_i_t )/(denom +eps )).clamp (0.0 ,1.0 )
            alpha_ij_map =alpha_ij .view (batch_size ,1 ,1 ,1 ).expand (batch_size ,1 ,onset .shape [-2 ],onset .shape [-1 ])

            if self .motion_mode =="endpoints_dt":
                if self .motion is None :
                    raise RuntimeError ("motion is not initialized for motion_mode='endpoints_dt'")
                motion_ij =self .motion (Fi ,Fj ,alpha_ij_map )
                D_ij =motion_ij .D_flow 
                T_ij =motion_ij .T_flow 
                flow_ap_to_i =alpha_ij_map *D_ij +T_ij 
                flow_ap_to_j =-(1.0 -alpha_ij_map )*D_ij +T_ij 
            elif self .motion_mode in {"apex_pair_flow","apex_pair_flow_dt","apex_neighborhood_flow","error_feedback_refine"}:
                if self .pair_flow is None :
                    raise RuntimeError ("pair_flow is not initialized for apex_pair_flow modes")
                flow_ap_to_i =self .pair_flow (Fi ,apex )
                flow_ap_to_j =self .pair_flow (Fj ,apex )
                T_ij =0.5 *(flow_ap_to_i +flow_ap_to_j )
                D_ij =0.5 *(flow_ap_to_i -flow_ap_to_j )
            elif self .motion_mode =="triplet_apex_cond_dt":
                if self .triplet_flow is None :
                    raise RuntimeError ("triplet_flow is not initialized for motion_mode='triplet_apex_cond_dt'")
                flow_cat_ij =self .triplet_flow (Fi ,apex ,Fj ,alpha_ij_map )
                flow_ap_to_i =flow_cat_ij [:,0 :2 ]
                flow_ap_to_j =flow_cat_ij [:,2 :4 ]
                T_ij =0.5 *(flow_ap_to_i +flow_ap_to_j )
                D_ij =0.5 *(flow_ap_to_i -flow_ap_to_j )
            else :
                if self .cost_volume_pair_flow is None :
                    raise RuntimeError ("cost_volume_pair_flow is not initialized for motion_mode='cost_volume_apex_pair_flow'")
                flow_ap_to_i =self .cost_volume_pair_flow (Fi ,apex )
                flow_ap_to_j =self .cost_volume_pair_flow (Fj ,apex )
                T_ij =0.5 *(flow_ap_to_i +flow_ap_to_j )
                D_ij =0.5 *(flow_ap_to_i -flow_ap_to_j )
            a_hat_i =warp (Fi ,flow_ap_to_i )
            a_hat_j =warp (Fj ,flow_ap_to_j )

            aux .update (
            {
            "alpha_ij":alpha_ij ,
            "t_i":t_i_t ,
            "t_j":t_j_t ,
            "D_ij":D_ij ,
            "T_ij":T_ij ,
            "flow_i":flow_ap_to_i ,
            "flow_j":flow_ap_to_j ,
            "a_hat_i":a_hat_i ,
            "a_hat_j":a_hat_j ,
            }
            )

            if sampled_pair_valid is not None :
                rv =sampled_pair_valid .to (device =device ,dtype =torch .float32 )
                if rv .ndim ==0 :
                    rv =rv .repeat (batch_size )
                elif rv .ndim ==2 and rv .shape [1 ]==1 :
                    rv =rv .squeeze (1 )
                if rv .ndim !=1 or rv .shape [0 ]!=batch_size :
                    raise ValueError (f"sampled_pair_valid must be [B] or [B,1], got {tuple(rv.shape)}")
                aux ["sampled_pair_valid"]=rv 

        return MotionWarpOutputs (logits =logits ,aux =aux )
