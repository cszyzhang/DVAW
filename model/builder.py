from __future__ import annotations 

import json 
import os 
import sys 
from pathlib import Path 
from typing import Any ,Dict ,List ,Literal 

import torch 
from model .dvaw_reco_model import DVAWReCoModel 

DatasetFilter =Literal ["casme2","samm","smic","all"]
SplitName =Literal ["test","train"]


def ensure_dir (path :Path )->Path :
    path .mkdir (parents =True ,exist_ok =True )
    return path 


def dump_yaml (path :Path ,cfg :Dict [str ,Any ])->None :
    try :
        import yaml 

        text =yaml .safe_dump (cfg ,sort_keys =False ,allow_unicode =True )
    except Exception :
        text =json .dumps (cfg ,ensure_ascii =False ,indent =2 )
    path .write_text (text ,encoding ="utf-8")


def _infer_image_channels (image_mode :str )->int :
    m =str (image_mode ).lower ().strip ()
    if m =="rgb":
        return 3 
    if m =="gray":
        return 1 
    raise ValueError (f"Unsupported image_mode: {image_mode}")


def _model_name (cfg :Dict [str ,Any ])->str :
    name =str (cfg .get ("model",{}).get ("name","embed_diff")).strip ().lower ()
    name =name .replace ("-","_")
    if name in {"motion_warp_mer","motion_warp","motionwarp"}:
        return "motion_warp_mer"
    raise ValueError (f"Unknown model.name: {name}")


def build_model_from_cfg (cfg :Dict [str ,Any ])->torch .nn .Module :
    model_name =_model_name (cfg )
    image_mode =str (cfg .get ("data",{}).get ("image_mode","rgb"))
    num_classes =int (cfg .get ("model",{}).get ("num_classes",3 ))
    if model_name =="motion_warp_mer":
        motion_cfg =cfg .get ("motion",{})
        motion_mode =str (motion_cfg .get ("motion_mode","endpoints_dt")).strip ().lower ().replace ("-","_")
        ds_cfg =motion_cfg .get ("cls_dual_stream",{})or {}
        ms_cfg =motion_cfg .get ("cls_motion_scale",{})or {}
        tb_cfg =motion_cfg .get ("cls_dual_branch",{})or {}
        rm_cfg =motion_cfg .get ("cls_region_masks",{})or {}
        roi_cfg =motion_cfg .get ("cls_roi",{})or {}
        ref_cfg =motion_cfg .get ("refine",{})or {}
        gl_cfg =motion_cfg .get ("global_local",{})or {}
        deformation_descriptor_cfg =motion_cfg .get ("deformation_descriptor",{})or {}
        basis_cfg =motion_cfg .get ("basis_displacement",{})or {}
        fusion_cfg =motion_cfg .get ("fusion",{})or {}
        loss_cfg =cfg .get ("loss",{})or {}
        photo_cfg =cfg .get ("photo",{})or {}
        routing_cfg =cfg .get ("routing",{})or {}
        trust_propagation_cfg =cfg .get ("trust_propagation",{})or {}
        gradient_classifier_cfg :Dict [str ,Any ]={}
        model_gradient_classifier_cfg =(cfg .get ("model",{})or {}).get ("gradient_classifier",{})or {}
        top_gradient_classifier_cfg =cfg .get ("gradient_classifier",{})or {}
        if isinstance (model_gradient_classifier_cfg ,dict ):
            gradient_classifier_cfg .update (model_gradient_classifier_cfg )
        if isinstance (top_gradient_classifier_cfg ,dict ):
            gradient_classifier_cfg .update (top_gradient_classifier_cfg )

        def _section (name :str )->Dict [str ,Any ]:
            v =motion_cfg .get (name ,None )
            if isinstance (v ,dict ):
                return v 
            v =cfg .get (name ,None )
            if isinstance (v ,dict ):
                return v 
            return {}


        region_reconstruction_cfg =_section ("region_reconstruction")
        texture_saliency_mask_cfg =_section ("texture_saliency_mask")
        midframe_cfg =_section ("midframe")
        round_trip_composition_cfg =_section ("round_trip_composition")
        dynamic_support_mask_cfg =_section ("dynamic_support_mask")
        superpoints_cfg =_section ("superpoints")
        descriptor_cfg =_section ("descriptor")
        two_stage_cfg =_section ("two_stage_cls")
        contrastive_cfg =_section ("contrastive")
        sc_cfg =cfg .get ("sc",{})or {}
        channel_swap_cfg =_section ("channel_swap")
        cross_view_mask_cfg =_section ("cross_view_mask")
        multiframe_cfg =_section ("multiframe")
        round_trip_consistency_cfg =_section ("round_trip_consistency")
        reco_cfg =_section ("reco")
        reco_transport_cfg =(reco_cfg .get ("trusted_transport",{})or {})if isinstance (reco_cfg ,dict )else {}

        cls_rep =str (motion_cfg .get ("cls_input_mode",motion_cfg .get ("cls_rep","raw")))
        deform_scales =deformation_descriptor_cfg .get ("scales",(1 ,))
        deform_ops =deformation_descriptor_cfg .get ("ops",("div","curl","shear1","shear2"))
        sc_enabled =bool (sc_cfg .get ("enabled",False ))
        if "enabled"not in sc_cfg :

            sc_enabled =float (loss_cfg .get ("lambda_sc",0.0 ))!=0.0 
        two_stage_enabled =bool (two_stage_cfg .get ("enabled",False ))
        two_stage_fuse =str (two_stage_cfg .get ("fuse","avg_logits"))
        two_stage_aux_ce_weight =float (two_stage_cfg .get ("aux_ce_weight",0.0 ))
        contrastive_enabled =bool (contrastive_cfg .get ("enabled",False ))and float (contrastive_cfg .get ("lambda",0.0 ))!=0.0 
        contrastive_mode =str (contrastive_cfg .get ("mode",""))
        contrastive_stop_grad_motion =bool (contrastive_cfg .get ("stop_grad_motion",False ))
        contrastive_proj_dim =int (contrastive_cfg .get ("proj_dim",0 ))
        sc_region_source =str (sc_cfg .get ("region_source","fixed"))
        sc_region_partition =str (sc_cfg .get ("region_partition","quad4"))
        sc_allow_dynamic_regions =bool (sc_cfg .get ("allow_dynamic_regions",False ))

        channel_swap_enabled =bool (channel_swap_cfg .get ("enabled",False ))
        channel_swap_partition_source =str (channel_swap_cfg .get ("partition_source","")).strip ().lower ()
        channel_swap_partition_fixed =str (channel_swap_cfg .get ("partition","grid3x3")).strip ().lower ()
        if not channel_swap_partition_source :

            channel_swap_partition_source ="reco"if channel_swap_partition_fixed .startswith ("reco")else "fixed"
        if channel_swap_partition_source not in {"fixed","reco"}:
            raise ValueError (f"channel_swap.partition_source must be 'fixed' or 'reco'. Got: {channel_swap_partition_source!r}")
        channel_swap_partition_eff ="reco"if channel_swap_partition_source =="reco"else channel_swap_partition_fixed 
        cross_view_mask_enabled =bool (cross_view_mask_cfg .get ("enabled",False ))
        multiframe_enabled =bool (multiframe_cfg .get ("enabled",False ))
        multiframe_variant =str (multiframe_cfg .get ("variant","extra_rec"))
        multiframe_extra_rec_in_loss =bool (multiframe_cfg .get ("extra_rec_in_loss",False ))


        round_trip_consistency_enabled =bool (round_trip_consistency_cfg .get ("enabled",False ))
        transport_variant =str (round_trip_consistency_cfg .get ("variant","")).strip ().lower ().replace ("-","_")
        predict_backward_eff =bool (motion_cfg .get ("predict_backward",False ))or (round_trip_consistency_enabled and transport_variant in {"fb","occ_mask"})
        round_trip_composition_enabled_eff =bool (round_trip_composition_cfg .get ("enabled",False ))or (round_trip_consistency_enabled and transport_variant in {"three_frame","three_frame_comp"})
        fb_conf_enabled =bool (round_trip_consistency_cfg .get ("fb_conf_enabled",True ))
        fb_conf_tau =float (round_trip_consistency_cfg .get ("fb_conf_tau",0.5 ))
        fb_conf_min =float (round_trip_consistency_cfg .get ("fb_conf_min",0.05 ))
        conf_dyn_mode =str (photo_cfg .get ("w_dyn_mode","none")).strip ().lower ().replace ("-","_")
        conf_dyn_enabled =bool (conf_dyn_mode =="id_err_topk")
        conf_dyn_topk_ratio =float (photo_cfg .get ("w_dyn_topk_ratio",0.15 ))
        conf_dyn_dilate_kernel =int (photo_cfg .get ("w_dyn_dilate_kernel",3 ))

        ms_stat =str (ms_cfg .get ("stat","mean_abs"))
        if "target"in ms_cfg :
            ms_target =float (ms_cfg .get ("target"))
        else :
            ms_target =1.0 if ms_stat .strip ().lower ().replace ("-","_")in {"median_abs","median","p75_abs","q75_abs"}else 0.5 
        ms_clip =ms_cfg .get ("gamma_clip",ms_cfg .get ("clip",[0.5 ,3.0 ]))
        ms_detach =bool (ms_cfg .get ("detach_gamma",ms_cfg .get ("detach",False )))

        gradient_classifier_enabled =bool (gradient_classifier_cfg .get ("enabled",gradient_classifier_cfg .get ("enabled",False )))
        gradient_classifier_hidden_dim =int (gradient_classifier_cfg .get ("hidden_dim",gradient_classifier_cfg .get ("hidden_dim",192 )))
        gradient_classifier_dropout =float (gradient_classifier_cfg .get ("dropout",gradient_classifier_cfg .get ("dropout",0.1 )))
        gradient_classifier_num_layers =int (gradient_classifier_cfg .get ("num_layers",gradient_classifier_cfg .get ("num_layers",2 )))
        gradient_classifier_mode =str (gradient_classifier_cfg .get ("mode",gradient_classifier_cfg .get ("mode","detach_backbone")))
        gradient_classifier_grad_scale =float (gradient_classifier_cfg .get ("grad_scale",gradient_classifier_cfg .get ("grad_scale_to_backbone",0.1 )))

        routing_token_hw =routing_cfg .get ("token_hw",[14 ,14 ])
        if not (isinstance (routing_token_hw ,(list ,tuple ))and len (routing_token_hw )==2 ):
            routing_token_hw =[14 ,14 ]
        routing_gate_thr =float (routing_cfg .get ("gate_thr",0.2 ))
        motion_view_mode =str (motion_cfg .get ("view_mode","dual"))
        routing_mode =str (routing_cfg .get ("mode","full"))
        transport_mode =str (reco_transport_cfg .get ("transport_mode","symmetric"))

        return DVAWReCoModel (
        image_channels =_infer_image_channels (image_mode ),
        num_classes =num_classes ,
        classifier_input_channels =int (cfg .get ("model",{}).get ("classifier_input_channels",3 )),
        flow_downscale =int (motion_cfg .get ("flow_downscale",4 )),
        classifier_in_channels =int (motion_cfg .get ("classifier_in_channels",4 )),
        motion_base_channels =int (motion_cfg .get ("base_channels",32 )),
        max_disp =float (motion_cfg .get ("max_disp",6.0 )),
        compose_mode =str (motion_cfg .get ("compose_mode","disp")),
        motion_mode =motion_mode ,
        corr_radius =int (motion_cfg .get ("corr_radius",2 )),
        cls_global_removal =str (motion_cfg .get ("cls_global_removal","none")),
        cls_lowpass_kernel =int (motion_cfg .get ("cls_lowpass_kernel",31 )),
        cls_dual_stream_enabled =bool (ds_cfg .get ("enabled",False )),
        cls_dual_stream_fusion =str (ds_cfg .get ("fusion","concat")),
        cls_use_err =bool (motion_cfg .get ("cls_use_err",False )),
        cls_err_norm =str (motion_cfg .get ("cls_err_norm",motion_cfg .get ("cls_err_normalize","mean"))),
        cls_motion_scale_enabled =bool (ms_cfg .get ("enabled",False )),
        cls_motion_scale_target =float (ms_target ),
        cls_motion_scale_clip =tuple (ms_clip ),
        cls_motion_scale_source =str (ms_cfg .get ("source","raw")),
        cls_motion_scale_stat =str (ms_stat ),
        cls_motion_scale_detach =bool (ms_detach ),
        cls_region_masks_enabled =bool (rm_cfg .get ("enabled",False )),
        cls_region_masks_k =int (rm_cfg .get ("K",3 )),
        cls_region_masks_include_base =bool (rm_cfg .get ("include_base",True )),
        cls_roi_enabled =bool (roi_cfg .get ("enabled",False )),
        cls_roi_source =str (roi_cfg .get ("source","err")),
        cls_roi_ratio =float (roi_cfg .get ("ratio",0.05 )),
        refine_base_channels =int (motion_cfg .get ("refine_base_channels",32 )),
        refine_enabled =bool (ref_cfg .get ("enabled",False )),
        refine_steps =int (ref_cfg .get ("steps",1 )),
        refine_delta_scale =float (ref_cfg .get ("delta_scale",0.5 )),
        global_local_enabled =bool (gl_cfg .get ("enabled",False )),
        global_local_base_channels =int (gl_cfg .get ("base_channels",32 )),
        global_local_theta_scale =float (gl_cfg .get ("theta_scale",0.1 )),
        cls_rep =cls_rep ,
        deformation_descriptor_scales =tuple (int (s )for s in (deform_scales if isinstance (deform_scales ,(list ,tuple ))else [deform_scales ])),
        deformation_descriptor_ops =tuple (str (o )for o in (deform_ops if isinstance (deform_ops ,(list ,tuple ))else [deform_ops ])),
        sc_enabled =bool (sc_enabled ),
        sc_region_source =str (sc_region_source ),
        sc_region_partition =str (sc_region_partition ),
        sc_allow_dynamic_regions =bool (sc_allow_dynamic_regions ),
        two_stage_cls_enabled =bool (two_stage_enabled ),
        two_stage_cls_fuse =str (two_stage_fuse ),
        two_stage_aux_ce_weight =float (two_stage_aux_ce_weight ),
        contrastive_enabled =bool (contrastive_enabled ),
        contrastive_mode =str (contrastive_mode ),
        contrastive_stop_grad_motion =bool (contrastive_stop_grad_motion ),
        contrastive_proj_dim =int (contrastive_proj_dim ),
        gradient_classifier_enabled =bool (gradient_classifier_enabled ),
        gradient_classifier_hidden_dim =int (gradient_classifier_hidden_dim ),
        gradient_classifier_dropout =float (gradient_classifier_dropout ),
        gradient_classifier_num_layers =int (gradient_classifier_num_layers ),
        gradient_classifier_mode =str (gradient_classifier_mode ),
        gradient_classifier_grad_scale =float (gradient_classifier_grad_scale ),
        cls_dual_branch_enabled =bool (tb_cfg .get ("enabled",False )),
        cls_dual_branch_branch_b =str (tb_cfg .get ("branch_b",tb_cfg .get ("branch_b","D_flow"))),
        cls_dual_branch_fuse =str (tb_cfg .get ("fuse","fixed_avg")),
        cls_dual_branch_wA =float (tb_cfg .get ("wA",0.5 )),
        cls_dual_branch_wB =float (tb_cfg .get ("wB",0.5 )),
        basis_displacement_enabled =bool (basis_cfg .get ("enabled",False )),
        basis_displacement_K =int (basis_cfg .get ("K",8 )),
        basis_displacement_base_res =int (basis_cfg .get ("base_res",28 )),
        fusion_enabled =bool (fusion_cfg .get ("enabled",False )),
        fusion_gate_hidden =int (fusion_cfg .get ("gate_hidden",32 )),
        predict_backward =bool (predict_backward_eff ),
        region_reconstruction_enabled =bool (region_reconstruction_cfg .get ("enabled",False )),
        region_reconstruction_mode =str (region_reconstruction_cfg .get ("mode","fixed_boxes")),
        region_reconstruction_apply_to_cls =bool (region_reconstruction_cfg .get ("apply_to_cls",False )),
        texture_saliency_mask_enabled =bool (texture_saliency_mask_cfg .get ("enabled",False )),
        texture_saliency_mask_grid =int (texture_saliency_mask_cfg .get ("grid",4 )),
        texture_saliency_mask_radius =int (texture_saliency_mask_cfg .get ("radius",9 )),
        texture_saliency_mask_score =str (texture_saliency_mask_cfg .get ("score","lap_var")),
        texture_saliency_mask_apply_to_cls =bool (texture_saliency_mask_cfg .get ("apply_to_cls",False )),
        midframe_enabled =bool (midframe_cfg .get ("enabled",False )),
        round_trip_composition_enabled =bool (round_trip_composition_enabled_eff ),
        dynamic_support_mask_enabled =bool (dynamic_support_mask_cfg .get ("enabled",False )),
        dynamic_support_mask_apply_to_cls =bool (dynamic_support_mask_cfg .get ("apply_to_cls",False )),
        superpoints_K =int (superpoints_cfg .get ("K",superpoints_cfg .get ("k",0 )))if bool (superpoints_cfg .get ("enabled",False ))else 0 ,
        superpoints_apply_to_cls =bool (superpoints_cfg .get ("apply_to_cls",False )),
        descriptor_norm_topk_ratio =float (motion_cfg .get ("descriptor_norm_topk_ratio",descriptor_cfg .get ("topk_ratio",0.10 ))),
        descriptor_detach_scale =bool (motion_cfg .get ("descriptor_detach_scale",False )),
        stopgrad_cls_to_motion =bool (motion_cfg .get ("stopgrad_cls_to_motion",True )),
        classifier_saliency_mask_enabled =bool (motion_cfg .get ("classifier_saliency_mask_enabled",False )),
        classifier_saliency_mask_ratio =float (motion_cfg .get ("classifier_saliency_mask_ratio",0.03 )),
        classifier_saliency_mask_blur_k =int (motion_cfg .get ("classifier_saliency_mask_blur_k",5 )),
        classifier_saliency_mask_blur_sigma =float (motion_cfg .get ("classifier_saliency_mask_blur_sigma",1.0 )),
        cls_dual_view_enabled =bool (motion_cfg .get ("cls_dual_view_enabled",False )),
        cls_dual_view_merge =str (motion_cfg .get ("cls_dual_view_merge","avg_logits")),
        cls_dual_view_w_unmasked =float (motion_cfg .get ("cls_dual_view_w_unmasked",0.5 )),
        conf_dyn_enabled =bool (conf_dyn_enabled ),
        conf_dyn_topk_ratio =float (conf_dyn_topk_ratio ),
        conf_dyn_dilate_kernel =int (conf_dyn_dilate_kernel ),
        view_mode =str (motion_view_mode ),
        routing_mode =str (routing_mode ),
        transport_mode =str (transport_mode ),
        feat_loss_enabled =float (loss_cfg .get ("lambda_feat_endpoints",0.0 ))!=0.0 ,
        feat_embed_channels =int (cfg .get ("model",{}).get ("embed_channels",16 )),
        channel_swap_enabled =bool (channel_swap_enabled ),
        channel_swap_partition =str (channel_swap_partition_eff ),
        channel_swap_sampling_source =str (channel_swap_cfg .get ("sampling_source","uncertainty")),
        channel_swap_mode =str (channel_swap_cfg .get ("mode","infonce")),
        channel_swap_proj_dim =int (channel_swap_cfg .get ("proj_dim",128 )),
        channel_swap_pred_dim =int (channel_swap_cfg .get ("pred_dim",256 )),
        channel_swap_k =int (channel_swap_cfg .get ("channel_swap_k",1 )),
        cross_view_mask_enabled =bool (cross_view_mask_enabled ),
        cross_view_mask_partition =str (cross_view_mask_cfg .get ("partition","grid3x3")),
        cross_view_mask_mode =str (cross_view_mask_cfg .get ("mode","on_from_off")),
        cross_view_mask_hidden =int (cross_view_mask_cfg .get ("hidden",64 )),
        cross_view_mask_k =int (cross_view_mask_cfg .get ("mask_k",1 )),
        multiframe_enabled =bool (multiframe_enabled ),
        multiframe_variant =str (multiframe_variant ),
        multiframe_extra_rec_in_loss =bool (multiframe_extra_rec_in_loss ),
        fb_conf_enabled =bool (fb_conf_enabled ),
        fb_conf_tau =float (fb_conf_tau ),
        fb_conf_min =float (fb_conf_min ),
        routing_token_hw =(int (routing_token_hw [0 ]),int (routing_token_hw [1 ])),
        routing_gate_thr =float (routing_gate_thr ),
        trust_propagation_enabled =bool (trust_propagation_cfg .get ("enabled",False )),
        trust_propagation_alpha =float (trust_propagation_cfg .get ("alpha",0.5 )),
        trust_propagation_steps =int (trust_propagation_cfg .get ("K",2 )),
        reco =reco_cfg if isinstance (reco_cfg ,dict )else None ,
        )


def write_resolved_config (run_dir :Path ,cfg :Dict [str ,Any ])->None :
    ensure_dir (run_dir )
    dump_yaml (run_dir /"config_resolved.yaml",cfg )


def env_info ()->Dict [str ,Any ]:
    return {
    "cwd":str (Path .cwd ()),
    "python":sys .version .split ()[0 ],
    "torch":getattr (torch ,"__version__",None ),
    "cuda_available":bool (torch .cuda .is_available ()),
    "cuda_device_count":int (torch .cuda .device_count ())if torch .cuda .is_available ()else 0 ,
    "hostname":os .uname ().nodename if hasattr (os ,"uname")else None ,
    }
