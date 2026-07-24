from __future__ import annotations 

from typing import Any ,Dict ,List ,Tuple 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 

from .contrast import PairDisc ,supcon_loss 
from .flow import compose_flow 
from .masking import make_entropy_dynamic_score ,make_residual_score ,patch_saliency_mask 
from .trusted_transport import TrustedTransportModule 
from .regions import RegionSelector ,region_pool 


class ReCoAuxiliary (nn .Module ):


    def __init__ (self ,cfg :Dict [str ,Any ],*,enable_regions :bool =True ):
        super ().__init__ ()
        self .cfg =cfg or {}
        self .enable_regions =bool (enable_regions )


        self .region_cfg =(self .cfg .get ("region",{})or {})if isinstance (self .cfg ,dict )else {}
        region_mode =str (self .region_cfg .get ("mode","quad4")).strip ().lower ()
        if self .enable_regions and region_mode !="quad4":
            self .region =RegionSelector (self .region_cfg )
        else :
            self .region =None 


        self .channel_swap_cfg =self .cfg .get ("channel_swap",{})or {}
        self .channel_swap_enabled =bool (self .channel_swap_cfg .get ("enabled",False ))
        self .channel_swap_mode =str (self .channel_swap_cfg .get ("mode","corrupt_xsample")).strip ().lower ()
        self .channel_swap_lambda =float (self .channel_swap_cfg .get ("lambda",0.05 ))
        self .channel_swap_temp =float (self .channel_swap_cfg .get ("temp",0.2 ))

        self .channel_swap_disc :PairDisc |None =None 
        if self .channel_swap_enabled and self .channel_swap_mode in {"corrupt_xsample","corrupt_xstage"}:
            cin =int (self .channel_swap_cfg .get ("disc_in_ch",16 ))
            hid =int (self .channel_swap_cfg .get ("disc_hid",64 ))
            self .channel_swap_disc =PairDisc (cin =cin ,hid =hid )


        self .cross_view_mask_cfg =self .cfg .get ("cross_view_mask",{})or {}
        self .cross_view_mask_enabled =bool (self .cross_view_mask_cfg .get ("enabled",False ))




        task =str (self .cross_view_mask_cfg .get ("mode","uniform_region")).strip ().lower ()
        mask_mode =str (self .cross_view_mask_cfg .get ("mask_mode","")).strip ().lower ()
        known_mask_modes ={"uniform_region","entropy_dyn_topk","residual_topk"}
        if task in known_mask_modes and not mask_mode :
            task ="completion"
            mask_mode =str (self .cross_view_mask_cfg .get ("mode","uniform_region")).strip ().lower ()
        if not task or task in {"completion"}:
            task ="completion"
            if not mask_mode :
                mask_mode ="uniform_region"
        else :
            raise ValueError (f"Unknown cross_view_mask.mode: {task!r} (expected 'completion')")
        if mask_mode not in known_mask_modes :
            raise ValueError (f"Unknown cross_view_mask.mask_mode: {mask_mode!r} (expected one of {sorted(known_mask_modes)})")
        self .cross_view_mask_task =task 
        self .cross_view_mask_selection_mode =mask_mode 
        self .cross_view_mask_lambda =float (self .cross_view_mask_cfg .get ("lambda",0.10 ))
        self .of_lambda =float (self .cross_view_mask_cfg .get ("lambda_of",0.05 ))
        self .mask_ratio =float (self .cross_view_mask_cfg .get ("mask_ratio",0.25 ))
        self .mask_topk =float (self .cross_view_mask_cfg .get ("mask_topk",self .mask_ratio ))
        self .score_temp =float (self .cross_view_mask_cfg .get ("score_temp",0.0 ))
        self .cross_view_mask_temp =float (self .cross_view_mask_cfg .get ("temp",0.0 ))
        self .patch_hw =tuple (self .cross_view_mask_cfg .get ("patch_hw",[16 ,16 ]))

        comp_hid =int (self .cross_view_mask_cfg .get ("hid",64 ))
        comp_in =int (self .cross_view_mask_cfg .get ("in_ch",16 ))
        comp_out =int (self .cross_view_mask_cfg .get ("out_ch",8 ))
        self .comp_in =int (comp_in )
        self .comp_out =int (comp_out )
        self .comp_net :nn .Module |None =None 
        self .of_net :nn .Module |None =None 
        if self .cross_view_mask_enabled :
            self .comp_net =nn .Sequential (
            nn .Conv2d (int (comp_in ),int (comp_hid ),3 ,padding =1 ),
            nn .ReLU (inplace =True ),
            nn .Conv2d (int (comp_hid ),int (comp_hid ),3 ,padding =1 ),
            nn .ReLU (inplace =True ),
            nn .Conv2d (int (comp_hid ),int (comp_out ),3 ,padding =1 ),
            )
            self .of_net =nn .Sequential (
            nn .Conv2d (int (comp_in ),int (comp_hid ),3 ,padding =1 ),
            nn .ReLU (inplace =True ),
            nn .Conv2d (int (comp_hid ),int (comp_out ),3 ,padding =1 ),
            )


        self .mf_cfg =self .cfg .get ("multiframe",{})or {}
        self .mf_enabled =bool (self .mf_cfg .get ("enabled",False ))
        self .mf_variant =str (self .mf_cfg .get ("variant","v_scalars")).strip ().lower ()
        self .mf_lambda =float (self .mf_cfg .get ("lambda",0.05 ))
        self .mf_temp =float (self .mf_cfg .get ("temp",0.2 ))
        self .s_mlp :nn .Module |None =None 
        if self .mf_enabled and self .mf_variant .startswith ("v_scalars"):
            self .s_mlp =nn .Sequential (nn .Linear (2 ,32 ),nn .ReLU (inplace =True ),nn .Linear (32 ,1 ))


        self .transport_cfg =self .cfg .get ("trusted_transport",{})or {}
        self .transport_enabled =bool (self .transport_cfg .get ("enabled",False ))
        self .transport_variant =str (self .transport_cfg .get ("variant","cycle_abspos")).strip ().lower ()
        self .transport_lambda =float (self .transport_cfg .get ("lambda",0.02 ))
        self .transport_align_lambda =float (self .transport_cfg .get ("lambda_align",0.02 ))
        self .transport_align_unit =str (self .transport_cfg .get ("align_unit","pixel")).strip ().lower ()
        if self .transport_align_unit not in {"pixel","token"}:
            raise ValueError (f"trusted_transport.align_unit must be 'pixel' or 'token', got {self.transport_align_unit!r}")
        self .transport_align_teacher =str (self .transport_cfg .get ("align_teacher","flow_teach_transport")).strip ().lower ()
        if self .transport_align_teacher in {"flow_to_transport","flow_to_transport","flow_teach"}:
            self .transport_align_teacher ="flow_teach_transport"
        if self .transport_align_teacher in {"transport_to_flow","transport_to_flow","transport_teach"}:
            self .transport_align_teacher ="transport_teach_flow"
        if self .transport_align_teacher not in {"flow_teach_transport","transport_teach_flow"}:
            raise ValueError (f"trusted_transport.align_teacher must be 'flow_teach_transport' or 'transport_teach_flow', got {self.transport_align_teacher!r}")
        self .transport_use_conf_tok =bool (self .transport_cfg .get ("use_conf_tok",False ))
        self .transport_conf_tok_min =float (self .transport_cfg .get ("conf_tok_min",0.0 ))
        if self .transport_conf_tok_min <0.0 :
            raise ValueError ("trusted_transport.conf_tok_min must be >= 0")
        token_hw =tuple (self .transport_cfg .get ("token_hw",[14 ,14 ]))
        feat_dim =int (self .transport_cfg .get ("feat_dim",64 ))
        temp =float (self .transport_cfg .get ("temp",0.07 ))
        pos_mode ="abs"if "abspos"in self .transport_variant else "rel"
        self .transport :TrustedTransportModule |None =None 
        if self .transport_enabled :
            self .transport =TrustedTransportModule (token_hw =token_hw ,feat_dim =feat_dim ,temp =temp ,pos_mode =pos_mode )


        if self .region is None :
            if self .channel_swap_enabled :
                raise ValueError ("reco.channel_swap.enabled=true requires reco.region.mode != 'quad4' (need region_wts).")
            if self .cross_view_mask_enabled and self .cross_view_mask_selection_mode =="uniform_region":
                raise ValueError ("reco.cross_view_mask.mask_mode='uniform_region' requires reco.region.mode != 'quad4' (need region_wts).")
            if self .mf_enabled and self .mf_variant =="multidelta_supcon":
                raise ValueError ("reco.multiframe.variant='multidelta_supcon' requires reco.region.mode != 'quad4' (need region_wts).")

    def enabled_any (self )->bool :
        return bool (self .region is not None or self .channel_swap_enabled or self .cross_view_mask_enabled or self .mf_enabled or self .transport_enabled )




    def build_regions (
    self ,*,feat_hw :tuple [int ,int ],batch_size :int ,img_ref :torch .Tensor ,dyn_map :torch .Tensor 
    )->tuple [torch .Tensor |None ,List [Tuple [int ,int ]],torch .Tensor ,Dict [str ,torch .Tensor ]]:
        if self .region is None :
            z =torch .zeros ([],device =img_ref .device ,dtype =img_ref .dtype )
            return None ,[],z ,{}
        wts ,sym_pairs ,reg_loss ,diag =self .region (feat_hw ,int (batch_size ),img_ref =img_ref ,dyn_map =dyn_map )
        return wts ,sym_pairs ,reg_loss ,diag 




    def channel_swap_losses (self ,M_OA :torch .Tensor ,M_AF :torch .Tensor ,region_wts :torch .Tensor )->tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:
        if not self .channel_swap_enabled :
            return torch .zeros ([],device =M_OA .device ,dtype =M_OA .dtype ),{}
        if region_wts is None :
            raise ValueError ("channel_swap_enabled requires region_wts")

        B ,C ,H ,W =M_OA .shape 
        R =int (region_wts .shape [1 ])
        rid =torch .randint (low =0 ,high =R ,size =(int (B ),),device =M_OA .device )
        mask =region_wts [torch .arange (int (B ),device =M_OA .device ),rid ].unsqueeze (1 )
        mask =(mask /(mask .amax (dim =(2 ,3 ),keepdim =True )+1e-6 )).clamp (0 ,1 )

        loss =torch .zeros ([],device =M_OA .device ,dtype =M_OA .dtype )
        diag :Dict [str ,torch .Tensor ]={"channel_swap_mode":torch .tensor (0 ,device =M_OA .device )}

        if self .channel_swap_mode =="corrupt_xsample":
            if self .channel_swap_disc is None :
                raise RuntimeError ("channel_swap_disc missing for corrupt_xsample")
            perm =torch .randperm (int (B ),device =M_OA .device )
            if bool ((perm ==torch .arange (int (B ),device =M_OA .device )).any ()):
                perm =(perm +1 )%int (B )
            M_other =M_OA [perm ]
            M_corrupt =M_OA *(1 -mask )+M_other *mask 

            x_pos =torch .cat ([M_OA ,M_AF ],dim =1 )
            x_neg =torch .cat ([M_corrupt ,M_AF ],dim =1 )
            logit_pos =self .channel_swap_disc (x_pos )
            logit_neg =self .channel_swap_disc (x_neg )
            loss_bce =F .binary_cross_entropy_with_logits (logit_pos ,torch .ones_like (logit_pos ))+F .binary_cross_entropy_with_logits (
            logit_neg ,torch .zeros_like (logit_neg )
            )
            loss =self .channel_swap_lambda *loss_bce 
            with torch .no_grad ():
                acc =0.5 *(((logit_pos >0 ).float ().mean ()+(logit_neg <0 ).float ().mean ()))
            diag .update ({"L_channel_swap":loss_bce .detach (),"channel_swap_acc":acc .detach (),"channel_swap_mode":torch .tensor (1 ,device =M_OA .device )})

        elif self .channel_swap_mode =="corrupt_xstage":
            if self .channel_swap_disc is None :
                raise RuntimeError ("channel_swap_disc missing for corrupt_xstage")

            def reg_norm (M :torch .Tensor )->torch .Tensor :
                return (M *mask ).pow (2 ).mean (dim =(1 ,2 ,3 )).sqrt ()+1e-6 

            n_oa =reg_norm (M_OA )
            n_af =reg_norm (M_AF )
            scale =(n_oa /n_af ).view (int (B ),1 ,1 ,1 )
            M_afn =M_AF *scale 
            M_corrupt =M_OA *(1 -mask )+M_afn *mask 

            x_pos =torch .cat ([M_OA ,M_AF ],dim =1 )
            x_neg =torch .cat ([M_corrupt ,M_AF ],dim =1 )
            logit_pos =self .channel_swap_disc (x_pos )
            logit_neg =self .channel_swap_disc (x_neg )
            loss_bce =F .binary_cross_entropy_with_logits (logit_pos ,torch .ones_like (logit_pos ))+F .binary_cross_entropy_with_logits (
            logit_neg ,torch .zeros_like (logit_neg )
            )
            loss =self .channel_swap_lambda *loss_bce 
            with torch .no_grad ():
                acc =0.5 *(((logit_pos >0 ).float ().mean ()+(logit_neg <0 ).float ().mean ()))
            diag .update ({"L_channel_swap":loss_bce .detach (),"channel_swap_acc":acc .detach (),"channel_swap_mode":torch .tensor (2 ,device =M_OA .device )})

        elif self .channel_swap_mode =="xstage_supcon":

            z_oa =region_pool (M_OA ,region_wts )
            z_af =region_pool (M_AF ,region_wts )

            z =torch .cat ([z_oa ,z_af ],dim =0 ).reshape (2 *int (B )*int (R ),-1 )

            stage =torch .repeat_interleave (torch .tensor ([0 ,1 ],device =M_OA .device ),int (B )*int (R ))
            b_id =(torch .arange (2 *int (B )*int (R ),device =M_OA .device )//int (R ))%int (B )
            r_id =torch .arange (2 *int (B )*int (R ),device =M_OA .device )%int (R )

            pos_mask =(b_id [:,None ]==b_id [None ,:])&(r_id [:,None ]==r_id [None ,:])&(stage [:,None ]!=stage [None ,:])

            loss_con ,d =supcon_loss (z ,pos_mask ,temp =self .channel_swap_temp )
            loss =self .channel_swap_lambda *loss_con 
            diag .update (
            {
            "L_channel_swap":loss_con .detach (),
            "L_xstage_sc":loss_con .detach (),
            "channel_swap_pos_sim":d ["pos_sim"],
            "channel_swap_neg_sim":d ["neg_sim"],
            "channel_swap_gap":d ["gap"],
            "channel_swap_mode":torch .tensor (3 ,device =M_OA .device ),
            }
            )
        else :
            raise ValueError (f"Unknown channel_swap.mode: {self.channel_swap_mode}")

        return loss ,diag 




    def cross_view_mask_losses (
    self ,
    M_OA :torch .Tensor ,
    M_AF :torch .Tensor ,
    M_OF :torch .Tensor ,
    *,
    img_ref :torch .Tensor ,
    dyn_map :torch .Tensor ,
    region_wts :torch .Tensor |None =None ,
    rng :torch .Generator |None =None ,
    )->tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:
        if not self .cross_view_mask_enabled :
            return torch .zeros ([],device =M_OA .device ,dtype =M_OA .dtype ),{}
        if self .comp_net is None or self .of_net is None :
            raise RuntimeError ("cross_view_mask_enabled but comp_net/of_net are not initialized")

        B ,C ,H ,W =M_OA .shape 

        if self .cross_view_mask_task !="completion":
            raise RuntimeError (f"Unexpected cross_view_mask_task: {self.cross_view_mask_task!r}")

        if self .cross_view_mask_selection_mode =="uniform_region":
            if region_wts is None :
                raise ValueError ("cross_view_mask.mask_mode=uniform_region requires region_wts")
            R =int (region_wts .shape [1 ])

            k_regions =int (max (1 ,min (R ,round (float (self .mask_ratio )*float (R )))))
            if rng is None :
                scores =torch .rand ((int (B ),int (R )),device =M_OA .device )
            else :
                scores =torch .rand ((int (B ),int (R )),generator =rng ,device =M_OA .device )
            rid =scores .topk (k =int (k_regions ),dim =1 ).indices 
            bidx =torch .arange (int (B ),device =M_OA .device )[:,None ].expand (int (B ),int (k_regions ))
            mask =region_wts [bidx ,rid ].sum (dim =1 ,keepdim =True )
            mask =(mask /(mask .amax (dim =(2 ,3 ),keepdim =True )+1e-6 )).clamp (0 ,1 )
        elif self .cross_view_mask_selection_mode =="entropy_dyn_topk":
            S =make_entropy_dynamic_score (img_ref ,dyn_map ,out_hw =(int (H ),int (W )))
            mask =patch_saliency_mask (S ,patch_hw =tuple (self .patch_hw ),topk_ratio =float (self .mask_topk ))
        elif self .cross_view_mask_selection_mode =="residual_topk":
            S =make_residual_score (dyn_map ,out_hw =(int (H ),int (W )))
            mask =patch_saliency_mask (S ,patch_hw =tuple (self .patch_hw ),topk_ratio =float (self .mask_topk ))
        else :
            raise ValueError (f"Unknown cross_view_mask.mask_mode: {self.cross_view_mask_selection_mode}")

        if int (M_OA .shape [1 ])!=int (self .comp_out )or int (M_AF .shape [1 ])!=int (self .comp_out )or int (M_OF .shape [1 ])!=int (self .comp_out ):
            raise ValueError (
            "cross_view_mask expects M_* channels to match comp_out. "
            f"Got M_OA={tuple(M_OA.shape)} M_AF={tuple(M_AF.shape)} M_OF={tuple(M_OF.shape)} comp_out={int(self.comp_out)}"
            )

        x_oa =torch .cat ([M_OA *(1 -mask ),M_AF ],dim =1 )
        x_af =torch .cat ([M_AF *(1 -mask ),M_OA ],dim =1 )
        if int (x_oa .shape [1 ])!=int (self .comp_in ):
            raise ValueError (f"cross_view_mask comp_in mismatch: expected {int(self.comp_in)} got {int(x_oa.shape[1])}")

        pred_oa =self .comp_net (x_oa )
        pred_af =self .comp_net (x_af )

        L_oa =(pred_oa -M_OA ).abs ()*mask 
        L_af =(pred_af -M_AF ).abs ()*mask 
        denom =(mask .sum ()*float (C )+1e-6 )
        L_oa =L_oa .sum ()/denom 
        L_af =L_af .sum ()/denom 

        x_of =torch .cat ([M_OA ,M_AF ],dim =1 )
        pred_of =self .of_net (x_of )
        L_of =(pred_of -M_OF ).abs ().mean ()

        loss =self .cross_view_mask_lambda *(L_oa +L_af )+self .of_lambda *L_of 
        diag ={"L_cross_view_mask":(L_oa +L_af ).detach (),"L_of_pred":L_of .detach (),"mask_ratio":mask .mean ().detach ()}
        return loss ,diag 




    def multiframe_losses (
    self ,
    flows_dict :Dict [str ,Any ],
    *,
    tau_on :torch .Tensor ,
    tau_off :torch .Tensor ,
    )->tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:
        if not self .mf_enabled :
            return torch .zeros ([],device =tau_on .device ,dtype =tau_on .dtype ),{}

        if self .mf_variant .startswith ("v_scalars"):
            if self .s_mlp is None :
                raise RuntimeError ("multiframe.v_scalars enabled but s_mlp is not initialized")
            f_onA =flows_dict .get ("f_onA",None )
            if f_onA is None or not torch .is_tensor (f_onA ):
                raise ValueError ("multiframe.v_scalars requires flows_dict['f_onA'] (B,2,H,W)")
            B =int (f_onA .shape [0 ])
            tau_on_v =tau_on .view (B ,1 )
            tau_off_v =tau_off .view (B ,1 )
            v =f_onA /(tau_on .view (B ,1 ,1 ,1 )+1e-6 )

            Lfit =torch .zeros ([],device =f_onA .device ,dtype =f_onA .dtype )
            cnt =0 
            scale_list :List [torch .Tensor ]=[]

            pre_list =flows_dict .get ("pre_list",[])or []
            for f_tA ,dt in pre_list :
                dt =dt .view (B ,1 )
                x =torch .stack ([(dt /(tau_on_v +1e-6 )),torch .zeros_like (dt )],dim =-1 ).squeeze (2 )
                mult =0.5 +torch .sigmoid (self .s_mlp (x ))
                s =dt .view (B ,1 ,1 ,1 )*mult .view (B ,1 ,1 ,1 )
                Lfit =Lfit +(f_tA -s *v ).abs ().mean ()
                cnt +=1 
                scale_list .append (mult .detach ().view (-1 ))

            post_list =flows_dict .get ("post_list",[])or []
            for f_At ,dt in post_list :
                dt =dt .view (B ,1 )
                x =torch .stack ([(dt /(tau_off_v +1e-6 )),torch .ones_like (dt )],dim =-1 ).squeeze (2 )
                mult =0.5 +torch .sigmoid (self .s_mlp (x ))
                s =dt .view (B ,1 ,1 ,1 )*mult .view (B ,1 ,1 ,1 )
                Lfit =Lfit +(f_At +s *v ).abs ().mean ()
                cnt +=1 
                scale_list .append (mult .detach ().view (-1 ))

            if cnt >0 :
                Lfit =Lfit /float (cnt )
            loss =self .mf_lambda *Lfit 
            if scale_list :
                s_all =torch .cat (scale_list ,dim =0 )
                s_mean =s_all .mean ()
                s_std =s_all .std (unbiased =False )
            else :
                s_mean =torch .zeros ([],device =f_onA .device ,dtype =f_onA .dtype )
                s_std =torch .zeros ([],device =f_onA .device ,dtype =f_onA .dtype )
            diag :Dict [str ,torch .Tensor ]={
            "L_mf":Lfit .detach (),
            "L_mf_vfit":Lfit .detach (),
            "vfit_scale_mean":s_mean .detach (),
            "vfit_scale_std":s_std .detach (),
            "mf_variant_id":torch .tensor (1 ,device =f_onA .device ),
            }
            return loss ,diag 

        if self .mf_variant =="round_trip_composition":
            f_onA =flows_dict .get ("f_onA",None )
            f_Aoff =flows_dict .get ("f_Aoff",None )
            f_onoff =flows_dict .get ("f_onoff",None )
            if f_onA is None or f_Aoff is None or f_onoff is None :
                raise ValueError ("multiframe.round_trip_composition requires f_onA,f_Aoff,f_onoff")
            if not (torch .is_tensor (f_onA )and torch .is_tensor (f_Aoff )and torch .is_tensor (f_onoff )):
                raise ValueError ("multiframe.round_trip_composition requires tensor flows")

            comp =compose_flow (f_onA ,f_Aoff )
            Lc =(comp -f_onoff ).abs ().mean ()
            loss =self .mf_lambda *Lc 
            return loss ,{"L_mf":Lc .detach (),"L_round_trip_composition_main":Lc .detach (),"mf_variant_id":torch .tensor (2 ,device =f_onA .device )}


        return torch .zeros ([],device =tau_on .device ,dtype =tau_on .dtype ),{"mf_variant_id":torch .tensor (3 ,device =tau_on .device )}

    def multidelta_supcon_losses (
    self ,
    *,
    embeds :torch .Tensor ,
    sample_ids :torch .Tensor ,
    region_ids :torch .Tensor ,
    )->tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:

        if not self .mf_enabled or self .mf_variant !="multidelta_supcon":
            return torch .zeros ([],device =embeds .device ,dtype =embeds .dtype ),{}
        if embeds .ndim !=2 :
            raise ValueError (f"embeds must be (M,D), got {tuple(embeds.shape)}")
        if sample_ids .ndim !=1 or region_ids .ndim !=1 or int (sample_ids .shape [0 ])!=int (embeds .shape [0 ])or int (region_ids .shape [0 ])!=int (embeds .shape [0 ]):
            raise ValueError ("sample_ids/region_ids must be (M,) aligned to embeds")

        sid =sample_ids .view (-1 )
        rid =region_ids .view (-1 )
        same_sample =sid [:,None ]==sid [None ,:]
        same_region =rid [:,None ]==rid [None ,:]
        self_mask =torch .eye (int (embeds .shape [0 ]),device =embeds .device ,dtype =torch .bool )
        pos_mask =same_sample &same_region &(~self_mask )
        loss ,d =supcon_loss (embeds ,pos_mask ,temp =self .mf_temp )
        diag ={
        "L_mf_supcon":loss .detach (),

        "L_multidelta_sc":loss .detach (),
        "pos_sim_mean":d ["pos_sim"],
        "neg_sim_mean":d ["neg_sim"],
        "mf_pos_sim":d ["pos_sim"],
        "mf_neg_sim":d ["neg_sim"],
        "mf_gap":d ["gap"],
        }
        return self .mf_lambda *loss ,diag 




    def trusted_transport_losses (
    self ,
    img_a :torch .Tensor ,
    img_b :torch .Tensor ,
    flow_ab :torch .Tensor ,
    *,
    full_hw_px :tuple [int ,int ],
    conf_map :torch .Tensor |None =None ,
    weight_map :torch .Tensor |None =None ,
    valid_map :torch .Tensor |None =None ,
    gate_thr :float =0.2 ,
    rng :torch .Generator |None =None ,
    )->tuple [torch .Tensor ,Dict [str ,torch .Tensor ]]:
        if not self .transport_enabled :
            return torch .zeros ([],device =img_a .device ,dtype =img_a .dtype ),{}
        if self .transport is None :
            raise RuntimeError ("trusted_transport.enabled=true but transport module is not initialized")

        do_lw =self .transport_variant =="labelwarp_abspos"
        Lcyc ,Llw ,flow_soft ,d =self .transport .forward_pair (img_a ,img_b ,full_hw_px =full_hw_px ,do_labelwarp =do_lw ,rng =rng )

        flow_ds =F .interpolate (flow_ab ,size =(int (flow_soft .shape [2 ]),int (flow_soft .shape [3 ])),mode ="bilinear",align_corners =False )
        if self .transport_align_unit =="token":



            Wpx =int (full_hw_px [1 ])
            w =int (flow_soft .shape [3 ])
            scale =float (Wpx )/float (max (w ,1 ))
            flow_ds =flow_ds /scale 
            flow_soft =flow_soft /scale 
        if self .transport_align_teacher =="flow_teach_transport":

            flow_ds_eff =flow_ds .detach ()
            flow_soft_eff =flow_soft 
        else :

            flow_ds_eff =flow_ds 
            flow_soft_eff =flow_soft .detach ()

        align_map =(flow_ds_eff -flow_soft_eff ).abs ().mean (dim =1 ,keepdim =True )
        transport_conf_mean =torch .zeros ([],device =img_a .device ,dtype =img_a .dtype )

        conf_tok =torch .ones_like (align_map )
        gate_tok =torch .ones_like (align_map ,dtype =torch .bool )
        if conf_map is not None :
            if (not torch .is_tensor (conf_map ))or conf_map .ndim !=4 or int (conf_map .shape [1 ])!=1 :
                raise ValueError (f"conf_map must be [B,1,H,W], got {type(conf_map).__name__} {getattr(conf_map, 'shape', None)}")
            conf_tok =F .interpolate (
            conf_map .to (device =align_map .device ,dtype =align_map .dtype ),
            size =(int (align_map .shape [2 ]),int (align_map .shape [3 ])),
            mode ="area",
            )
            transport_conf_mean =conf_tok .mean ().detach ()
            gate_tok =gate_tok &(conf_tok >float (max (float (self .transport_conf_tok_min ),float (gate_thr ))))

        if valid_map is not None :
            if (not torch .is_tensor (valid_map ))or valid_map .ndim !=4 or int (valid_map .shape [1 ])!=1 :
                raise ValueError (f"valid_map must be [B,1,H,W], got {type(valid_map).__name__} {getattr(valid_map, 'shape', None)}")
            valid_tok =F .interpolate (
            valid_map .to (device =align_map .device ,dtype =align_map .dtype ),
            size =(int (align_map .shape [2 ]),int (align_map .shape [3 ])),
            mode ="area",
            )>0.99 
            gate_tok =gate_tok &valid_tok 

        if weight_map is not None :
            if (not torch .is_tensor (weight_map ))or weight_map .ndim !=4 or int (weight_map .shape [1 ])!=1 :
                raise ValueError (
                f"weight_map must be [B,1,H,W], got {type(weight_map).__name__} {getattr(weight_map, 'shape', None)}"
                )
            weight_tok =F .interpolate (
            weight_map .to (device =align_map .device ,dtype =align_map .dtype ),
            size =(int (align_map .shape [2 ]),int (align_map .shape [3 ])),
            mode ="area",
            )
        elif self .transport_use_conf_tok :
            weight_tok =conf_tok .clamp_min (float (self .transport_conf_tok_min ))
        else :
            weight_tok =torch .ones_like (align_map )

        active =gate_tok .to (dtype =align_map .dtype )
        if float (active .sum ().detach ().cpu ().item ())>0.0 :
            weight_eff =(weight_tok *active ).clamp_min (0.0 )
            if float (weight_eff .sum ().detach ().cpu ().item ())<=0.0 :
                weight_eff =active 
            Lalign =(align_map *weight_eff ).sum ()/weight_eff .sum ().clamp_min (1e-6 )
        else :
            Lalign =align_map .mean ()

        loss =self .transport_lambda *Lcyc +self .transport_lambda *Llw +self .transport_align_lambda *Lalign 

        diag_mass_t_mean =d .get ("diag_mass_T_mean",torch .zeros ([],device =img_a .device ,dtype =img_a .dtype ))
        entropy_t_mean =d .get ("entropy_T_mean",torch .zeros ([],device =img_a .device ,dtype =img_a .dtype ))
        active_tok_cnt_mean =active .view (int (active .shape [0 ]),-1 ).sum (dim =1 ).mean ().detach ()
        Pab =d .get ("transport_T_matrix",None )
        if torch .is_tensor (Pab )and Pab .ndim ==3 :
            diag_vec =torch .diagonal (Pab ,dim1 =1 ,dim2 =2 )
            p_safe =Pab .clamp_min (1e-8 )
            ent =-(p_safe *p_safe .log ()).sum (dim =-1 )
            ent =ent /torch .log (torch .tensor (float (max (int (Pab .shape [-1 ]),2 )),device =ent .device ))
            active_flat =gate_tok .view (int (gate_tok .shape [0 ]),-1 ).to (dtype =diag_vec .dtype )
            den =active_flat .sum ().clamp_min (1e-6 )
            diag_mass_t_mean =(diag_vec *active_flat ).sum ()/den 
            entropy_t_mean =(ent *active_flat ).sum ()/den 

        diag ={
        "L_round_trip_consistency":Lcyc .detach (),
        "L_transport_lw":Llw .detach (),
        "L_transport_align":Lalign .detach (),
        "transport_conf_mean":transport_conf_mean ,
        "diag_mass_T_mean":diag_mass_t_mean .detach (),
        "entropy_T_mean":entropy_t_mean .detach (),
        "active_tok_cnt_mean":active_tok_cnt_mean ,
        **d ,
        }
        return loss ,diag 
