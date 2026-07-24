from __future__ import annotations 

from typing import List ,Tuple 

import torch 


def build_region_id_map (
partition :str ,
H :int ,
W :int ,
*,
device :torch .device |str |None =None ,
)->tuple [torch .Tensor ,int ,List [Tuple [int ,int ]]]:

    part =str (partition ).strip ().lower ()
    device =device or "cpu"
    rid =torch .zeros ((int (H ),int (W )),dtype =torch .long ,device =device )

    if part in {"quad4","quad","quadrant","quadrants"}:
        h2 ,w2 =int (H )//2 ,int (W )//2 
        rid [:h2 ,:w2 ]=0 
        rid [:h2 ,w2 :]=1 
        rid [h2 :,:w2 ]=2 
        rid [h2 :,w2 :]=3 
        return rid ,4 ,[(0 ,1 ),(2 ,3 )]

    if part in {"vstripes3","vstripes","vstripe3"}:
        w1 =int (W )//3 
        w2 =2 *int (W )//3 
        rid [:,:w1 ]=0 
        rid [:,w1 :w2 ]=1 
        rid [:,w2 :]=2 
        return rid ,3 ,[(0 ,2 )]

    if part in {"grid3x3","grid_3x3","grid9"}:
        hs =[0 ,int (H )//3 ,2 *int (H )//3 ,int (H )]
        ws =[0 ,int (W )//3 ,2 *int (W )//3 ,int (W )]
        k =0 
        for i in range (3 ):
            for j in range (3 ):
                rid [hs [i ]:hs [i +1 ],ws [j ]:ws [j +1 ]]=k 
                k +=1 


        return rid ,9 ,[(0 ,2 ),(3 ,5 ),(6 ,8 )]

    if part in {"grid3x3_15","grid3x3-15","grid5x3_merge","grid5x3_to_3x3"}:



        hs =[0 ,int (H )//3 ,2 *int (H )//3 ,int (H )]
        ws =[0 ,2 *int (W )//5 ,3 *int (W )//5 ,int (W )]
        k =0 
        for i in range (3 ):
            for j in range (3 ):
                rid [hs [i ]:hs [i +1 ],ws [j ]:ws [j +1 ]]=k 
                k +=1 
        return rid ,9 ,[(0 ,2 ),(3 ,5 ),(6 ,8 )]

    if part in {"au6_fixed","au6","au6fixed"}:

        rid .fill_ (0 )

        def box (x0 :float ,y0 :float ,x1 :float ,y1 :float ,val :int )->None :
            xx0 ,xx1 =int (x0 *int (W )),int (x1 *int (W ))
            yy0 ,yy1 =int (y0 *int (H )),int (y1 *int (H ))
            rid [yy0 :yy1 ,xx0 :xx1 ]=int (val )


        box (0.30 ,0.70 ,0.70 ,0.95 ,5 )

        box (0.10 ,0.15 ,0.45 ,0.45 ,1 )
        box (0.55 ,0.15 ,0.90 ,0.45 ,2 )

        box (0.10 ,0.45 ,0.45 ,0.70 ,3 )
        box (0.55 ,0.45 ,0.90 ,0.70 ,4 )

        return rid ,6 ,[(1 ,2 ),(3 ,4 )]


    return build_region_id_map ("quad4",H ,W ,device =device )


def region_mean (x :torch .Tensor ,rid :torch .Tensor ,num_regions :int )->tuple [torch .Tensor ,torch .Tensor ]:

    if x .ndim !=4 :
        raise ValueError (f"x must be (B,C,H,W), got {tuple(x.shape)}")
    if rid .ndim !=2 :
        raise ValueError (f"rid must be (H,W), got {tuple(rid.shape)}")
    B ,C ,H ,W =x .shape 
    if tuple (rid .shape )!=(int (H ),int (W )):
        raise ValueError (f"rid shape must match x spatial: rid={tuple(rid.shape)} x={tuple(x.shape)}")
    R =int (num_regions )
    if R <=0 :
        raise ValueError (f"num_regions must be > 0, got {num_regions}")

    rid_flat =rid .reshape (-1 ).to (device =x .device ,dtype =torch .long )
    x_flat =x .reshape (int (B ),int (C ),-1 )

    out =torch .zeros ((int (B ),int (C ),int (R )),device =x .device ,dtype =x .dtype )
    rid_bc =rid_flat .view (1 ,1 ,-1 ).expand (int (B ),int (C ),-1 )
    out .scatter_add_ (2 ,rid_bc ,x_flat )

    ones =torch .ones ((1 ,1 ,rid_flat .numel ()),device =x .device ,dtype =x .dtype )
    cnt =torch .zeros ((1 ,1 ,int (R )),device =x .device ,dtype =x .dtype )
    cnt .scatter_add_ (2 ,rid_flat .view (1 ,1 ,-1 ),ones )

    out =out /(cnt +1e-6 )
    out =out .permute (0 ,2 ,1 ).contiguous ()

    counts_ratio =(cnt .reshape (-1 )/float (rid_flat .numel ())).detach ()
    return out ,counts_ratio 
