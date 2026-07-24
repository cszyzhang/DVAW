from __future__ import annotations 

import torch 


def propagate_trust_tokens (
trust_tok :torch .Tensor ,
T :torch .Tensor ,
active_mask_tok :torch .Tensor ,
*,
alpha :float =0.5 ,
K :int =2 ,
)->torch .Tensor :

    if trust_tok .ndim !=4 or int (trust_tok .shape [1 ])!=1 :
        raise ValueError (f"trust_tok must be [B,1,H,W], got {tuple(trust_tok.shape)}")
    if active_mask_tok .ndim !=4 or int (active_mask_tok .shape [1 ])!=1 :
        raise ValueError (f"active_mask_tok must be [B,1,H,W], got {tuple(active_mask_tok.shape)}")
    if T .ndim !=3 :
        raise ValueError (f"T must be [B,N,N], got {tuple(T.shape)}")

    B ,_ ,H ,W =trust_tok .shape 
    N =int (H )*int (W )
    if int (T .shape [0 ])!=int (B )or int (T .shape [1 ])!=int (N )or int (T .shape [2 ])!=int (N ):
        raise ValueError (
        f"T shape mismatch: trust_tok N={N}, expected [B={B},{N},{N}], got {tuple(T.shape)}"
        )

    a =float (alpha )
    if a <0.0 :
        a =0.0 
    if a >1.0 :
        a =1.0 
    k_steps =max (int (K ),0 )

    w0 =trust_tok .view (int (B ),int (N ),1 )
    wk =w0 
    for _ in range (k_steps ):
        wk =(1.0 -a )*w0 +a *torch .bmm (T ,wk )

    active_rows =active_mask_tok .view (int (B ),int (N )).to (dtype =torch .bool )
    out =w0 .clone ()
    out [active_rows ]=wk [active_rows ]
    return out .view (int (B ),1 ,int (H ),int (W ))

