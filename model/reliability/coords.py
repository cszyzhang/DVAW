from __future__ import annotations 

import torch 


def make_grid_norm (h :int ,w :int ,*,device =None ,dtype =torch .float32 )->torch .Tensor :

    ys =torch .linspace (-1.0 ,1.0 ,int (h ),device =device ,dtype =dtype )
    xs =torch .linspace (-1.0 ,1.0 ,int (w ),device =device ,dtype =dtype )
    yy ,xx =torch .meshgrid (ys ,xs ,indexing ="ij")
    return torch .stack ([xx ,yy ],dim =-1 )[None ,...]


def make_grid_pixel (h :int ,w :int ,*,device =None ,dtype =torch .float32 )->torch .Tensor :

    ys =torch .arange (int (h ),device =device ,dtype =dtype )+0.5 
    xs =torch .arange (int (w ),device =device ,dtype =dtype )+0.5 
    yy ,xx =torch .meshgrid (ys ,xs ,indexing ="ij")
    return torch .stack ([xx ,yy ],dim =-1 )


def flatten_hw (x :torch .Tensor )->torch .Tensor :

    b ,c ,h ,w =x .shape 
    return x .permute (0 ,2 ,3 ,1 ).reshape (b ,h *w ,c )


def unflatten_hw (x :torch .Tensor ,h :int ,w :int )->torch .Tensor :

    b ,n ,c =x .shape 
    if int (n )!=int (h )*int (w ):
        raise ValueError (f"Expected N=H*W, got N={int(n)} H={int(h)} W={int(w)}")
    return x .reshape (b ,int (h ),int (w ),c ).permute (0 ,3 ,1 ,2 )

