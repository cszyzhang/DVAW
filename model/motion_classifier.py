from __future__ import annotations 

import torch 
import torch .nn as nn 
import torch .nn .functional as F 
from torch .nn import init 
from torch .nn .parameter import Parameter 


class SpatialAttention (nn .Module ):
    def __init__ (self ,in_channels ,norm_layer ,up_kwargs ):
        super (SpatialAttention ,self ).__init__ ()
        self .in_channels =in_channels 
        self .norm_layer =norm_layer 
        self ._up_kwargs =up_kwargs 


        self .pool_short =nn .AdaptiveAvgPool2d ((1 ,None ))
        self .pool_medium =nn .AdaptiveAvgPool2d ((2 ,None ))
        self .pool_long =nn .AdaptiveAvgPool2d ((4 ,None ))


        self .conv_short =nn .Sequential (
        nn .Conv2d (in_channels ,in_channels ,(1 ,3 ),1 ,padding =(0 ,1 ),bias =False ),
        norm_layer (in_channels ),
        nn .ReLU (True ),
        )
        self .conv_medium =nn .Sequential (
        nn .Conv2d (in_channels ,in_channels ,(2 ,3 ),1 ,padding =(0 ,1 ),bias =False ),
        norm_layer (in_channels ),
        nn .ReLU (True ),
        )
        self .conv_long =nn .Sequential (
        nn .Conv2d (in_channels ,in_channels ,(4 ,3 ),1 ,padding =(0 ,1 ),bias =False ),
        norm_layer (in_channels ),
        nn .ReLU (True ),
        )

    def forward (self ,x ):
        _ ,_ ,h ,w =x .size ()
        x_pooled_short =F .interpolate (
        self .conv_short (self .pool_short (x )),size =(h ,w ),mode ="bilinear",align_corners =True 
        )
        x_pooled_medium =F .interpolate (
        self .conv_medium (self .pool_medium (x )),size =(h ,w ),mode ="bilinear",align_corners =True 
        )
        x_pooled_long =F .interpolate (self .conv_long (self .pool_long (x )),size =(h ,w ),mode ="bilinear",align_corners =True )
        x_concatenated =F .relu_ (x_pooled_short +x_pooled_medium +x_pooled_long )
        return F .relu_ (x_concatenated +x )


class ChannelAttention (nn .Module ):
    def __init__ (self ,channel =512 ,G =8 ):
        super ().__init__ ()
        self .G =G 
        self .channel =channel 
        self .avg_pool =nn .AdaptiveAvgPool2d (1 )
        self .gn =nn .GroupNorm (channel //(2 *G ),channel //(2 *G ))
        self .cweight =Parameter (torch .zeros (1 ,channel //(2 *G ),1 ,1 ))
        self .cbias =Parameter (torch .ones (1 ,channel //(2 *G ),1 ,1 ))
        self .sigmoid =nn .Sigmoid ()
        self .spatialattn =SpatialAttention (4 ,nn .BatchNorm2d ,{"mode":"bilinear","align_corners":True })
        self .max_pool =nn .AdaptiveMaxPool2d (1 )

    def init_weights (self ):
        for m in self .modules ():
            if isinstance (m ,nn .Conv2d ):
                init .kaiming_normal_ (m .weight ,mode ="fan_out")
                if m .bias is not None :
                    init .constant_ (m .bias ,0 )
            elif isinstance (m ,nn .BatchNorm2d ):
                init .constant_ (m .weight ,1 )
                init .constant_ (m .bias ,0 )
            elif isinstance (m ,nn .Linear ):
                init .normal_ (m .weight ,std =0.001 )
                if m .bias is not None :
                    init .constant_ (m .bias ,0 )

    @staticmethod 
    def channel_shuffle (x ,groups ):
        b ,c ,h ,w =x .shape 
        x =x .reshape (b ,groups ,-1 ,h ,w )
        x =x .permute (0 ,2 ,1 ,3 ,4 )


        x =x .reshape (b ,-1 ,h ,w )
        return x 

    def forward (self ,x ):
        b ,c ,h ,w =x .size ()


        x =x .view (b *self .G ,-1 ,h ,w )


        x_0 ,x_1 =x .chunk (2 ,dim =1 )


        x_channel0 =self .avg_pool (x_0 )
        x_channel1 =self .max_pool (x_0 )
        x_channel =x_channel0 +x_channel1 
        x_channel =self .cweight *x_channel +self .cbias 
        x_channel =x_0 *self .sigmoid (x_channel )

        x_spatial =self .spatialattn (x_1 )
        x_spatial =x_1 *self .sigmoid (x_spatial )


        out =torch .cat ([x_channel ,x_spatial ],dim =1 )
        out =out .contiguous ().view (b ,-1 ,h ,w )


        out =self .channel_shuffle (out ,2 )
        return out 


class MotionClassifier (nn .Module ):
    def __init__ (self ,in_channels =3 ,out_channels =3 ,num_classes =None ):
        super (MotionClassifier ,self ).__init__ ()
        if num_classes is not None :
            out_channels =int (num_classes )
        self .conv1 =nn .Conv2d (in_channels ,out_channels =3 ,kernel_size =3 ,padding =2 )
        self .conv2 =nn .Conv2d (in_channels ,out_channels =5 ,kernel_size =3 ,padding =2 )
        self .conv3 =nn .Conv2d (in_channels ,out_channels =8 ,kernel_size =3 ,padding =2 )
        self .relu =nn .ReLU ()
        self .bn1 =nn .BatchNorm2d (3 )
        self .bn2 =nn .BatchNorm2d (5 )
        self .bn3 =nn .BatchNorm2d (8 )
        self .maxpool =nn .MaxPool2d (kernel_size =3 ,stride =3 ,padding =1 )
        self .dropout =nn .Dropout (p =0.5 )
        self .gap =nn .AdaptiveAvgPool2d ((1 ,1 ))
        self .fc =nn .Linear (in_features =16 ,out_features =out_channels )
        self .channelattn =ChannelAttention (channel =16 ,G =2 )

    def forward (self ,x ,return_features :bool =False ):
        x1 =self .conv1 (x )
        x1 =self .relu (x1 )
        x1 =self .bn1 (x1 )
        x1 =self .maxpool (x1 )
        x1 =self .dropout (x1 )
        x2 =self .conv2 (x )
        x2 =self .relu (x2 )
        x2 =self .bn2 (x2 )
        x2 =self .maxpool (x2 )
        x2 =self .dropout (x2 )
        x3 =self .conv3 (x )
        x3 =self .relu (x3 )
        x3 =self .bn3 (x3 )
        x3 =self .maxpool (x3 )
        x3 =self .dropout (x3 )
        x =torch .cat ((x1 ,x2 ,x3 ),1 )

        xraw =x 
        x =self .channelattn (x )
        feat_map =xraw +x 

        feat_vec =self .gap (feat_map ).flatten (1 )
        logits =self .fc (feat_vec )
        if bool (return_features ):
            return logits ,feat_map ,feat_vec 
        return logits 
