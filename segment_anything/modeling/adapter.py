
import torch
import torch.nn as nn

class Spatial(nn.Module):
    def __init__(self, embed_dim:int) -> None:
        super().__init__()

        self.spatial = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU(),
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def forward(self, x):
        return self.spatial(x)
    

class Adapter_Channel_Spatial_Attention(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=0.25, norm_layer = nn.LayerNorm, skip_connect=True):
        super().__init__()
        
        self.skip_connect = skip_connect
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = norm_layer(embed_dim)
        
        # for channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim, bias=False),
        )

        self.sigmoid =  nn.Sigmoid()
        
        # for spatial attention
        self.spatial_kernel = 7
        
        self.conv = nn.Sequential( 
            nn.Conv2d(2,1,kernel_size=self.spatial_kernel,stride = 1,padding=(self.spatial_kernel-1)//2,bias=False),
            nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True)               
        )
            
        self.spatial = Spatial(embed_dim=embed_dim)
        
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
       
                
    def forward(self, x):
        
        #x -> （B, H, W, C）-> （B, C, H, W）
        x = x.permute(0,3,1,2)  
        
        B, C, _, _ = x.size()
    
        # channel attention

        avg_out = self.mlp(self.avg_pool(x).view(B,C))
        max_out = self.mlp(self.max_pool(x).view(B,C))
        
        channel_attn = self.sigmoid(avg_out + max_out)
        
        x_channel = channel_attn.view(B,C,1,1) * x

        # spatial attention
        spatial_attn = self.sigmoid(
            self.conv(
                    torch.cat( (torch.max(x_channel,1)[0].unsqueeze(1), torch.mean(x_channel,1).unsqueeze(1)), dim=1 )
                )
            )
        
        x_spatial = spatial_attn * x_channel
 
        x_spatial = self.spatial(x_spatial)
            
        if self.skip_connect:
            x = x + x_spatial
        else:
            x = x_spatial
            
        # (B, C, H, W） -> (B, H, W, C)
        x = x.permute(0,2,3,1) 
        
        return self.norm(x)
      