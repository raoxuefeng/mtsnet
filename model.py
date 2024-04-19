from segment_anything.modeling.image_encoder import ImageEncoderViT 
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling import TwoWayTransformer
from segment_anything.modeling.mask_decoder import MaskDecoder as MaskDecoder
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple
import clip
from GFM import GatedFusion
from LTEN import LTEN

class MTSNet(nn.Module):
    def __init__(
        self, 
        config, 
        label_text_dict = {},
        device = 'cuda:0',
        ):
        super().__init__()
        
        print(config)
        
        self.device = device
        self.img_size = config['sam']['img_size']
        self.num_classes = config['sam']['num_classes']
        self.adapter_train = config['sam']['adapter_train']
        self.label_dict = label_text_dict
        self.im_type = config['img_type']

        ## define hyperparameters, can be taken to a config later
        prompt_embed_dim=256
        image_embedding_size=16
        mask_in_chans=16

        
        ## define the  LTEN and GFM
        self.lten  = LTEN()

        self.gfm = GatedFusion(input_dim=256)
        
        ## define the components of MTG-SAM
        self.sam_encoder = ImageEncoderViT(img_size=self.img_size,adapter_train=self.adapter_train)

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=768
        )
        
        self.prompt_encoder=PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(self.img_size, self.img_size),
        mask_in_chans=mask_in_chans,
        )

        self.clip_model, _  = clip.load("ViT-B/32", device=device)

        #define text prompt layers
        self.Text_Embedding_Affine = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256)
            )
           

        ## initialize sam with pretrained weights
        sam_ckpt='/home/guest/RaoXuefeng/biastuning/ckpt/sam_hq_vit_b.pth' # SAM HQ

        sam_state_dict = torch.load(sam_ckpt)
        for k in list(sam_state_dict.keys()):
            if self.img_size!=1024:
                #pos embed can be loaded only when image size is 1024
                if "pos_embed" in k:
                    full_matrix = sam_state_dict.pop(k)
                    adapted_matrix = nn.functional.adaptive_avg_pool2d(full_matrix.permute(0,3,1,2), (self.sam_encoder.pos_embed.shape[1], self.sam_encoder.pos_embed.shape[2]))
                    adapted_matrix = adapted_matrix.permute(0,2,3,1)
                    sam_state_dict[k] = adapted_matrix
            elif "image_encoder." in k:
                sam_state_dict[k[14:]] = sam_state_dict.pop(k)
            elif "prompt_encoder." in k:
                sam_state_dict[k[15:]] = sam_state_dict.pop(k)
            elif "mask_decoder." in k:
                sam_state_dict[k[13:]] = sam_state_dict.pop(k)
        
    
        self.sam_encoder.load_state_dict(sam_state_dict,strict=False)
        self.prompt_encoder.load_state_dict(sam_state_dict, strict=False)
        self.mask_decoder.load_state_dict(sam_state_dict,strict=False)

    def forward(self, x_img, x_text):
        B, C, H, W = x_img.shape
        x_text = list(x_text)
        

        image_embeddings, interm_embeddings = self.sam_encoder(x_img)

        lten_features = self.lten(x_img)

        image_embeddings = self.gfm(image_embeddings,lten_features)


        text_inputs = (clip.tokenize(x_text)).to(self.device)
    
        text_features = self.clip_model.encode_text(text_inputs)
    
        text_features_affine = self.Text_Embedding_Affine(text_features.float())

        text_features_affine = text_features_affine.unsqueeze(1)


        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
               
        sparse_embeddings = sparse_embeddings.to(self.device).repeat(B,1,1)

        sparse_embeddings = torch.cat(
            [sparse_embeddings,text_features_affine], dim=1)


        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
                use_gsam=False
            )
        
        high_res_masks = self.postprocess_masks(low_res_masks, (self.img_size,self.img_size), (self.img_size,self.img_size))
        
        return high_res_masks

    def get_image_embeddings(self, x_img):
        with torch.no_grad():
            B, C, H, W = x_img.shape
            image_embeddings,interm_embeddings = self.sam_encoder(x_img)

            lten_features = self.lten(x_img)

            image_embeddings = self.gfm(image_embeddings,lten_features)
            
            return image_embeddings,interm_embeddings

    def get_masks_for_multiple_labels(self, img_interm_embeds, x_text):
        '''
        img_embeds - image embeddings obtained from get_imgae_embeddings function
        xtext - text prompts. image encoder wont be run and only the decoder will be run for each of these
        '''
        img_embeds,interm_embeds = img_interm_embeds
        B = img_embeds.shape[0]
        with torch.no_grad():
            x_text = list(x_text)

            text_inputs = (clip.tokenize(x_text)).to(self.device)

            text_features = self.clip_model.encode_text(text_inputs)
            
            text_features_affine = self.Text_Embedding_Affine(text_features.float())
            
            text_features_affine = text_features_affine.unsqueeze(1)


            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
            
            sparse_embeddings = sparse_embeddings.to(self.device).repeat(B,1,1)

            sparse_embeddings = torch.cat(
                [sparse_embeddings,text_features_affine], dim=1)
            

            low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=img_embeds,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    hq_token_only=True,
                    interm_embeddings=interm_embeds,
                    use_gsam=False
                )
            
            high_res_masks = self.postprocess_masks(low_res_masks, (self.img_size,self.img_size), (self.img_size,self.img_size))
            
            return high_res_masks


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.sam_encoder.img_size, self.sam_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        masks = torch.sigmoid(masks)
        return masks.squeeze(1)