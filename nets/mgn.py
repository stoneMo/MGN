import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from nets.grouping import ModalityTrans


class MGN_Net(nn.Module):

    def __init__(self, args):
        super(MGN_Net, self).__init__()

        self.fc_a =  nn.Linear(128, args.dim)
        self.fc_v = nn.Linear(2048, args.dim)
        self.fc_st = nn.Linear(512, args.dim)
        self.fc_fusion = nn.Linear(args.dim * 2, args.dim)

        # hard or soft assignment
        self.unimodal_assgin = args.unimodal_assign
        self.crossmodal_assgin = args.crossmodal_assign

        unimodal_hard_assignment = True if args.unimodal_assign == 'hard' else False
        crossmodal_hard_assignment = True if args.crossmodal_assign == 'hard' else False

        # learnable tokens
        self.audio_token = nn.Parameter(torch.zeros(25, args.dim))
        self.visual_token = nn.Parameter(torch.zeros(25, args.dim))

        # class-aware uni-modal grouping
        self.audio_cug = ModalityTrans(
                            args.dim,
                            depth=args.depth_aud,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=True
                        )

        self.visual_cug = ModalityTrans(
                            args.dim,
                            depth=args.depth_vis,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=False
                        )

        # modality cross-modal grouping
        self.av_mcg = ModalityTrans(
                            args.dim,
                            depth=args.depth_av,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=crossmodal_hard_assignment,
                            use_han=False                        
                        )

        # prediction
        self.fc_prob = nn.Linear(args.dim, 1)
        self.fc_prob_a = nn.Linear(args.dim, 1)
        self.fc_prob_v = nn.Linear(args.dim, 1)

        self.fc_cls = nn.Linear(args.dim, 25)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, audio, visual, visual_st):

        x1_0 = self.fc_a(audio)

        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        vid_st = self.fc_st(visual_st)
        x2_0 = torch.cat((vid_s, vid_st), dim=-1)
        x2_0 = self.fc_fusion(x2_0)

        # visual uni-modal grouping
        x2, attn_visual_dict, x2_attn = self.visual_cug(x2_0, self.visual_token, return_attn=True)

        # audio uni-modal grouping
        x1, attn_audio_dict, x1_attn = self.audio_cug(x1_0, self.audio_token, x2_0, return_attn=True)

        # modality-aware cross-modal grouping
        x, attn_av_dict, _ = self.av_mcg(x1, x2, return_attn=True)

        # prediction
        av_prob = torch.sigmoid(self.fc_prob(x))                                # [B, 25, 1]
        global_prob = av_prob.sum(dim=-1)                                       # [B, 25]

        # cls token prediction
        aud_cls_prob = self.fc_cls(self.audio_token)                            # [25, 25]
        vis_cls_prob = self.fc_cls(self.visual_token)                           # [25, 25]

        # attentions
        attn_audio = attn_audio_dict[self.unimodal_assgin].squeeze(1)                    # [25, 10]
        attn_visual = attn_visual_dict[self.unimodal_assgin].squeeze(1)                  # [25, 10]

        # audio prediction
        a_prob = torch.sigmoid(self.fc_prob_a(x1))                                # [B, 25, 1]
        a_frame_prob = (a_prob * attn_audio).permute(0, 2, 1)                     # [B, 10, 25]
        a_prob = a_prob.sum(dim=-1)                                               # [B, 25]

        # visual prediction
        v_prob = torch.sigmoid(self.fc_prob_v(x2))                                # [B, 25, 1]
        v_frame_prob = (v_prob * attn_visual).permute(0, 2, 1)                    # [B, 10, 25]
        v_prob = v_prob.sum(dim=-1)                                               # [B, 25]

        return aud_cls_prob, vis_cls_prob, global_prob, a_prob, v_prob, a_frame_prob, v_frame_prob

