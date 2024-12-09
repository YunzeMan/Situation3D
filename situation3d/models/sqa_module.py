import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.join(os.getcwd(), 'lib'))
from lib.config import CONF
from models.mcan_sqa_module import MCAN_ED, AttFlat, LayerNorm, SA, SGA
from models.sep_lang_module_bert import LangModule
import numpy as np
from .mink_unet import DisNet


def quaternions_to_rotation_matrices(quaternions): # scipy style, assuming [batch_size, 4], [x,y,z,w]
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    # Preallocate a tensor for the batch of rotation matrices
    batch_size = quaternions.shape[0]
    rotation_matrices = torch.zeros((batch_size, 3, 3), device=quaternions.device, dtype=quaternions.dtype)
    # Compute each element of the rotation matrix
    rotation_matrices[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    rotation_matrices[:, 0, 1] = 2 * (x * y - z * w)
    rotation_matrices[:, 0, 2] = 2 * (x * z + y * w)

    rotation_matrices[:, 1, 0] = 2 * (x * y + z * w)
    rotation_matrices[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    rotation_matrices[:, 1, 2] = 2 * (y * z - x * w)

    rotation_matrices[:, 2, 0] = 2 * (x * z - y * w)
    rotation_matrices[:, 2, 1] = 2 * (y * z + x * w)
    rotation_matrices[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    return rotation_matrices


def batch_rotation_vector_to_matrix(batch_rot_vec):
    # Compute the magnitude (theta) of each rotation vector
    theta = torch.norm(batch_rot_vec, dim=1, keepdim=True)

    # Handle the case of very small rotations
    close_to_zero = theta < 1e-6
    identity_matrices = torch.eye(3).repeat(batch_rot_vec.shape[0], 1, 1).to(batch_rot_vec.device)
    if close_to_zero.all():
        return identity_matrices

    # Normalize the rotation vectors to get the unit axis of rotation
    u = batch_rot_vec / theta

    # Construct the skew-symmetric matrices for each rotation vector
    zeros = torch.zeros(batch_rot_vec.shape[0], 1).to(batch_rot_vec.device)

    K = torch.cat([
        torch.cat([zeros, -u[:, 2:3], u[:, 1:2]], dim=1),
        torch.cat([u[:, 2:3], zeros, -u[:, 0:1]], dim=1),
        torch.cat([-u[:, 1:2], u[:, 0:1], zeros], dim=1)
    ], dim=1).view(-1, 3, 3)

    # Expand theta for broadcasting
    theta = theta.view(-1, 1, 1)

    # Compute the rotation matrices using Rodrigues' formula
    R = identity_matrices + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

    # Handle the case where rotations are close to zero
    R[close_to_zero.squeeze(), :, :] = identity_matrices[close_to_zero.squeeze(), :, :]

    return R


def create_sinusoidal_embeddings(n_pos, dim, out):
    with torch.no_grad():
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False 


class Embeddings(nn.Module):
    def __init__(
        self, d_model, language_len, vision_len, dropout, sinusoidal_pos_embds
    ):
        super().__init__()
        max_position_embeddings = 2*language_len + vision_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                out=self.position_embeddings.weight,
            )
        self.modality_embedding = nn.Embedding(3, d_model)
        self.language_len = language_len
        self.vision_len = vision_len
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(
            embeddings[:, :, 0]
        )  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)
        modality_embeddings = self.modality_embedding(
            torch.tensor(
                [0] * self.language_len + [1] * self.language_len + [2] * self.vision_len, dtype=torch.long
            ).to(embeddings.device)
        )
        embeddings = (
            embeddings + position_embeddings + modality_embeddings
        )  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class SIG3D(nn.Module):
    def __init__(self, num_answers,
        situation_loss_tag,
        # qa
        answer_pdrop=0.3,
        mcan_num_layers=2,
        mcan_num_heads=8,
        mcan_pdrop=0.1,
        mcan_flat_mlp_size=512,
        mcan_flat_glimpses=1,
        mcan_flat_out_size=1024,
        # lang
        lang_use_bidir=False,
        lang_num_layers=1,
        lang_emb_size=300,
        lang_pdrop=0.1,
        # common
        hidden_size=768,
        # option
        use_answer=True,
        use_bert=True,
        bert_model_name='sentence-transformers/all-mpnet-base-v2',
        freeze_bert=False,
        finetune_bert_last_layer=True,
        finetune_bert_full=False
    ):
        super().__init__()

        # Option
        self.use_answer = use_answer
        self.situation_loss_tag = situation_loss_tag
        self.use_bert = use_bert
        lang_size = hidden_size * (1 + lang_use_bidir)
        # Language encoding
        self.lang_net = LangModule(use_bidir=lang_use_bidir, num_layers=lang_num_layers,
                                    emb_size=lang_emb_size, hidden_size=hidden_size, pdrop=lang_pdrop,
                                    use_bert=use_bert, bert_model_name=bert_model_name, freeze_bert=freeze_bert, 
                                    finetune_bert_last_layer=finetune_bert_last_layer, finetune_bert_full=finetune_bert_full)
        # Vision encoding
        self.openscene_net = DisNet(CONF.OPENSCENE.feature_2d_extractor)

        # Feature projection
        self.lang_feat_linear = nn.Sequential(
            nn.Linear(lang_size, hidden_size),
            nn.GELU()
        )

        self.s_feat_linear = nn.Sequential(
            nn.Linear(lang_size, hidden_size),
            nn.GELU()
        )
        self.q_feat_linear = nn.Sequential(
            nn.Linear(lang_size, hidden_size),
            nn.GELU()
        )
        self.scene_feat_linear = nn.Sequential(
            nn.Linear(CONF.OPENSCENE.feat_dim, hidden_size),
            nn.GELU()
        )

        self.enc_list_s = nn.ModuleList([SA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])
        self.enc_list_q = nn.ModuleList([SA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])
        self.dec_list = nn.ModuleList([SGA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])
        self.dec_list_2 = nn.ModuleList([SGA(hidden_size, mcan_num_heads, mcan_pdrop) for _ in range(mcan_num_layers)])
        # --------------------------------------------

        # Language classifier
        if '__quat__' in situation_loss_tag:
            if '__class__' in situation_loss_tag:
                self.aux_cls = nn.Sequential(
                    nn.Linear(2*mcan_flat_out_size, hidden_size), # 1024 -> 768
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, 5) # 768 -> 5
                )
            else:
                self.aux_reg = nn.Sequential(
                    nn.Linear(2*mcan_flat_out_size, hidden_size), # 1024 -> 768
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, 7) # 768 -> 7
                )
        elif '__angle__' in situation_loss_tag:
            if '__class__' in situation_loss_tag:
                self.aux_cls = nn.Sequential(
                    nn.Linear(2*mcan_flat_out_size, hidden_size), # 1024 -> 768
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, 3) # 768 -> 3
                )
            else:
                self.aux_reg = nn.Sequential(
                    nn.Linear(2*mcan_flat_out_size, hidden_size), # 1024 -> 768
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, 5) # 768 -> 5
                )
        elif '__6d__' in situation_loss_tag:
            if '__class__' in situation_loss_tag:
                self.aux_cls = nn.Sequential(
                    nn.Linear(2*mcan_flat_out_size, hidden_size), # 1024 -> 768
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, 7) # 768 -> 7
                )
            else:
                self.aux_reg = nn.Sequential(
                    nn.Linear(2*mcan_flat_out_size, hidden_size), # 1024 -> 768
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, 9) # 768 -> 9
                )
        else:
            raise NotImplementedError

        # QA head
        self.attflat_visual = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.attflat_s = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        self.attflat_q = AttFlat(hidden_size, mcan_flat_mlp_size, mcan_flat_glimpses, mcan_flat_out_size, 0.1)
        if CONF.TRAIN.NO3D:
            self.answer_cls = nn.Sequential(
                    nn.Linear(2*mcan_flat_out_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(answer_pdrop),
                    nn.Linear(hidden_size, num_answers)
            )
        else:
            self.answer_cls = nn.Sequential(
                    nn.Linear(3*mcan_flat_out_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(answer_pdrop),
                    nn.Linear(hidden_size, num_answers)
            )
        self.position_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Position likelihood should be between 0 and 1
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6)  # 6D rotation representation
        )


        self.position = Embeddings(768, 100, 256, 0.1, True)
        self.pos_embed = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, 256) # 256 is the hidden size from the openscene backbone
        )  


    def forward(self, data_dict):
        data_dict = self.lang_net(data_dict)
        data_dict = self.openscene_net(data_dict)

        # unpack outputs from question encoding branch
        s_feat = data_dict["s_out"]  # torch.Size([32, X(48), 256])  BERT: (torch.Size([32, 100, 768]))
        q_feat = data_dict["q_out"]  # torch.Size([32, Y(16), 256])  BERT: (torch.Size([32, 100, 768]))
        s_mask = data_dict["s_mask"] # torch.Size([32, X(48)])   BERT: (torch.Size([32, 100]))
        q_mask = data_dict["q_mask"] # torch.Size([32, Y(16)])   BERT: (torch.Size([32, 100]))

        # unpack outputs from vision branch
        if not CONF.TRAIN.NO3D:
            scene_feat_original = data_dict['feat_bottleneck'] # [2357, 4]
            list_of_coords, list_of_featurs = scene_feat_original.decomposed_coordinates_and_features
            scene_feat = []
            scene_positions = []
            for batch_idx, (coords, feats) in enumerate(zip(list_of_coords, list_of_featurs)):
                reduced_coords = coords[:, [0, 1]]
                unique_coords, indices = reduced_coords.unique(dim=0, return_inverse=True)
                reduced_feats = torch.zeros(unique_coords.size(0), feats.size(1), device=feats.device)
                reduced_feats = reduced_feats.scatter_reduce_(0, indices.unsqueeze(-1).expand_as(feats), feats, reduce='mean') # [514, 256]
                # Sample feature points from the unique coordinates
                if CONF.OPENSCENE.num_points < unique_coords.size(0):
                    sampled_indices = torch.randperm(unique_coords.size(0))[:CONF.OPENSCENE.num_points]
                else:
                    unique_indices = torch.randperm(unique_coords.size(0))
                    duplicate_indices = torch.randint(0, unique_coords.size(0), (CONF.OPENSCENE.num_points - unique_coords.size(0),))
                    sampled_indices = torch.cat([unique_indices, duplicate_indices])
                sampled_unique_coords = unique_coords[sampled_indices] # [256, 2]
                sampled_scene_feats = reduced_feats[sampled_indices] # [256, 256]
                positions = (sampled_unique_coords + torch.tensor(scene_feat_original.tensor_stride[0:2], device=feats.device)/2) * CONF.OPENSCENE.voxel_size # [256, 2] actual coordinates in meters
                # Add the positional embedding to the sampled_reduced_feats
                # sampled_scene_feats = torch.cat((sampled_scene_feats, positions), dim=1) # [256, 258]
                scene_feat.append(sampled_scene_feats.unsqueeze(0)) # [1, 256, 256]
                scene_positions.append(positions.unsqueeze(0)) # [1, 256, 2]
            scene_feat = torch.cat(scene_feat, dim=0) # [32, 256, 256]
            scene_positions = torch.cat(scene_positions, dim=0) # [32, 256, 2]
            data_dict['scene_positions'] = scene_positions
            scene_pos_embeddings = self.pos_embed(scene_positions) # [32, 256, 256]
            data_dict["att_feat_pre"] = scene_feat # torch.Size([32, 256, 256])
            scene_feat = scene_feat + scene_pos_embeddings # [32, 256, 256]
        if s_mask.dim() == 2:
            s_mask = s_mask.unsqueeze(1).unsqueeze(2) # batch, 1, 1, num_words(48)
        if q_mask.dim() == 2:
            q_mask = q_mask.unsqueeze(1).unsqueeze(2) # batch, 1, 1, num_words(16)
        
        # --------- Add situational PE to the scene_feat ---------
        gt_translation = data_dict["auxiliary_task"][:, :3]     # [32, 3]
        gt_rotation = data_dict["auxiliary_task"][:, 3:]        # [32, 4]

        # calculate among scene_positions. Only compare with the first 2 dimensions
        gt_translation = gt_translation.unsqueeze(1)[:, :, :2] # [32, 1, 2]
        distance = torch.norm(scene_positions - gt_translation, dim=2) # [32, 256]

        weights = torch.exp(-distance**2 / (2 * 0.16**2)) # [32, 256]
        weights = weights / weights.sum(dim=1, keepdim=True) # [32, 256]
        # save the weights as cross-entropy loss ground truth
        data_dict["auxiliary_task_loc_gt"] = weights
        
        # Pre-process Lanauge & Image Feature
        s_feat = self.lang_feat_linear(s_feat) # torch.Size([32, X(48), 256])   BERT: [32, 100, 768]
        q_feat = self.lang_feat_linear(q_feat) # torch.Size([32, Y(16), 256])   BERT: [32, 100, 768]
        if not CONF.TRAIN.NO3D:
            scene_feat = self.scene_feat_linear(scene_feat) # BS, N_proposal, hidden  [32, 256, 768]

        for enc in self.enc_list_s:
            s_feat = enc(s_feat, s_mask)  # torch.Size([32, X(48), 256])  BERT: [32, 100, 768]
        for enc in self.enc_list_q: 
            q_feat = enc(q_feat, q_mask)  # torch.Size([32, Y(16), 256])  BERT: [32, 100, 768]
        if not CONF.TRAIN.NO3D:
            for dec in self.dec_list:
                scene_feat = dec(scene_feat, s_feat, None, s_mask)  # torch.Size([32, 256, 768])
            for dec in self.dec_list_2:
                scene_feat = dec(scene_feat, q_feat, None, q_mask)  # torch.Size([32, 256, 768])

        data_dict["att_feat_ori"] = scene_feat # torch.Size([32, 256, 768])

        pos_likelihood = self.position_head(scene_feat)  
        rotation_pred = self.rotation_head(scene_feat)
        data_dict["pred_pos_likelihood"] = pos_likelihood.squeeze(-1)  # [32, 256]
        data_dict["pred_rotation"] = rotation_pred  # [32, 256, 6]

        s_feat, data_dict["satt"] = self.attflat_s(
                s_feat,
                s_mask
        )  # torch.Size([32, 512])   torch.Size([32, 48, 1])

        q_feat, data_dict["qatt"] = self.attflat_q(
                q_feat,
                q_mask,
        )  # torch.Size([32, 512])   torch.Size([32, 16, 1])
        if not CONF.TRAIN.NO3D:
            scene_feat, data_dict["oatt"] = self.attflat_visual(
                    scene_feat,
                    None
            )  # torch.Size([32, 512])   torch.Size([32, 256, 1])

        if not CONF.TRAIN.NO3D:
            fuse_feat = torch.cat((s_feat, q_feat, scene_feat), dim=1)  # torch.Size([32, 1536])  512*3
        else:
            fuse_feat = torch.cat((s_feat, q_feat), dim=1)  

        if '__class__' in self.situation_loss_tag:
            data_dict["aux_scores"] = self.aux_cls(scene_feat)  # torch.Size([32, 256, 5]) / torch.Size([32, 256, 7])
        else:
            temp = torch.cat((s_feat, scene_feat), dim=1) # torch.Size([32, 1024])
            data_dict["aux_scores"] = self.aux_reg(temp)  # torch.Size([32, 7]) / torch.Size([32, 5])
        
        if self.use_answer:
            data_dict["answer_scores"] = self.answer_cls(fuse_feat) # batch_size, num_answers [32, 707]

        return data_dict

# test the Embeddings class
if __name__ == '__main__':
    position = Embeddings(768, 100, 256, 0.1, True)
