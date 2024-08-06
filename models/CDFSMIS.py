import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .encoder import Res50Encoder


class AttentionMacthcing(nn.Module):
    def __init__(self, feature_dim=512, seq_len=5000):
        super(AttentionMacthcing, self).__init__()
        self.fc_spt = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_qry = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(seq_len * 2, seq_len // 5),

            nn.ReLU(),
            nn.Linear(seq_len // 5, 2 * seq_len),
        )
        self.sigmoid = nn.Sigmoid()


    def correlation_matrix(self, spt_fg_fts, qry_fg_fts):
        """
        Calculates the correlation matrix between the spatial foreground features and query foreground features.

        Args:
            spt_fg_fts (torch.Tensor): The spatial foreground features. 
            qry_fg_fts (torch.Tensor): The query foreground features. 

        Returns:
            torch.Tensor: The cosine similarity matrix. Shape: [1, 1, N].
        """

        spt_fg_fts = F.normalize(spt_fg_fts, p=2, dim=1)  # shape [1, 512, 900]
        qry_fg_fts = F.normalize(qry_fg_fts, p=2, dim=1)  # shape [1, 512, 900]

        cosine_similarity = torch.sum(spt_fg_fts * qry_fg_fts, dim=1, keepdim=True)  # shape: [1, 1, N]

        return cosine_similarity

    def forward(self, spt_fg_fts, qry_fg_fts, band):
        """
        Args:
            spt_fg_fts (torch.Tensor): Spatial foreground features. 
            qry_fg_fts (torch.Tensor): Query foreground features. 
            band (str): Band type, either 'low', 'high', or other.

        Returns:
            torch.Tensor: Fused tensor. Shape: [1, 512, 5000].
        """

        spt_proj = F.relu(self.fc_spt(spt_fg_fts))  # shape: [1, 512, 900]
        qry_proj = F.relu(self.fc_qry(qry_fg_fts))  # shape: [1, 512, 900]

        similarity_matrix = self.sigmoid(self.correlation_matrix(spt_fg_fts, qry_fg_fts))
        
        if band == 'low' or band == 'high':
            weighted_spt = (1 - similarity_matrix) * spt_proj  # shape: [1, 512, 900]
            weighted_qry = (1 - similarity_matrix) * qry_proj  # shape: [1, 512, 900]
        else:
            weighted_spt = similarity_matrix * spt_proj  # shape: [1, 512, 900]
            weighted_qry = similarity_matrix * qry_proj  # shape: [1, 512, 900]

        combined = torch.cat((weighted_spt, weighted_qry), dim=2)  # shape: [1, 1024, 900]
        fused_tensor = F.relu(self.fc_fusion(combined))  # shape: [1, 512, 900]

        return fused_tensor

class FAM(nn.Module):
    def __init__(self, feature_dim=512, N=900):
        super(FAM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.attention_matching = AttentionMacthcing(feature_dim, N)
        self.adapt_pooling = nn.AdaptiveAvgPool1d(N)

    def forward(self, spt_fg_fts, qry_fg_fts):
        """
        Forward pass of the FAM module.

        Args:
            spt_fg_fts (list): List of spatial foreground features.
            qry_fg_fts (list): List of query foreground features.

        Returns:
            tuple: A tuple containing the fused low, mid, and high frequency features.
        """
        if qry_fg_fts[0].shape[2] == 0:
            qry_fg_fts[0] = F.pad(qry_fg_fts[0], (0, 1))

        spt_fg_fts = [[self.adapt_pooling(fts) for fts in way] for way in spt_fg_fts]
        qry_fg_fts = [self.adapt_pooling(fts) for fts in qry_fg_fts]

        spt_fg_fts_low, spt_fg_fts_mid, spt_fg_fts_high = self.filter_frequency_bands(spt_fg_fts[0][0], cutoff=0.30)
        qry_fg_fts_low, qry_fg_fts_mid, qry_fg_fts_high = self.filter_frequency_bands(qry_fg_fts[0], cutoff=0.30)

        fused_fts_low = self.attention_matching(spt_fg_fts_low, qry_fg_fts_low, 'low')
        fused_fts_mid = self.attention_matching(spt_fg_fts_mid, qry_fg_fts_mid, 'mid')
        fused_fts_high = self.attention_matching(spt_fg_fts_high, qry_fg_fts_high, 'high')

        return fused_fts_low, fused_fts_mid, fused_fts_high
    


    def reshape_to_square(self, tensor):
        """
        Reshapes a tensor to a square shape.

        Args:
            tensor (torch.Tensor): The input tensor of shape (B, C, N), where B is the batch size,
                C is the number of channels, and N is the number of elements.

        Returns:
            tuple: A tuple containing:
                - square_tensor (torch.Tensor): The reshaped tensor of shape (B, C, side_length, side_length),
                  where side_length is the length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - side_length (int): The length of each side of the square tensor.
                - N (int): The original number of elements in the input tensor.
        """
        B, C, N = tensor.shape
        side_length = int(np.ceil(np.sqrt(N)))
        padded_length = side_length ** 2
        
        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        padded_tensor[:, :, :N] = tensor

        square_tensor = padded_tensor.view(B, C, side_length, side_length)
        
        return square_tensor, side_length, side_length, N
    


    def filter_frequency_bands(self, tensor, cutoff=0.2):
            """
            Filters the input tensor into low, mid, and high frequency bands.

            Args:
                tensor (torch.Tensor): The input tensor to be filtered.
                cutoff (float, optional): The cutoff value for frequency band filtering.

            Returns:
                torch.Tensor: The low frequency band of the input tensor.
                torch.Tensor: The mid frequency band of the input tensor.
                torch.Tensor: The high frequency band of the input tensor.
            """

            tensor = tensor.float()
            tensor, H, W, N = self.reshape_to_square(tensor)
            B, C, _, _ = tensor.shape

            max_radius = np.sqrt((H // 2)**2 + (W // 2)**2)
            low_cutoff = max_radius * cutoff
            high_cutoff = max_radius * (1 - cutoff)

            fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

            def create_filter(shape, low_cutoff, high_cutoff, mode='band', device=self.device):
                rows, cols = shape
                center_row, center_col = rows // 2, cols // 2
                
                y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing='ij')
                distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
                
                mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)
                
                if mode == 'low':
                    mask[distance <= low_cutoff] = 1
                elif mode == 'high':
                    mask[distance >= high_cutoff] = 1
                elif mode == 'band':
                    mask[(distance > low_cutoff) & (distance < high_cutoff)] = 1
                
                return mask

            low_pass_filter = create_filter((H, W), low_cutoff, None, mode='low')[None, None, :, :]
            high_pass_filter = create_filter((H, W), None, high_cutoff, mode='high')[None, None, :, :]
            mid_pass_filter = create_filter((H, W), low_cutoff, high_cutoff, mode='band')[None, None, :, :]

            low_freq_fft = fft_tensor * low_pass_filter
            high_freq_fft = fft_tensor * high_pass_filter
            mid_freq_fft = fft_tensor * mid_pass_filter

            low_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
            high_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
            mid_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real

            low_freq_tensor = low_freq_tensor.view(B, C, H * W)[:, :, :N]
            high_freq_tensor = high_freq_tensor.view(B, C, H * W)[:, :, :N]
            mid_freq_tensor = mid_freq_tensor.view(B, C, H * W)[:, :, :N]

            return low_freq_tensor, mid_freq_tensor, high_freq_tensor
    
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feature, K_feature):

        B, C, N = Q_feature.shape

        Q_feature = Q_feature.permute(0, 2, 1)
        K_feature = K_feature.permute(0, 2, 1)


        Q = self.query(Q_feature)  # shape: [B, N, C]
        K = self.key(K_feature)    # shape: [B, N, C]
        V = self.value(K_feature)  # shape: [B, N, C]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        attention_weights = self.softmax(attention_scores)  # shape: [B, N, N]

        attended_features = torch.matmul(attention_weights, V)  # shape: [B, N, C]
        attended_features = attended_features.permute(0, 2, 1)

        return attended_features

class MSFM(nn.Module): # Attention-based Feature Fusion Module
    def __init__(self, feature_dim):
        super(MSFM, self).__init__()
        self.CA1 = CrossAttentionFusion(feature_dim)
        self.CA2 = CrossAttentionFusion(feature_dim)
        self.relu = nn.ReLU()
    
    def forward(self, low, mid, high):
        low_new = self.CA1(mid, low)
        high_new = self.CA2(mid, high)
        fused_features = self.relu(low_new + mid + high_new)
        return fused_features
    



class FewShotSeg(nn.Module):

    def __init__(self, args):
        super().__init__()

        # Encoder
        self.encoder = Res50Encoder(replace_stride_with_dilation=[True, True, False],
                                    pretrained_weights="COCO")  # or "ImageNet"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.args = args
        self.scaler = 20.0
        self.criterion = nn.NLLLoss(ignore_index=255, weight=torch.FloatTensor([0.1, 1.0]).cuda())

        self.N = 900
        self.FAM = FAM(feature_dim=512, N=self.N)
        self.MSFM = MSFM(feature_dim=512)


    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, opt, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors  (1, 3, 257, 257)
            qry_mask: label
                N x 2 x H x W, tensor
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W
        ## Feature Extracting With ResNet Backbone
        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])
        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])
        
        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  # t for support features
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        outputs_qry = []
        coarse_loss = torch.zeros(1).to(self.device)
        for epi in range(supp_bs):

            """
            supp_fts[[epi], way, shot]: (B, C, H, W) 
            """

            if supp_mask[[0], 0, 0].max() > 0.:

                spt_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                             for shot in range(self.n_shots)] for way in range(self.n_ways)]
                spt_fg_proto = self.getPrototype(spt_fts_)

                
                # CPG module *******************
                qry_pred = torch.stack(
                    [self.getPred(qry_fts[way], spt_fg_proto[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

                qry_pred_coarse = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

                if train:
                    log_qry_pred_coarse = torch.cat([1 - qry_pred_coarse, qry_pred_coarse], dim=1).log()

                    coarse_loss = self.criterion(log_qry_pred_coarse, qry_mask)
              


                # ************************************************

                spt_fg_fts = [[self.get_fg(supp_fts[way][shot], supp_mask[[0], way, shot])
                               for shot in range(self.n_shots)] for way in range(self.n_ways)]  # (1, 512, N)
                

                qry_fg_fts = [self.get_fg(qry_fts[way], qry_pred_coarse[epi])
                              for way in range(self.n_ways)]  # (1, 512, N)
                
                fused_fts_low, fused_fts_mid, fused_fts_high = self.FAM(spt_fg_fts, qry_fg_fts)

                fused_fg_fts = self.MSFM(fused_fts_low, fused_fts_mid, fused_fts_high)

 
                fg_proto = [self.get_proto_new(fused_fg_fts)]


                pred = torch.stack(
                    [self.getPred(qry_fts[way], fg_proto[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'

                pred_up = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
                pred = torch.cat((1.0 - pred_up, pred_up), dim=1)
                outputs_qry.append(pred)


            else:
                ########################acquiesce prototypical network ################
                supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                              for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_prototypes = self.getPrototype(supp_fts_)  # the coarse foreground

                qry_pred = torch.stack(
                    [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
                ########################################################################

                # Combine predictions of different feature maps #
                qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)

                outputs_qry.append(preds)

        output_qry = torch.stack(outputs_qry, dim=1)
        output_qry = output_qry.view(-1, *output_qry.shape[2:])


        return output_qry, coarse_loss

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes


    def get_fg(self, fts, mask):

        """
        :param fts: (1, C, H', W')
        :param mask: (1, H, W)
        :return:
        """

        mask = torch.round(mask)
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        mask = mask.unsqueeze(1).bool()
        result_list = []

        for batch_id in range(fts.shape[0]):
            tmp_tensor = fts[batch_id]  
            tmp_mask = mask[batch_id]  

            foreground_features = tmp_tensor[:, tmp_mask[0]]  

            if foreground_features.shape[1] == 1:  

                foreground_features = torch.cat((foreground_features, foreground_features), dim=1)

            result_list.append(foreground_features)  

        foreground_features = torch.stack(result_list)

        return foreground_features

    def get_proto_new(self, fts):
        """
        :param fts:  (1, 512, N)
        :return: 1, 512, 1
        """
        N = fts.size(2)
        proto = torch.sum(fts, dim=2) / (N + 1e-5)

        return proto


    


    