import torch
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import x_transformers
import csv
import os
import torchmetrics
from x_transformers import Encoder

class BatchNormLastDim(nn.Module):
    def __init__(self, d, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(d, **kwargs)

    def forward(self, x):
        if x.ndim == 2:
            return self.batch_norm(x)
        elif x.ndim == 3:
            return self.batch_norm(x.transpose(1,2)).transpose(1,2)
        else:
            raise NotImplementedError("BatchNormLastDim not implemented for ndim > 3 yet")

def simple_mlp(d_in, d_out, n_hidden, d_hidden, final_activation=False, input_batch_norm=False,
        hidden_batch_norm=False, dropout=0., activation=nn.ReLU):
    # Could add options for different activations, batch norm, etc. as needed
    layers = []
    if input_batch_norm:
        layers.append(BatchNormLastDim(d_in))
    layers.append(nn.Linear(d_in, d_hidden if n_hidden > 0 else d_out))
    if n_hidden > 0:
        layers.append(activation())
    for _ in range(n_hidden - 1):
        if hidden_batch_norm:
            layers.append(BatchNormLastDim(d_hidden))
        layers.extend([nn.Linear(d_hidden, d_hidden), activation(), nn.Dropout(dropout)])
    if n_hidden > 0:
        layers.append(nn.Linear(d_hidden, d_out))
    if final_activation:
        layers.append(activation())
    return nn.Sequential(*layers)

class DynamicTransformer(nn.Module):
    def __init__(self, base_dim, depth=1, heads=4, max_seq_len=512):
        super().__init__()
        self.base_dim = base_dim
        self.depth = depth
        self.heads = heads
        self.max_seq_len = max_seq_len
        self.transformer = self.create_transformer(base_dim)

    def create_transformer(self, dim):
        # Initializes a new transformer with the given dimension
        return Encoder(
            dim=dim,
            depth=self.depth,
            heads=self.heads,
            max_seq_len=self.max_seq_len,
            ff_glu=True,  # Using GLU variant in feedforward
            ff_mult=4,    # Feedforward dimension multiplier
            attn_dropout=0.1,
            ff_dropout=0.1
        )

    def forward(self, x):
        # Check if the current transformer's dimension matches x's last dimension
        current_dim = x.size(-1)
        if current_dim != self.base_dim:
            # Adjust transformer dimensions if input dimension has changed
            self.base_dim = current_dim
            self.transformer = self.create_transformer(current_dim)
        return self.transformer(x)

def pretrain_model(d_static_num, d_time_series_num, d_target, **kwargs):
    return Model(d_static_num, d_time_series_num, d_target, **kwargs)

def fine_tune_model(ckpt_path, **kwargs):
    return Model.load_from_checkpoint(ckpt_path, pretrain=False, aug_noise=0., aug_mask=0.5, transformer_dropout=0.5,
            lr=1.e-4, weight_decay=1.e-5, fusion_method='rep_token', **kwargs)

class Model(pl.LightningModule):
    def __init__(self, d_static_num, d_time_series_num, d_target, lr=3.e-4, weight_decay=1.e-1, glu=False,
            scalenorm=True, n_hidden_mlp_embedding=1, d_hidden_mlp_embedding=64, d_embedding=192, d_feedforward=512,
            max_len=48, n_transformer_head=2, n_duett_layers=2, d_hidden_tab_encoder=128, n_hidden_tab_encoder=1,
            norm_first=True, fusion_method='masked_embed', n_hidden_head=1, d_hidden_head=64, aug_noise=0., aug_mask=0.,
            pretrain=True, pretrain_masked_steps=1, pretrain_n_hidden=0, pretrain_d_hidden=64, pretrain_dropout=0.5,
            pretrain_value=True, pretrain_presence=True, pretrain_presence_weight=0.2, predict_events=True,
            transformer_dropout=0., pos_frac=None, freeze_encoder=False, seed=0, save_representation=None,
            masked_transform_timesteps=32,**kwargs):
        super().__init__()
        #self.d_embedding = d_embedding
        #self.d_embedding = output_embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.d_time_series_num = d_time_series_num
        self.d_target = d_target
        self.d_embedding = d_embedding
        self.max_len = max_len
        self.pretrain = pretrain
        self.pretrain_masked_steps = pretrain_masked_steps
        self.pretrain_dropout = pretrain_dropout
        self.freeze_encoder = freeze_encoder
        self.set_pos_frac(pos_frac)
        self.rng = np.random.default_rng(seed)
        self.aug_noise = aug_noise
        self.aug_mask = aug_mask
        self.fusion_method = fusion_method
        self.pretrain_presence = pretrain_presence
        self.pretrain_presence_weight = pretrain_presence_weight
        self.predict_events = predict_events
        self.masked_transform_timesteps = masked_transform_timesteps
        self.pretrain_value = pretrain_value
        self.save_representation = save_representation
        self.test_roc_auc = torchmetrics.AUROC(pos_label=1)
        self.test_pr_auc = torchmetrics.AveragePrecision(pos_label=1)
        self.register_buffer("MASKED_EMBEDDING_KEY", torch.tensor(0)) # For multi-gpu training
        self.register_buffer("REPRESENTATION_EMBEDDING_KEY", torch.tensor(1))
        self.full_event_embedding = nn.Embedding(128, 1584)

        # For any special timesteps, e.g., masked, static, [CLS], etc.
        self.special_embeddings = nn.Embedding(8, d_embedding)
        self.embedding_layers = nn.ModuleList([
            simple_mlp(2, d_embedding, n_hidden_mlp_embedding, d_hidden_mlp_embedding, hidden_batch_norm=True)
            #print(d_time_series_num)
            for _ in range(d_time_series_num)])

        self.n_obs_embedding = nn.Embedding(16, 1)
        
        self.embedding_transform = nn.Linear(6336, self.d_embedding)

        if d_feedforward is None:
            d_feedforward = d_embedding * 4

        et_dim = d_embedding*(masked_transform_timesteps+1)
        tt_dim = d_embedding*(d_time_series_num+1)

        self.event_transformers = nn.ModuleList([DynamicTransformer(d_embedding) for _ in range(n_duett_layers)])
        #self.event_transformers = nn.ModuleList([x_transformers.Encoder(dim=et_dim, depth=1,
                #heads=n_transformer_head, pre_norm=norm_first, use_scalenorm=scalenorm,
                #attn_dim_head=d_embedding//n_transformer_head, ff_glu=glu,
                #ff_mult=d_feedforward/et_dim, attn_dropout=transformer_dropout,
                #ff_dropout=transformer_dropout) for _ in range(n_duett_layers)])
        self.full_event_embedding = nn.Embedding(d_time_series_num + 1, et_dim)
        self.time_transformers = nn.ModuleList([x_transformers.Encoder(dim=tt_dim, depth=1,
                heads=n_transformer_head, pre_norm=norm_first, use_scalenorm=scalenorm,
                attn_dim_head=d_embedding//n_transformer_head, ff_glu=glu,
                ff_mult=d_feedforward/tt_dim, attn_dropout=transformer_dropout,
                ff_dropout=transformer_dropout) for _ in range(n_duett_layers)])
        self.full_time_embedding =  self.cve(batch_norm=True, d_embedding=tt_dim)
        self.full_rep_embedding = nn.Embedding(tt_dim, 1)

        d_representation = d_embedding * (d_time_series_num + 1) # time_series + static
        #actual_d_representation = z.shape[1] 
        self.head = simple_mlp(192, d_target, n_hidden_head, d_hidden_head,
                hidden_batch_norm=True, final_activation=False, activation=nn.ReLU)
        self.pretrain_value_proj = simple_mlp(d_representation, d_time_series_num,
                pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True)
        if self.pretrain_presence:
            self.pretrain_presence_proj = simple_mlp(d_representation, d_time_series_num,
                    pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True)
        if self.predict_events:
            self.predict_events_proj = simple_mlp(et_dim, masked_transform_timesteps,
                    pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True)
            if self.pretrain_presence:
                self.predict_events_presence_proj = simple_mlp(et_dim, masked_transform_timesteps,
                        pretrain_n_hidden, pretrain_d_hidden, hidden_batch_norm=True)
        self.d_static_num = d_static_num
        self.tab_encoder = simple_mlp(d_static_num, d_embedding, n_hidden_tab_encoder,
                    d_hidden_tab_encoder, hidden_batch_norm=True)
        self.adjust_dims = None
        
        if isinstance(self.tab_encoder, nn.Sequential):
            print("Tab Encoder Configuration:")
            for idx, module in enumerate(self.tab_encoder):
                if hasattr(module, 'in_features'):
                    print(f"  Layer {idx} - Linear, in_features: {module.in_features}, out_features: {module.out_features}")
                else:
                    print(f"  Layer {idx} - {module.__class__.__name__}")
        else:
            print(f"Tab Encoder in_features: {self.tab_encoder.in_features}")

        self.pretrain_loss = F.mse_loss
        self.loss_function = F.binary_cross_entropy_with_logits
        self.pretrain_presence_loss = F.binary_cross_entropy_with_logits
        num_classes = None if d_target == 1 else d_target
        self.train_auroc = torchmetrics.AUROC(num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(num_classes=num_classes)
        self.train_ap = torchmetrics.AveragePrecision(num_classes=num_classes)
        self.val_ap = torchmetrics.AveragePrecision(num_classes=num_classes)
        self.test_auroc = torchmetrics.AUROC(num_classes=num_classes)
        self.test_ap = torchmetrics.AveragePrecision(num_classes=num_classes)

    def set_pos_frac(self, pos_frac):
        if type(pos_frac) == list:
            pos_frac = torch.tensor(pos_frac, device=torch.device('cuda'))
        self.pos_frac = pos_frac
        if pos_frac != None:
            self.pos_weight = 1 / (2 * pos_frac)
            self.neg_weight = 1 / (2 * (1 - pos_frac))

    def cve(self, d_embedding=None, batch_norm=False):
        if d_embedding == None:
            d_embedding = self.d_embedding
        d_hidden = int(np.sqrt(d_embedding))
        if batch_norm:
            return nn.Sequential(nn.Linear(1, d_hidden), nn.Tanh(), BatchNormLastDim(d_hidden), nn.Linear(d_hidden, d_embedding))
        return nn.Sequential(nn.Linear(1, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_embedding))

    def feats_to_input(self, x, batch_size, limits=None):
        assert len(x) == 3, f"Expected tuple of length 3, got {len(x)}"
        xs_ts, xs_static, times = x
        #print(f"Initial shapes - xs_ts: {xs_ts.shape}, xs_static: {xs_static.shape}, times: {times.shape}")

        # Ensure 'times' is at least 1D and properly formatted
        if times.dim() == 0:
            times = times.unsqueeze(0)  # Convert to at least 1D if completely scalar
        elif times.dim() == 1:
            times = times.unsqueeze(1)
        elif times.dim() > 1:
            times = times.squeeze()  # Ensure 'times' is no more than 1D

        #print(f"After times adjustment - xs_ts: {xs_ts.shape}, xs_static: {xs_static.shape}, times: {times.shape}")

        # Check if the data is missing a temporal dimension
        if xs_ts.dim() == 2:  # [batch, features]
            xs_ts = xs_ts.unsqueeze(1)  # Adding a time step dimension
            #print(f"After adding time dimension - xs_ts: {xs_ts.shape}")

        # Augmentation with noise and masking
        if self.training and self.aug_noise > 0 and not self.pretrain:
            xs_static += self.aug_noise * torch.randn_like(xs_static)
            #print(f"After static noise augmentation - xs_static: {xs_static.shape}")

        #print(f"Final output shapes - xs_static: {xs_static.shape}, xs_ts: {xs_ts.shape}, times: {times.shape}")
        return xs_static, xs_ts, times

    def pretrain_prep_batch(self, x, batch_size):
        xs_static, xs_ts, xs_times = self.feats_to_input(x, batch_size)
        #print(f"Initial pretrain prep shapes - xs_static: {xs_static.shape}, xs_ts: {xs_ts.shape}, times: {xs_times.shape}")

        n_steps = xs_ts.shape[1]
        n_vars = (xs_ts.shape[2] - 1) // 2

        y_ts = []
        y_ts_n_obs = []
        y_events = []  # Initialize as empty or as required
        y_events_mask = []  # Initialize as empty or as required
        xs_ts_clipped = xs_ts.clone()

        # Use torch operations to handle the masking
        for batch_i in range(batch_size):
            mask_i = torch.randperm(n_steps)[:self.pretrain_masked_steps]
            
            y_ts.append(xs_ts[batch_i, mask_i, :n_vars])
            y_ts_n_obs.append(xs_ts[batch_i, mask_i, n_vars:2*n_vars])
            
            xs_ts_clipped[batch_i, mask_i, :] = 0  # Set masked elements to zero

        y_ts = torch.stack(y_ts)
        y_ts_n_obs = torch.stack(y_ts_n_obs)
        y_ts_masks = y_ts_n_obs.clip(0, 1)  # Ensure masks are binary

        # Assuming y_events and y_events_mask are not used or are placeholders
        y_events = torch.empty(0)  # Adjust based on actual use
        y_events_mask = torch.empty(0)  # Adjust based on actual use

        #print(f"Final pretrain prep batch function xs_static shape: {xs_static.shape}, xs_ts_clipped: {xs_ts_clipped.shape}, xs_times: {xs_times.shape}, y_ts: {y_ts.shape}, y_ts_masks: {y_ts_masks.shape}, y_events: {y_events.shape}, y_events_mask: {y_events_mask.shape}")
        return (xs_static, xs_ts_clipped, xs_times), y_ts, y_ts_masks, y_events, y_events_mask

    def forward(self, x, pretrain=False, representation=False):
        #print(x.shape)
        xs_static,xs_feats,xs_times = x
        if xs_feats.dim() == 2:
            xs_feats = xs_feats.unsqueeze(1)
        n_vars = xs_feats.shape[2] // 2
        #print(f"Initial xs_static shape: {xs_static.shape}, xs_feats shape: {xs_feats.shape}")

        #print(f"Shape of xs_feats: {xs_feats.shape}")  # Debugging output
        print(f"n_vars: {n_vars}") 

        if xs_static.shape[1] != self.d_static_num:
            raise ValueError(f"Input dimension mismatch in static features, expected {self.d_static_num}, got {xs_static.shape[1]}")

    # Process with tab_encoder
        encoded_static = self.tab_encoder(xs_static)


        # Embedding layer inputs preparation
        embedding_layer_input = torch.empty(xs_feats.shape[0], xs_feats.shape[1], n_vars, 2, device=xs_feats.device)
        embedding_layer_input[:, :, :, 0] = xs_feats[:, :, :n_vars]

        second_half_features = xs_feats[:, :, n_vars:]
        target_shape_last_dim = embedding_layer_input[:, :, :, 1].shape[-1]
        #print("Embedding layer input shape:", embedding_layer_input.shape)

        if pretrain:
            # Adjust shapes for pretraining specifically
            if second_half_features.shape[-1] > target_shape_last_dim:
                # Truncate extra dimensions if necessary
                second_half_features = second_half_features[:, :, :target_shape_last_dim]
            elif second_half_features.shape[-1] < target_shape_last_dim:
                # Pad with zeros if necessary
                padding = torch.zeros(xs_feats.shape[0], xs_feats.shape[1], target_shape_last_dim - second_half_features.shape[-1], device=xs_feats.device)
                second_half_features = torch.cat([second_half_features, padding], dim=-1)
        else:
            # Ensure validation does not fail due to dimension mismatch
            if second_half_features.shape[-1] != target_shape_last_dim:
                raise ValueError(f"Expected dimension size {target_shape_last_dim}, but got {second_half_features.shape[-1]}")

        embedding_layer_input[:, :, :, 1] = second_half_features

        # Embedding processing
        psi = torch.zeros(xs_feats.shape[0], xs_feats.shape[1]+1, n_vars, self.d_embedding, device=xs_feats.device)
        for i, el in enumerate(self.embedding_layers):
            if i < n_vars:
                psi[:, :-1, i, :] = el(embedding_layer_input[:, :, i, :])

        psi[:, :-1, -1, :] = self.tab_encoder(xs_static).unsqueeze(1).expand(-1, psi.shape[1]-1, -1)
        psi[:, -1, :, :] = self.special_embeddings(self.REPRESENTATION_EMBEDDING_KEY.to(self.device)).expand_as(psi[:, -1, :, :])

        # Flatten psi for transformer processing
        psi = psi.view(xs_feats.shape[0], -1, self.d_embedding)

        # Transformer processing
        for transformer in self.event_transformers:
            psi = transformer(psi)
            if psi.nelement() == 0:  # Check if transformer output is empty
                print("Warning: Transformer output is empty.")
                break  # Exit the loop or handle differently

        # Fusion method handling
        if self.fusion_method == 'rep_token':
            z_ts = psi[:, -1, :]
        elif self.fusion_method == 'masked_embed':
            masked_ind = (xs_feats[:, :, -1] > 0).unsqueeze(-1).expand(-1, -1, self.d_embedding)
            z_ts = psi[masked_ind].view(xs_feats.shape[0], -1, self.d_embedding) if masked_ind.any() else torch.empty(0, self.d_embedding, device=psi.device)
        elif self.fusion_method == 'averaging':
            z_ts = torch.mean(psi, dim=1)

        z = z_ts
        if representation:
            return z

        if pretrain:
            if z.nelement() == 0:
                print("Warning: z is empty after processing.")
            y_hat_presence = self.pretrain_presence_proj(z).squeeze() if self.pretrain_presence and z.nelement() != 0 else None
            y_hat_value = self.pretrain_value_proj(z).squeeze(1) if self.pretrain_value and z.nelement() != 0 else None
            #print("Shape of y_hat_presence:", y_hat_presence.shape if y_hat_presence is not None else "None")
            #print("Shape of y_hat_value:", y_hat_value.shape if y_hat_value is not None else "None")
            y_hat_events = None  # Define if needed
            y_hat_events_presence = None  # Define if needed
            return y_hat_value, y_hat_presence, y_hat_events, y_hat_events_presence
        else:
            #print("Shape of z before head:", z.shape)
            if z is not None and z.nelement() != 0:
                out = self.head(z).squeeze(1)
                #print("Output shape from head:", out.shape if out is not None else "None")  # This prints the output shape from self.head
            else:
                #print("z is empty or None, skipping processing through self.head")
                out = None  # Proper handling when z is empty or None
            if self.save_representation:
                return out, z
            return out
 
        
    def prepare_transformed_embedding(self, target_shape):
    # Dummy method to create a compatible shape embedding for addition
        return torch.zeros(target_shape, device=self.device)

    def configure_optimizers(self):
        optimizers = [torch.optim.AdamW([p for l in self.modules() for p in l.parameters()],
                lr=self.lr, weight_decay=self.weight_decay)]
        return optimizers

    def training_step(self, batch, batch_idx):
        *x, y = batch
        if len(x) != 3:
            raise ValueError(f"Expected 3 input feature sets, got {len(x)}")
        
        static_features, time_series_features, times = [tensor.to(self.device) for tensor in x]
        y = y.to(self.device)

        # Debug output
        print(f"Static Features Shape training step: {static_features.shape}")
        print(f"Time Series Features Shape training step: {time_series_features.shape}")
        print(f"Times Shape training step: {times.shape}")

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)  # Initialize with requires_grad

        if self.pretrain:
            features_tuple = (time_series_features, static_features, times)
            x_pretrain, y_masked, mask, y_events, y_events_mask = self.pretrain_prep_batch(features_tuple, y.shape[0])
            y_hat_value, y_hat_presence, y_hat_events, y_hat_events_presence = self.forward(x_pretrain, pretrain=True)

            if self.pretrain_value and y_hat_value is not None:
                loss += (self.pretrain_loss(y_hat_value[mask.bool()], y_masked[mask.bool()]) * mask.float()).mean()

            if self.pretrain_presence and y_hat_presence is not None:
                presence_loss = self.pretrain_presence_loss(y_hat_presence[mask.bool()], mask.float()) * self.pretrain_presence_weight
                loss += presence_loss

            if self.predict_events and y_hat_events is not None:
                event_loss = self.pretrain_loss(y_hat_events[y_events_mask.bool()], y_events[y_events_mask.bool()])
                loss += event_loss

            # Debug output for dimensions
            print(f"y_hat_value: {'None' if y_hat_value is None else y_hat_value.shape}")
            print(f"y_hat_presence: {'None' if y_hat_presence is None else y_hat_presence.shape}")
            print(f"y_hat_events: {'None' if y_hat_events is None else y_hat_events.shape}")
            print(f"y_hat_events_presence: {'None' if y_hat_events_presence is None else y_hat_events_presence.shape}")

        else:
            y_hat = self.forward((static_features,time_series_features, times))
            if y_hat is None:
                return {'loss': torch.tensor(0.0, device=self.device)}
            weight = torch.where(y > 0, self.pos_weight, self.neg_weight) if self.pos_frac is not None else None
            loss = self.loss_function(y_hat, y, weight)
            self.train_auroc.update(y_hat, y.int())
            self.train_ap.update(y_hat, y.int())

        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # Unpack batch based on the structure created by custom_collate_fn
        *x, y = batch  # x contains all feature tensors, y contains the labels
        if len(x) != 3:
            raise ValueError(f"Expected 3 input feature sets, got {len(x)}")
        
        static_features, time_series_features, times = [tensor.to(self.device) for tensor in x]
        # Unpack features; adjust indices below as needed
        #static_features, time_series_features, times = x

        # Debugging output (consider wrapping in a debug flag)
        print(f"Static Features Shape validation step: {static_features.shape}")
        print(f"Time Series Features Shape validation step: {time_series_features.shape}")
        print(f"Times Shape validation step: {times.shape}")

        # Ensure tensors are on the right device
        #static_features = static_features.to(self.device)
        #time_series_features = time_series_features.to(self.device)
        #times = times.unsqueeze(1) if times.dim() == 1 else times.to(self.device)
        y = y.to(self.device)
        batch_size = y.shape[0]

        if self.pretrain:
            # Properly pass features as a tuple for pretraining
            features_tuple = (time_series_features, static_features, times)

            # Call pretrain prep batch function with features tuple
            x_pretrain, y_masked, mask, y_events, y_events_mask = self.pretrain_prep_batch(features_tuple, batch_size)
            

            # Forward pass
            y_hat_value, y_hat_presence, y_hat_events, y_hat_events_presence = self.forward(x_pretrain, pretrain=True)

            # Initialize and calculate loss
            loss = 0
            if self.pretrain_value and y_hat_value is not None:
                valid_steps = min(y_hat_value.size(1), self.pretrain_masked_steps)  # Ensure we don't go out of bounds
                for i in range(valid_steps):
                    loss += self.pretrain_loss(y_hat_value[:, i] * mask[:, i], y[:, i] * mask[:, i])
                loss /= valid_steps

            if self.pretrain_presence and y_hat_presence is not None:
                presence_loss = 0
                valid_steps = min(y_hat_presence.size(1), self.pretrain_masked_steps)  # Ensure we don't go out of bounds
                for i in range(valid_steps):
                    presence_loss += self.pretrain_presence_loss(y_hat_presence[:, i], mask[:, i]) * self.pretrain_presence_weight
                presence_loss /= valid_steps
                loss += presence_loss

            if self.predict_events and y_hat_events is not None:
                event_loss = self.pretrain_loss(y_hat_events * y_events_mask, y_events * y_events_mask)
                loss += event_loss

            # Logging
            self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)
            if 'presence_loss' in locals():
                self.log('val_presence_loss', presence_loss, on_epoch=True, sync_dist=True)
            if 'event_loss' in locals():
                self.log('val_event_loss', event_loss, on_epoch=True, sync_dist=True)
        else:
            # Handle the non-pretrain case
            print(f"ts:{time_series_features.shape}, x_static:{static_features.shape}, x_times:{times.shape}), label_y:{y.shape[0]}")
            y_hat = self.forward(self.feats_to_input((time_series_features, static_features, times), y.shape[0]))
            print(f"y_hat: {'None' if y_hat is None else y_hat.shape}")
            if y_hat is None:
                loss = torch.tensor(0.0, device=self.device)  # No predictions to evaluate
            else:
                loss = self.loss_function(y_hat, y, weight=torch.where(y > 0, self.pos_weight, self.neg_weight) if self.pos_frac is not None else None)
                self.val_auroc.update(y_hat, y.int())
                self.val_ap.update(y_hat, y.int())
                self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)

            # Additional logging for metrics
            self.log('val_ap', self.val_ap.compute(), on_epoch=True, sync_dist=True, rank_zero_only=True)
            self.log('val_auroc', self.val_auroc.compute(), on_epoch=True, sync_dist=True, rank_zero_only=True)

        return {'val_loss': loss}

    def training_epoch_end(self, training_step_outputs):
        if not self.pretrain:
            self.log('train_auroc', self.train_auroc, sync_dist=True, rank_zero_only=True)
            self.log('train_ap', self.train_ap, sync_dist=True, rank_zero_only=True)

    def validation_epoch_end(self, validation_step_outputs):
        if not self.pretrain:
            print("val_auroc", self.val_auroc.compute(), "val_ap", self.val_ap.compute())

    def test_step(self, batch, batch_idx):
        *x, y = batch  # This assumes the batch structure is similar to training/validation
        #print("Initial x shapes:", [xi.shape for xi in x])  # Print the shapes of the input feature tensors
        #print("Initial y shape:", y.shape)  # Print the shape of the labels tensor
        x = [xi.unsqueeze(-1) if xi.dim() == 1 else xi for xi in x]  # Ensure all tensors have at least 2 dimensions
        #print("Adjusted x shapes for concatenation:", [xi.shape for xi in x])  # Print adjusted shapes
        x_concat = torch.cat(x, dim=-1).to(self.device)  # Concatenate along the last dimension
        #print("Concatenated x shape:", x_concat.shape)  # Print the shape after concatenation
        y = y.to(self.device)
        #print("Device-adjusted y shape:", y.shape)  # Print the shape of y after moving to device
        # Continue with your forward pass and other processing
        y_hat = self.forward(x)
        if y_hat is not None:
            #print("Output y_hat shape:", y_hat.shape)
            # Calculate loss
            loss = self.loss_function(y_hat, y)
            self.log('test_loss', loss, on_epoch=True, sync_dist=True, rank_zero_only=True)

            # Calculate ROC-AUC and PR-AUC
            y_hat_prob = torch.sigmoid(y_hat)  # Convert logits to probabilities
            #print("Predicted probabilities shape:", y_hat_prob.shape)
            self.test_roc_auc(y_hat_prob, y.int())
            self.test_pr_auc(y_hat_prob, y.int())
            
            # Extracting logits and probabilities
            y_hat_logits = y_hat.detach().cpu().numpy()
            y_hat_prob = y_hat_prob.detach().cpu().numpy()
            y_np = y.cpu().numpy()
            # Assuming that 'x' includes an ethnicity feature in its last column
            ethnicity_np = x[0][:, -1].cpu().numpy()  # Modify this if the position of ethnicity is different
            print("Ethnicity data sample:", ethnicity_np[:10])

            # Write results to CSV for analysis
            if batch_idx == 0:  # write headers if first batch
                with open('test_results_early_stop.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Logit', 'Probability', 'True Label', 'Ethnicity'])
            with open('test_results_early_stop.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                for logit, prob, true_label, eth in zip(y_hat_logits, y_hat_prob, y_np, ethnicity_np):
                    writer.writerow([logit, prob, true_label, eth])

            return {'test_loss': loss}
        else:
            # Handle cases where y_hat is None
            print("y_hat is None, skipping loss calculation and logging.")
            self.log('test_loss', torch.tensor(0.0), on_epoch=True, sync_dist=True)
            return {'test_loss': torch.tensor(0.0, device=self.device)}
    
    def test_epoch_end(self, outputs):
        roc_auc = self.test_roc_auc.compute()
        pr_auc = self.test_pr_auc.compute()
        self.log('test_roc_auc', roc_auc, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log('test_pr_auc', pr_auc, on_epoch=True, sync_dist=True, rank_zero_only=True)
        print(f"Test ROC-AUC: {roc_auc:.4f}, Test PR-AUC: {pr_auc:.4f}")


    def on_load_checkpoint(self, checkpoint):
        # Ignore errors from size mismatches in head, since those might change between pretraining
        # and supervised training
        # Adapted from https://github.com/PyTorchLightning/pytorch-lightning/issues/4690#issuecomment-731152036
        print('Loading from checkpoint')
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in model_state_dict:
            if k not in state_dict:
                state_dict[k] = model_state_dict[k]
                is_changed = True
        for k in state_dict:
            if k in model_state_dict:
                if k.startswith('head') and state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

        if self.freeze_encoder:
            self.freeze()

    def freeze(self):
        print('Freezing')
        for n, w in self.named_parameters():
            if "head" not in n:
                w.requires_grad = False
            else:
                print("Skip freezing:", n)
