import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl

import duett
from physionet import MIMICDataModule  # Ensure you import the MIMICDataModule correctly
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class WarmUpCallback(pl.callbacks.Callback):
    """Linear warmup over warmup_steps batches, tries to auto-detect the base lr."""
    def __init__(self, steps=1000, base_lr=None, invsqrt=True, decay=None):
        super().__init__()
        self.warmup_steps = steps
        self.base_lr = base_lr if base_lr is not None else 0.0003
        self.invsqrt = invsqrt
        self.decay = decay if decay is not None else steps
        self.state = {'steps': 0}  # Initialize the state dictionary with steps

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, unused=0):
        # Get optimizers
        optimizers = pl_module.optimizers()

        # Increment step count
        self.state['steps'] += 1

        # Calculate the new learning rate
        if self.state['steps'] < self.warmup_steps:
            lr_scale = self.state['steps'] / self.warmup_steps
            lr = lr_scale * self.base_lr
        else:
            decay_steps = self.state['steps'] - self.warmup_steps
            if self.invsqrt:
                lr = self.base_lr * (self.decay / (decay_steps + self.decay)) ** 0.5
            else:
                lr = self.base_lr * (self.decay / (decay_steps + self.decay))

        # Check if optimizers is a list or a single optimizer and apply the learning rate accordingly
        if isinstance(optimizers, list):
            for optimizer in optimizers:
                self.set_lr(optimizer, lr)
        else:
            self.set_lr(optimizers, lr)  # Apply the learning rate to a single optimizer

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)  # Safely update the state

    def state_dict(self):
        return self.state.copy()

def average_models(models):
    """Averages model weights and loads the resulting weights into the first model, returning it."""
    models = list(models)
    n = len(models)
    sds = [m.state_dict() for m in models]
    averaged = {}
    for k in sds[0]:
        averaged[k] = sum(sd[k] for sd in sds) / n
    models[0].load_state_dict(averaged)
    return models[0]

seed = 2020
pl.seed_everything(seed)
dm = MIMICDataModule(
    file_path_features='/home/anand/DuETT/Duett/mimic_iv_DueTT_features.csv',
    file_path_labels='/home/anand/DuETT/Duett/mimic_iv_DueTT_labels.csv'
)
dm.setup()

train_loader = dm.train_dataloader()

for batch in train_loader:
    print(batch)  # print or inspect your batch structure and content
    break  # br

print("Positive fraction of the dataset:", dm.pos_frac)


early_stop_callback = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    min_delta=0.00,      # Minimum change to qualify as an improvement
    patience=10,          # Number of epochs with no improvement after which training will be stopped
    verbose=True,        # To print logs
    mode='min'           # Minimize the monitored metric (val_loss in this case)
)

checkpoint = pl.callbacks.ModelCheckpoint(
    save_last=True,
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    dirpath='checkpoints'
)

warmup = WarmUpCallback(steps=2000, base_lr=0.001)  # Adjust base_lr based on your model's specific needs

trainer = pl.Trainer(
    gpus=1,
    logger=False,
    num_sanity_val_steps=2,
    max_epochs=50,
    gradient_clip_val=1.0,
    callbacks=[warmup, checkpoint,early_stop_callback]
)


pretrained_model_path = 'checkpoints/pretrained_model.ckpt'
if os.path.isfile(pretrained_model_path):
    print("Loading pretrained model...")
    model = duett.Model.load_from_checkpoint(
        checkpoint_path=pretrained_model_path,
        d_static_num=dm.d_static_num(),
        d_time_series_num=dm.d_time_series_num(),
        d_target=dm.d_target())
else:
    print("Starting pretraining...")
    pretrain_model = duett.pretrain_model(
        d_static_num=dm.d_static_num(),
        d_time_series_num=dm.d_time_series_num(),
        d_target=dm.d_target(),
        pos_frac=dm.pos_frac,
        seed=seed
    )
    try:
        last_checkpoint = checkpoint.last_model_path
        if last_checkpoint:
            print(f"Resuming from checkpoint: {last_checkpoint}")
            trainer.fit(pretrain_model, dm, ckpt_path=last_checkpoint)
            trainer.save_checkpoint(pretrained_model_path)
        else:
            trainer.fit(pretrain_model, dm)
            trainer.save_checkpoint(pretrained_model_path)
    except Exception as e:
        print(f"Training interrupted: {e}")

    #trainer.fit(model, dm)
    




#trainer.fit(pretrain_model, dm)

#trainer.fit(pretrain_model, dm)
#pretrained_path = checkpoint.best_model_path
for seed in range(2020, 2023):
    pl.seed_everything(seed)
    fine_tune_model = duett.fine_tune_model(pretrained_model_path, d_static_num=dm.d_static_num(),
            d_time_series_num=dm.d_time_series_num(), d_target=dm.d_target(), pos_frac=dm.pos_frac, seed=seed)
    checkpoint = pl.callbacks.ModelCheckpoint(save_top_k=5, save_last=False, mode='max', monitor='val_ap', dirpath='checkpoints')
    warmup = WarmUpCallback(steps=1000)
    trainer = pl.Trainer(gpus=1, logger=False, max_epochs=50, gradient_clip_val=1.0,
            callbacks=[warmup, checkpoint,early_stop_callback])
    trainer.fit(fine_tune_model, dm)
    final_model = average_models([duett.fine_tune_model(path, d_static_num=dm.d_static_num(),
            d_time_series_num=dm.d_time_series_num(), d_target=dm.d_target(), pos_frac=dm.pos_frac)
            for path in checkpoint.best_k_models.keys()])
    trainer.test(final_model, dataloaders=dm)