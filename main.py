from torch.utils.data import DataLoader
from torch import nn
import argparse
import torch

from datasets import OmniglotDataset, MiniImageNet
from core import NShotTaskSampler, create_nshot_task_label, EvaluateFewShot
from models import VAE
from maml import meta_gradient_step
from train import fit
from callbacks import *
from utils import setup_dirs
from config import PATH

import torch.nn.functional as F
from torchsummary import summary

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

####################
#### Parameters ####
####################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='omniglot')
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--inner-val-steps', default=3, type=int)
parser.add_argument('--inner-lr', default=0.4, type=float)
parser.add_argument('--meta-lr', default=0.001, type=float)
parser.add_argument('--meta-batch-size', default=32, type=int)
parser.add_argument('--order', default=1, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch-len', default=100, type=int)
parser.add_argument('--eval-batches', default=20, type=int)

args = parser.parse_args()

if args.dataset == 'omniglot':
    dataset_class = OmniglotDataset
    fc_layer_size = 64
    num_input_channels = 1
elif args.dataset == 'miniImageNet':
    dataset_class = MiniImageNet
    fc_layer_size = 1600
    num_input_channels = 3
else:
    raise(ValueError('Unsupported dataset'))

param_str = f'{args.dataset}_order={args.order}_n={args.n}_k={args.k}_metabatch={args.meta_batch_size}_' \
            f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}'
print(param_str)

###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, args.epoch_len, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size)
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                   num_tasks=args.meta_batch_size)
)

############
# Training #
############
print(f'Training MAML on {args.dataset}...')
meta_model = VAE(in_channels=num_input_channels, input_size=28, z_dim=16).to(device, dtype=torch.double)
meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)

print([x for x, y in meta_model.named_parameters()])
print(summary(meta_model, (1, 28, 28)))

def loss_fn(recon_x, x, mu, logvar):
    #print(recon_x.shape, x.shape)
    BCE = F.binary_cross_entropy(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_

callbacks = [
    EvaluateFewShot(
        eval_fn=meta_gradient_step,
        num_tasks=args.eval_batches,
        n_shot=args.n,
        k_way=args.k,
        q_queries=args.q,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
        # MAML kwargs
        inner_train_steps=args.inner_val_steps,
        inner_lr=args.inner_lr,
        device=device,
        order=args.order,
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/maml/{param_str}.pth',
        monitor=f'val_{args.n}-shot_{args.k}-way_acc'
    ),
    ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
    CSVLogger(PATH + f'/logs/maml/{param_str}.csv'),
]


fit(
    meta_model,
    meta_optimiser,
    loss_fn,
    epochs=args.epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=meta_gradient_step,
    fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                         'train': True,
                         'order': args.order, 'device': device, 'inner_train_steps': args.inner_train_steps,
                         'inner_lr': args.inner_lr},
)