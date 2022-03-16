import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    ### training options
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    # (only valid when not using --cuda_ray)
    parser.add_argument('--num_steps', type=int, default=128)
    parser.add_argument('--upsample_steps', type=int, default=128)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    ### dataset options
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    # (default is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box(-bound, bound)")
    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()
    print(opt)
    
    seed_everything(opt.seed)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork


    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=opt.cuda_ray,
    )
    #model = NeRFNetwork(encoding="frequency", encoding_dir="frequency", num_layers=4, hidden_dim=256, geo_feat_dim=256, num_layers_color=4, hidden_dim_color=128)
    print(model)
    
#        model.name = name
#        model.conf = conf
#        model.mute = mute
#        model.metrics = metrics
#        model.local_rank = local_rank
#        model.world_size = world_size
#        model.workspace = workspace
#        model.ema_decay = ema_decay
#        model.fp16 = fp16
#        model.best_mode = best_mode
#        model.use_loss_as_metric = use_loss_as_metric
#        model.max_keep_ckpt = max_keep_ckpt
#        model.eval_interval = eval_interval
#        model.use_checkpoint = use_checkpoint
#        model.use_tensorboardX = use_tensorboardX
#        model.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
#        model.scheduler_update_every_step = scheduler_update_every_step
#        model.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
#        model.console = Console()

#        if model.world_size > 1:
#            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
#        self.model = model

    # Marching Cubes from Tim Chen
    N = 512
    t = np.linspace(-1.2, 1.2, N+1)
    chunk = 100
    device = torch.device('cuda')
    model = model.to(device)
    bound = 2
    with torch.no_grad():
        query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
        query_pts = query_pts.reshape(-1, N+1, 3)
        raw = []
        for i in range(query_pts.shape[0] // chunk):
            pts = query_pts[i*chunk: (i+1)*chunk, :]
            raw_chunk = model.density(torch.tensor(pts).to(device),bound).cpu().numpy()
            raw.append(raw_chunk)
        raw = np.concatenate(raw)
        raw = raw.reshape(N+1,N+1,N+1)
        #raw = model.density(torch.tensor(pts).to(device),bound).cpu().numpy()
    print('Bingo')
    sigma = np.maximum(raw[...,-1], 0.)

    sigma = sigma.reshape(sh[:-1])

    threshold = 5.
    print('fraction occupied', np.mean(sigma > threshold))
    vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    print('done', vertices.shape, triangles.shape)

    mesh = trimesh.Trimesh(vertices / N + 1  - .5, triangles)
    mesh.show()
    ### test mode
    if opt.test:

        trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='latest')

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

            trainer.test(test_loader)
    
    else:

        criterion = torch.nn.HuberLoss(delta=0.1)

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

        # need different milestones for GUI/CMD mode.
        scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 1500, 2000] if opt.gui else [50, 100, 150], gamma=0.33)

        trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=10)

        # need different dataset type for GUI/CMD mode.

        if opt.gui:
            train_dataset = NeRFDataset(opt.path, type='all', mode=opt.mode, scale=opt.scale)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            trainer.train_loader = train_loader # attach dataloader to trainer

            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.mode, downscale=2, scale=opt.scale)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

            trainer.train(train_loader, valid_loader, 200)

            # also test
            test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

            trainer.test(test_loader)
