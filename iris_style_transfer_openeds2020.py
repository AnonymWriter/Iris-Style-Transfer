import tqdm
import torch
import wandb
import argparse
from PIL import Image
import torchvision.transforms.v2 as transforms

# self-defined functions
from pipelines import nst
from data_preprocessing import load_data_openeds2020
from utils import seed, angular_distance, prepare_dir, crop_image
from models import GazeEstimator1, GazeEstimator2, EfficientNet, VGG19

def iris_style_transfer_openeds2020(
    args: argparse.Namespace,
    dataloader: torch.utils.data.DataLoader,
    vgg: torch.nn.Module,
    efficientnet: torch.nn.Module,
    estimator1: torch.nn.Module,
    estimator2: torch.nn.Module,
    s_iris: torch.Tensor,
    c_loss_weight: float,
    s_loss_weight: float,
    nst_epoch: int,
    metric_prefix: str,
    save_dir: str,
    save_period: int = 50,
    ):
    """
    Main function for investigating the influence of iris style transfer (gaze estimation) using OpenEDS2020 dataset.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        dataloader (torch.utils.data.DataLoader): data loader.
        vgg (torch.nn.Module): vgg network.
        efficientnet (torch.nn.Module): efficientnet model.
        estimator1 (torch.nn.Module): model-based gaze estimator.
        estimator1 (torch.nn.Module): appearance-based gaze estimator.
        c_loss_weight (float): content loss weight (alpha).
        s_loss_weight (float): style loss weight (beta).
        nst_epoch (int): iris style transfer iterations.
        metric_prefix (str): wandb log prefix.
        save_dir (str): path to save directory.
        save_period (int): how often to save.
    """
    
    preds1_pre_nst, preds2_pre_nst, preds1_post_nst, preds2_post_nst = [], [], [], []
    labelss, c_losses, s_losses = [], [], []
    t_resize = transforms.Resize((224, 224))
    t_toPIL = transforms.ToPILImage()
    
    # gaze estimation before 
    
    for batch_id, (c_imgs, labels) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            # save the 1st image of the batch
            if batch_id % save_period == 0:
                t_toPIL(c_imgs[0]).save(save_dir + 'batch_' + str(batch_id) + '_raw.png')
                
            labelss.append(labels)
            c_imgs = c_imgs.to(args.device)
            batch_wandb_log = {}
            
            # gaze estimation before iris style transfer
            c_segs = efficientnet(c_imgs)
            preds1 = estimator1(c_segs)
            preds2 = estimator2(c_imgs)
            preds1_pre_nst.append(preds1)
            preds2_pre_nst.append(preds2)
            radian_distances1, degree_distances1 = angular_distance(preds1.cpu(), labels)
            radian_distances2, degree_distances2 = angular_distance(preds2.cpu(), labels)
            batch_wandb_log[metric_prefix + '/batch/pre/radian_distance1'] = radian_distances1.mean()
            batch_wandb_log[metric_prefix + '/batch/pre/degree_distance1'] = degree_distances1.mean()
            batch_wandb_log[metric_prefix + '/batch/pre/radian_distance2'] = radian_distances2.mean()
            batch_wandb_log[metric_prefix + '/batch/pre/degree_distance2'] = degree_distances2.mean()
            
            # collect content irises from batch
            c_irises, c_iris_shapes, c_iris_bbs, c_ms_iris = [], [], [], []
            for c_img, c_seg in zip(c_imgs, c_segs):
                # compute masks
                c_m_efficientnet = c_seg == 2
                c_m_ritnet = c_img <= args.glint_threshold
                c_m_iris = c_m_efficientnet * c_m_ritnet
                c_ms_iris.append(c_m_iris)
                
                # apply iris mask
                c_img = c_img * c_m_iris

                # apply bounding box
                x_min, y_min, x_max, y_max = crop_image(c_img, return_idx = True)
                c_iris_bbs.append((x_min, y_min, x_max, y_max))
                c_iris = c_img[:, x_min: x_max + 1, y_min: y_max + 1]
                c_iris_shapes.append(c_iris.shape[-2:])

                # resize to (224, 224)
                c_iris = t_resize(c_iris)
                
                c_irises.append(c_iris)
            c_irises = torch.stack(c_irises)
            c_irises = c_irises.repeat(1, 3, 1, 1)
        
        # apply nst to content iris
        new_c_irises, _, c_loss_hist, s_loss_hist = nst(c_irises, 
                                                        s_iris, 
                                                        c_loss_weight = c_loss_weight, 
                                                        s_loss_weight = s_loss_weight, 
                                                        epochs = nst_epoch, # args.nst_epochs, 
                                                        vgg = vgg, 
                                                        use_tqdm = False, 
                                                        device = args.device)
        
        with torch.no_grad():
            c_loss = c_loss_hist[-1]
            s_loss = s_loss_hist[-1]
            c_losses.append(c_loss)
            s_losses.append(s_loss)
            batch_wandb_log[metric_prefix + '/batch/c_loss'] = c_loss
            batch_wandb_log[metric_prefix + '/batch/s_loss'] = s_loss
            batch_wandb_log[metric_prefix + '/batch/cs_loss'] = c_loss * c_loss_weight + s_loss * s_loss_weight
            
            # RGB to grayscale
            new_c_irises = transforms.functional.rgb_to_grayscale(new_c_irises)

            # iris operation for each image in the batch
            for i in range(len(new_c_irises)):
                new_c_iris = new_c_irises[i] 
                
                # resize new iris texture to its original size
                raw_shape = c_iris_shapes[i]
                new_c_iris = transforms.Resize(raw_shape)(new_c_iris)
                
                # apply mask again
                x_min, y_min, x_max, y_max = c_iris_bbs[i]
                c_m_iris = c_ms_iris[i][:, x_min: x_max + 1, y_min: y_max + 1] 
                new_c_iris = new_c_iris * c_m_iris

                # replace the old iris with the new iris
                c_imgs[i, :, x_min: x_max + 1, y_min: y_max + 1] *= ~c_m_iris
                c_imgs[i, :, x_min: x_max + 1, y_min: y_max + 1] += new_c_iris

            # save the 1st image of the batch
            if batch_id % save_period == 0:
                t_toPIL(c_imgs[0]).save(save_dir + 'batch_' + str(batch_id) + '_new.png')
            
            # gaze estimation after iris style transfer
            c_segs = efficientnet(c_imgs)
            preds1 = estimator1(c_segs)
            preds2 = estimator2(c_imgs)
            preds1_post_nst.append(preds1)
            preds2_post_nst.append(preds2)
            radian_distances1, degree_distances1 = angular_distance(preds1.cpu(), labels)
            radian_distances2, degree_distances2 = angular_distance(preds2.cpu(), labels)
            batch_wandb_log[metric_prefix + '/batch/post/radian_distance1'] = radian_distances1.mean()
            batch_wandb_log[metric_prefix + '/batch/post/degree_distance1'] = degree_distances1.mean()
            batch_wandb_log[metric_prefix + '/batch/post/radian_distance2'] = radian_distances2.mean()
            batch_wandb_log[metric_prefix + '/batch/post/degree_distance2'] = degree_distances2.mean()
            
        # batch log
        wandb.log(batch_wandb_log)

    # metrics after iris style transfer loop
    wandb_log = {}
    with torch.no_grad():
        # concatenate all tensors and save to files
        preds1_pre_nst = torch.cat(preds1_pre_nst).detach().cpu() ; torch.save(preds1_pre_nst, save_dir + 'preds1_pre.pt')
        preds2_pre_nst = torch.cat(preds2_pre_nst).detach().cpu() ; torch.save(preds2_pre_nst, save_dir + 'preds2_pre.pt')
        preds1_post_nst = torch.cat(preds1_post_nst).detach().cpu() ; torch.save(preds1_post_nst, save_dir + 'preds1_post.pt')
        preds2_post_nst = torch.cat(preds2_post_nst).detach().cpu() ; torch.save(preds2_post_nst, save_dir + 'preds2_post.pt')
        labelss = torch.cat(labelss).detach().cpu() ; torch.save(labelss, save_dir + 'labels.pt')
        
        # angular distances
        radian_distances1, degree_distances1 = angular_distance(preds1_pre_nst, labelss)
        radian_distances2, degree_distances2 = angular_distance(preds2_pre_nst, labelss)
        wandb_log[metric_prefix + '/pre/radian_distance1'] = radian_distances1.mean()
        wandb_log[metric_prefix + '/pre/degree_distance1'] = degree_distances1.mean()
        wandb_log[metric_prefix + '/pre/radian_distance2'] = radian_distances2.mean()
        wandb_log[metric_prefix + '/pre/degree_distance2'] = degree_distances2.mean()
        
        radian_distances1, degree_distances1 = angular_distance(preds1_post_nst, labelss)
        radian_distances2, degree_distances2 = angular_distance(preds2_post_nst, labelss)
        wandb_log[metric_prefix + '/post/radian_distance1'] = radian_distances1.mean()
        wandb_log[metric_prefix + '/post/degree_distance1'] = degree_distances1.mean()
        wandb_log[metric_prefix + '/post/radian_distance2'] = radian_distances2.mean()
        wandb_log[metric_prefix + '/post/degree_distance2'] = degree_distances2.mean()        
        
        # content and style losses
        c_loss = torch.nanmean(torch.as_tensor(c_losses))
        s_loss = torch.nanmean(torch.as_tensor(s_losses))
        cs_loss = c_loss * c_loss_weight + s_loss * s_loss_weight
        wandb_log[metric_prefix + '/c_loss'] = c_loss
        wandb_log[metric_prefix + '/s_loss'] = s_loss
        wandb_log[metric_prefix + '/cs_loss'] = cs_loss
        
    wandb.log(wandb_log)
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    
    # general parameters
    parser.add_argument('-P', '--project', type = str, default = 'iris-style-transfer-openeds2020', help = 'project name')
    parser.add_argument('-seed', '--seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-device', '--device', type = int, default = 0, help = 'GPU index. -1 stands for CPU.')
    parser.add_argument('-path1', '--estimator1_path', type = str, default = './models/weights/seed_42_GazeEstimator1_lr_1e-05_epoch_500.pth', help = 'pretrained estimator1 weight path')
    parser.add_argument('-path2', '--estimator2_path', type = str, default = './models/weights/seed_42_GazeEstimator2_lr_1e-05_epoch_150.pth', help = 'pretrained estimator2 weight path')
    parser.add_argument('--eval_train', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether also evaluate on the training set (super large and hence slow)')
    parser.add_argument('--eval_test', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether also evaluate on the test set (super large and hence slow)')
    parser.add_argument('-W', '--num_workers', type = int, default = 16, help = 'number of workers for data loader')
    parser.add_argument('-M', '--pin_memory', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use pin memory for data loader')
    
    # hyperparameters
    parser.add_argument('-bs', '--bs', type = int, default = 128, help = 'batch size')
    parser.add_argument('-cw', '--c_loss_weight', type = int, default = 1, help = 'cw')
    # parser.add_argument('-sw', '--s_loss_weight', type = int, default = 1, help = 'sw')
    parser.add_argument('--glint_threshold', type = float, default = 0.8, help = 'glint threshold')
    
    args = parser.parse_args()
    args.device = 'cuda:' + str(args.device) if args.device >= 0 else 'cpu'

    # reproducibility
    seed(args.seed)

    
    # models
    vgg = VGG19()
    vgg.to(args.device)
    efficientnet = EfficientNet()
    efficientnet.to(args.device)
    estimator1 = GazeEstimator1(extract_feature = True)
    estimator1.load_state_dict(torch.load(args.estimator1_path, weights_only = True, map_location = 'cpu'))
    estimator1.to(args.device)
    estimator1.eval()
    estimator2 = GazeEstimator2(extract_feature = True, freeze_resnet = False)
    estimator2.load_state_dict(torch.load(args.estimator2_path, weights_only = True, map_location = 'cpu'))
    estimator2.to(args.device)
    estimator2.eval()
    
    # a randomly chosen but fixed style image (one-for-all, i.e., always transfering the same style)
    s_img = Image.open('../data/openeds2020/openEDS2020-GazePrediction/test/sequences/2577/023.png').convert('L')
    s_img = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale = True)])(s_img)
    s_img = s_img.to(args.device)
    
    # extract iris from style image
    s_m_efficientnet = efficientnet(s_img)
    s_m_ritnet = s_m_efficientnet == 2
    s_m_glint = s_img <= args.glint_threshold
    s_img = s_img * s_m_ritnet * s_m_glint
    s_iris = crop_image(s_img)
    t = transforms.Resize((224, 224))
    s_iris = t(s_iris)

    # beta range and epoch range
    s_loss_weights = [1]
    nst_epochs = [200]
    
    # for training, validation, and test sets
    postfixes = ['validation/']
    if args.eval_train: # skip training set because it is too large
        postfixes.append('train/')
    if args.eval_test: # skip test set because it is toooooo large
        postfixes.append('test/')
    for postfix in postfixes:
        # load data
        print('loading ' + postfix[:-1] + ' set...')
        c_imgs, labels = load_data_openeds2020(extract_feature = False, postfix = postfix, device = args.device)
        print('number of samples in ' + postfix + ' set:', len(c_imgs))
        
        # dataset and dataloader
        dataset = torch.utils.data.TensorDataset(c_imgs, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.bs, shuffle = False, num_workers = args.num_workers, pin_memory = args.pin_memory)        
     
        for sw in s_loss_weights:
            for nst_epoch in nst_epochs:
                # prepare folder
                save_dir = 'saved/openeds2020/sw_' + str(sw) + '_epoch_' + str(nst_epoch) + '/' + postfix
                prepare_dir(save_dir)
                
                # save all labels to file
                torch.save(labels, save_dir + 'gts.pt')
        
                # wandb init
                args.name = 'seed ' + str(args.seed) + ' sw ' + str(sw) + ' epoch ' + str(nst_epoch) + ' test'
                wandb.init(project = args.project, name = args.name, config = args.__dict__, anonymous = "allow")
                
                # iris style transfer main function
                iris_style_transfer_openeds2020(args, dataloader, vgg, efficientnet, estimator1, estimator2, s_iris, args.c_loss_weight, sw, nst_epoch, postfix, save_dir)
                wandb.finish()