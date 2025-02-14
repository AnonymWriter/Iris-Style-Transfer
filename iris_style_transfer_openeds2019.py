import tqdm
import torch
import wandb
import argparse
import torchvision.transforms.v2 as transforms

# self-defined functions
from pipelines import nst
from utils import seed, cal_metrics, cal_IoUs, prepare_dir
from models import VGG19, Classifier1, Classifier2, RITnet
from data_preprocessing import load_data_openeds2019, OpenEDS2019ISTDataset

def iris_style_transfer_openeds2019(
    args: argparse.Namespace,
    dataloader: torch.utils.data.DataLoader,
    vgg: torch.nn.Module,
    ritnet: torch.nn.Module,
    classifier1: torch.nn.Module,
    classifier2: torch.nn.Module,
    c_loss_weight: float,
    s_loss_weight: float,
    nst_epoch: int,
    metric_prefix: str,
    save_dir: str,
    save_period: int = 50,
    ):
    """
    Main function for investigating the influence of iris style transfer (iris recognition and eye segmentation) using OpenEDS2019 dataset.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        dataloader (torch.utils.data.DataLoader): data loader.
        vgg (torch.nn.Module): vgg network.
        ritnet (torch.nn.Module): ritnet model.
        classifier1 (torch.nn.Module): CNN feature-based classifier.
        classifier2 (torch.nn.Module): style feature-based classifier.
        c_loss_weight (float): content loss weight (alpha).
        s_loss_weight (float): style loss weight (beta).
        nst_epoch (int): iris style transfer iterations.
        metric_prefix (str): wandb log prefix.
        save_dir (str): path to save directory.
        save_period (int): how often to save.
    """
    
    c_preds1_pre_nst, c_preds2_pre_nst, c_preds1_post_nst, c_preds2_post_nst = [], [], [], []
    c_labelss, s_labelss = [], []
    ious0, ious1, ious2, ious3, mious = [], [], [], [], []
    c_losses, s_losses = [], []
    t_resize = transforms.Resize((224, 224))
    t_toPIL = transforms.ToPILImage()

    for batch_id, (c_imgs, c_labels, c_ms_iris, c_iris_bbs, c_ms_gt, s_irises, s_labels) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            # save the 1st image of the batch
            if batch_id % save_period == 0:
                t_toPIL(c_imgs[0]).save(save_dir + 'batch_' + str(batch_id) + '_raw.png')
                t_toPIL(s_irises[0]).save(save_dir + 'batch_' + str(batch_id) + '_sty.png')
            
            batch_wandb_log = {}
            c_labelss.append(c_labels)
            s_labelss.append(s_labels)

            # collect content irises from batch
            c_irises = []
            c_iris_shapes = []
            for c_img, c_m_iris, (x_min, y_min, x_max, y_max) in zip(c_imgs, c_ms_iris, c_iris_bbs):
                # apply iris mask
                c_img = c_img * c_m_iris

                # apply bounding box
                c_iris = c_img[:, x_min: x_max + 1, y_min: y_max + 1]
                c_iris_shapes.append(c_iris.shape[-2:])

                # resize to (224, 224)
                c_iris = t_resize(c_iris)
                
                c_irises.append(c_iris)
            c_irises = torch.stack(c_irises)
            c_irises = c_irises.repeat(1, 3, 1, 1)

            # classifications before iris style transfer
            x, x_c, x_s = vgg(c_irises)
            pred1 = classifier1(x)
            pred2 = classifier2(x_s)
            c_preds1_pre_nst.append(pred1)
            c_preds2_pre_nst.append(pred2)
            cal_metrics(c_labels.cpu(), pred1.cpu(), batch_wandb_log, metric_prefix + 'pre/c1/batch/')
            cal_metrics(c_labels.cpu(), pred2.cpu(), batch_wandb_log, metric_prefix + 'pre/c2/batch/')
            cal_metrics(s_labels.cpu(), pred1.cpu(), batch_wandb_log, metric_prefix + 'pre/c1/mis/batch/')
            cal_metrics(s_labels.cpu(), pred2.cpu(), batch_wandb_log, metric_prefix + 'pre/c2/mis/batch/')
        
        # apply nst to content iris
        new_c_irises, _, c_loss_hist, s_loss_hist = nst(c_irises, 
                                                        s_irises.repeat(1, 3, 1, 1), 
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
            new_c_irises2 = []
            for i in range(len(new_c_irises)):
                new_c_iris = new_c_irises[i] 
                
                # resize new iris texture to its original size
                raw_shape = c_iris_shapes[i]
                new_c_iris = transforms.Resize(raw_shape)(new_c_iris)
                
                # apply mask again
                x_min, y_min, x_max, y_max = c_iris_bbs[i]
                c_m_iris = c_ms_iris[i, :, x_min: x_max + 1, y_min: y_max + 1] 
                new_c_iris = new_c_iris * c_m_iris

                # replace the old iris with the new iris
                c_imgs[i, :, x_min: x_max + 1, y_min: y_max + 1] *= ~c_m_iris
                c_imgs[i, :, x_min: x_max + 1, y_min: y_max + 1] += new_c_iris
                
                # resize to (224, 224)
                new_c_iris = t_resize(new_c_iris)

                new_c_irises2.append(new_c_iris)
            new_c_irises = torch.stack(new_c_irises2)
            new_c_irises = new_c_irises.repeat(1, 3, 1, 1)

            # save the 1st image of the batch
            if batch_id % save_period == 0:
                t_toPIL(c_imgs[0]).save(save_dir + 'batch_' + str(batch_id) + '_new.png')
            
            # classifications after iris style transfer
            x, x_c, x_s = vgg(new_c_irises)
            pred1 = classifier1(x)
            pred2 = classifier2(x_s)
            c_preds1_post_nst.append(pred1)
            c_preds2_post_nst.append(pred2)
            cal_metrics(c_labels.cpu(), pred1.cpu(), batch_wandb_log, metric_prefix + 'post/c1/batch/')
            cal_metrics(c_labels.cpu(), pred2.cpu(), batch_wandb_log, metric_prefix + 'post/c2/batch/')
            cal_metrics(s_labels.cpu(), pred1.cpu(), batch_wandb_log, metric_prefix + 'post/c1/mis/batch/')
            cal_metrics(s_labels.cpu(), pred2.cpu(), batch_wandb_log, metric_prefix + 'post/c2/mis/batch/')

            # compute mIoU after nst
            c_ms_ritnet = [ritnet(c_img) for c_img in c_imgs]
            c_ms_ritnet = torch.stack(c_ms_ritnet).squeeze(1)
            iou_per_class, miou = cal_IoUs(c_ms_ritnet, c_ms_gt)
            ious0.append(iou_per_class[0]) ; batch_wandb_log[metric_prefix + 'post/batch/iou0'] = torch.nanmean(iou_per_class[0])
            ious1.append(iou_per_class[1]) ; batch_wandb_log[metric_prefix + 'post/batch/iou1'] = torch.nanmean(iou_per_class[1])
            ious2.append(iou_per_class[2]) ; batch_wandb_log[metric_prefix + 'post/batch/iou2'] = torch.nanmean(iou_per_class[2])
            ious3.append(iou_per_class[3]) ; batch_wandb_log[metric_prefix + 'post/batch/iou3'] = torch.nanmean(iou_per_class[3])
            mious.append(miou) ; batch_wandb_log[metric_prefix + 'post/batch/miou'] = torch.nanmean(miou)

        # batch log
        wandb.log(batch_wandb_log)

    # metrics
    wandb_log = {}
    with torch.no_grad():
        # save ious to files
        ious0 = torch.cat(ious0) ; torch.save(ious0, save_dir + 'ious0_post.pt') ; wandb_log[metric_prefix + 'post/mean_iou0'] = torch.nanmean(ious0)
        ious1 = torch.cat(ious1) ; torch.save(ious1, save_dir + 'ious1_post.pt') ; wandb_log[metric_prefix + 'post/mean_iou1'] = torch.nanmean(ious1)
        ious2 = torch.cat(ious2) ; torch.save(ious2, save_dir + 'ious2_post.pt') ; wandb_log[metric_prefix + 'post/mean_iou2'] = torch.nanmean(ious2)
        ious3 = torch.cat(ious3) ; torch.save(ious3, save_dir + 'ious3_post.pt') ; wandb_log[metric_prefix + 'post/mean_iou3'] = torch.nanmean(ious3)
        mious = torch.cat(mious) ; torch.save(mious, save_dir + 'mious_post.pt') ; wandb_log[metric_prefix + 'post/mean_miou'] = torch.nanmean(mious)
        
        # content and style losses
        c_loss = torch.nanmean(torch.as_tensor(c_losses))
        s_loss = torch.nanmean(torch.as_tensor(s_losses))
        cs_loss = c_loss * c_loss_weight + s_loss * s_loss_weight
        wandb_log[metric_prefix + '/c_loss'] = c_loss
        wandb_log[metric_prefix + '/s_loss'] = s_loss
        wandb_log[metric_prefix + '/cs_loss'] = cs_loss
        
        # classification performance before and after iris style transfer
        c_labelss = torch.cat(c_labelss).cpu()
        cal_metrics(c_labelss, torch.cat(c_preds1_pre_nst ).cpu(), wandb_log, metric_prefix + 'pre/c1/' )
        cal_metrics(c_labelss, torch.cat(c_preds2_pre_nst ).cpu(), wandb_log, metric_prefix + 'pre/c2/' )
        cal_metrics(c_labelss, torch.cat(c_preds1_post_nst).cpu(), wandb_log, metric_prefix + 'post/c1/')
        cal_metrics(c_labelss, torch.cat(c_preds2_post_nst).cpu(), wandb_log, metric_prefix + 'post/c2/')
        
        # false acceptance rate
        s_labelss = torch.cat(s_labelss).cpu()
        cal_metrics(s_labelss, torch.cat(c_preds1_pre_nst ).cpu(), wandb_log, metric_prefix + 'pre/c1/mis/' )
        cal_metrics(s_labelss, torch.cat(c_preds2_pre_nst ).cpu(), wandb_log, metric_prefix + 'pre/c2/mis/' )
        cal_metrics(s_labelss, torch.cat(c_preds1_post_nst).cpu(), wandb_log, metric_prefix + 'post/c1/mis/')
        cal_metrics(s_labelss, torch.cat(c_preds2_post_nst).cpu(), wandb_log, metric_prefix + 'post/c2/mis/')
    
    wandb.log(wandb_log)
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    
    # general parameters
    parser.add_argument('-P', '--project', type = str, default = 'iris-style-transfer-openeds2019', help = 'project name')
    parser.add_argument('-seed', '--seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-device', '--device', type = int, default = 0, help = 'GPU index. -1 stands for CPU.')
    parser.add_argument('-path1', '--classifier1_path', type = str, default = './models/weights/seed_42_Classifier1_lr_1e-05_prob_0.0_epoch_100.pth', help = 'pretrained classifier1 weight path')
    parser.add_argument('-path2', '--classifier2_path', type = str, default = './models/weights/seed_42_Classifier2_lr_1e-05_prob_0.0_epoch_500.pth', help = 'pretrained classifier2 weight path')
    parser.add_argument('--eval_train', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether also evaluate on the training set')

    # hyperparameters
    parser.add_argument('-T', '--test_split_ratio', type = float, default = 0.2, help = 'train-test-split ratio')
    # parser.add_argument('-E', '--nst_epochs', type = int, default = 200, help = 'number of epochs for neural style transfer')
    parser.add_argument('-bs', '--bs', type = int, default = 64, help = 'batch size')
    parser.add_argument('-cw', '--c_loss_weight', type = int, default = 1, help = 'cw')
    # parser.add_argument('-sw', '--s_loss_weight', type = int, default = 1, help = 'sw')
    
    args = parser.parse_args()
    args.device = 'cuda:' + str(args.device) if args.device >= 0 else 'cpu'

    # reproducibility
    seed(args.seed)

    # load data
    train_x, train_y, train_m, test_x, test_y, test_m, num_class = load_data_openeds2019(test_split_ratio = args.test_split_ratio, load_seg = True)
    print('number of classes:', num_class)
    
    # models
    vgg = VGG19()
    vgg.to(args.device)
    ritnet = RITnet()
    ritnet.to(args.device)
    classifier1 = Classifier1(num_class = num_class)
    classifier1.load_state_dict(torch.load(args.classifier1_path, weights_only = True, map_location = 'cpu'))
    classifier1.to(args.device)
    classifier1.eval()
    classifier2 = Classifier2(num_class = num_class)
    classifier2.load_state_dict(torch.load(args.classifier2_path, weights_only = True, map_location = 'cpu'))
    classifier2.to(args.device)
    classifier2.eval()

    # beta range and epoch range
    s_loss_weights = [1] # [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    nst_epochs = [200] # [1, 5, 10, 20, 50, 100, 150, 200]
    
    # for test dataset
    dataset = OpenEDS2019ISTDataset(test_x , test_y , test_m , ritnet = ritnet, device = args.device)
    dataloader = torch.utils.data.DataLoader(dataset , batch_size = args.bs, shuffle = False)
    for sw in s_loss_weights:
        for nst_epoch in nst_epochs:
            # prepare folder and wandb init
            save_dir = 'saved/openeds2019/sw_' + str(sw) + '_epoch_' + str(nst_epoch) + '/test/'
            prepare_dir(save_dir)
            args.name = 'seed ' + str(args.seed) + ' sw ' + str(sw) + ' epoch ' + str(nst_epoch) + ' test'
            wandb.init(project = args.project, name = args.name, config = args.__dict__, anonymous = "allow")
            
            # record mious
            ious0 = torch.cat(dataset.ious0) ; torch.save(ious0, save_dir + 'ious0_pre.pt') ; wandb.log({'test/pre/mean_iou0' : torch.nanmean(ious0)})
            ious1 = torch.cat(dataset.ious1) ; torch.save(ious1, save_dir + 'ious1_pre.pt') ; wandb.log({'test/pre/mean_iou1' : torch.nanmean(ious1)})
            ious2 = torch.cat(dataset.ious2) ; torch.save(ious2, save_dir + 'ious2_pre.pt') ; wandb.log({'test/pre/mean_iou2' : torch.nanmean(ious2)})
            ious3 = torch.cat(dataset.ious3) ; torch.save(ious3, save_dir + 'ious3_pre.pt') ; wandb.log({'test/pre/mean_iou3' : torch.nanmean(ious3)})
            mious = torch.cat(dataset.mious) ; torch.save(mious, save_dir + 'mious_pre.pt') ; wandb.log({'test/pre/mean_miou' : torch.nanmean(mious)})

            # iris style transfer main function
            iris_style_transfer_openeds2019(args, dataloader , vgg, ritnet, classifier1, classifier2, args.c_loss_weight, sw, nst_epoch, 'test/', save_dir)
            wandb.finish()

    # for training dataset
    if args.eval_train:
        dataset = OpenEDS2019ISTDataset(train_x, train_y, train_m, ritnet = ritnet, device = args.device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.bs, shuffle = False)    
        for sw in s_loss_weights:
            for nst_epoch in nst_epochs:
                # prepare folder and wandb init
                save_dir = 'saved/openeds2019/sw_' + str(sw) + '_epoch_' + str(nst_epoch) + '/train/'
                prepare_dir(save_dir)
                args.name = 'seed ' + str(args.seed) + ' sw ' + str(sw) + ' epoch ' + str(nst_epoch) + ' train'
                wandb.init(project = args.project, name = args.name, config = args.__dict__, anonymous = "allow")
                
                # record mious
                ious0 = torch.cat(dataset.ious0) ; torch.save(ious0, save_dir + 'ious0_pre.pt') ; wandb.log({'train/pre/mean_iou0' : torch.nanmean(ious0)})
                ious1 = torch.cat(dataset.ious1) ; torch.save(ious1, save_dir + 'ious1_pre.pt') ; wandb.log({'train/pre/mean_iou1' : torch.nanmean(ious1)})
                ious2 = torch.cat(dataset.ious2) ; torch.save(ious2, save_dir + 'ious2_pre.pt') ; wandb.log({'train/pre/mean_iou2' : torch.nanmean(ious2)})
                ious3 = torch.cat(dataset.ious3) ; torch.save(ious3, save_dir + 'ious3_pre.pt') ; wandb.log({'train/pre/mean_iou3' : torch.nanmean(ious3)})
                mious = torch.cat(dataset.mious) ; torch.save(mious, save_dir + 'mious_pre.pt') ; wandb.log({'train/pre/mean_miou' : torch.nanmean(mious)})
                
                # iris style transfer main function
                iris_style_transfer_openeds2019(args, dataloader, vgg, ritnet, classifier1, classifier2, args.c_loss_weight, sw, nst_epochs, 'train/', save_dir)
                wandb.finish()