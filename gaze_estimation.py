import tqdm
import torch
import wandb
import argparse

# self-defined functions
from utils import seed, angular_distance
from models import GazeEstimator1, GazeEstimator2
from data_preprocessing import load_data_openeds2020

def gaze_estimation(args: argparse.Namespace, lrs: list[float] = [1e-6, 1e-5, 1e-4]) -> None:
    """
    Main function for training gaze estimation models (OpenEDS2020 dataset).

    Arguments:
        args (argparse.Namespace): parsed argument object. 
        lrs (list[float]): learning rates to try.
    """

    # reproducibility
    seed(args.seed)
    
    # datasets and dataloaders
    print('loading training set...')
    images, labels = load_data_openeds2020(extract_feature = args.estimator == 1, estimator = args.estimator, postfix = 'train/', device = args.device)
    train_d = torch.utils.data.TensorDataset(images, labels)
    train_l = torch.utils.data.DataLoader(train_d, batch_size = args.bs, shuffle = True, num_workers = args.num_workers, pin_memory = args.pin_memory)
    print('number of samples in training set:', len(train_d))
    
    print('loading validation set...')
    images, labels = load_data_openeds2020(extract_feature = args.estimator == 1, estimator = args.estimator, postfix = 'validation/', device = args.device)
    valid_d = torch.utils.data.TensorDataset(images, labels)
    valid_l = torch.utils.data.DataLoader(valid_d, batch_size = args.bs, shuffle = False, num_workers = args.num_workers, pin_memory = args.pin_memory)
    print('number of samples in validation set:', len(valid_d))
    
    if args.test:
        print('loading test set...')
        images, labels = load_data_openeds2020(extract_feature = args.estimator == 1, estimator = args.estimator, postfix = 'test/', device = args.device)
        test_d  = torch.utils.data.TensorDataset(images, labels)
        test_l  = torch.utils.data.DataLoader(valid_d, batch_size = args.bs, shuffle = False, num_workers = args.num_workers, pin_memory = args.pin_memory)
        print('number of samples in test set:', len(test_d))
    
    # try each learning rate
    for lr in lrs:
        # reproducibility
        seed(args.seed)   
        
        # wandb initialization
        args.lr = lr
        args.name = 'seed ' + str(args.seed)
        args.name += ' model-based' if args.estimator == 1 else ' appearance-based'
        args.name += ' lr ' + str(args.lr)
        wandb.init(project = args.project, name = args.name, config = args.__dict__, anonymous = "allow")
    
        # models, optimizer, and loss function
        if args.estimator == 1: # model-based gaze estimator
            model = GazeEstimator1(extract_feature = False)
        else: # appearance-based gaze estimator
            model = GazeEstimator2(extract_feature = True, freeze_resnet = False)
                    
        model.to(args.device)    
        optim = torch.optim.Adam(model.parameters(), lr = args.lr)
        loss_func = torch.nn.CosineEmbeddingLoss()
        
        # main loop
        for e in tqdm.tqdm(range(args.epochs)):
            wandb_log = {}
            
            # training
            model.train()
            preds, labels = [], []
            for batch_id, (x, y) in enumerate(train_l):
                optim.zero_grad()
                x = x.to(args.device)
                y = y.to(args.device)
                o = model(x)
                loss = loss_func(o, y, torch.tensor([1], device = args.device))
                loss.backward()
                optim.step()
                preds.append(o)
                labels.append(y)    
            preds = torch.cat(preds).detach().cpu()
            labels = torch.cat(labels).detach().cpu()
            losses = loss_func(preds, labels, torch.tensor([1]))
            radian_distances, degree_distances = angular_distance(preds, labels)
            wandb_log['train/loss'] = losses
            wandb_log['train/radian_distance'] = radian_distances.mean()
            wandb_log['train/degree_distance'] = degree_distances.mean()

            # validation
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for batch_id, (x, y) in enumerate(valid_l):
                    x = x.to(args.device)
                    o = model(x)
                    preds.append(o)
                    labels.append(y)
                preds = torch.cat(preds).detach().cpu()
                labels = torch.cat(labels).detach().cpu()
                losses = loss_func(preds, labels, torch.tensor([1]))
                radian_distances, degree_distances = angular_distance(preds, labels)
                wandb_log['valid/loss'] = losses
                wandb_log['valid/radian_distance'] = radian_distances.mean()
                wandb_log['valid/degree_distance'] = degree_distances.mean()
                
            # test
            if args.test:
                preds, labels = [], []
                with torch.no_grad():
                    for batch_id, (x, y) in enumerate(test_l):
                        x = x.to(args.device)
                        o = model(x)
                        preds.append(o)
                        labels.append(y)
                    preds = torch.cat(preds).detach().cpu()
                    labels = torch.cat(labels).detach().cpu()
                    losses = loss_func(preds, labels, torch.tensor([1]))
                    radian_distances, degree_distances = angular_distance(preds, labels)
                    wandb_log['test/loss'] = losses
                    wandb_log['test/radian_distance'] = radian_distances.mean()
                    wandb_log['test/degree_distance'] = degree_distances.mean()
            
            wandb.log(wandb_log)

            # save model
            if args.save_period > 0 and (e + 1) % args.save_period == 0:
                torch.save(model.state_dict(), './models/weights/seed_' + str(args.seed) + '_GazeEstimator' + str(args.estimator) + '_lr_' + str(args.lr) + '_epoch_' + str(e + 1) + '.pth')
                
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    
    # general parameters
    parser.add_argument('-P', '--project', type = str, default = 'iris-style-transfer', help = 'project name')
    parser.add_argument('-seed', '--seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-device', '--device', type = int, default = 0, help = 'GPU index. -1 stands for CPU.')
    parser.add_argument('-estimator', '--estimator', type = int, default = 1, help = 'train gaze estimator 1 (model-based) or 2 (appearance-based)')
    parser.add_argument('-SP', '--save_period', type = int, default = 10, help = 'how often the trained model should be saved. -1 stands for no saving.')
    parser.add_argument('-T', '--test', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether to evaluate model performance on test set (super large!)')
    parser.add_argument('-W', '--num_workers', type = int, default = 16, help = 'number of workers for data loader')
    parser.add_argument('-M', '--pin_memory', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use pin memory for data loader')
    
    # hyperparameters
    parser.add_argument('-E', '--epochs', type = int, default = 150, help = 'number of epochs')
    parser.add_argument('-bs', '--bs', type = int, default = 128, help = 'batch size')
    parser.add_argument('-lr', '--lr', type = float, default = 1e-5, help = 'learning rate')
    
    args = parser.parse_args()
    assert(args.estimator in [1, 2])
    args.device = 'cuda:' + str(args.device) if args.device >= 0 else 'cpu'
    
    gaze_estimation(args)