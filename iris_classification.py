import tqdm
import torch
import wandb
import argparse
import torch.nn.functional as F

# self-defined functions
from utils import seed, cal_metrics
from models import VGG19, Classifier1, Classifier2
from data_preprocessing import load_data_openeds2019, OpenEDS2019IRDataset

def iris_classification(args: argparse.Namespace) -> None:
    """
    Main function for training iris classifiers using CNN features and style features (OpenEDS2019 dataset).

    Arguments:
        args (argparse.Namespace): parsed argument object. 
    """

    # reproducibility
    seed(args.seed)

    # datasets and dataloader
    train_x, train_y, _, test_x, test_y, _, num_class = load_data_openeds2019(test_split_ratio = args.test_split_ratio)
    train_d = OpenEDS2019IRDataset(
        train_x, train_y, 
        rotation_prob   = args.rotation_prob, 
        rotation_degree = args.rotation_degree,
        perspect_prob   = args.perspect_prob,
        perspect_degree = args.perspect_degree,
        device = args.device
        )
    test_d  = OpenEDS2019IRDataset(test_x, test_y, 
        rotation_prob   = args.rotation_prob, 
        rotation_degree = args.rotation_degree,
        perspect_prob   = args.perspect_prob,
        perspect_degree = args.perspect_degree,
        device = args.device
        )
    train_l = torch.utils.data.DataLoader(train_d, batch_size = args.bs, shuffle = True)
    test_l  = torch.utils.data.DataLoader(test_d , batch_size = args.bs, shuffle = False)
    print('number of classes:', num_class)
    
    # models and optimizer
    vgg = VGG19()
    vgg.to(args.device)
    classifier1 = Classifier1(num_class = num_class)
    classifier2 = Classifier2(num_class = num_class)
    classifier1.to(args.device)
    classifier2.to(args.device)
    params = list(classifier1.parameters()) + list(classifier2.parameters())
    if not args.freeze_vgg:
        for param in vgg.parameters():
            param.requires_grad = True    
        params += list(vgg.parameters())
    optim = torch.optim.Adam(params, lr = args.lr)
    
    # main loop
    for e in tqdm.tqdm(range(args.epochs)):
        # training
        classifier1.train()
        classifier2.train()
        if not args.freeze_vgg:
            vgg.train()
        preds1, preds2, labels = [], [], []
        for batch_id, (x, y) in enumerate(train_l):
            optim.zero_grad()
            x = x.repeat(1, 3, 1, 1) # grayscale to RGB  
            x, x_c, x_s = vgg(x)
            p1 = classifier1(x)
            p2 = classifier2(x_s)
            
            loss = F.cross_entropy(p1, y) + F.cross_entropy(p2, y)
            loss.backward()
            optim.step()

            preds1.append(p1)
            preds2.append(p2)
            labels.append(y)
        preds1 = torch.cat(preds1).detach().cpu()
        preds2 = torch.cat(preds2).detach().cpu()
        labels = torch.cat(labels).detach().cpu()
        wandb_log = {}
        cal_metrics(labels, preds1, wandb_log, 'train/c1/')
        cal_metrics(labels, preds2, wandb_log, 'train/c2/')

        # test
        classifier1.eval()
        classifier2.eval()
        if not args.freeze_vgg:
            vgg.eval()
        preds1, preds2, labels = [], [], []
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(test_l):
                # x = x.to(args.device)
                x, x_c, x_s = vgg(x)
                p1 = classifier1(x)
                p2 = classifier2(x_s)
                preds1.append(p1)
                preds2.append(p2)
                labels.append(y)
            preds1 = torch.cat(preds1).detach().cpu()
            preds2 = torch.cat(preds2).detach().cpu()
            labels = torch.cat(labels).detach().cpu()
            cal_metrics(labels, preds1, wandb_log, 'test/c1/')
            cal_metrics(labels, preds2, wandb_log, 'test/c2/')
        
        wandb.log(wandb_log)

        # save model
        if args.save_period > 0 and args.rotation_prob == args.perspect_prob == 0 and (e + 1) % args.save_period == 0:
            torch.save(classifier1.state_dict(), './models/weights/seed_' + str(args.seed) + '_Classifier1_lr_' + str(args.lr) + '_prob_' + str(args.rotation_prob) + '_epoch_' + str(e + 1) + '.pth')
            torch.save(classifier2.state_dict(), './models/weights/seed_' + str(args.seed) + '_Classifier2_lr_' + str(args.lr) + '_prob_' + str(args.rotation_prob) + '_epoch_' + str(e + 1) + '.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    
    # general parameters
    parser.add_argument('-P', '--project', type = str, default = 'iris-style-transfer', help = 'project name')
    parser.add_argument('-seed', '--seed', type = int, default = 42, help = 'random seed')
    parser.add_argument('-device', '--device', type = int, default = 0, help = 'GPU index. -1 stands for CPU.')
    parser.add_argument('-SP', '--save_period', type = int, default = 50, help = 'how often the trained model should be saved. -1 stands for no saving.')
    
    # hyperparameters
    parser.add_argument('-E', '--epochs', type = int, default = 500, help = 'number of epochs')
    parser.add_argument('-T', '--test_split_ratio', type = float, default = 0.2, help = 'train-test-split ratio')
    parser.add_argument('-bs', '--bs', type = int, default = 64, help = 'batch size')
    parser.add_argument('-lr', '--lr', type = float, default = 1e-5, help = 'learning rate')
    parser.add_argument('-rp', '--rotation_prob', type = float, default = 0, help = 'image random rotation probabilty')
    parser.add_argument('-rd', '--rotation_degree', type = float, default = 180, help = 'image random rotation degree')
    parser.add_argument('-pp', '--perspect_prob', type = float, default = 0, help = 'image random perspective transformation probabilty')
    parser.add_argument('-pd', '--perspect_degree', type = float, default = 0.3, help = 'image random perspective transformation degree')
    parser.add_argument('--freeze_vgg', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'freeze vgg19 or not')

    args = parser.parse_args()
    args.device = 'cuda:' + str(args.device) if args.device >= 0 else 'cpu'
    args.name = 'seed ' + str(args.seed) + ' rd ' + str(args.rotation_degree) + ' pd ' + str(args.perspect_degree) + ' lr ' + str(args.lr)
    wandb.init(project = args.project, name = args.name, config = args.__dict__, anonymous = "allow")
    
    iris_classification(args)