import tqdm
import torch

# self-defined functions
from utils import crop_image
from models import VGG19, ContentLoss_L2, StyleLoss_BN, StyleLoss_Gram, RITnet

def nst(c_img: torch.Tensor, 
        s_img: torch.Tensor, 
        clone_content: bool = True, 
        BN_loss: bool = True,
        c_loss_weight: float = 1,
        s_loss_weight: float = 1,
        lr: float = 1,
        epochs: int = 200,
        vgg: torch.nn.Module = None,
        use_tqdm: bool = True,
        device: str = 'cuda:0',
        ) ->  tuple[torch.Tensor, list, list, list]:
    """
    Neural style transfer pipeline.

    Arguments:
        c_img (torch.Tensor): content image tensor. 
        s_img (torch.Tensor): style image tensor.
        clone_content (bool): whether the initilization is a clone on content image or random noise. 
        BN_loss (bool): whether to use BN style loss or gram matrix style loss.
        c_loss_weight (float): content loss weight (alpha).
        s_loss_weight (float): style loss weight (beta).
        lr (float): learning rate.
        epochs (int): style transfer iterations.
        vgg (torch.nn.Module): vgg model.
        use_tqdm (bool): whether to use tqdm for showing style transfer progress.
        device (str): CPU or GPU.
        
    Returns:
        x (torch.Tensor): stylized image tensor.
        x_hist (list[torch.Tensor]): stylized image tensor history. 
        c_loss_hist (list[float]): content loss history.
        s_loss_hist (list[float]): style loss history.
    """
    
    # vgg model
    if vgg is None:
        vgg = VGG19()
    vgg.to(device)

    # input and output images
    c_img = c_img.to(device)
    s_img = s_img.to(device)
    if clone_content:
        x = c_img.clone()
    else:
        x = torch.rand(c_img.shape).to(device)
    x = x.contiguous() # sometimes x is not contiguous for some unknown reason...
    x.requires_grad = True
     
    # optimizer
    optim = torch.optim.LBFGS([x], lr = lr)
    
    # content and style loss functions
    _, c_features, _ = vgg(c_img)
    _, _, s_features = vgg(s_img)
    c_loss_func = ContentLoss_L2(targets = c_features)
    if BN_loss:
        s_loss_func = StyleLoss_BN(targets = s_features)
    else:
        s_loss_func = StyleLoss_Gram(targets = s_features)
    
    # style transfer main loop
    x_hist = []
    c_loss_hist = []
    s_loss_hist = []
    current_epoch = [0]
    
    # nst loop
    if use_tqdm:
        pbar = tqdm.tqdm(total = epochs)
    while current_epoch[0] < epochs:
        def closure():
            with torch.no_grad():
                x.clamp_(0, 1)
                # x.mul_(mask)
                
            optim.zero_grad()
            _, x_c, x_s = vgg(x)
            c_loss = c_loss_func(x_c)
            s_loss = s_loss_func(x_s)
            loss = c_loss * c_loss_weight + s_loss * s_loss_weight
            loss.backward()

            # records
            x_hist.append(x.detach().cpu())
            c_loss_hist.append(c_loss.item())
            s_loss_hist.append(s_loss.item())
            
            current_epoch[0] += 1
            if use_tqdm:
                pbar.update(1)

            return loss
            
        optim.step(closure)

    if use_tqdm:
        pbar.close()

    x = x.detach()
    x.clamp_(0, 1)
    return x, x_hist, c_loss_hist, s_loss_hist

def mask_and_crop_iris(x: torch.Tensor, 
                       ritnet: torch.nn.Module = None,
                       glint_threshold: float = 0.8,
                       device: str = 'cuda:0',
                       ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    """
    Helper function for masking out the non-iris part of an eye image, trimming black border, and removing glints.

    Arguments:
        x (torch.Tensor): eye image tensor. 
        ritnet (torch.nn.Module): RITnet model.
        glint_threshold (float): threshold for glints.
        device (str): CPU or GPU.

    Returns:
        x (torch.Tensor): iris texture tensor.
        m (torch.Tensor): iris mask.
        x_min (int): iris bounding box corner. 
        y_min (int): iris bounding box corner.
        x_max (int): iris bounding box corner.
        y_max (int): iris bounding box corner.
    """
    
    x = x.to(device)

    # ritnet model
    if ritnet is None:
        ritnet = RITnet()
    ritnet.to(device)
    
    # compute ritnet mask
    m_ritnet = ritnet(x)
    m_ritnet = m_ritnet == 2 # 2 represents iris part

    # compute non-glint mask
    m_glint = x <= glint_threshold
    
    # apply mask
    m = m_ritnet * m_glint
    x = x * m
    
    # crop image
    x_min, y_min, x_max, y_max = crop_image(x, return_idx = True)
    x = x[:, x_min: x_max + 1, y_min: y_max + 1]
    m = m[:, x_min: x_max + 1, y_min: y_max + 1]

    # grayscale to RGB
    x = x.repeat(3, 1, 1)
    
    return x, m, x_min, y_min, x_max, y_max