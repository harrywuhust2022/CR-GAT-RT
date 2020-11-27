import torch
import torch.nn.functional as F
from utils.helpers import colorize_mask
import cv2
import os
import matplotlib.pyplot as plt
from tools import calculate_correct_map,getw,get_CR
class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True,p=1):
        '''
        Initialize the attacker step with a given perturbation magnitude.
        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.p=p 

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set
        Args:
            ch.tensor x : the input to project back into the feasible set.
        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).
        Parameters:
            g (ch.tensor): the raw gradient
        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:
    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=self.p, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1),self.p,dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        l = len(x.shape) - 1
        rp = torch.randn_like(x)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        return torch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)

def from_zeroone_to_miu(images):
    images2=images.permute(0,2,3,1)
    MEAN = torch.tensor([0.45734706, 0.43338275, 0.40058118])
    STD = torch.tensor([0.23965294, 0.23532275, 0.2398498])
    images2=(images2-MEAN)/STD
    images2=images2.permute(0,3,1,2)
    return images2

def save(ts,str):
    img=ts[0].permute(1,2,0)
    img=img.detach().cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str,img)

def save2d(o, output_path, name):
    o = abs(o[0])
    o = (o - o.min()) / (o.max() - o.min())
    plt.imshow(o, cmap='hot')
    plt.savefig(os.path.join(output_path, name + '.png'), aspect='auto', bbox_inches='tight')

def save_images(output, output_path, name, palette):
    # Saves the image, the model output and the results after the post processing
    mask = output.detach().squeeze(0).cpu().numpy()
    mask = F.softmax(torch.from_numpy(mask), dim=0).argmax(0).cpu().numpy()
    w, h = mask.shape
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, name + '.png'))
cnt=0
def get_adv_examples(data, target, model, lossfunc, eps, step_size, iterations,palette):
    global cnt
    cnt+=1
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step_count=0
    
    output = model(x)['out']
    #save(x*255,"outputs//%d_0_atk.png"%cnt)

    step = L2Step(x, eps, step_size)
    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        step_count+=1
        output = model(x)['out']
        correct=calculate_correct_map(output,target,21)
        #print(correct.sum())
        CR1, CR2, surrogate = getw(output)
        #save2d(CR2,"outputs","%d_%d_CR"%(cnt,step_count-1))
        
        #save_images(output,"outputs","%d_%d_res"%(cnt,step_count-1),palette)
        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)

        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        #save2d(torch.abs(grad[0][0].unsqueeze(0))/torch.abs(grad.max()),"outputs","%d_%d_GR"%(cnt,step_count-1))
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
        #save(x*255,"outputs//%d_%d_atk.png"%(cnt,step_count))
    return x

def get_adv_examples_L2(data, target, model, lossfunc, eps, step_size, iterations,palette):
    global cnt
    cnt+=1
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step_count=0
    
    output = model(x)['out']

    step = L2Step(x, eps, step_size,True,2)
    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        output = model(x)['out']
        correct=calculate_correct_map(output,target,21)
        
        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)

        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x
def get_adv_examples_LINF(data, target, model, lossfunc, eps, step_size, iterations,palette):
    global cnt
    cnt+=1
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step_count=0
    delta=0
    output = model(x)['out']


    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        output = model(x)['out']
        correct=calculate_correct_map(output,target,21)
        
        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)

        
        grad, = torch.autograd.grad(m * loss, [x])
        with torch.no_grad():
            delta=x-data
            delta=delta+step_size*torch.sign(grad)
            delta=torch.clamp(delta,-eps,eps)
            x=torch.clamp(x+delta,0,1)
    return x   
def get_adv_examples_FGSM(data, target, model, lossfunc, eps, step_size, iterations,palette):
    global cnt
    cnt+=1
    m = 1  # for untargeted 1
    iterations=1
    step_size=eps*2
    iterator = range(iterations)
    x = data.clone()
    step_count=0
    
    output = model(x)['out']
    #save(x*255,"outputs//%d_0_atk.png"%cnt)
    
    step = L2Step(x, eps, step_size,True,1)
    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        step_count+=1
        output = model(x)['out']
        correct=calculate_correct_map(output,target,21)
        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)

        grad, = torch.autograd.grad(m * loss, [x])
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    #print(torch.norm(x-data,p=1,dim=(1,2,3)))
    return x    
def get_adv_examples_SL2(data, target, model, lossfunc, eps, step_size, iterations,palette):
    global cnt
    cnt+=1
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step_count=0
    lastx=0
    output = model(x)['out']
    #save(x*255,"outputs//S%d_0_atk.png"%cnt)
    array=[]
    step = L2Step(x, eps, step_size,True,2)
    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        step_count+=1
        output = model(x)['out']
        
        CR1, CR2, surrogate = getw(output)

        array.append(CR1.clone().cpu())

        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)
        
        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        if(step_count!=1):
            mp=array[step_count-2]-array[step_count-1]
            
            grad=grad.mul(torch.abs(mp))
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)
    return x
def get_adv_examples_SLINF(data, target, model, lossfunc, eps, step_size, iterations,palette):
    global cnt
    cnt+=1
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step_count=0
    delta=0
    array=[]
    step = L2Step(x, eps, step_size,True,2)
    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        step_count+=1
        output = model(x)['out']
        
        CR1, CR2, surrogate = getw(output)

        array.append(CR1.clone().cpu())

        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)

        grad, = torch.autograd.grad(m * loss, [x])
        if(step_count!=1):
            mp=array[step_count-2]-array[step_count-1]
            grad=grad.mul(torch.abs(mp))
        with torch.no_grad():
            delta=x-data
            delta=delta+step_size*torch.sign(grad)
            delta=torch.clamp(delta,-eps,eps)
            x=torch.clamp(x+delta,0,1)
    return x
def get_adv_examples_S(data, target, model, lossfunc, eps, step_size, iterations,palette):
    global cnt
    cnt+=1
    m = 1  # for untargeted 1
    iterator = range(iterations)
    x = data.clone()
    step_count=0
    lastx=0
    output = model(x)['out']
    #save(x*255,"outputs//S%d_0_atk.png"%cnt)
    array=[]
    step = L2Step(x, eps, step_size)
    for _ in iterator:
        x = x.clone().detach().requires_grad_(True)
        step_count+=1
        output = model(x)['out']
        correct=calculate_correct_map(output,target,21)
        #print(correct.sum())
        CR1, CR2, surrogate = getw(output)
        #save2d(CR2,"outputs","S%d_%d_CR"%(cnt,step_count-1))
        array.append(CR1.clone().cpu())
        #save_images(output,"outputs","S%d_%d_res"%(cnt,step_count-1),palette)
        losses = lossfunc(output, target.cuda())

        loss = torch.mean(losses)
        
        if step.use_grad:
            grad, = torch.autograd.grad(m * loss, [x])
        if(step_count!=1):
            mp=array[step_count-2]-array[step_count-1]
            #save2d(torch.abs(grad[0][0].unsqueeze(0))/torch.abs(grad.max()),"outputs","S%d_%d_GR"%(cnt,step_count-1))
            grad=grad.mul(torch.abs(mp))
            #save2d(torch.abs(grad[0][0].unsqueeze(0))/torch.abs(grad.max()),"outputs","S%d_%d_GD"%(cnt,step_count-1))
        #lastx=x.clone()
        with torch.no_grad():
            x = step.step(x, grad)
            x = step.project(x)

        #save(x*255,"outputs//S%d_%d_atk.png"%(cnt,step_count))
    return x