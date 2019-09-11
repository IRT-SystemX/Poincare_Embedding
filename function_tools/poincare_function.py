import torch
from torch.autograd import Function
from function_tools import torch_function

# redefining the hyperbolic distance forward and grad
class PoincareDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        with torch.no_grad():
            x_norm = torch.sum(x ** 2, dim=-1)
            y_norm = torch.sum(y ** 2, dim=-1)
            d_norm = torch.sum((x-y) ** 2, dim=-1)
            cc = 1+2*d_norm/((1-x_norm)*(1-y_norm)) 
            ctx.save_for_backward(cc, x, y , x_norm, y_norm)
            return  torch.log(cc + torch.sqrt(cc**2-1))

    @staticmethod
    def get_grad(output, theta, x, theta_norm, x_norm ):
        with torch.no_grad():
            beta = (1-x_norm)
            alpha = (1-theta_norm)
            coef_beta = 4/((torch.sqrt((output+1e-6)**2-1)) * beta)
            b = (x_norm - 2* (theta*x).sum(-1) + 1 )/(alpha**2)
            c = x/alpha.unsqueeze(-1).expand_as(x)
            b_c = b.unsqueeze(-1).expand_as(x)*theta - c
            return coef_beta.unsqueeze(-1).expand_as(x) * b_c
    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            output,x, y, x_norm,y_norm = ctx.saved_tensors
            return (PoincareDistance.get_grad(output,x, y, x_norm, y_norm)* grad_output.unsqueeze(-1).expand_as(x),
                PoincareDistance.get_grad(output, y, x, y_norm, x_norm)* grad_output.unsqueeze(-1).expand_as(x))


def poincare_distance(x, y):
    return PoincareDistance.apply(x, y)


def add(x, y):
    nx = torch.sum(x ** 2, dim=-1, keepdim=True).expand_as(x)
    ny = torch.sum(y ** 2, dim=-1, keepdim=True).expand_as(x)
    xy = (x * y).sum(-1, keepdim=True).expand_as(x)
    return ((1 + 2*xy+ ny)*x + (1-nx)*y)/(1+2*xy+nx*ny)


def log(k, x):
    kpx = add(-k,x)
    norm_kpx = kpx.norm(2,-1, keepdim=True).expand_as(kpx)
    norm_k = k.norm(2,-1, keepdim=True).expand_as(kpx)
    return (1-norm_k)* ((torch_function.arcTanh(norm_kpx))) * (kpx/norm_kpx)


def exp(k, x):
    norm_k = k.norm(2,-1, keepdim=True).expand_as(k)
    lambda_k = 2/(1-norm_k)
    norm_x = x.norm(2,-1, keepdim=True).expand_as(x)
    direction = x/norm_x
    factor = torch.tanh((lambda_k * norm_x)/2)
    return add(k,direction*factor)