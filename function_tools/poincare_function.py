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

def renorm_projection(x, eps=1e-4):
    x_n = x.norm(2, -1)
    if(len(x[x_n>=1.])>0):
        x[x_n>=1.] /= (x_n.unsqueeze(-1).expand_as(x[x_n>=1.]) + eps)
    return x

def add(x, y):
    nx = torch.sum(x ** 2, dim=-1, keepdim=True).expand_as(x)
    ny = torch.sum(y ** 2, dim=-1, keepdim=True).expand_as(x)
    xy = (x * y).sum(-1, keepdim=True).expand_as(x)
    return ((1 + 2*xy+ ny)*x + (1-nx)*y)/(1+2*xy+nx*ny)

def log(k, x):
    kpx = add(-k,x)
    norm_kpx = kpx.norm(2,-1, keepdim=True).expand_as(kpx)
    norm_k = k.norm(2,-1, keepdim=True).expand_as(kpx)
    return (1-norm_k**2)* ((torch_function.arcTanh(norm_kpx))) * (kpx/norm_kpx)

def exp(k, x):
    norm_k = k.norm(2,-1, keepdim=True).expand_as(k)
    lambda_k = 2/(1-norm_k**2)
    norm_x = x.norm(2,-1, keepdim=True).expand_as(x)
    direction = x/norm_x
    factor = torch.tanh((lambda_k * norm_x)/2)
    res = add(k,direction*factor)
    if(0 != len((norm_x==0).nonzero())):
        res[norm_x == 0] = k[norm_x == 0]
    return res



def test():
    import numpy as np
    from torch import nn
    from function_tools import poincare_function as tf
    from function_tools.numpy_function import RiemannianFunction as nf
    import cmath
    def sigmoid (x):
        return 1/(1+cmath.exp(-x))


    x = torch.rand(3,2)/1.5 
    y = torch.rand(3,2)/1.5 
    xn = x[:,0].detach().numpy() + x[:,1].detach().numpy() *1j
    yn = y[:,0].detach().numpy() + y[:,1].detach().numpy() *1j
    x = nn.Parameter(x)
    y = nn.Parameter(y)

    print("LOG : ")
    print("   Torch version")
    print("   "+str(tf.log(x, y)))
    print("   numpy version")
    print("   "+str(nf.log(xn, yn)))

    print("EXP: ")
    print("   Torch version")
    print("   "+str(tf.exp(x, y)))
    print("   numpy version")
    print("   "+str(np.array([nf.exp(xn[0], yn[0]), nf.exp(xn[1], yn[1]), nf.exp(xn[2], yn[2])])))

    print("Poincare DIST : ")
    print("   Torch version")
    print("   "+str(tf.riemannian_distance(x, y)))
    print("   numpy version")
    print("   "+str(nf.riemannian_distance(xn, yn)))

    print("Poincare Grad : ")
    print("   Torch version")
    x = torch.rand(1,2)/1.5 
    y = torch.rand(1,2)/1.5 
    xn = x[:,0].detach().numpy() + x[:,1].detach().numpy() *1j
    yn = y[:,0].detach().numpy() + y[:,1].detach().numpy() *1j
    x = nn.Parameter(x)
    y = nn.Parameter(y)
    l = tf.riemannian_distance(x, y)
    l.backward()
    print(" Gradient Angle")
    print("   "+str(x.grad/x.grad.norm(2,-1)))
    print("   numpy version")
    print("   "+str(nf.riemannian_distance_grad(xn, yn)/abs(nf.riemannian_distance_grad(xn, yn))))
    print(" Real value not angle")
    print(x.grad)
    g_xn = nf.riemannian_distance_grad(xn, yn)
    print(nf.riemannian_distance_grad(xn, yn))
    print("factor ",g_xn[0].real/x.grad[0,0].item())
    print(nf.riemannian_distance_grad(xn, yn)/(g_xn[0].real/x.grad[0,0].item() ))
    ln = nf.riemannian_distance_grad(xn, yn) 
    #* -sigmoid(nf.riemannian_distance(xn, yn))
    print(ln)
    print("new num" ,nf.exp(xn, -ln))
    xr = ((1 - torch.sum(x.data ** 2, dim=-1)))/4
    print(-x.grad * xr)
    print(x - (x.grad * xr))
    print("new num" ,tf.exp(x, - x.grad ))


if __name__ == "__main__":
    test()