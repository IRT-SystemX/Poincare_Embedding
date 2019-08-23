import torch
from torch.nn import Module
from torch.autograd import Function


# redefining the hyperbolic distance forward and grad
class HyperbolicDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        with torch.no_grad():
            x_norm = torch.sum(x ** 2, dim=-1)
            y_norm = torch.sum(y ** 2, dim=-1)
            d_norm = torch.sum((x-y) ** 2, dim=-1)
            cc = 1+2*d_norm/((1-x_norm)*(1-y_norm)) 
            ctx.save_for_backward(cc,x, y , x_norm, y_norm)
        
            return  torch.log(cc + torch.sqrt(cc**2-1))

    @staticmethod
    def get_grad(output, theta, x, theta_norm, x_norm ):
        with torch.no_grad():
            beta = (1-x_norm)
            alpha = (1-theta_norm)
            coef_beta = 4/((torch.sqrt((output+1e-4)**2-1)) * beta)
            b = (x_norm - 2* (theta*x).sum(-1) + 1 )/(alpha**2)
            c = x/alpha.unsqueeze(-1).expand_as(x)
            b_c = b.unsqueeze(-1).expand_as(x)*theta - c
            return coef_beta.unsqueeze(-1).expand_as(x) * b_c
    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            output,x, y, x_norm,y_norm = ctx.saved_tensors
            return (HyperbolicDistanceFunction.get_grad(output,x, y, x_norm, y_norm)* grad_output.unsqueeze(-1).expand_as(x),
                HyperbolicDistanceFunction.get_grad(output,y, x, y_norm, x_norm)* grad_output.unsqueeze(-1).expand_as(x))

def hyperbolicDistance(x, y):
    return HyperbolicDistanceFunction.apply(x,y)

# defined as O_1 in the associated paper

class NegativeLogDistance(Module):
    def __init__(self, distance=hyperbolicDistance, alpha=1.0):
        super(NegativeLogDistance, self).__init__()
        self.d = distance
        self.a = alpha

    def forward(self, x, y, keepdim=False):
        assert(x.size(0)==y.size(0) and x.size(1)==y.size(1))
        assert(x.dim() == y.dim())
        if(keepdim):
            return -self.a * torch.log(torch.sigmoid(-self.d(x, y)))
        return -self.a * torch.sum(torch.log(torch.sigmoid(-self.d(x, y))), -1)

def negativeLogDistance(x, y, distance=hyperbolicDistance, alpha=1.0, keepdim=False):
    return NegativeLogDistance(distance=distance, alpha=alpha)(x, y, keepdim=keepdim)

# defined as O_2 in the associated paper
class DeepWalkLoss(Module):
    def __init__(self, distance=hyperbolicDistance, alpha=1.0):
        super(DeepWalkLoss, self).__init__()
        self.d = distance
        self.a = alpha        

    # x input , y belong in C and z 
    def forward(self, x, y, z):
        pos = negativeLogDistance(x, y, keepdim=True) 
        neg = negativeLogDistance(y.unsqueeze(-1).expand_as(z), z, keepdim=True).sum(-1)
        # TODO: Faster use alpha
        return (pos-neg).sum(-1).sum()

class CommunityLoss(Module):
    # mu = K\times N
    # sigma = K \times N \times N
    def __init__(self, pi, mu, sigma, distance=hyperbolicDistance, alpha=1.0):
        super(CommunityLoss, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        self.d = distance
        self.a = self.alpha
    # x input , y belong in C and z 
    def forward(self, x):
        # TODO: to finish use alpha
        # Not sure using the real loss function
        # mixt = self.pi *(self.d(x.unsqueeze(1).expand("TODO"), self.mu.unsqueeze(0).expand("TODO"))) 
        # return a * mixt
        pass