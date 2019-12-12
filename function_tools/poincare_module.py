import torch
from torch import nn
from torch.autograd import Function
from function_tools import poincare_function as pf
from function_tools import function as f
class PoincareEmbedding(nn.Module):
    def __init__(self, N, M=3):
        super(PoincareEmbedding, self).__init__()
        self.l_embed = nn.Embedding(N, M, max_norm=1-1e-3)
        with torch.no_grad():
            self.l_embed.weight.data[:,:] = (torch.rand(N, M) *2 -1) / (M*2)
            self.l_embed.weight.data[:,:] = self.l_embed(torch.arange(0,N)).data
    def forward(self, x):
        return self.l_embed(x)


class PoincareNN(nn.Module):
    def __init__(self):
        super(PoincareNN, self).__init__()
    def poincare_parameters(self):
        return self.poincare_parameters_list
    def euclidean_parameters(self):
        return self.euclidean_parameters_list

class PoincareMLR(PoincareNN):
    def  __init__(self, features_size, output_size):
        super(PoincareMLR, self).__init__()
        self.a = nn.Parameter(torch.randn(output_size, features_size))
        self.p = nn.Parameter(torch.randn(output_size, features_size) * 1e-3)

    def poincare_parameters(self):
        return [self.p]
    
    def euclidean_parameters(self):
        return [self.a]

    def forward(self, x):
        # batch size
        N = x.size(0)
        # input size
        I = x.size(-1)
        # output size
        O = self.a.size(0)

        # use p to 
        p = self.p.unsqueeze(0).expand(N, O, I)
        
        lambda_p = pf.lambda_k(self.p)
        # print(self.p.size())
        # print("all1",torch.sum(self.p ** 2, dim=-1))
        # print("all2",self.p.norm(2, -1)**2)
        # print("max",self.p.norm(2, -1).max().item())
        # print("min",self.p.norm(2, -1).min().item())

        # print("----------------------")
        # print("all",lambda_px[0].norm(2, -1))
        # print("max",lambda_px[0].norm(2, -1).max().item())
        # print("min",lambda_px[0].norm(2, -1).min().item())
        if(lambda_p.sum() != lambda_p.sum()):
            print("Error lambda px is none")
            print(self.p.size())
            print(self.p)
        assert(lambda_p.sum() == lambda_p.sum())
        a_h = ((2*self.a)/lambda_p.expand( O, I)).unsqueeze(0).expand(N, O, I)
        # print(a_h[0].norm(2,-1))
        lambda_p = lambda_p.squeeze().unsqueeze(0).expand(N, O)


        # print(p.norm(2,-1).max())
        assert(p.norm(2,-1).max() < 1.0)
        # print(N, O, I)
        # print(x.size())
        xrs = x.unsqueeze(1).expand(N, O, I)
        # print(self.p.size(), x.size())
        minus_p_plus_x = pf.add(-p, xrs)
        assert(minus_p_plus_x.sum()  == minus_p_plus_x.sum())
        norm_a = a_h.norm(2,-1)


        # print(minus_p_plus_x.size(), a_h.size())
        try:
            px_dot_a = (minus_p_plus_x * a_h).sum(-1)
            if(px_dot_a.sum() != px_dot_a.sum()):
                raise Exception
        except:
            print("Poincare Module.py ")
            print("a_h ", a_h[0].norm(2,-1))
            print("p ",self.p.norm(2, -1)**2)
            print("p_lh ",lambda_p[0].norm(2, -1))
            print("a ",self.a)
            print(minus_p_plus_x.sum() )
            quit()
        lambda_px = pf.lambda_k(minus_p_plus_x).squeeze()
        # print(lambda_px.size())
        # print(norm_a.size())
        # print(px_dot_a.size())
        # print("norm_a max ", norm_a.max())
        # print("norm_a min ", norm_a.min())
        te = f.arc_sinh((2 * px_dot_a) * lambda_px * (1/norm_a))
        if(te.sum() != te.sum()):
            print("te",te.mean(0))
        llp = lambda_p*norm_a 
        if(llp.sum() != llp.sum()):
            print("te",llp.mean(0))
        logit = lambda_p*norm_a * f.arc_sinh((2 * px_dot_a) * lambda_px * (1/norm_a))

        return  logit

def poincareMLR_test():
    import tqdm
    from torch import optim
    from optim_tools import optimizer as ph
    x = torch.randn(100, 10) *50
    y = (torch.rand(100, 5)).round()

    model = PoincareMLR(x.size(-1), y.size(-1))

    g = model(x)
    print(g.norm(2,-1))

    optimizer_euclidean = optim.Adam(model.euclidean_parameters(), lr=1e-2)
    optimizer_hyperbolic = ph.PoincareBallSGDExp(model.poincare_parameters(), lr=1e-3)

    criterion = torch.nn.BCEWithLogitsLoss()

    progress_bar = tqdm.trange(5000)
    for i in progress_bar:
        optimizer_euclidean.zero_grad()
        optimizer_hyperbolic.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        print(pred.max())
        optimizer_euclidean.step()
        optimizer_hyperbolic.step()
        progress_bar.set_postfix({"loss": loss.item()})


if __name__ == "__main__":
    # execute only if run as a script
    # run all the unit test of the file
    poincareMLR_test()