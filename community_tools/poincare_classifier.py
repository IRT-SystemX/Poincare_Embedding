import torch
import tqdm

from torch import optim
from optim_tools import optimizer as ph
from function_tools import poincare_module as pm

class PoincareClassifier(object):
    def __init__(self, n_classes):
        self._n_c = n_classes
    def fit(self, X, Y=None, iteration=700):
        Y = Y.float()
        self.model = pm.PoincareMLR(X.size(-1), Y.size(-1))
        print(X.size(), Y.size(-1))
        optimizer_euclidean = optim.Adam(self.model.euclidean_parameters(), lr=1e-1)
        optimizer_hyperbolic = ph.PoincareBallSGDExp(self.model.poincare_parameters(), lr=5e-2)

        criterion = torch.nn.BCEWithLogitsLoss()


        progress_bar = tqdm.trange(iteration)
        for i in progress_bar:
            optimizer_euclidean.zero_grad()
            optimizer_hyperbolic.zero_grad()

            pred = self.model(X)
            # print(pred.min())

            loss = criterion(pred,Y)
            # print(loss.mean())
            loss.backward()

            optimizer_euclidean.step()
            optimizer_hyperbolic.step()
            progress_bar.set_postfix({"loss":loss.item(), "max_pred":pred.max().item()})
    def probs(self, z):
        with torch.no_grad():
            return self.model(z).sigmoid()
    def predict(self, z):
        with torch.no_grad():
            return self.model(z).max(-1)[1]