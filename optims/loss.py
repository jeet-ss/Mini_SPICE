import torch as t
import torch.nn as nn


class Huber_loss(nn.Module):
    def __init__(self, tau=0.1):
        super(Huber_loss, self).__init__()
        self.tau = tau

    def forward(self, error):
        ###
        # for pitch loss , error is a scalar value
        ##
        loss = 0
        for e in error:
            if t.abs(e) <= self.tau:
                loss += t.square(e)/2
            else:
                loss += self.tau*(t.abs(e) - self.tau) + (self.tau**2)/2

        return loss
        

class Recons_loss(nn.Module):
    def __init__(self) :
        super().__init__()

    def forward(self, x_1, x_2, hat_x_1, hat_x_2 ):
        #print("recons", x_1.size(), hat_x_1.size())
        error = t.add(t.square(t.linalg.norm(t.sub(x_1, hat_x_1), dim=1, ord=2)),
                        t.square(t.linalg.norm(t.sub(x_2, hat_x_2), dim=1, ord=2)))
        loss = t.mean(error)
        return loss
    

class Conf_loss(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, c_1, c_2, e_t, sigma):
        # mean along batch dimension
        loss = t.mean(t.square(t.abs((1 - c_1) - (e_t/sigma))) +\
                       t.square(t.abs((1 - c_2) - (e_t/sigma))))
        return loss