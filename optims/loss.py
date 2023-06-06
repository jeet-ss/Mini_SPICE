import torch as t


class Huber_loss(t.nn.Module):
    def __init__(self, tau=0.1):
        super(Huber_loss, self).__init__()
        self.tau = tau

    def forward(self, error):
        ###
        # for pitch loss , error is a scalar value
        ##
        loss = 0
        if t.abs(error) <= self.tau:
            loss = t.square(error)/2
        else:
            loss = self.tau*(t.abs(error) - self.tau) + (self.tau**2)/2

        return loss
        

class Recons_loss(t.nn.Module):
    def __init__(self) :
        super(Recons_loss, self).__init__()

    def forward(self, x_1, x_2, hat_x_1, hat_x_2 ):
        error = t.add(t.square(t.linalg.norm(t.sub(x_1, hat_x_1), dim=1, ord=2)),
                        t.square(t.linalg.norm(t.sub(x_2, hat_x_2), dim=1, ord=2)))
        #print("inside recon los", error.size())
        loss = t.mean(error)
        return loss
    

class Conf_loss(t.nn.Module):
    def __init__(self, ) -> None:
        super(Conf_loss, self).__init__()

    def forward(self, c_1, c_2, e_t, sigma):
        # mean along batch dimension
        loss = t.mean(t.square(t.abs((1 - c_1) - (e_t/sigma))) +\
                       t.square(t.abs((1 - c_2), (e_t/sigma))))
        return loss