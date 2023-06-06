from os import mkdir, path
import torch
import sys
import filters
import os
import numpy as np
import matplotlib.pyplot as plt

_v0 = None 
_x0 = None

def get_x0(b_size, x_dim, sigma):
    global _x0
    if _x0 is None:
        _x0 = 3*torch.ones(b_size, x_dim)\
             + sigma * torch.randn(b_size, x_dim)
    x0 = _x0
    return x0

def get_x0_test(b_size, x_dim, sigma):
    _x0_test = 3*torch.ones(b_size, x_dim)\
               + sigma * torch.randn(b_size, x_dim)
    x0 = _x0_test
    return x0

def get_ha0(b_size, h_dim):
    global _v0
    if _v0 is None:
        _v0 = torch.zeros(1,h_dim)
    ha0 = torch.zeros(b_size, h_dim)
    for b in range(b_size):
        ha0[b,:] = _v0
    return ha0

def set_tensor_type(tensor_type,cuda):
    print("")
    print("#######################################")
    print("Simulation")
    print("PU: ", cuda)
    print("#######################################")
    print("")
    if (tensor_type == "double") and cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    elif (tensor_type == "float") and cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif (tensor_type == "double") and (not cuda):
        torch.set_default_tensor_type(torch.DoubleTensor)
    elif (tensor_type == "float") and (not cuda):
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        raise NameError("Unknown tensor_type")
    
def pre_train_full(net,b_size,h_dim,x_dim,
                   sigma0,optimizer_classname,optimizer_kwargs):
    """
    Pre-train c at t=0
    # learn the parameters in net.c using ha0 and x0
    # by minimizing the L_0(q_0^a) loss
    """
    
    
    print("#######################################")
    print('Pre-train c at t=0')
    
 
    x0 = get_x0(b_size, x_dim, sigma0)
    
    # create an optimizer optimizer0 for the paramerters in c
    optimizer0 = eval(optimizer_classname)(net.c.parameters(), **optimizer_kwargs) #creer un optimizeur avec le fichier python 
    
    # TODO minimize L_0(q_0^a), check how small is the loss
    ite = 0
    
    # Initlize h0
    ha0 = get_ha0(b_size, h_dim)
    
    # Use closure0 to compute the loss and gradients
    def closure0():
        # TODO first use optimizer0 to set all the gradients to zero
        # then compute the loss logpdf_a0 = L_0(q_0^a), by using x0, h0, and c
        # perform back-propogation of the loss
        # return the loss logpdf_a0


        optimizer0.zero_grad()  #a chaque fois qu'on fait un backward on accumule les gradients dans l'attribut grad donc on doit remettre les gradients à 0 a chasque nouvelle étape
        logpdf_a0 = -torch.mean(net.c.forward(ha0).log_prob(x0), dim = 0) #somme dans le cours
        logpdf_a0.backward() # différenciation automatique 
        nonlocal ite
        ite = ite + 1
        
        return logpdf_a0

    print("-------------Start pretraining------------------")
    optimizer0.step(closure0)  #Obligée pour LBFGS
    print("-------------End pretraining------------------")

    # print out the final mean and covariance of q_0^a
    q_0_a = net.c(ha0)
    print('## INIT q_0^a mean', q_0_a.mean[0,:])  # first sample
    print('## INIT q_0^a var', q_0_a.variance[0,:])  # first sample
    print('## INIT q_0^a covar', q_0_a.covariance_matrix[0,:,:]) # first sample
    
    print("#######################################")
    print("")


def train_full(net, b_size, h_dim, x_dim,
               T, checkpoint, direxp,
               prop, obs, sigma0,
               optimizer_classname, optimizer_kwargs):
    
    """
    Train over full time 0..T with BPTT
    # learn the parameters in net.a, net.b, net.c using t=0..T
    # by minimizing the total loss
    """
    if not path.exists(direxp):
        mkdir(direxp)          
    
    print("#######################################")
    print("Training on 0..T:")

    # generate training data seq for t=0..T
    # TODO rewrite xt and yt


    x0 = get_x0(b_size, x_dim, sigma0)
    x = torch.clone(x0)
    xt, yt = [],[]
    for i in range(1,T+1):  #creation d'un jeu de donnée , prop = propagateur de la dyamique, obs = rotation sur x donc bruit 
        x = prop.forward(x).sample() 
        y = obs.forward(x).sample()
        xt.append(x)
        yt.append(y)
    
    # Train net using xt and yt, t = 1 .. T and x0
    # TODO miminize total loss, by constructing optimizer and rewriting closure    
    

    ite = 0
    ha_0 = get_ha0(b_size, h_dim)
    optimizer = eval(optimizer_classname)(net.parameters(), **optimizer_kwargs)

    def closure():  #Une fonction pour (calculer le gradient et la loss pour Optimiseur LBFGS)      
        # TODO

        optimizer.zero_grad()
        ha_clone = torch.clone(ha_0)
        L_0 = -torch.mean(net.c.forward(ha_clone).log_prob(x0), dim = 0)


        loss = torch.zeros_like(L_0)
        for t in range(1,T+1):
            tmp, ha_clone = net.forward(ha_clone, xt[t-1], yt[t-1])
            loss += tmp
        loss = (loss/T) + L_0

        loss.backward()

        # Checkpoint
        nonlocal ite
        if ite == 1 or (ite % checkpoint == 0):
            print("## Train iteration nbr " + str(ite)+" ##")
            save_dict(direxp,scores=net.scores)
            print_scores(net.scores)

        ite +=  1
        
        return loss
    
    # TODO run optimizer
    print("-------------Start training------------------")
    optimizer.step(closure)
    print("-------------Stop training------------------")
    print("#######################################")

    ### Affichage d'un trajectoire d'une données d'entraînement ###  ##On parcour la liste des x et des y et on les plots
    ha_clone = torch.clone(ha_0)
    plt.figure()
    plt.plot(x0[0,0].item(), x0[0,1].item(), 'r.', label = 'Model')
    plt.plot(net.c.forward(ha_clone).mean[0,0].item(), net.c.forward(ha_clone).mean[0,1].item(), 'bo', label = 'Analyse')
    for t in range(1,T+1):
        loss_, ha_clone = net.forward(ha_clone, xt[t-1], yt[t-1])
        plt.plot(xt[t-1][0,0].item(), xt[t-1][0,1].item(), 'r.')
        if (t == 1):
            plt.plot(yt[t-1][0,0].item(), yt[t-1][0,1].item(), 'gx', label = 'Observations')
        else:
            plt.plot(yt[t-1][0,0].item(), yt[t-1][0,1].item(), 'gx')
        plt.plot(net.c.forward(ha_clone).mean[0,0].item(), net.c.forward(ha_clone).mean[0,1].item(), 'bo')
    plt.legend()
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.title(r"Plot de $x_{t}$, $y_{t}$ and $h_{t}^{a}$ pour un échantillon du jeu d'entraînement.")
    plt.savefig("figure_train")
    plt.show()

    ### Affichage de la trajectoire pour une donnée de test ###  ##on affiche la moyenne mu car en espérance c'est le meilleur point à prendre 
    x0_test = get_x0_test(1, x_dim, sigma0)
    ha0_test = get_ha0(1, h_dim)
    x_test = torch.clone(x0_test)
    ha_test = torch.clone(ha0_test) 
    xt_test, yt_test, sample_analysis = [],[], []
    for i in range(1, 2*T + 1):
        x_test = prop.forward(x_test).sample()
        y_test = obs.forward(x_test).sample()
        xt_test.append(x_test)
        yt_test.append(y_test)

    plt.figure()
    plt.plot(x0_test[0,0].item(), x0_test[0,1].item(), 'r.', label = 'Model')
    plt.plot(net.c.forward(ha0_test).mean[0,0].item(), net.c.forward(ha0_test).mean[0,1].item(), 'bo', label = 'Analyse')
    for t in range(1, 2*T + 1):
        loss_test, ha_test = net.forward(ha_test, xt_test[t-1], yt_test[t-1])
        plt.plot(xt_test[t-1][0,0].item(), xt_test[t-1][0,1].item(), 'r.')
        if (t == 1):
            plt.plot(yt_test[t-1][0,0].item(), yt_test[t-1][0,1].item(), 'gx', label = 'Observations')
        else:
            plt.plot(yt_test[t-1][0,0].item(), yt_test[t-1][0,1].item(), 'gx')
        sp = net.c.forward(ha_test)
        sample_analysis.append(sp)
        plt.plot(sp.mean[0,0].item(), sp.mean[0,1].item(), 'bo')
    plt.legend()
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.title(r"Plot de $x_{t}$, $y_{t}$ and $h_{t}^{a}$ pour un échantillon de test.")
    plt.savefig("figure_test")
    plt.show()
    return xt_test, yt_test, sample_analysis
    
def train_online(net, b_size, h_dim, x_dim,
                 T, checkpoint, direxp,
                 prop, obs, sigma0,
                 optimizer_classname, optimizer_kwargs, 
                 scheduler_classname, scheduler_kwargs):   
    """
    Train functions for the DAN, online and truckated BPTT   #dès qu'on passe a t+1 on remet à 0 les gradients car système chaotique 
    """
    if not path.exists(direxp):
        mkdir(direxp)
        
    # TODO construct optimizer and scheduler
    print("#######################################")
    print("Simulation:")
    assert(optimizer_classname != "NONE")
    print("Optimisor: ", optimizer_classname)
    
    optimizer = eval(optimizer_classname)(net.parameters(), **optimizer_kwargs)

    assert(scheduler_classname != "NONE")
    print("Scheduler: ", scheduler_classname)
   
    scheduler = eval(scheduler_classname)(optimizer, **scheduler_kwargs)
    print("#######################################")
    print("")
    
    x0 = get_x0(b_size, x_dim, sigma0)
    ha0 = get_ha0(b_size, h_dim)
    x = torch.clone(x0)
    ha_clone = torch.clone(ha0)

    print("#######################################")
    print("-------------Start training online mode 0..T:------------------")
    for t in range(1, T+1):
        # TODO
        # on the fly data generation
        # Truncated back propagation through time
        

        x = prop.forward(x).sample()
        y = obs.forward(x).sample()


        optimizer.zero_grad()


        loss, ha_clone = net.forward(ha_clone, x, y)

 
        loss.backward()


        optimizer.step()
        scheduler.step()

        ### Truncated BPTT ###
        ha_clone = ha_clone.detach()
        
        # Checkpoint
        if (t % checkpoint == 0) or (t == T):
            if ha_clone is not None:
                print("## Train Cycle " + str(t)+" ##")
                save_dict(direxp,
                          net=net.state_dict(),
                          ha_clone=ha_clone,
                          x=x,
                          scores=net.scores,
                          optimizer=optimizer.state_dict())
                print_scores(net.scores)

    print("-------------Stop training------------------")
    print("#######################################")
    print("")

@torch.no_grad()
def test(net, b_size, h_dim, x_dim,
         T, checkpoint, direxp,
         prop, obs, sigma0):
    x = get_x0_test(b_size, x_dim, sigma0)
    ha_clone = get_ha0(b_size, h_dim)
    for t in range(1, T+1):
        # on the fly data generation
        x = prop(x)\
            .sample(sample_shape=torch.Size([1]))\
            .squeeze(0)
        y = obs(x)\
            .sample(sample_shape=torch.Size([1]))\
            .squeeze(0)

        # Evaluate the loss
        _, ha_clone = net(ha_clone, x, y)

        # Checkpoint
        if (t % checkpoint == 0) or (t == T):
            print("## Test Cycle " + str(t)+" ##")
            save_dict(direxp,
                      test_scores=net.scores)
            print_scores(net.scores)

def experiment(tensor_type, seed,
               net_classname, net_kwargs,
               sigma0, prop_kwargs, obs_kwargs,
               train_kwargs, test_kwargs,
               optimizer_classname, optimizer_kwargs,
               scheduler_classname, scheduler_kwargs,
               directory, nameexp):

    # CPU or GPU tensor
    cuda = False # torch.cuda.is_available()
    set_tensor_type(tensor_type,cuda)

    # Reproducibility
    torch.manual_seed(seed)

    net = eval(net_classname)(**net_kwargs)
    prop = filters.Constructor(**prop_kwargs)
    obs = filters.Constructor(**obs_kwargs)
    b_size = train_kwargs['b_size']
    h_dim = train_kwargs['h_dim']
    x_dim = train_kwargs['x_dim']
    T = train_kwargs['T']
    checkpoint = train_kwargs['checkpoint']
    direxp = directory + nameexp
    xt = []
    yt = []
    sample_analyse = []
    
    if train_kwargs["mode"] == "full":
        pre_train_full(net,b_size,h_dim,x_dim,sigma0,
                       optimizer_classname,optimizer_kwargs)        
        xt, yt, sample_analyse = train_full(net, b_size, h_dim, x_dim,
                   T, checkpoint, direxp,
                   prop, obs, sigma0,
                   optimizer_classname, optimizer_kwargs)
    else:
        train_online(net, b_size, h_dim, x_dim,
                     T, checkpoint, direxp,
                     prop, obs, sigma0,
                     optimizer_classname, optimizer_kwargs, 
                     scheduler_classname, scheduler_kwargs)
    
    # Clear scores
    net.clear_scores()

    # Testing
    b_size = test_kwargs['b_size']
    h_dim = test_kwargs['h_dim']
    x_dim = test_kwargs['x_dim']
    T = test_kwargs['T']
    checkpoint = test_kwargs['checkpoint']
    test(net, b_size, h_dim, x_dim,
         T, checkpoint, direxp,
         prop, obs, sigma0)    

    return xt, yt, sample_analyse, net

def save_dict(prefix, **kwargs):
    """
    saves the arg dict val with name "prefix + key + .pt"
    """
    for key, val in kwargs.items():
        torch.save(val, prefix + key + ".pt")


def print_scores(scores):
    for key, val in scores.items():
        if len(val) > 0:
            print(key+"= "+str(val[-1]))


def update(k_default, k_update):
    """Update a default dict with another dict
    """
    for key, value in k_update.items():
        if isinstance(value, dict):
            k_default[key] = update(k_default[key], value)
        else:
            k_default[key] = value
    return k_default


def update_and_save(k_default, list_k_update, name_fun):
    """update and save a default dict for each dict in list_k_update,
    generates a name for the exp with name_fun: dict -> string
    returns the exp names on stdout
    """
    out, directory = "", k_default["directory"]
    for k_update in list_k_update:
        nameexp = name_fun(k_update)
        if not os.path.exists(nameexp):
            os.mkdir(nameexp)
        k_default["nameexp"] = nameexp + "/"
        torch.save(update(k_default, k_update), nameexp + "/kwargs.pt")
        out += directory + "," + nameexp

    # return the dir and nameexp
    sys.stdout.write(out)


if __name__ == "__main__":
    """
    the next argument is the experiment name
    - launch the exp
    """
    torch.autograd.set_detect_anomaly(True)
    cuda = torch.cuda.is_available()
    device = "cpu"#torch.device("cuda" if cuda else "cpu")
    experiment(**torch.load(sys.argv[1] + "/kwargs.pt",
                            map_location=torch.device(device)))
