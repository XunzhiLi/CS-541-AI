import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def generate_wt(w,lr,w0,y,x):
    return w-lr*(x.T.dot((x.dot(w0)-y)))
if __name__ == '__main__':
    n = 100
    d = 40
    random_data_matrix = np.random.randn(n,d)
    random_y = np.random.randn(n,1)
    XtX = np.dot(random_data_matrix.T,random_data_matrix)
    w_star= np.linalg.inv(XtX).dot(random_data_matrix.T).dot(random_y)

    Hessian_matrix = XtX
    eigenvalue, featurevector = np.linalg.eig(Hessian_matrix)
    Max_eigenvalue = max(eigenvalue)
    Min_elgenvalue = min(eigenvalue)
    lr = 2/(Max_eigenvalue+Min_elgenvalue)   #learning rate
    lr_list = [0.01*lr,0.1*lr,lr,2*lr,20*lr,100*lr]
    w0 = 0
    delta_w_normlist = [[0 for col in range(100)] for row in range(6)]
    t_list = []

    #set a t_list for plotting graph
    for l in range(1,101):
        t_list.append(l)

    for j in range(6):
        #
        lr_ = lr_list[j]
        for i in range(1,101):
            new_w = generate_wt(w0,lr_,w0,random_y,random_data_matrix)
            delta_w = new_w-w_star
            delta_w_norm =LA.norm(delta_w,ord=2)
            delta_w_normlist[j][i-1] = delta_w_norm
            w0= new_w

    plt.xlabel('t')
    plt.ylabel('||wt − w∗||')

    plt.plot(t_list,delta_w_normlist[0], alpha=0.2, label='0.01lr',color='red')
    plt.plot(t_list,delta_w_normlist[1], alpha=0.2, label='0.1lr',color='black')
    plt.plot(t_list,delta_w_normlist[2], alpha=0.2, label='lr',color='green')
    # plt.plot(t_list,delta_w_normlist[3], alpha=0.2, label='2lr',color='blue')
    # plt.plot(t_list,delta_w_normlist[4], alpha=0.2, label='20lr',color='grey')
    # plt.plot(t_list,delta_w_normlist[5], alpha=0.2, label='100lr',color='purple')
    plt.title('Homework 3', fontsize=20)
    plt.legend()
    plt.show()