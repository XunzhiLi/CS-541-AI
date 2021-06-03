import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def GD(U0,V0,times,learing_rate,Omega1,lamda):
    F_list  = []
    U,V = U0,V0
    for t in range(times):
        F = 0

        # U deviate
        for index,row in Omega1.iterrows():
            i = row['row']
            j = row['col']
            e = M[i,j] - np.dot(U[:,i].T, V[:,j])
            deviate_U = e * V[:, j] - lamda * U[:, i]
            U[:, i] = U[:, i] + learing_rate * deviate_U
            F += e*e

        # V deviate
        for index,row in Omega1.iterrows():
            i = row['row']
            j = row['col']
            e = M[i,j] - np.dot(U[:,i].T, V[:,j])
            deviate_V = e * U[:, i] - lamda * V[:, j]
            V[:, j] = V[:, j] + learing_rate * deviate_V
            F += e*e

        F = 0.5*F + 0.5 * lamda *(LA.norm(U,'fro')**2 + LA.norm(V,'fro')**2)
        F_list.append(F)

    return F_list,F,U,V

def RMSE(U_star,V_star):
    X = np.dot(U_star.T, V_star)
    RMSE_lemda = 0
    for index,row in Omega2.iterrows():
        a = row['row']
        b = row['col']
        RMSE_lemda += (M[a,b]-X[a,b])**2
    return RMSE_lemda**0.5



if __name__ == '__main__':
    # First Step : Dataset pre-process
    dataset = pd.read_csv('ratings.csv', usecols=[0, 1,2])
    data_number = len(dataset["rating"])
    n = max(dataset["userId"])    # users_number
    movie_id_list = dataset["movieId"].tolist()
    movie_id_list = list(set(movie_id_list))
    p = len(list(set(movie_id_list)))   #movies_number
    M = np.zeros((n,p))
    for i in range(data_number):
        one_raw_info = dataset.loc[i]
        user_id = int(one_raw_info["userId"])
        movie_id = int(one_raw_info["movieId"])

        movie_index = movie_id_list.index(movie_id)
        rating = one_raw_info["rating"]
        M[user_id-1][movie_index] = rating

    # create omega: the index set of observed entries
    col1 = dataset['userId'].map(lambda x: x - 1)
    col2 = dataset['movieId'].map(lambda x: movie_id_list.index(x))
    userid_movie = [col1,col2]
    Omega = pd.concat(userid_movie, axis=1)
    Omega.columns = ['row','col']

    # divide omega into two parts for training and testing seperatively
    Omega1,Omega2 = train_test_split(Omega,train_size=0.9)


    #Second Step: Learning
    learing_rate = 0.1
    lamda = 1
    r = 5
    U0 = np.random.randn(r, n)
    V0 = np.random.randn(r, p)
    times = 25


    F_list0,F0,U_star0,V_star0 = GD(U0, V0, times, learing_rate, Omega1, 0.1)
    F_list1,F1,U_star1,V_star1 = GD(U0,V0,times,learing_rate,Omega1,0.5)
    F_list2,F2,U_star2,V_star2 = GD(U0,V0,times,learing_rate,Omega1,1)
    F_list3,F3,U_star3,V_star3 = GD(U0,V0,times,learing_rate,Omega1,2)
    F_list4,F4,U_star4,V_star4 = GD(U0,V0,times,learing_rate,Omega1,5)
    F_list5,F5,U_star5,V_star5 = GD(U0,V0,times,learing_rate,Omega1,10)
    F_list6,F6,U_star6,V_star6 = GD(U0,V0,times,learing_rate,Omega1,20)

    plt.xlabel('GD time')
    plt.ylabel('F(U,V)')
    plt.plot(range(1, times + 1), F_list0, alpha=0.2, label='lamda = 0.1,lr = 0.1', color='black')
    plt.plot(range(1,times+1), F_list1, alpha=0.2, label='lamda = 0.5,lr = 0.1', color='pink')
    plt.plot(range(1,times+1), F_list2, alpha=0.2, label='lamda = 1,lr = 0.1', color='green')
    plt.plot(range(1,times+1), F_list3, alpha=0.2, label='lamda = 2,lr = 0.1', color='blue')
    plt.plot(range(1,times+1), F_list4, alpha=0.2, label='lamda = 5,lr = 0.1', color='yellow')
    plt.plot(range(1,times+1), F_list5, alpha=0.2, label='lamda = 10,lr = 0.1', color='orange')
    plt.plot(range(1,times+1), F_list6, alpha=0.2, label='lamda = 20,lr = 0.1', color='red')
    plt.title('Homework 4', fontsize=20)
    plt.legend()
    plt.show()

    #
    #Step3 evaluation
    RMSE_list = []
    RMSE_list.append(RMSE(U_star0,V_star0))
    RMSE_list.append(RMSE(U_star1,V_star1))
    RMSE_list.append(RMSE(U_star2,V_star2))
    RMSE_list.append(RMSE(U_star3,V_star3))
    RMSE_list.append(RMSE(U_star4,V_star4))
    RMSE_list.append(RMSE(U_star5,V_star5))
    RMSE_list.append(RMSE(U_star6,V_star6))
    print(RMSE_list)


    lamda_list = [0.1, 0.5, 1, 2, 5, 10, 20]
    plt.xlabel('Lamda')
    plt.ylabel('RMSE')

    plt.plot(lamda_list,RMSE_list, alpha=0.2, label='RMSE-λ', color='red')
    plt.title('Homework 4', fontsize=20)
    plt.legend()
    plt.show()













