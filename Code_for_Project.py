
import numpy as np
import matplotlib.pyplot as plt
## data reading
a=open('/Users/zhangqihang/Downloads/data/Part1_training_data.txt',"r")
data=a.read()
data_list=data.split()
data_value_list=[]
for i in data_list:
    data_value_list.append(eval(i))
#print(data_value_list)
b=open('/Users/zhangqihang/Downloads/data/Part1_testing_data.txt',"r")
data_of_reality=b.read()
data_real_list=data_of_reality.split()
data_real_value_list=[]
for i in data_real_list:
    data_real_value_list.append(eval(i))
#print(data_real_value_list)
#print(len(data_real_value_list))



def get_the_training(N):
    row_of_matrix_x_not_t=[]
    all_rows_for_training=[]
    row_of_matrix_x_t=[]
    for i in data_value_list[N::]:
        row_of_matrix_x_not_t=[]
        for j in data_value_list[data_value_list.index(i)-N:data_value_list.index(i):]:
            row_of_matrix_x_not_t.append(j)
        row_of_matrix_x_t.append(i)
        all_rows_for_training.append(row_of_matrix_x_not_t)
    X=np.mat(all_rows_for_training)
    Y=np.mat(row_of_matrix_x_t)
    return X,Y



#N=5
#X,Y=get_the_training(N)
def get_cofficient_matrix_B(X,Y):
    #X_inverse=X.I
    _X=X.transpose()
    Y=Y.transpose()
    #X_inverse=X_inverse.transpose()
    B=np.dot(_X,X).I
    #print(B)
    C=np.dot(B,_X)
    #print(C)
    #print(Y)
    D=np.dot(C,Y)
    #print(D)
    return D


#D=get_cofficient_matrix_B(X,Y)
#print(B)
#print(D)
def start_game(X,N,D):
    final_game_answer=[]
    for i in range(100):
        matrix_needed=X[:len(X)-2:-1]
        Answer=np.dot(matrix_needed,D)
        #print(matrix_needed)
        mid_matrix=X
        #print(type(mid_matrix))
        #print(X)
        mid_matrix=mid_matrix[:len(mid_matrix)-2:-1]
        #print(mid_matrix)
        for j in range(N):
            if j!=N-1:
                mid_matrix[0,j]=mid_matrix[0,j+1]
            else:
                mid_matrix[0,j]=Answer
        #print(mid_matrix)
        #mid_matrix[0].expend(Answer)
        #mid_matrix[1].pop(0)
        #mid_matrix[1].append(float(Answer))
        X=np.vstack((X,mid_matrix))
        final_game_answer.append(float(Answer))
        #print(final_game_answer)
        #print(type(final_game_answer))
        #print(len(final_game_answer))
        #print(Answer)
    return final_game_answer,X
#H=[]


#final_game_answer,X=start_game(X,N,D)

#print(len(final_game_answer))
def caculate(final_game_answer,data_real_value_list):
    MSE=0
    for i in range(100):
        MSE=MSE+(final_game_answer[i]-data_real_value_list[i])**2
    MSE=MSE/100
    return MSE

#caculate(final_game_answer,data_real_value_list)
#print(D[::])
#print(H[:len(H)-N-1:-1])



def main():
    global data_value_list,data_real_value_list
    MSE_list=[]
    for N in range(1,101):
        #print(N)
        X,Y=get_the_training(N)
        D=get_cofficient_matrix_B(X,Y)
        final_game_answer,X=start_game(X,N,D)
        MSE_for_N=caculate(final_game_answer,data_real_value_list)
        #print(MSE_for_N)
        #print(type(MSE_list))
        MSE_list.append([MSE_for_N,N])
        #print(type(MSE_list))
        MSE_list.sort(key=lambda x: x[0])
        if N==44:
            
            x=np.arange(201,301)
            y = np.array(final_game_answer)[x-201]
            x=np.arange(201,301)
            y2 = np.array(data_real_value_list)[x-201]
            #y2 = data_real_value_list[i-201]
            #plt.title("sine wave form")  
            #plt.plot(x, y,y2) 
            #plt.show()



            #x = np.linspace(0,2,100)
            fig,ax = plt.subplots()
            ax.plot(x,y,label='Prediction')
            ax.plot(x,y2,label='Realitly')
            ax.set_xlabel('time')
            ax.set_ylabel('price')
            ax.set_title('simple plot for N=44')
            ax.legend()
            plt.show()
            

    print(MSE_list)



main()






