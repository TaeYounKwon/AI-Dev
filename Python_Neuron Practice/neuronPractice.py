import numpy as np
import matplotlib.pyplot as plt
N,D_in, H, D_out = 2,100,50,1
x = np.random.randn(N,D_in)
y = np.random.randn(N,D_out)
print(x.shape)
print(y.shape)
w1 = np.random.randn(D_in,H)
w2 = np.random.randn(H,D_out)
print(w1.shape)
print(w2.shape)

y_pred_list = list()
loss_list = list()
learning_rate = 1e-6
for t in range(500):
    h =x.dot(w1)
    h_relu = np.maximum(h,0)
    y_pred = h_relu.dot(w2)
    y_pred_list.append(y_pred[0][0])
    
    loss = np.square(y_pred -y).sum()
    loss_list.append(loss)
    print(t,loss)
    
    grad_y_pred = 2.0 *(y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0]=0
    grad_w1 = x.T.dot(grad_h)
    w1-= learning_rate * grad_w1
    w2-= learning_rate * grad_w2
    
#Comprehention    
step_list = [i for i in range(len(loss_list))]    
plt.plot(step_list, loss_list)
plt.ylabel('Cost')
plt.xlabel('Step')
plt.show()    