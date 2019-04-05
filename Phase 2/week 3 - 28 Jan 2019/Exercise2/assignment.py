import os
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import optimize
import utils
# define the submission/grader object for this exercise
grader = utils.Grader()


data=np.genfromtxt("Data/ex2data1.txt",delimiter=',')
X,y =data[:,:2],data[:,2]
X=np.concatenate((np.array([1.0 for _ in range(y.shape[0])])[:,np.newaxis],X),axis=1)
m,n=X.shape


def plotData(X,y):
	fig=plt.figure()
	pos = y == 1
	neg = y == 0
	plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
	plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
	plt.xlabel('Exam 1 score')
	plt.ylabel('Exam 2 score')
	plt.legend(['admitted','Not admitted'])
	


def sigmoid(z):
	return 1/(1+math.exp(-z))


# Test the implementation of sigmoid function here
z = 0
g = sigmoid(z)

print('g(', z, ') = ', g)


grader[1] = sigmoid

def costFunction(theta,X,y):
	m=y.size
	J=0
	grad=np.zeros(theta.shape)
	for i in range(m):
		J-=((y[i]*np.log(sigmoid(np.matmul(theta.T,X[i]))))+(1-y[i])*np.log(1-sigmoid(np.matmul(theta.T,X[i]))))
		grad+=((sigmoid(np.matmul(theta.T,X[i]))-y[i])*X[i])
	J/=m;grad/=m
	return J,grad


# Initialize fitting parameters
initial_theta = np.zeros(n)

cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros): {:.3f}'.format(cost))
print('Expected cost (approx): 0.693\n')

print('Gradient at initial theta (zeros):')
print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('Cost at test theta: {:.3f}'.format(cost))
print('Expected cost (approx): 0.218\n')

print('Gradient at test theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')

grader[2] = costFunction
grader[3] = costFunction


initial_theta=np.zeros(n)
options={'maxiter': 400}
res=optimize.minimize(costFunction,initial_theta,(X,y),jac=True,method='TNC',options=options)
cost = res.fun
theta= res.x

print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
print('Expected cost (approx): 0.203\n');

print('theta:')
print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')

utils.plotDecisionBoundary(plotData, theta, X, y)
plt.show()
def predict(theta,X):
	m=X.shape[0]
	p=np.zeros(m)
	for i in range(m):
		if np.dot(X[i], theta)>=0.5:
			p[i]=1
	return p

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 
prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85,'
      'we predict an admission probability of {:.3f}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')

p=predict(theta,X)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
print('Expected accuracy (approx): 89.00 %')



grader[4] = predict
# grader.grade()