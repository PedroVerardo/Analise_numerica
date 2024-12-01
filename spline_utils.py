import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def open_or_closed(arr: np.array) -> bool:
    return np.array_equal(arr[0], arr[-1])

def vector_norm(arr):
    num_points = len(arr)
    h = np.zeros(num_points + 1)

    if np.array_equal(arr[0], arr[-1]):
        print("closed")
        temp = np.linalg.norm(arr[-2] - arr[0])
        h[0] = temp
        h[-1] = temp
        for i in range(1, num_points):
            h[i] = np.linalg.norm(arr[i] - arr[i-1])

    else:
        print("open")
        for i in range(1, num_points):
            h[i] = np.linalg.norm(arr[i] - arr[i-1])
    print("h: ", h)
    return h

def calculate_gamma(h: np.array, v: np.array) -> np.array:
    if v is None:
        v = np.zeros(len(h) - 1)

    gamma = np.zeros(len(h) - 1)

    for i in range(len(h)-1):
        result = 2*(h[i] + h[i+1])/(v[i]*h[i]*h[i+1] + 2*(h[i] + h[i+1]))
        gamma[i] = result

    print("gamma: ",gamma)
    return gamma

def _calculate_lambda(h: np.array, gamma: np.array) -> np.array:
    lambda_ = np.zeros(len(h) - 1)

    for i in range(1, len(h) - 1):
        result = (gamma[i-1]*h[i-1] + h[i])/(gamma[i-1]*h[i-1] + h[i] + gamma[i]*h[i+1])
        lambda_[i] = result
    
    return lambda_

def _calculate_mu(h: np.array, gamma: np.array) -> np.array:
    mu = np.zeros(len(h) - 1)

    for i in range(1, len(h) - 2):
        result = gamma[i]*h[i]/(gamma[i]*h[i] + h[i+1] + gamma[i+1]*h[i+2])
        mu[i] = result
    
    return mu

def calculate_mu(arr: np.array, h: np.array, gamma: np.array) -> np.array:
    mu = _calculate_mu(h, gamma)
    if not np.array_equal(arr[0], arr[-1]): #open
        mu[0] = 0
        mu[-1] = 0
    else: #closed
        mu[0] =  (gamma[0]*h[0])/(gamma[0]*h[0] + h[1] + gamma[1]*h[2])
        mu[-1] = (gamma[-1]*h[-1])/(gamma[-1]*h[-1] + h[-2] + gamma[-2]*h[-3])
    print("mu: ", mu)
    return mu

def calculate_lambda(arr: np.array, h: np.array, gamma: np.array) -> np.array:
    lambda_ = _calculate_lambda(h, gamma)
    if not np.array_equal(arr[0], arr[-1]): #open
        lambda_[0] = 1
        lambda_[-1] = 1
    else: #closed
        lambda_[0] = (gamma[-1]*h[-1] + h[0])/(gamma[-1]*h[-1] + h[0] + gamma[0]*h[1])
        print("lambda conta 0 : ", (gamma[-1]*h[-1] + h[0]))
        lambda_[-1] = (gamma[-2]*h[-2] + h[-1])/(gamma[-2]*h[-2] + h[-1] + gamma[-1]*h[0])
        print("lambda conta 1 : ", (gamma[-2]*h[-2] + h[-1])/(gamma[-2]*h[-2] + h[-1] + gamma[-1]*h[0]))

    print("lambda: ", lambda_)
    return np.array(lambda_)

def calculate_Ri_Li(D: np.array, mu: np.array, lambda_: np.array) -> tuple:
    R = []
    L = []
    
    for i in range(len(D) - 1):
        
        Ri = (1 - mu[i]) * D[i] + mu[i] * D[i + 1]
        R.append(Ri)
        
        
        Li = (1 - lambda_[i + 1]) * D[i] + lambda_[i + 1] * D[i + 1]
        L.append(Li)

    print("L: ", L)
    print("R: ", R)
    return np.array(R), np.array(L)

def casteljau(d_points: np.array, t: float) -> np.array:
    d0, d1, d2, d3 = d_points
    
    d10 = (1 - t) * d0 + t * d1
    d11 = (1 - t) * d1 + t * d2
    d12 = (1 - t) * d2 + t * d3
    
    d20 = (1 - t) * d10 + t * d11
    d21 = (1 - t) * d11 + t * d12
    
    P_t = (1 - t) * d20 + t * d21
    
    return P_t

def calculate_control_points(P, h, gamma, mu, lambda_):
    n = len(P)
    
    matrix = np.zeros((n, n))

   
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)

    
    delta = np.zeros(n)
    for i in range(n):
        delta[i] = h[i] / (h[i] + h[i + 1])

   
    for i in range(n):
        a[i] = (1 - delta[i]) * (1 - lambda_[i])
        b[i] = (1 - delta[i]) * lambda_[i] + delta[i] * (1 - mu[i])
        c[i] = delta[i] * mu[i]
    
    print("a: ", a)
    print("b: ", b)
    print("c: ", c)
    print("delta: ", delta)


    for i in range(1, n):
        matrix[i, i - 1] = a[i]  
    matrix[0, -1] = a[0]       

    for i in range(n):
        matrix[i, i] = b[i]     

    for i in range(n - 1):
        matrix[i, i + 1] = c[i]  
    matrix[-1, 0] = c[-1]      

    np.savetxt('matrix.txt', matrix, fmt='%0.5f')

    try:
        D = np.linalg.solve(matrix, P)
    except np.linalg.LinAlgError as e:
        raise ValueError("The matrix is singular and cannot be solved. Check your coefficients or input data.") from e

    return D

def plot_spline(P: np.array, V:np.array = None, num_points: int=100):

    h = vector_norm(P)
    
    gamma = calculate_gamma(h, V)
    mu = calculate_mu(P, h, gamma)
    lambda_ = calculate_lambda(P, h, gamma)

    D = calculate_control_points(P, h, gamma, mu, lambda_)

    R, L = calculate_Ri_Li(D, mu, lambda_)

    plt.figure(figsize=(10, 6))
    
    plt.plot(P[:, 0], P[:, 1], 'o--', color='gray', label='Control Points D_i')
    
    for i in range(len(P) - 1):
        d_points = np.array([P[i], R[i], L[i], P[i+1]])
        
        spline_segment = np.array([casteljau(d_points, t) for t in np.linspace(0, 1, num_points)])
        
        plt.plot(spline_segment[:, 0], spline_segment[:, 1], '-', label=f'Segment {i+1}')

    plt.scatter(R[:, 0], R[:, 1], color='blue', label='R_i')
    plt.scatter(L[:, 0], L[:, 1], color='red', label='L_i')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic Spline Interpolation')
    plt.legend()
    plt.grid(True)
    plt.show()

