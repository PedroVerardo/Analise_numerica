import numpy as np
import matplotlib.pyplot as plt


#TODo em caso de erro a parte do mu e do lambda estão esquisitas, já que vai de 1 até n-2
def open_or_closed(arr: np.array) -> bool:
    return np.array_equal(arr[0], arr[-1])

def vector_norm(arr):
    num_points = len(arr)
    h = np.zeros(num_points + 1)

    if np.array_equal(arr[0], arr[-1]):
        print("closed")
        temp = np.linalg.norm(arr[0] - arr[-1])
        h[0] = temp
        h[-1] = temp
        for i in range(1, num_points):
            h[i] = np.linalg.norm(arr[i] - arr[i-1])

    else:
        print("open")
        for i in range(1, num_points):
            h[i] = np.linalg.norm(arr[i] - arr[i-1])
    
    return h

def calculate_gamma(h: np.array, v: np.array) -> np.array:
    if v is None:
        v = np.zeros_like(h)

    gamma = np.zeros_like(h)

    for i in range(len(h)-1):
        result = 2*(h[i] + h[i+1])/(v[i]*h[i]*h[i+1] + 2*(h[i] + h[i+1]))
        gamma[i] = result

    return gamma

def _calculate_lambda(h: np.array, gamma: np.array) -> np.array:
    lambda_ = np.zeros_like(h)

    for i in range(1, len(h) - 1):
        result = (gamma[i-1]*h[i-1] + h[i])/(gamma[i-1]*h[i-1] + h[i] + gamma[i]*h[i+1])
        lambda_[i] = result
    
    return lambda_

def _calculate_mu(h: np.array, gamma: np.array) -> np.array:
    mu = np.zeros_like(h)

    for i in range(1, len(h) - 2):
        result = gamma[i]*h[i]/(gamma[i]*h[i] + h[i+1] + gamma[i+1]*h[i+2])
        mu[i] = result
    
    return mu

def calculate_mu(arr: np.array, h: np.array, gamma: np.array) -> np.array:
    mu = _calculate_mu(h, gamma)
    if open_or_closed(arr):
        mu[0] = 0
        mu[-1] = 0
    else:
        mu[0] =  gamma[0]*h[0]/(gamma[0]*h[0] + h[1] + gamma[1]*h[2])
        mu[-1] = gamma[-1]*h[-1]/(gamma[-1]*h[-1] + h[-2] + gamma[-2]*h[-3])
    
    return mu

def calculate_lambda(arr: np.array, h: np.array, gamma: np.array) -> np.array:
    lambda_ = _calculate_lambda(h, gamma)
    if open_or_closed(arr):
        lambda_[0] = 1
        lambda_[-1] = 1
    else:
        lambda_[0] = (gamma[-1]*h[-1] + h[0])/(gamma[-1]*h[-1] + h[0] + gamma[0]*h[1])
        lambda_[-1] = (gamma[-2]*h[-2] + h[-1])/(gamma[-2]*h[-2] + h[-1] + gamma[-1]*h[0])
    
    return np.array(lambda_)

def calculate_Ri_Li(D: np.array, mu: np.array, lambda_: np.array) -> tuple:
    R = []
    L = []
    
    for i in range(len(D) - 1):
        
        Ri = (1 - mu[i]) * D[i] + mu[i] * D[i + 1]
        R.append(Ri)
        
        
        Li = (1 - lambda_[i + 1]) * D[i] + lambda_[i + 1] * D[i + 1]
        L.append(Li)
    
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

def plot_spline(P: np.array, V:np.array = None, num_points: int=100):

    h = vector_norm(P)
    
    gamma = calculate_gamma(h, V)
    mu = calculate_mu(P, h, gamma)
    lambda_ = calculate_lambda(P, h, gamma)

    R, L = calculate_Ri_Li(P, mu, lambda_)

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

