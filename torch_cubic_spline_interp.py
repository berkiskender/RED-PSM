import matplotlib.pylab as pylab
import torch

def h_poly_helper(tt):
    A = torch.tensor([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype)
    return [
    sum( A[i, j]*tt[j] for j in range(4) )
    for i in range(4) ]

def h_poly(t):
    tt = [ None for _ in range(4) ]
    tt[0] = 1
    for i in range(1, 4):
        tt[i] = tt[i-1]*t
    return h_poly_helper(tt)

def H_poly(t):
    tt = [ None for _ in range(4) ]
    tt[0] = t
    for i in range(1, 4):
        tt[i] = tt[i-1]*t*i/(i+1)
    return h_poly_helper(tt)

# def interp(x, y, xs):
#     m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
#     m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
#     I = torch.searchsorted(x[1:], xs)
#     dx = (x[I+1]-x[I])
#     hh = h_poly((xs-x[I])/dx)
#     return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx

def interp(x, y, xs):
    m = (y[..., 1:] - y[..., :-1])/(x[1:] - x[:-1])
    m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1])/2, m[..., [-1]]], dim=-1)
    I = torch.searchsorted(x[1:], xs)
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[..., I] + hh[1]*m[..., I]*dx + hh[2]*y[..., I+1] + hh[3]*m[..., I+1]*dx

def integ(x, y, xs):
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
    I = pylab.searchsorted(x[1:], xs)
    Y = torch.zeros_like(y)
    Y[1:] = (x[1:]-x[:-1])*(
      (y[:-1]+y[1:])/2 + (m[:-1] - m[1:])*(x[1:]-x[:-1])/12
      )
    Y = Y.cumsum(0)
    dx = (x[I+1]-x[I])
    hh = H_poly((xs-x[I])/dx)
    return Y[I] + dx*(
      hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
      )

# Example
if __name__ == "__main__":
    x = torch.linspace(0, 6, 7)
    y = x.sin()
    xs = torch.linspace(0, 6, 101)
    ys = interp(x, y, xs)
    Ys = integ(x, y, xs)
    P.scatter(x, y, label='Samples', color='purple')
    P.plot(xs, ys, label='Interpolated curve')
    P.plot(xs, xs.sin(), '--', label='True Curve')
    P.plot(xs, Ys, label='Spline Integral')
    P.plot(xs, 1-xs.cos(), '--', label='True Integral')
    P.legend()
    P.show()