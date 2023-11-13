import numpy as np
import sympy

class QuinticSpline:
  """Defines a quintic spline with G2 continuity as r(u) = [x(u) y(u)]^T
     x(u) = a0 + a1*u + a2*u^2 + a3*u^3 + a4*u^4 + a5*u^5
     y(u) = b0 + b1*u + b2*u^2 + b3*u^3 + b4*u^4 + b5*u^5
     for u in [0,1]
  """
  def __init__(self, n: list, pi: list, pf: list):
    """
    args:
      n: [n0, n1, n2, n3]
         n0 and n1 are the curve 'velocity' at the beginning and end, respectively
         n2 and n3 are the curve 'twist' at the beginning and end, respectively
      pi: [xi, yi, thi, ki] - Initial point location, angle and curvature
      pf: [xf, yf, thf, kf] - Final point location, angle and curvature
    """
    # Unpack initial and final points to get locations, angles and curvatures
    self._pi = pi
    self._pf = pf
    xi, yi, thi, ki = pi
    xf, yf, thf, kf = pf

    # Calculate coefficients for x(u) polynomial
    self._a = np.zeros([6])
    self._a[0] = xi
    self._a[1] = n[0] * np.cos(thi)
    self._a[2] = 0.5 * (n[2] * np.cos(thi) - n[0] ** 2 * ki * np.sin(thi))
    self._a[3] = 10 * (xf - xi) - (6 * n[0] + 1.5 * n[2]) * np.cos(thi) \
                -(4 * n[1] - 0.5 * n[3]) * np.cos(thf) + 1.5 * n[0] ** 2 * ki * np.sin(thi) \
                - 0.5 * n[1] ** 2 * kf * np.sin(thf)
    self._a[4] = -15 * (xf - xi) + (8 * n[0] + 1.5 * n[2]) * np.cos(thi) \
                + (7 * n[1] - n[3]) * np.cos(thf) - 1.5 * n[0] ** 2 * ki * np.sin(thi) \
                + n[1] ** 2 * kf * np.sin(thf)
    self._a[5] = 6 * (xf - xi) - (3 * n[0] + 0.5 * n[2]) * np.cos(thi) \
                - (3 * n[1] - 0.5 * n[3]) * np.cos(thf) + 0.5 * n[0] ** 2 * ki * np.sin(thi) \
                - 0.5 * n[1] ** 2 * kf * np.sin(thf)

    # Calculate coefficients for y(u) polynomial
    self._b = np.zeros([6])
    self._b[0] = yi
    self._b[1] = n[0] * np.sin(thi)
    self._b[2] = 0.5 * (n[2] * np.sin(thi) + n[0] ** 2 * ki * np.cos(thi))
    self._b[3] = 10 * (yf - yi) - (6 * n[0] + 1.5 * n[2]) * np.sin(thi) \
                -(4 * n[1] - 0.5 * n[3]) * np.sin(thf) - 1.5 * n[0] ** 2 * ki * np.cos(thi) \
                + 0.5 * n[1] ** 2 * kf * np.cos(thf)
    self._b[4] = -15 * (yf - yi) + (8 * n[0] + 1.5 * n[2]) * np.sin(thi) \
                + (7 * n[1] - n[3]) * np.sin(thf) + 1.5 * n[0] ** 2 * ki * np.cos(thi) \
                - n[1] ** 2 * kf * np.cos(thf)
    self._b[5] = 6 * (yf - yi) - (3 * n[0] + 0.5 * n[2]) * np.sin(thi) \
                - (3 * n[1] - 0.5 * n[3]) * np.sin(thf) - 0.5 * n[0] ** 2 * ki * np.cos(thi) \
                + 0.5 * n[1] ** 2 * kf * np.cos(thf)

  def pi(self) -> float:
    return self._pi

  def pf(self) -> float:
    return self._pf

  def x(self, u: float) -> float:
    return   self._a[0]          \
           + self._a[1] * u      \
           + self._a[2] * u ** 2 \
           + self._a[3] * u ** 3 \
           + self._a[4] * u ** 4 \
           + self._a[5] * u ** 5 \

  def x_dot1(self, u: float) -> float:
    return   self._a[1]              \
           + self._a[2] * 2 * u      \
           + self._a[3] * 3 * u ** 2 \
           + self._a[4] * 4 * u ** 3 \
           + self._a[5] * 5 * u ** 4 \

  def x_dot2(self, u: float) -> float:
    return   self._a[2] * 2           \
           + self._a[3] * 6  * u      \
           + self._a[4] * 12 * u ** 2 \
           + self._a[5] * 20 * u ** 3 \

  def y(self, u: float) -> float:
    return   self._b[0]          \
           + self._b[1] * u      \
           + self._b[2] * u ** 2 \
           + self._b[3] * u ** 3 \
           + self._b[4] * u ** 4 \
           + self._b[5] * u ** 5 \

  def y_dot1(self, u: float) -> float:
    return   self._b[1]              \
           + self._b[2] * 2 * u      \
           + self._b[3] * 3 * u ** 2 \
           + self._b[4] * 4 * u ** 3 \
           + self._b[5] * 5 * u ** 4 \

  def y_dot2(self, u: float) -> float:
    return   self._b[2] * 2           \
           + self._b[3] * 6  * u      \
           + self._b[4] * 12 * u ** 2 \
           + self._b[5] * 20 * u ** 3 \

  def k(self, u: float) -> float:
    denom = (self.x_dot1(u) ** 2 + self.y_dot1(u) ** 2) ** 1.5
    if denom == 0:
      return np.inf
    else:
      num = self.x_dot1(u) * self.y_dot2(u) - self.y_dot1(u) * self.x_dot2(u)
      return num / denom

  def __call__(self, u: float) -> tuple:
    return self.x(u), self.y(u)

def calc_n(p: list) -> float:

  delta_x = p[1][0] - p[0][0]
  delta_y = p[1][1] - p[0][1]
  delta_th = p[1][2] - p[0][2]

  delta_max = 0.0
  if delta_x > delta_y:
    delta_max = delta_x
  else:
    delta_max = delta_y

  if delta_th > np.pi:
    delta_th = 2 * np.pi - delta_th

  n = 0.0
  if delta_th > np.pi / 2:
    n = (  2 * delta_max ** 2 \
         * 2 * (delta_th / np.pi)
        ) ** 0.5
  else:
    n = (delta_max ** 2 +  delta_max ** 2) ** 0.5

  print("n = ", n)
  return n


def figure_8(h: float) -> tuple:
  p = []
  p.append([0, 0, 0, 0])
  p.append([0, 0.5 * h, 0.75 * np.pi, 0])
  p.append([0, h, 0, 0])
  p.append([0, 0.5 * h, -0.75 * np.pi, 0])

  n1 = n2 = calc_n(p)

  s1 = QuinticSpline(n = [n1, n2, 0, 0],
                     pi = p[0],
                     pf = p[1]
                    )
  s2 = QuinticSpline(n = [n1, n2, 0, 0],
                     pi = p[1],
                     pf = p[2]
                    )
  s3 = QuinticSpline(n = [n1, n2, 0, 0],
                     pi = p[2],
                     pf = p[3]
                    )
  s4 = QuinticSpline(n = [n1, n2, 0, 0],
                     pi = p[3],
                     pf = p[0]
                    )
  u = sympy.symbols('u')
  sympy.plot_parametric(s1(u), s2(u), s3(u), s4(u),
                        (u, 0, 1),
                        xlim = (-0.5 * h * 1.5, 0.5 * h * 1.5),
                        ylim = (0, h * 1.5)
                       )

def circle(r: float):
  p = []
  p.append([0, 0, 0, 0])
  p.append([r, r, 0.5 * np.pi, 0])
  p.append([0, 2 * r, np.pi, 0])
  p.append([-r, r, 1.5 * np.pi, 0])

  n1 = n2 = calc_n(p)

  s1 = QuinticSpline(n = [n1, n2, 0, 0],
                     pi = p[0],
                     pf = p[1]
                    )
  s2 = QuinticSpline(n = [n1, n2, 0, 0],
                     pi = p[1],
                     pf = p[2]
                    )
  s3 = QuinticSpline(n = [n1, n2, 0, 0],
                     pi = p[2],
                     pf = p[3]
                    )
  s4 = QuinticSpline(n = [n1, n2, 0, 0],
                     pi = p[3],
                     pf = p[0]
                    )
  u = sympy.symbols('u')
  sympy.plot_parametric(s1(u), s2(u), s3(u), s4(u),
                        (u, 0, 1),
                        xlim = (-r * 1.5, r * 1.5),
                        ylim = (0, 2 * r * 1.5)
                       )

def point_to_point(x: float, y: float, thi: float, thf: float):
  p = []
  p.append([0, 0, thi, 0])
  p.append([x, y, thf, 0])

  n1 = n2 = calc_n(p) * 5

  s = QuinticSpline(n = [n1, n2, 0, 0],
                    pi = p[0],
                    pf = p[1]
                   )
  u = sympy.symbols('u')
  sympy.plot_parametric(s(u),
                        (u, 0, 1),
                        xlim = (0, x * 1.5),
                        ylim = (0, x * 1.5)
                       )

if __name__ == '__main__':
  #figure_8(3.0)
  #circle(5)
  point_to_point(10, 10, 0.5 * np.pi, 1.5 * np.pi)