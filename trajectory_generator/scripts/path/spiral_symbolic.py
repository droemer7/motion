import numpy as np
import sympy as sym
from sympy import simplify
import scipy.optimize
import math
import csv
import io

class CubicSpiralEquations:
  def __init__(self):
    # Symbols
    self._pi = list(sym.symbols('xi, yi, thi'))
    self._pf = list(sym.symbols('xf, yf, thf'))
    self._kpe = list(sym.symbols('kxe, kye, kthe'))
    s = sym.symbols('s')
    p = list(sym.symbols('p0, p1, p2, p3, p4'))
    self._a = list(sym.symbols('a0, a1, a2, a3, a4'))

    # Parameter remap
    self._a[0] = p[0]
    self._a[1] = -(11*p[0]/2 - 9*p[1] + 9*p[2]/2 - p[3])/p[4]
    self._a[2] = (9*p[0] - 45*p[1]/2 + 18*p[2] - 9*p[3]/2)/(p[4]**2)
    self._a[3] = -(9*p[0]/2 - 27*p[1]/2 + 27*p[2]/2 - 9*p[3]/2)/(p[4]**3)
    self._a[4] = p[4]

    # Sub in p0 = p3 = 0.0 for zero starting and ending curvature
    p_sub = [(p[0], 0.0), (p[3], 0.0)]
    self._a = [a.subs(p_sub) for a in self._a]

    # Objective function components
    p = (p[1], p[2], p[4])
    obj_be = sym.simplify(sym.integrate((  self._a[0]
                                         + self._a[1]*s
                                         + self._a[2]*s**2
                                         + self._a[3]*s**3)**2,
                                         (s, 0, self._a[4])
                                       )
                         )
    obj_xe = sym.simplify(self._kpe[0] * (self.x(self._a[4]) - self._pf[0])**2)
    obj_ye = sym.simplify(self._kpe[1] * (self.y(self._a[4]) - self._pf[1])**2)
    obj_the = sym.simplify(self._kpe[2] * (self.th(self._a[4]) - self._pf[2])**2)

    obj_be_grad = sym.simplify(sym.derive_by_array(obj_be, p))
    obj_xe_grad = sym.simplify(sym.derive_by_array(obj_xe, p))
    obj_ye_grad = sym.simplify(sym.derive_by_array(obj_ye, p))
    obj_the_grad = sym.simplify(sym.derive_by_array(obj_the, p))

    obj_be_hess = sym.simplify(sym.hessian(obj_be, p))
    obj_xe_hess = sym.hessian(obj_xe, p)
    obj_ye_hess = sym.hessian(obj_ye, p)
    obj_the_hess = sym.simplify(sym.hessian(obj_the, p))

    with open('spiral_equations.csv', 'w', newline = '') as file:
      writer = csv.writer(file, delimiter=' ')
      writer.writerow(["obj_be = ", obj_be])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_xe = ", obj_xe])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_ye = ", obj_ye])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_the = ", obj_the])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_be_grad = ", obj_be_grad])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_xe_grad = ", obj_xe_grad])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_ye_grad = ", obj_ye_grad])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_the_grad = ", obj_the_grad])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_be_hess = ", obj_be_hess])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_xe_hess = ", obj_xe_hess])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_ye_hess = ", obj_ye_hess])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])
      writer.writerow(["obj_the_hess = ", obj_the_hess])
      writer.writerow([""])
      writer.writerow(["=============================================================="])
      writer.writerow([""])

  def th(self, s):
    return (        self._pi[2]
            +       self._a[0]*s
            + (1/2)*self._a[1]*s**2
            + (1/3)*self._a[2]*s**3
            + (1/4)*self._a[3]*s**4
           )

  def x(self, s):
    return self._pi[0] + s/24 * (    sym.cos(self._pi[2])
                                 + 4*sym.cos(self.th(s/8))
                                 + 2*sym.cos(self.th(2*s/8))
                                 + 4*sym.cos(self.th(3*s/8))
                                 + 2*sym.cos(self.th(4*s/8))
                                 + 4*sym.cos(self.th(5*s/8))
                                 + 2*sym.cos(self.th(6*s/8))
                                 + 4*sym.cos(self.th(7*s/8))
                                 +   sym.cos(self.th(s))
                                )

  def y(self, s):
    return self._pi[1] + s/24 * (    sym.sin(self._pi[2])
                                 + 4*sym.sin(self.th(s/8))
                                 + 2*sym.sin(self.th(2*s/8))
                                 + 4*sym.sin(self.th(3*s/8))
                                 + 2*sym.sin(self.th(4*s/8))
                                 + 4*sym.sin(self.th(5*s/8))
                                 + 2*sym.sin(self.th(6*s/8))
                                 + 4*sym.sin(self.th(7*s/8))
                                 +   sym.sin(self.th(s))
                                )

CubicSpiralEquations()