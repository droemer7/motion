from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import math
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=30, linewidth=100, floatmode='maxprec_equal')

EPSILON = 1e-10
VM_MIN = 0.1

# This is a prototype that will be ported to C++
# It has no function in the ROS package

# ==================
# Profile Generator
# ==================
class VelProfile:
  @dataclass
  class Point:
    x: float = 0.0
    v: float = 0.0
    a: float = 0.0
    j: float = 0.0
    t: float = 0.0

  def gen(self, dir, dist, x0, v0, a0, vd, vm, am, jm):
    vm = abs(vm)
    am = abs(am)
    jm = abs(jm)

    assert vm > 0, "vm cannot be zero"
    assert am > 0, "am cannot be zero"
    assert jm > 0, "jm cannot be zero"

    p = self._gen(dir, dist, x0, v0, a0, vd, vm, am, jm)

    return p

  # Private
  def __init__(self, max_iter=50):
    self._max_iter = max_iter

  def _calc_a(self, p: Point, dt: float):
    return p.a + p.j*dt

  def _calc_v(self, p: Point, dt: float):
    return p.v + p.a*dt + (p.j/2)*dt*dt

  def _calc_x(self, p: Point, dt: float):
    return p.x + p.v*dt + (p.a/2)*dt*dt + (p.j/6)*dt*dt*dt

  def _calc_p(self, p: Point, dt: float):
    assert (dt >= 0), "dt ({:.6f}) < 0.0".format(dt)
    return self.Point(self._calc_x(p, dt),
                      self._calc_v(p, dt),
                      self._calc_a(p, dt),
                      p.j,
                      p.t + dt)

  def _dir(self, p0, vd):
    # If v0 and vd are the same, sign(vd - v0) will initially be 0; however, after a small time step, sign(vd - v0) will be opposite to the sign of a0. Thus in this case we initialize sign(dv) = -sign(a0).
    # This step is necessary to ensure we do not have zero jerk, leading to the inability to construct the curve (and divide-by-zero errors).
    # If both sign(vd - v0) and sign(a0) are 0, then the curve degenerates to a single point and we will bypass the curve generation logic.
    if np.sign(vd - p0.v) != 0:
      return np.sign(vd - p0.v)
    elif np.sign(p0.a) != 0:
      return -np.sign(p0.a)
    else:
      return 0

  def _join(self, s):
    m = []
    for i in range(len(s)):
      for j in range(len(s[i])):

        # The profile needs to have at least one point, even if everything is zeros.
        # Once we've added one point, we start requiring t > 0 to ensure the next point is something real.
        if len(m) <= 1 or s[i][j].t > 0:

          # Always overwrite the previous point (last segment end point) once we begin a new segment (j = 0). These points overlap
          # in x, v and a, but the value of j is undefined at the end of a segment, and therefore isn't correct. Thus we take all
          # values from the new segment's start point to copy the correct jerk value (j).
          if i > 0 and j == 0:
            m[len(m)-1] = s[i][j]

          # Otherwise, just append the new point
          else:
            m.append(s[i][j])
        else:
          break

    return m

  def _gen(self, dir, dist, x0, v0, a0, vd, vm, am, jm):
    print()
    print("----------------------------------------------------------------------------------------------------------")

    self._print_parameters(dir, dist, x0, v0, a0, vd, abs(vm), abs(am), abs(jm)) # TODO: remove

    # Initialize segment array
    # A segment is an array of points where the control action changes (jerk)
    s = [[self.Point()] for i in range(5)]

    # Limits and distance are assumed to be in magnitude - calculations determine the appropriate direction
    vm = abs(vm)
    am = abs(am)
    jm = abs(jm)
    dist = abs(dist)

    # Increase vm to match vd if necessary
    if abs(vd) > abs(vm):
      vm = abs(vd)

    # Set the sign of vm to the direction of travel
    vm = dir*abs(vm)
    vm_init = vm
    vm_last = vm

    # Calculate xd based on user's desired direction of travel and distance
    xd = dir*dist + x0
    dx = xd - x0

    # Create a profile that ends at xd, if possible
    i = 0
    n = 1
    while (n < self._max_iter):
      print("Iteration {}: dx = {:.2f}, vm: {:.2e} ==> {:.2e}, vd = {:.2f}".format(n, dx, vm_last, vm, vd))
      print()

      # Initialize the current end point to the starting point
      pe = self.Point(x0, v0, a0)

      if (dist > 0 or
          (vm > v0 and vm > vd) or
          (vm < v0 and vm < vd)):
        # Create segment 0 v0 ==> vm (if v0 = vm and a0 = 0 this returns an array with one point)
        self._print_seg_info(i, 1, pe.v, vm, vd)
        s[i] = self._profile_seg(pe, vm, am, jm)
        pe = s[i][-1]
        i += 1

        # Create segment 1 vm ==> vd (if vm = vd this returns an array with one point)
        self._print_seg_info(i, 1, pe.v, vd, vd)
        s[i] = self._profile_seg(pe, vd, am, jm)
        pe = s[i][-1]
        i += 1
      else:
        # Create segment 0 v0 ==> vd
        self._print_seg_info(i, 2, pe.v, vd, vd)
        s[i] = self._profile_seg(pe, vd, am, jm)
        pe = s[i][-1]
        i += 1

      # If dist is specified, attempt to create a profile that ends at xd
      if dist > 0:
        # Recalculate how far away we are from xd
        dx = xd - pe.x

        # xd was exceeded: try to reduce overshoot of xd
        if dir*dx < 0:
          print("Exceeded distance: dx = {:.2f} (xe = {:.2f}, xd = {:.2f})".format(abs(dx), pe.x, xd))
          print()

          # If vm > vd & minimum then we can reduce vm to try to reduce overshoot of xd
          if abs(vm) > abs(vd) and abs(vm) > VM_MIN:
            # Recalculate a new vm proportional to xd/xe
            vm_last = vm
            vm = abs(vm-vd)*abs(xd)/abs(pe.x)
            vm = dir*max(abs(vm), abs(vd), VM_MIN)

            # Reset the segment index back to 0
            i = 0
            n += 1
          # vm is already equal to vd or minimum value, so we can't reduce overshoot of xd any further
          else:
            break

        # xd was not met: insert a constant velocity segment for the duration needed to achieve xd with all segments combined
        elif dir*dx > 0:
          print("Short distance: dx = {:.2f} (xe = {:.2f}, xd = {:.2f})".format(abs(dx), pe.x, xd))
          print()
          # Copy the previous segment to the current segment
          s[i] = s[i-1]

          # Copy the last point from segment 0 to which we will connect our new constant velocity segment
          pe = deepcopy(s[0][-1])

          # Calculate the amount of time needed to travel dx (the distance we are short by)
          dt = dx/pe.v

          # Insert the new constant velocity segment, overwriting the existing segment
          s[1] = [self.Point(pe.x, pe.v, pe.a, 0.0, pe.t), self.Point(pe.x + dx, pe.v, 0.0, 0.0, pe.t + dt)]

          # Shift the final segment's x and t values by the amounts in the inserted segment
          for p in range(len(s[i])):
            s[i][p].x += dx
            s[i][p].t += dt

          i += 1

          # At this point xe = xd, so we can stop here
          break
      # If dist is not specified, use the profile as-is
      else:
        break

    # Join all of the individual segments so they are connected properly
    p = self._join(s)

    self._print_profile(p)

    # Sub-sample the profile segment points into the final profile
    p = self._sample(p)

    print("Iterations: {}".format(n))
    print("Number of segments: {}".format(i))
    print("Final position: {:.2f}".format(p[0][-1]))
    print("----------------------------------------------------------------------------------------------------------")

    return p

  def _profile_seg(self, p0, vd, am, jm):
    print("  x0 = {:.2f}".format(p0.x))
    print("  v0 = {:.2f}, vd = {:.2f}".format(p0.v, vd))
    print("  a0 = {:.2f}, am = {:.2f}".format(p0.a, am))
    print("  j0 = {:.2f}".format(p0.j))
    print("  t0 = {:.2f}".format(p0.t))
    print()

    # Create list for points
    p = [self.Point() for i in range(7)]

    # Copy point 0
    i = 0
    p[i] = deepcopy(p0)

    # Determine direction of the desired change in v
    dv = vd - p[i].v
    dir = self._dir(p[i], vd)

    # Case 1
    if ((dv > 0 and p[i].a <= 0) or
        (dv < 0 and p[i].a >= 0)):
      print("  Case 1: dv = {:.2f}, a[{}] = {:.2f}".format(dv, i, p[i].a))

      p = self._profile_seg_base(p0, vd, am, jm)

    # Case 2
    elif ((dv > 0 and p[i].a < am) or # and p[i].a > 0
          (dv < 0 and p[i].a > -am)): # and p[i].a < 0
      print("  Case 2: dv = {:.2f}, a[{}] = {:.2f}".format(dv, i, p[i].a))

      # Create segment 0: jerk a0 ==> 0
      p[i].j = -dir*jm
      p[i+1] = self._calc_p(p[i], -p[i].a/p[i].j)
      i += 1

      # Recalculate where we are with respect to the target velocity vd
      dv_new = vd - p[i].v
      dir_new = self._dir(p[i], vd)

      # Haven't exceeded the target velocity vd: add jerk to get there faster
      if dir_new != 0 and (dir_new == dir):
        print("  Case 2-1: v: {:.2f} ==> {:.2f}, vd = {:.2f}".format(p[0].v, p[i].v, vd))
        # Change segment 0: jerk a0 ==> dir*am
        i = 0
        p[i].j = dir*jm
        p[i+1] = self._calc_p(p[i], (dir*am - p[i].a)/p[i].j)
        i += 1

        # Create segment 1: jerk dir*am ==> 0
        p[i].j = -dir*jm
        p[i+1] = self._calc_p(p[i], -p[i].a/p[i].j)
        i += 1

        # Recalculate where we are with respect to the target velocity vd
        dv_new = vd - p[i].v
        dir_new = self._dir(p[i], vd)

        # Still haven't reached the target velocity vd: add another segment
        if dir_new != 0 and (dir_new == dir):
          print("  Case 2-1-1: v: {:.2f} ==> {:.2f}, vd = {:.2f}".format(p[0].v, p[i].v, vd))
          # Change segment 1: acceleration so dv_new ==> 0
          i -= 1
          p[i].j = 0
          p[i+1] = self._calc_p(p[i], dv_new/p[i].a)
          i += 1

          # Create segment 2: jerk dir*am ==> 0
          # Paramters are the same as the first segment 1 we created
          p[i].j = -dir*jm
          p[i+1] = self._calc_p(p[i], -p[i].a/p[i].j)
          i += 1

        # Target velocity vd was exceeded, so we need to shorten the previous two jerk segments
        elif abs(dv_new) != 0.0:
          print("  Case 2-1-2: v: {:.2f} ==> {:.2f}, vd = {:.2f}".format(p[0].v, p[i].v, vd))
          i -= 2
        # a_join = -dir_new*math.sqrt(2*p[i].a**2 + 4*-dir_new*jm*vd - 4*-dir_new*jm*p[i].v)/2
          a_join = dir*math.sqrt(2*p[i].a**2 + 4*dir*jm*vd - 4*dir*jm*p[i].v)/2

          # Change segment 0: jerk a0 ==> a_join
          p[i].j = dir*jm
          p[i+1] = self._calc_p(p[i], (a_join - p[i].a)/p[i].j)
          i += 1

          # Create segment 1: jerk a_join ==> 0 and v ==> vd
          p[i].j = -dir*jm
          p[i+1] = self._calc_p(p[i], -a_join/p[i].j)
          i += 1

      # Target velocity vd was exceeded, reverse course
      elif abs(dv_new) != 0.0:
        print("  Case 2-2  : v: {:.2f} ==> {:.2f}, vd = {:.2f}".format(p[0].v, p[i].v, vd))

        p_add = self._profile_seg_base(p[i], vd, am, jm)
        p[i:len(p_add)+i] = p_add

    # Case 3
    elif dv != 0: # and abs(p[i].a) >= am
      print("  Case 3: dv = {:.2f}, a[{}] = {:.2f}".format(dv, i, p[i].a))

      # da may be zero, in which case we can skip to the 'next' segment
      if abs(dir*am - p[i].a) > 0:
        # Create segment 0: jerk a0 ==> dir*am
        p[i].j = -dir*jm
        p[i+1] = self._calc_p(p[i], (dir*am - p[i].a)/p[i].j)
        i += 1

      # Create segment 1: jerk dir*am ==> 0
      p[i].j = -dir*jm
      p[i+1] = self._calc_p(p[i], -p[i].a/p[i].j)
      i += 1

      # Recalculate where we are with respect to the target velocity vd
      dv_new = vd - p[i].v
      dir_new = self._dir(p[i], vd)

      # Still haven't reached the target velocity vd: add another segment
      if dir_new != 0 and (dir_new == dir):
        print("  Case 3-1: v: {:.2f} ==> {:.2f}, vd = {:.2f}".format(p[0].v, p[i].v, vd))
        # Change segment 1: acceleration so dv_new ==> 0
        i -= 1
        p[i].j = 0
        p[i+1] = self._calc_p(p[i], dv_new/p[i].a)
        i += 1

        # Create segment 2: jerk dir*am ==> 0
        # Paramters are the same as the first segment 1 we created
        p[i].j = -dir*jm
        p[i+1] = self._calc_p(p[i], -p[i].a/p[i].j)
        i += 1

      # Target velocity vd was exceeded, reverse course
      elif dir_new != 0:
        print("  Case 3-2: v: {:.2f} ==> {:.2f}, vd = {:.2f}".format(p[0].v, p[i].v, vd))

        p_add = self._profile_seg_base(p[i], vd, am, jm)
        p[i:len(p_add)+i] = p_add

    # Case 4
    else: # dv = 0, p[i].a = anything
      print("  Case 4: dv = {:.2f}, a[{}] = {:.2f}".format(dv, i, p[i].a))

      p_add = self._profile_seg_base(p[i], vd, am, jm)
      p[i:len(p_add)+i] = p_add

    # Delete unused segments (t = 0)
    p_trim = [p[0]]

    for i in range(1, len(p)):
      if p[i].t != 0:
        p_trim.append(p[i])

    p = p_trim

    print()
    self._print_profile(p)

    print()
    assert abs(p[-1].v - vd) < EPSILON, "v[{}] = {:.2f}, vd = {:.2f}".format(i, p[-1].v, vd)

    return p

  def _profile_seg_base(self, p0, vd, am, jm):
    # Create list for points
    p = [self.Point() for i in range(5)]

    # Copy point 0
    i = 0
    p[i] = deepcopy(p0)

    # Determine direction of the desired change in v
    dv = vd - p[i].v
    dir = self._dir(p[i], vd)

    # This function assumes that we are generating a simple curve which requires a0 = 0 or a0 opposite in sign to the direction of the desired change in v.
    # Other cases require different math to generate optimal (fastest) curves to vd, so they are not intended to be handled here.
    assert p[i].a == 0 or np.sign(p[i].a) != dir, "a0 = {:.1f}, dir = {}".format(p[i].a, dir)

    # If both dir is 0 here, the curve degenerates to a single point and we bypass all logic and return that point
    if dir != 0:
      # a may be zero, in which case we can skip to the 'next' segment
      if p[i].a != 0:
        # # Create segment 0: jerk a0 ==> 0
        # Note that v > vd at the end of this segment
        p[i].j = dir*jm
        p[i+1] = self._calc_p(p[i], -p[i].a/p[i].j)
        i += 1

      # Create segment 1: jerk 0 ==> dir*am
      p[i].j = dir*jm
      p[i+1] = self._calc_p(p[i], dir*am/p[i].j)
      i += 1

      # Create segment 2: jerk dir*am ==> 0
      p[i].j = -dir*jm
      p[i+1] = self._calc_p(p[i], -p[i].a/p[i].j)
      i += 1

      # Recalculate where we are with respect to the target velocity vd
      dv_new = vd - p[i].v
      dir_new = self._dir(p[i], vd)

      # Still haven't reached the target velocity vd: add another segment
      if dir_new != 0 and (dir_new == dir):
        print("  Base 1: v: {:.2f} ==> {:.2f}, vd = {:.2f}".format(p[0].v, p[i].v, vd))
        # Change segment 2: acceleration so dv_new ==> 0
        i -= 1
        p[i].j = 0
        p[i+1] = self._calc_p(p[i], dv_new/p[i].a)
        i += 1

        # Create segment 3: jerk dir*am ==> 0
        # Parameters are the same as the first segment 2 we created
        p[i].j = -dir*jm
        p[i+1] = self._calc_p(p[i], -p[i].a/p[i].j)
        i += 1

      # Target velocity vd was exceeded, so we need to shorten the previous two jerk segments
      elif dir_new != 0.0:
        print("  Base 2: v: {:.2f} ==> {:.2f}, vd = {:.2f}".format(p[0].v, p[i].v, vd))
        i -= 2
        dt = math.sqrt(abs(vd - p[i].v)/jm)

        # Change segment 1: jerk 0 ==> -a
        p[i].j = dir*jm
        p[i+1] = self._calc_p(p[i], dt)
        i += 1

        # Change segment 2: jerk -a ==> 0 and v ==> vd
        p[i].j = -dir*jm
        p[i+1] = self._calc_p(p[i], dt)
        i += 1

    return p[:i+1]

  def _sample(self, p, dt=0.01):
    # Find the number of segments in the profile, and calculate the number of sampled points in between
    # Add the number of segments to the number of points so when we generate
    # the sample times, we can add points at exactly each segment end
    num_seg = len(p) - 1
    num_pts = round(p[num_seg].t/dt + 1 + num_seg)

    # Create list for sampled points
    pts = [self.Point() for i in range(num_pts)]
    pts[0] = p[0]

    # Create the sample time array with jerk values
    i = 1
    j = 1
    while (i < num_pts and
           j <= num_seg):

      if p[j].t > 0 or j == 1:
        t = pts[i-1].t + dt

        if t > p[j].t:
          t = p[j].t
          j += 1

        pts[i] = self.Point(0, 0, 0, p[j-1].j, t)
        i += 1

    # Remove unused elements from the end
    for i in range(num_pts-1, 0, -1):
      if pts[i].t == 0:
        pts.pop()
      else:
        break

    # Calculate each point based on the previous one and the sample time
    for i in range(1, len(pts)):
      # Save the pre-calculated j because calc_pt will overwrite it
      j = pts[i].j

      # Calculate the new point
      pts[i] = self._calc_p(pts[i-1], pts[i].t - pts[i-1].t)
      #print("v(t={:.1f})={:.2f}".format(pts[i].t, pts[i].v))

      # Reassign j
      pts[i].j = j

    return self._to_array(pts)

  def _to_array(self, pts_list):
    pts_array = np.zeros((5, len(pts_list)))

    for i in range(len(pts_list)):
      pts_array[0][i] = pts_list[i].x
      pts_array[1][i] = pts_list[i].v
      pts_array[2][i] = pts_list[i].a
      pts_array[3][i] = pts_list[i].j
      pts_array[4][i] = pts_list[i].t

    return pts_array

  def _print_profile(self, s):
    # TODO change so to remove handling of unjoined segments in profile
    num_seg = 0
    x = []
    v = []
    a = []
    j = []
    t = []
    for i in range(len(s)):
      if isinstance(s[i], self.Point):
        num_seg = 1
        x.append(s[i].x)
        v.append(s[i].v)
        a.append(s[i].a)
        j.append(s[i].j)
        t.append(s[i].t)
      else:
        num_seg += 1
        for k in range(len(s[i])):
          x.append(s[i][k].x)
          v.append(s[i][k].v)
          a.append(s[i][k].a)
          j.append(s[i][k].j)
          t.append(s[i][k].t)

    print("Profile")
    print("x:", x)
    print("v:", v)
    print("a:", a)
    print("j:", j)
    print("t:", t)
    print()

  def _print_parameters(self, dir, dist, x0, v0, a0, vd, vm, am, jm):
    if dir > 0:
      print("Direction: (+)")
    else:
      print("Direction: (-)")
    print("Initial State: x = {:.1f}, v = {:.1f}, a = {:.1f}, j = ---".format(x0, v0, a0))
    print("Desired State: x = {:.1f}, v = {:.1f}, a = ---, j = ---".format(dir*dist + x0, vd))
    print("Limits:        x = ---, v = {:.1f}, a = {:.1f}, j = {:.1f}".format(vm, am, jm))
    print()

  def _print_seg_info(self, i, case, v0, vm, vd):
    print("Segment {}".format(i))
    print("Case {}: v0 = {:.2f}, vm = {:.2f}, vd = {:.2f}".format(case, v0, vm, vd))
    print()

# =============
# Unit Testing
# =============
class VelProfileTestCaseVariation():
  def __init__(self, title, dist, x0, v0, a0, vd, vm, am, jm):
    print()
    print(title)
    self.title = title
    profile = VelProfile()
    self.data = [profile.gen(dir=1,  dist=dist, x0=x0, v0=v0,  a0=a0,  vd=vd,  vm=vm, am=am, jm=jm),
                 profile.gen(dir=-1, dist=dist, x0=x0, v0=-v0, a0=-a0, vd=-vd, vm=vm, am=am, jm=jm)]
    self.dist_des = [dist, -dist]
    self.dist_act = [self.data[0][0][-1] - self.data[0][0][0],
                     self.data[1][0][-1] - self.data[1][0][0]]
    if dist > 0:
      self.dist_err = [self.dist_des[0] - self.dist_act[0],
                       self.dist_des[1] - self.dist_act[1]]
    else:
      self.dist_err = [0, 0]

class VelProfileTestCase():
  def __init__(self, title, dist, x0, v0, vd, vm, am, jm):
    print()
    print(title)
    self.title = title
    self.variations = [VelProfileTestCaseVariation("a0 > 0",       dist=dist, x0=x0, v0=v0, a0=abs(0.5*am),  vd=vd, vm=vm, am=am, jm=jm),
                       VelProfileTestCaseVariation("a0 = 0",       dist=dist, x0=x0, v0=v0, a0=0,            vd=vd, vm=vm, am=am, jm=jm),
                       VelProfileTestCaseVariation("-am < a0 < 0", dist=dist, x0=x0, v0=v0, a0=-0.5*abs(am), vd=vd, vm=vm, am=am, jm=jm),
                       VelProfileTestCaseVariation("a0 = -am",     dist=dist, x0=x0, v0=v0, a0=-am,          vd=vd, vm=vm, am=am, jm=jm),
                       VelProfileTestCaseVariation("a0 < -am",     dist=dist, x0=x0, v0=v0, a0=-1.5*abs(am), vd=vd, vm=vm, am=am, jm=jm)]

class VelProfileTestSet():
  def __init__(self, title, dist, x0, v0, vd, am, jm):
    self.title = title
    self.cases = [VelProfileTestCase(title + "\n\nCase 1: vm > v0 & vd\n",       dist=dist, x0=x0, v0=v0, vd=vd, vm=self._set_vm_high(v0, vd), am=am, jm=jm),
                  VelProfileTestCase(title + "\n\nCase 2: vm between v0 & vd\n", dist=dist, x0=x0, v0=v0, vd=vd, vm=self._set_vm_mid(v0, vd),  am=am, jm=jm),
                  VelProfileTestCase(title + "\n\nCase 3: vm < v0 & vd\n",       dist=dist, x0=x0, v0=v0, vd=vd, vm=self._set_vm_low(v0, vd),  am=am, jm=jm)]

  def _set_vm_high(self, v0, vd):
    vm = 2*max(abs(v0),abs(vd))
    if vm == 0:
      vm = 1
    return vm

  def _set_vm_mid(self, v0, vd):
    vm =  v0 + 0.5*(vd-v0)
    if vm == 0:
      vm = 1
    return vm

  def _set_vm_low(self, v0, vd):
    vm = 0.5*min(abs(v0),abs(vd))
    if vm == 0:
      vm = 1
    return vm

# Generate and run tests
test = VelProfileTestSet(title="Test Set 1: dist = 0",            dist=0,   x0=0, v0=0, vd=15, am=3, jm=1)
        #VelProfileTestSet(title="Test Set 2: dist > 0 & short",    dist=150, x0=0, v0=0, vd=15, am=3, jm=1)
        #VelProfileTestSet(title="Test Set 3: dist > 0 & exceeded", dist=10, x0=0, v0=0, vd=15, am=3, jm=1)

# Plot results
for case in test.cases:
  figure, axis = plt.subplots(2, len(case.variations), constrained_layout=True, figsize=(16,10))
  figure.suptitle(case.title)

  for v in range(len(case.variations)):

    for i in range(len(case.variations[v].data)):
      axis[i,v].set_title(case.variations[v].title)
      axis[i,v].plot(case.variations[v].data[i][4], case.variations[v].data[i][1:4].T)
      axis[i,v].annotate("Distance:\nDesired = {:.2f}\nActual = {:.2f}\nError = {:.2f}".
                         format(case.variations[v].dist_des[i], case.variations[v].dist_act[i], case.variations[v].dist_err[i]),
                         xy=(1, 1), xycoords='data', xytext=(0.02, .98), textcoords='axes fraction', va='top', ha='left')

# Display results
plt.show()
