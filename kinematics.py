#python 2.7
# modules
from math import sin, cos, tan, sqrt, radians, degrees, asin, acos, atan, pi
from scipy.optimize import fsolve

# python 2
Author = 'Harith Al-safi'
Date = "5/5/2020"

# inputs
# projectile motion vs normal
print('This is a kinematics calculater by Harith Al-Safi')
print('Input x for unknown values, [g] is -9.81')
print('-------')
print('[1] for normal kinematics')
print('[2] for horizantal projectile')
print('[3] for angled projectile')
p = input('Enter your choice: ')
print('-------')

# initials
x = 'non'
g = -9.81


# normal kinematics
if p == 1:
    u = input('[u] Initial velocity : ')
    v = input('[v] Final velocity : ')
    s = input('[s] Displacement : ')
    a = input('[a] Acceleration : ')
    t = input('[t] Time : ')
    print('--------')

# horizantal projectile motion
if p == 2:
    ux = input('[ux] Horizantal Initial velocity: ')
    sx = input('[sx] Horizantal displacement: ')
    u = 0
    v = input('[v] Vertical Final velocity: ')
    s = input('[s] Verical displacement: ')
    a = g
    t = input('[t] Time: ')
    V = input('[V] Resultant final velocity: ')
    th = input('[th] Angle of landing in degrees: ')
    print('--------')
    if th != x:
        th = radians(th)

# angled projectile
if p == 3:
    U = input('[U] Resultant initial velocity: ')
    ux = input('[ux] Horizantal Initial velocity: ')
    u = input('[u] Vertical Initial velocity: ')
    th = input('[th] Angle of launching in degrees: ')
    th1 = input('[th1] Angle of landing in degrees: ')
    V = input('[V] Resultant final velocity: ')
    v = input('[v] Vertical Final velocity: ')
    hm = input('[hm] Maximum height above launch: ')
    sx = input('[sx] Horizantal displacement: ')
    s = input('[s] Verical displacement at any point: ')
    a = g
    t = input('[t] Time: ')
    print('--------')
    if th != x:
        th = radians(th)
    if th1 != x:
        th1 = radians(th1)


# functions and classes:
# kinematics
class kinematics:
    # s, a, t are given
    def kin_s_a_t(self, u, v, s, a, t):
        del u
        del v
        v = float(s + 0.5*a*t**2)/float(t)
        u = v-a*t
        return u, v

    # u, v, a are given
    @classmethod
    def kin_u_v_a(cls, u, v, s, a, t):
        del s
        del t
        t = float(v-u)/float(a)
        s = (float(u+v)/float(2))*t
        return s, t

    # v, u, t are given
    @classmethod
    def kin_u_v_t(cls, u, v, s, a, t):
        del s
        del a
        a = float(v-u)/float(t)
        s = (float(u+v)/float(2))*t
        return s, a

    # u, v, s are given
    @classmethod
    def kin_u_v_s(cls, u, v, s, a, t):
        del a
        del t
        a = float(v**2-u**2)/float(2*s)
        t = float(v-u)/float(a)
        return a, t

    # u, a, t are given
    @classmethod
    def kin_u_a_t(cls, u, v, s, a, t):
        del v
        del s
        v = u + a*t
        s = float(v**2-u**2)/float(2*a)
        return v, s

    # v, a, t are given
    @classmethod
    def kin_v_a_t(cls, u, v, s, a, t):
        del u
        del s
        u = v-a*t
        s = float(v**2-u**2)/float(2*a)
        return u, s

    # v, s, t are given
    @classmethod
    def kin_v_s_t(cls, u, v, s, a, t):
        del u
        del a
        a = float(-0.5*(s-v*t))/float(t**2)
        u = v-a*t
        return u, a

    # v, s, a are givem
    @classmethod
    def kin_v_s_a(cls, u, v, s, a, t):
        del u
        del t
        if s < 0:
            u = -sqrt(v**2-2*a*s)
        else:
            u = sqrt(v**2-2*a*s)
        t = float(v-u)/float(a)
        return u, t

    # u, s, a are given
    @classmethod
    def kin_u_s_a(cls, u, v, s, a, t):
        del v
        del t
        if s < 0:
            v = -sqrt(u**2+2*a*s)
        else:
            v = sqrt(u**2+2*a*s)
        t = float(v-u)/float(a)
        return v, t

    # u, s, t are given
    @classmethod
    def kin_u_s_t(cls, u, v, s, a, t):
        del v
        del a
        v = 2*(float(s)/float(t))-u
        a = float(v-u)/float(t)
        return v, a


# horizantal projectile
class hor_projectile:
    # ux and sx are given
    def projt_sx_ux(self, ux, sx, t):
        del t
        t = float(sx)/float(ux)
        return t

    # ux and t are given
    @classmethod
    def projt_ux_t(cls, ux, sx, t):
        del sx
        sx = ux*t
        return sx

    # sx and t are given
    @classmethod
    def projt_sx_t(cls, ux, sx, t):
        del ux
        ux = float(sx)/float(t)
        return ux

    # resultant final velocity
    @classmethod
    def projt_a_s(cls, V, v, ux):
        del V
        V = -sqrt(ux**2 + v**2)
        return V

    # vertical final from V and ux
    @classmethod
    def projt_V_ux(cls, V, v, ux):
        del v
        v = -sqrt(V**2-ux**2)
        return v

    # horizantal velocity
    @classmethod
    def projt_V_v(cls, V, v, ux):
        del ux
        ux = sqrt(V**2-v**2)
        return ux

    # angle at hitting
    @classmethod
    def protj_th(cls, ux, V):
        th = acos(float(ux)/float(abs(V)))
        return th


# angled projectile
class angle_projectile:
    # initial launching
    # U and th are given
    def pro_U_th(self, U, th, ux, u):
        del u
        del ux
        u = sin(th)*U
        ux = cos(th)*U
        return u, ux

    # U and ux are given
    @classmethod
    def pro_U_ux(cls, U, th, ux, u):
        del th
        del u
        u = sqrt(U**2-ux**2)
        th = asin(float(u)/float(U))
        return u, th

    # U and u are given
    @classmethod
    def pro_U_u(cls, U, th, ux, u):
        del th
        del ux
        ux = sqrt(U**2-u**2)
        th = asin(float(u)/float(U))
        return ux, th

    # ux and u are given
    @classmethod
    def pro_u_ux(cls, U, th, ux, u):
        del th
        del U
        U = sqrt(ux**2+u**2)
        th = asin(float(u)/float(U))
        return U, th

    # ux and th are given
    @classmethod
    def pro_ux_th(cls, U, th, ux, u):
        del U
        del u
        U = float(ux)/float(cos(th))
        u = sqrt(U**2-ux**2)
        return U, u

    # u and th are given
    @classmethod
    def pro_u_th(cls, U, th, ux, u):
        del U
        del ux
        U = float(u)/float(sin(th))
        ux = sqrt(U**2-u**2)
        return U, ux


# numerical solutions
class numer:
    def solve_U(self, th):
        z = sx*tan(th) + (sx**2*a)/(2*U**2*(cos(th))**2)-s
        return z

    @classmethod
    def solve_u(cls, th):
        z = u*(float(sx*tan(th))/float(u))+0.5*a*(float(sx*tan(th))/float(u))**2-s
        return z

    @classmethod
    def solve_V(cls, th):
        z = -sx*tan(th)-0.5*a*(float(sx)/float(-V*cos(th)))**2-s
        return z

    @classmethod
    def solve_v(cls, th1):
        z = v*float(sx*tan(th1))/float(-v)-0.5*a*(float(sx*tan(th1)/float(-v)))**2-s
        return z


# class initiations
k = kinematics()
h = hor_projectile()
o = angle_projectile()
q = numer()


# print function
# Colored output
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def prt():
    if p == 1:
        print(bcolors.ENDC + '[u] Initial velocity is: ' + bcolors.WARNING + str(u))
        print(bcolors.ENDC + '[v] Final velocity is: ' + bcolors.WARNING + str(v))
        print(bcolors.ENDC + '[s] Displacement is: ' + bcolors.WARNING + str(s))
        print(bcolors.ENDC + '[a] Acceleration is: ' + bcolors.WARNING + str(a))
        print(bcolors.ENDC + '[t] Time is: ' + bcolors.WARNING + str(t))
        exit()
    if p == 2:
        print(bcolors.ENDC + '[ux] Horizantal Initial velocity is: ' + bcolors.WARNING + str(ux))
        print(bcolors.ENDC + '[sx] Horizantal displacement is: ' + bcolors.WARNING + str(sx))
        print(bcolors.ENDC + '[v] Final velocity is: ' + bcolors.WARNING + str(v))
        print(bcolors.ENDC + '[s] Displacement is: ' + bcolors.WARNING + str(s))
        print(bcolors.ENDC + '[t] Time is: ' + bcolors.WARNING + str(t))
        print(bcolors.ENDC + '[V] Resultant final velocity is: ' + bcolors.WARNING + str(V))
        print(bcolors.ENDC + '[th] Angle of impact is: ' + bcolors.WARNING + str(degrees(th)))
        exit()


def prt1():
    print(bcolors.ENDC + '[U] Initial resultant velocity is: ' + bcolors.WARNING + str(U))
    print(bcolors.ENDC + '[ux] Horizantal Initial velocity is: ' + bcolors.WARNING + str(ux))
    print(bcolors.ENDC + '[u] Vertical initial velocity is: ' + bcolors.WARNING + str(u))
    print(bcolors.ENDC + '[th] Angle of launch is: ' + bcolors.WARNING + str(degrees(th)))
    print(bcolors.ENDC + '[th1] Angle of impact is: ' + bcolors.WARNING + str(degrees(th1)))
    print(bcolors.ENDC + '[V] Resultant final velocity is: ' + bcolors.WARNING + str(V))
    print(bcolors.ENDC + '[v] Final velocity is: ' + bcolors.WARNING + str(v))
    if U > 0:
        print(bcolors.ENDC + '[hm] Maximum height is: ' + bcolors.WARNING + str(hm))
    print(bcolors.ENDC + '[sx] Horizantal displacement is: ' + bcolors.WARNING + str(sx))
    if s != 0:
        print(bcolors.ENDC + '[s] Displacement is: ' + bcolors.WARNING + str(s))
    print(bcolors.ENDC + '[t] Time is: ' + bcolors.WARNING + str(t))
    exit()


# conditions
# kinematics
if p == 1:
    # 3 values are given
    if u == x and v == x or s == x and t == x or s == x and a == x or a == x and t == x or v == x and s == x or u == x and s == x or u == x and a == x or u == x and t == x or v == x and t == x or v == x and a == x:
        if u == x and v == x:
            u, v = k.kin_s_a_t(u, v, s, a, t)
            prt()
        if s == x and t == x:
            s, t = k.kin_u_v_a(u, v, s, a, t)
            prt()
        if s == x and a == x:
            s, a = k.kin_u_v_t(u, v, s, a, t)
            prt()
        if a == x and t == x:
            a, t = k.kin_u_v_s(u, v, s, a, t)
            prt()
        if v == x and s == x:
            v, s = k.kin_u_a_t(u, v, s, a, t)
            prt()
        if u == x and s == x:
            u, s = k.kin_v_a_t(u, v, s, a, t)
            prt()
        if u == x and a == x:
            u, a = k.kin_v_s_t(u, v, s, a, t)
            prt()
        if u == x and t == x:
            u, t = k.kin_v_s_a(u, v, s, a, t)
            prt()
        if v == x and t == x:
            v, t = k.kin_u_s_a(u, v, s, a, t)
            prt()
        if v == x and a == x:
            v, a = k.kin_u_s_t(u, v, s, a, t)
            prt()
    # 4 values given
    if u == x and v != x and s != x and a != x and t != x or v == x and u != x and s != x and a != x and t != x:
        u, v = k.kin_s_a_t(u, v, s, a, t)
        prt()
    if s == x and v != x and u != x and a != x and t != x or t == x and v != x and u != x and a != x and s != x:
        s, t = k.kin_u_v_a(u, v, s, a, t)
        prt()
    if a == x and v != x and u != x and s != x and t != x:
        s, a = k.kin_u_v_t(u, v, s, a, t)
        prt()

# angled projectile
while p == 3:
    # normal projectila
    if s == 0:
        # if none are given
        if U == x and ux == x and u == x and th == x:
            if sx != x and t != x:
                ux = float(sx)/float(t)
                th = atan(float(0.5*-a*t)/float(ux))
                U, u = o.pro_ux_th(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
        # U is given
        if U != x and ux == x and u == x and th == x:
            # sx and t are given
            if sx != x and t != x:
                ux = float(sx)/float(t)
                u, th = o.pro_U_ux(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # hm given
            if hm != x:
                u = sqrt(-2*a*hm)
                ux, th = o.pro_U_u(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                prt1()
            # if sx is only given
            if sx != x and t == x:
                th = float(asin(float(sx*-a)/float(U**2)))/float(2)
                u, ux = o.pro_U_th(U, th, ux, u)
                t = float(sx)/float(ux)
                th1 = th
                V = -U
                v = -u
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
        # th is given
        if th != x and U == x and ux == x and u == x:
            if sx != x and t != x:
                ux = float(sx)/float(t)
                U, u = o.pro_ux_th(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            if hm != x:
                u = sqrt(-2*a*hm)
                U, ux = o.pro_u_th(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                prt1()
        # if u is given
        if u != x and U == x and th == x and ux == x:
            if sx != x and t != x:
                ux = float(sx)/float(t)
                U, th = o.pro_u_ux(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
        # if ux is given
        if ux != x and U == x and u == x and th == x:
            if hm != x:
                u = sqrt(-2*a*hm)
                U, th = o.pro_u_ux(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                prt1()
            if t != x:
                sx = ux*t
                th = atan(float(0.5*-a*t)/float(ux))
                U, u = o.pro_ux_th(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
        # if 2 or more are given
        if U != x and th != x or U != x and ux != x or U != x and u != x or ux != x and u != x or ux != x and th != x or u != x and th != x:
            # 2 only
            # U and th are given
            if U != x and th != x and ux == x and u == x:
                u, ux = o.pro_U_th(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # U and ux are given
            if U != x and ux != x and u == x and th == x:
                u, th = o.pro_U_ux(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # U and u are given
            if U != x and u != x and ux == x and th == x:
                ux, th = o.pro_U_u(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # ux and u are given
            if ux != x and u != x and U == x and th == x:
                U, th = o.pro_u_ux(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # ux and th are given
            if ux != x and th != x and U == x and u == x:
                U, u = o.pro_ux_th(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # u and th
            if u != x and th != x and U == x and ux == x:
                U, ux = o.pro_u_th(U, th, ux, u)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # 3 are given
            # U is unknown
            if U == x and ux != x and u != x and th != x:
                U = sqrt(ux**2+u**2)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # ux is uknown
            if ux == x and U != x and th != x and u != x:
                ux = sqrt(U**2-u**2)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # u is unkown
            if u == x and U != x and th != x and ux != x:
                u = sqrt(U**2-ux**2)
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # th is unknown
            if th == x and U != x and ux != x and u != x:
                th = asin(float(u)/float(U))
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()
            # 4 aree given
            if th != x and U != x and ux != x and u != x:
                th1 = th
                V = -U
                v = -u
                if t == x:
                    t = float(v-u)/float(a)
                if sx == x:
                    sx = ux*t
                if hm == x:
                    hm = float(-u**2)/float(2*a)
                prt1()

    # angled projectile
    if s != 0:
        # Initials only
        if V == x and v == x and th1 == x:
            # one initial only
            # U is given
            if U != x and ux == x and u == x and th == x:
                # sx and t are given
                if sx != x and t != x:
                    ux = float(sx)/float(t)
                    u, th = o.pro_U_ux(U, th, ux, u)
                    p = 1
                if s != x and t != x:
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    ux, th = o.pro_U_u(U, th, ux, u)
                    p = 1
                if hm != x:
                    u = sqrt(-2*a*hm)
                    ux, th = o.pro_U_u(U, th, ux, u)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # s and sx are given
                if sx != x and s != x:
                    x2 = 0.001
                    x3 = fsolve(q.solve_U, x2)
                    th = x3[0]
                    u, ux = o.pro_U_th(U, th, ux, u)
                    t = float(sx)/float(ux)
                    p = 1
            # th is given
            if th != x and U == x and ux == x and u == x:
                # sx and s are only given
                if sx != x and s != x:
                    U = sqrt(float(a*sx**2)/float(2*s*(cos(th))**2-2*(cos(th))**2*sx*tan(th)))
                    u, ux = o.pro_U_th(U, th, ux, u)
                    t = float(sx)/float(ux)
                    p = 1
                # sx and t are given
                if sx != x and t != x:
                    ux = float(sx)/float(t)
                    U, u = o.pro_ux_th(U, th, ux, u)
                    p = 1
                # s and t are given
                if s != x and t != x:
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    U, ux = o.pro_u_th(U, th, ux, u)
                    p = 1
                if hm != x:
                    u = sqrt(-2*a*hm)
                    U, ux = o.pro_u_th(U, th, ux, u)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
            # ux is given
            if ux != x and U == x and u == x and th == x:
                if s != x and t != x:
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    U, th = o.pro_u_ux(U, th, ux, u)
                    p = 1
                if hm != x:
                    u = sqrt(-2*a*hm)
                    U, th = o.pro_u_ux(U, th, ux, u)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                if sx != x and s != x:
                    t = float(sx)/float(ux)
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    U, th = o.pro_u_ux(U, th, ux, u)
                    p = 1
            # if u is given
            if u != x and U == x and ux == x and th == x:
                if sx != x and t != x:
                    ux = float(sx)/float(t)
                    U, th = o.pro_u_ux(U, th, ux, u)
                    p = 1
                if s != x and t != x:
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    U, th = o.pro_u_ux(U, th, ux, u)
                    p = 1
                if sx != x and s != x:
                    x2 = 1.570796
                    x3 = fsolve(q.solve_u, x2)
                    th = x3[0]
                    U, ux = o.pro_u_th(U, th, ux, u)
                    t = float(sx)/float(ux)
                    p = 1

            # 2 or more Initial angled quantities given
            if U != x and th != x or U != x and ux != x or U != x and u != x or ux != x and u != x or ux != x and th != x or u != x and th != x:
                # 2 only
                # U and th are given
                if U != x and th != x and ux == x and u == x:
                    u, ux = o.pro_U_th(U, th, ux, u)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # U and ux are given
                if U != x and ux != x and u == x and th == x:
                    u, th = o.pro_U_ux(U, th, ux, u)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # U and u are given
                if U != x and u != x and ux == x and th == x:
                    ux, th = o.pro_U_u(U, th, ux, u)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # ux and u are given
                if ux != x and u != x and U == x and th == x:
                    U, th = o.pro_u_ux(U, th, ux, u)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # ux and th are given
                if ux != x and th != x and U == x and u == x:
                    U, u = o.pro_ux_th(U, th, ux, u)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # u and th
                if u != x and th != x and U == x and ux == x:
                    U, ux = o.pro_u_th(U, th, ux, u)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # 3 are given
                # U is unknown
                if U == x and ux != x and u != x and th != x:
                    U = sqrt(ux**2+u**2)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # ux is uknown
                if ux == x and U != x and th != x and u != x:
                    ux = sqrt(U**2-u**2)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # u is unkown
                if u == x and U != x and th != x and ux != x:
                    u = sqrt(U**2-ux**2)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # th is unknown
                if th == x and U != x and ux != x and u != x:
                    th = asin(float(u)/float(U))
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # 4 given
                if th != x and U != x and ux != x and u != x:
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
        # finals only
        if U == x and u == x and th == x:
            # one is given
            # V is given
            if V != x and ux == x and v == x and th1 == x:
                # sx and t are given
                if sx != x and t != x:
                    ux = float(sx)/float(t)
                    v, th1 = o.pro_U_ux(V, th1, ux, v)
                    v = -v
                    th1 = -th1
                    p = 1
                # s and t
                if s != x and t != x:
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    ux, th1 = o.pro_U_u(V, th1, ux, v)
                    p = 1
                # sx and s
                if sx != x and s != x:
                    x2 = 1.570796
                    x3 = fsolve(q.solve_V, x2)
                    th1 = x3[0]
                    v, ux = o.pro_U_th(V, th1, ux, v)
                    ux = -ux
                    t = float(sx)/float(ux)
                    p = 1
            # th1 is given
            if th1 != x and ux == x and v == x and V == x:
                # sx, s
                if sx != x and s != x:
                    V = -sqrt(float(a*sx**2)/float((cos(th1))**2*(-2*(s+sx*tan(th1)))))
                    v, ux = o.pro_U_th(V, th1, ux, v)
                    ux = -ux
                    t = float(sx)/float(ux)
                    p = 1
                # sx and t are given
                if sx != x and t != x:
                    ux = float(sx)/float(t)
                    V, v = o.pro_ux_th(V, th1, ux, v)
                    p = 1
                # s and t are given
                if s != x and t != x:
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    V, ux = o.pro_u_th(V, th1, ux, v)
                    p = 1
            # ux is given
            if ux != x and V == x and v == x and th1 == x:
                # if sx and s
                if sx != x and s != x:
                    t = float(sx)/float(ux)
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    V, th1 = o.pro_u_ux(V, th1, ux, v)
                    V = -V
                    p = 1
                # s and t
                if s != x and t != x:
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    V, th1 = o.pro_u_ux(V, th1, ux, v)
                    V = -V
                    p = 1
            # if v is given
            if v != x and V == x and th1 == x and ux == x:
                # sx and t given
                if sx != x and t != x:
                    ux = float(sx)/float(t)
                    V, th1 = o.pro_u_ux(V, th1, ux, v)
                    V = - V
                    p = 1
                # s and t
                if s != x and t != x:
                    u, v = k.kin_s_a_t(u, v, s, a, t)
                    V, th1 = o.pro_u_ux(V, th1, ux, v)
                    V = -V
                    p = 1
                # sx and s
                if sx != x and s != x:
                    x2 = 1.570796
                    x3 = fsolve(q.solve_v, x2)
                    th1 = x3[0]
                    V, ux = o.pro_u_th(V, th1, ux, v)
                    t = float(sx)/float(ux)
                    p = 1
            # 2 or more final
            if V != x and th1 != x or V != x and ux != x or V != x and v != x or ux != x and v != x or ux != x and th1 != x or v != x and th1 != x:
                # 2 only
                # V and th
                if V != x and th1 != x and ux == x and v == x:
                    v, ux = o.pro_U_th(V, th1, ux, v)
                    ux = -ux
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # V and ux
                if V != x and ux != x and th1 == x and v == x:
                    v, th1 = o.pro_U_ux(V, th1, ux, v)
                    v = -v
                    th1 = -th1
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # V and v are given
                if V != x and v != x and th1 == x and ux == x:
                    ux, th1 = o.pro_U_u(V, th1, ux, v)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # ux and v
                if ux != x and v != x and th1 == x and V == x:
                    V, th1 = o.pro_u_ux(V, th1, ux, v)
                    V = -V
                    th1 = -th1
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # ux and th1
                if ux != x and th1 != x and V == x and v == x:
                    V, v = o.pro_ux_th(V, th1, ux, v)
                    V = -V
                    v = -v
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # v and th1
                if v != x and th1 != x and V == x and ux == x:
                    V, ux = o.pro_u_th(V, th1, ux, v)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # 3 values are given
                # V is unknown
                if V == x and th1 != x and v != x and ux != x:
                    V = -sqrt(ux**2+v**2)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # ux is uknown
                if ux == x and v != x and V != x and th1 != x:
                    ux = sqrt(V**2-v**2)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # v is unkown
                if v == x and V != x and th1 != x and ux != x:
                    v = -sqrt(V**2-ux**2)
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # th1 is unknown
                if th1 == x and V != x and v != x and ux != x:
                    th1 = asin(float(v)/float(V))
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
                # 4 given
                if th1 != x and V != x and v != x and ux != x:
                    if sx != x and v == x and s == x and t == x:
                        t = float(sx)/float(ux)
                        p = 1
                    else:
                        p = 1
    break

while p == 1:
    # 3 values are given
    if u == x and v == x or s == x and t == x or s == x and a == x or a == x and t == x or v == x and s == x or u == x and s == x or u == x and a == x or u == x and t == x or v == x and t == x or v == x and a == x:
        # t, s, a are given
        if u == x and v == x and t != x and s != x and a != x:
            u, v = k.kin_s_a_t(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
        # a, v, u are given
        if s == x and t == x and u != x and v != x and a != x:
            s, t = k.kin_u_v_a(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
        # u, v, t are given
        if s == x and a == x and v != x and u != x and t != x:
            s, a = k.kin_u_v_t(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
        # u, v, s are given
        if a == x and t == x and u != x and v != x and s != x:
            a, t = k.kin_u_v_s(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
        # u, a, t
        if v == x and s == x and u != x and a != x and t != x:
            v, s = k.kin_u_a_t(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
        # v, a, t are given
        if u == x and s == x and a != x and v != x and t != x:
            u, s = k.kin_v_a_t(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
        # v, t, s
        if u == x and a == x and v != x and t != x and s != x:
            u, a = k.kin_v_s_t(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
        # v, s, a
        if u == x and t == x and v != x and s != x and a != x:
            u, t = k.kin_v_s_a(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
        # u, a, s
        if v == x and t == x and s != x and a != x and u != x:
            v, t = k.kin_u_s_a(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
        # u, s, t
        if v == x and a == x and u != x and s != x and t != x:
            v, a = k.kin_u_s_t(u, v, s, a, t)
            if sx == x:
                sx = ux * t
            if V == x or th1 == x:
                if V == x and th1 == x:
                    V = -sqrt(ux**2+v**2)
                    th1 = asin(float(v)/float(V))
                if V == x and th1 != x:
                    V = -sqrt(ux**2+v**2)
                if th1 == x and V != x:
                    th1 = asin(float(v)/float(V))
            if U == x or th == x:
                if U == x and th == x:
                    U = sqrt(ux**2+u**2)
                    th = asin(float(u)/float(U))
                if U == x and th != x:
                    U = sqrt(ux**2+u**2)
                if th == x and U != x:
                    th = asin(float(u)/float(U))
            if hm == x:
                hm = float(-u**2)/float(2*a)
            prt1()
            break
    # 4 values given
    if u == x and v != x and s != x and a != x and t != x or v == x and u != x and s != x and a != x and t != x:
        u, v = k.kin_s_a_t(u, v, s, a, t)
        if sx == x:
            sx = ux * t
        if V == x or th1 == x:
            if V == x and th1 == x:
                V = -sqrt(ux**2+v**2)
                th1 = asin(float(v)/float(V))
            if V == x and th1 != x:
                V = -sqrt(ux**2+v**2)
            if th1 == x and V != x:
                th1 = asin(float(v)/float(V))
        if U == x or th == x:
            if U == x and th == x:
                U = sqrt(ux**2+u**2)
                th = asin(float(u)/float(U))
            if U == x and th != x:
                U = sqrt(ux**2+u**2)
            if th == x and U != x:
                th = asin(float(u)/float(U))
        if hm == x:
            hm = float(-u**2)/float(2*a)
        prt1()
        break
    if s == x and v != x and u != x and a != x and t != x or t == x and v != x and u != x and a != x and s != x:
        s, t = k.kin_u_v_a(u, v, s, a, t)
        if sx == x:
            sx = ux * t
        if V == x or th1 == x:
            if V == x and th1 == x:
                V = -sqrt(ux**2+v**2)
                th1 = asin(float(v)/float(V))
            if V == x and th1 != x:
                V = -sqrt(ux**2+v**2)
            if th1 == x and V != x:
                th1 = asin(float(v)/float(V))
        if U == x or th == x:
            if U == x and th == x:
                U = sqrt(ux**2+u**2)
                th = asin(float(u)/float(U))
            if U == x and th != x:
                U = sqrt(ux**2+u**2)
            if th == x and U != x:
                th = asin(float(u)/float(U))
        if hm == x:
            hm = float(-u**2)/float(2*a)
        prt1()
        break
    if a == x and v != x and u != x and s != x and t != x:
        s, a = k.kin_u_v_t(u, v, s, a, t)
        if sx == x:
            sx = ux * t
        if V == x or th1 == x:
            if V == x and th1 == x:
                V = -sqrt(ux**2+v**2)
                th1 = asin(float(v)/float(V))
            if V == x and th1 != x:
                V = -sqrt(ux**2+v**2)
            if th1 == x and V != x:
                th1 = asin(float(v)/float(V))
        if U == x or th == x:
            if U == x and th == x:
                U = sqrt(ux**2+u**2)
                th = asin(float(u)/float(U))
            if U == x and th != x:
                U = sqrt(ux**2+u**2)
            if th == x and U != x:
                th = asin(float(u)/float(U))
        if hm == x:
            hm = float(-u**2)/float(2*a)
        prt1()
        break
    # 5 values given
    if a != x and s != x and u != x and t != x and v != x:
        if sx == x:
            sx = ux * t
        if V == x or th1 == x:
            if V == x and th1 == x:
                V = -sqrt(ux**2+v**2)
                th1 = asin(float(v)/float(V))
            if V == x and th1 != x:
                V = -sqrt(ux**2+v**2)
            if th1 == x and V != x:
                th1 = asin(float(v)/float(V))
        if U == x or th == x:
            if U == x and th == x:
                U = sqrt(ux**2+u**2)
                th = asin(float(u)/float(U))
            if U == x and th != x:
                U = sqrt(ux**2+u**2)
            if th == x and U != x:
                th = asin(float(u)/float(U))
        if hm == x:
            hm = float(-u**2)/float(2*a)
        prt1()
        break
    break
# horizantal projectile
if p == 2:
    # 1 given horizantal
    if ux != x and sx == x and V == x or sx != x and ux == x and V == x or V != x and ux == x and sx == x:
        # only 0 vert uknown
        if s != x and t != x and v != x:
            if ux != x and sx == x and V == x:
                sx = h.projt_ux_t(ux, sx, t)
                V = h.projt_a_s(a, V, s, ux)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if sx != x and ux == x and V == x:
                ux = h.projt_sx_t(ux, sx, t)
                V = h.projt_a_s(a, V, s, ux)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if V != x and ux == x and sx == x:
                ux = h.projt_V_v(V, v, ux)
                sx = h.projt_ux_t(ux, sx, t)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
        # only 1 vert unkown
        if s == x and t != x and v != x or t == x and v != x and s != x or v == x and t != x and s != x:
            if s == x and t != x and v != x:
                s = float(v**2)/float(2*a)
                if ux != x and sx == x and V == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    V = h.projt_a_s(a, V, s, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and ux == x and V == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    V = h.projt_a_s(a, V, s, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux == x and sx == x:
                    ux = h.projt_V_v(V, v, ux)
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
            if t == x and s != x and v != x:
                t = float(v)/float(a)
                if ux != x and sx == x and V == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    V = h.projt_a_s(a, V, s, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and ux == x and V == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    V = h.projt_a_s(a, V, s, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux == x and sx == x:
                    ux = h.projt_V_v(V, v, ux)
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
            if v == x and s != x and t != x:
                v = a*t
                if ux != x and sx == x and V == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and ux == x and V == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux == x and sx == x:
                    ux = h.projt_V_v(V, v, ux)
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
        # only 2 vertical unknown
        if s == x and t == x and v != x or v == x and s == x and t != x or v == x and t == x and s != x:
            if s == x and t == x:
                s, t = k.kin_u_v_a(u, v, s, a, t)
                if ux != x and sx == x and V == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and ux == x and V == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux == x and sx == x:
                    ux = h.projt_V_v(V, v, ux)
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
            if v == x and s == x:
                v, s = k.kin_u_a_t(u, v, s, a, t)
                if ux != x and sx == x and V == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and ux == x and V == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux == x and sx == x:
                    ux = h.projt_V_v(V, v, ux)
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
            if v == x and t == x:
                v, t = k.kin_u_s_a(u, v, s, a, t)
                if ux != x and sx == x and V == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and ux == x and V == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux == x and sx == x:
                    ux = h.projt_V_v(V, v, ux)
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
        # 3 vertical are unkown
        if s == x and t == x and v == x:
            if V != x and th != x and sx == x and ux == x:
                v = sin(th)*V
                ux = -cos(th)*V
                s, t = k.kin_u_v_a(u, v, s, a, t)
                sx = h.projt_ux_t(ux, sx, t)
                prt()
            if ux != x and th != x and V == x and sx == x:
                V = float(ux)/float(cos(th))
                v = sqrt(V**2-ux**2)
                s, t = k.kin_u_v_a(u, v, s, a, t)
                sx = h.projt_ux_t(ux, sx, t)
                prt()
            if sx != x and th != x and V == x and ux == x:
                V = -sqrt(float(a*sx)/float(-cos(th)*sin(th)))
                ux = cos(th)*-V
                v = sqrt(V**2-ux**2)
                s, t = k.kin_u_v_a(u, v, s, a, t)
                prt()

    # 2 horizantal given are given
    if ux != x and sx != x and V == x or ux != x and V != x and sx == x or V != x and sx != x and ux or ux != x and V != x and sx != x:
        # 0 vertical are unkown
        if s != x and t != x and v != x:
            if ux != x and sx != x and V == x:
                V = h.projt_a_s(V, v, ux)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if sx != x and V != x and ux == x:
                ux = h.projt_sx_t(ux, sx, t)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if V != x and ux != x and sx == x:
                sx = h.projt_ux_t(ux, sx, t)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
        # 1 vertical is unknown
        if s == x and t != x and v != x or t == x and v != x and s != x or v == x and t != x and s != x:
            if s == x and t != x and v != x:
                s = float(v**2)/float(2*a)
                if ux != x and sx != x and V == x:
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and V != x and ux == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux != x and sx == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
            if t == x and s != x and v != x:
                t = float(v)/float(a)
                if ux != x and sx != x and V == x:
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and V != x and ux == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux != x and sx == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
            if v == x and s != x and t != x:
                v = a*t
                if ux != x and sx != x and V == x:
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and V != x and ux == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux != x and sx == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
        # 2 vertical are unknown
        if s == x and t == x or v == x and s == x or v == x and t == x:
            if s == x and t == x and v != x:
                s, t = k.kin_u_v_a(u, v, s, a, t)
                if ux != x and sx != x and V == x:
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and V != x and ux == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux != x and sx == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
            if v == x and s == x and t != x:
                v, s = k.kin_u_a_t(u, v, s, a, t)
                if ux != x and sx != x and V == x:
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and V != x and ux == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux != x and sx == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
            if v == x and t == x and s != x:
                v, t = k.kin_u_s_a(u, v, s, a, t)
                if ux != x and sx != x and V == x:
                    V = h.projt_a_s(V, v, ux)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if sx != x and V != x and ux == x:
                    ux = h.projt_sx_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()
                if V != x and ux != x and sx == x:
                    sx = h.projt_ux_t(ux, sx, t)
                    if th == x:
                        th = h.protj_th(ux, V)
                    prt()

        # 3 vertical unknown
        if s == x and t == x and v == x:
            if ux != x and sx != x and V == x:
                t = h.projt_sx_ux(ux, sx, t)
                v, s = k.kin_u_a_t(u, v, s, a, t)
                V = h.projt_a_s(V, v, ux)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if ux != x and V != x and sx == x:
                v = h.projt_V_ux(V, v, ux)
                s, t = k.kin_u_v_a(u, v, s, a, t)
                V = h.projt_a_s(V, v, ux)
                sx = h.projt_ux_t(ux, sx, t)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if V != x and sx != x and ux == x and th != x:
                ux = cos(th)*V
                t = h.projt_sx_ux(ux, sx, t)
                v, s = k.kin_u_a_t(u, v, s, a, t)
                prt()

    # 3 given horizantal
    if ux != x and sx != x and V != x:
        # 1 vertical unkown
        if s == x and t != x and v != x or t == x and v != x and s != x or v == x and t != x and s != x:
            if s == x and t != x and v != x:
                s = float(v**2)/float(2*a)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if t == x and s != x and v != x:
                t = float(v)/float(a)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if v == x and s != x and t != x:
                v = a*t
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
        # 2 vertical are unknown
        if s == x and t == x or v == x and s == x or v == x and t == x:
            if s == x and t == x and v != x:
                s, t = k.kin_u_v_a(u, v, s, a, t)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if v == x and s == x and t != x:
                v, s = k.kin_u_a_t(u, v, s, a, t)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()
            if v == x and t == x and s != x:
                v, t = k.kin_u_s_a(u, v, s, a, t)
                if th == x:
                    th = h.protj_th(ux, V)
                prt()

        # 3 vertical unknown
        if s == x and t == x and v == x:
            t = h.projt_sx_ux(ux, sx, t)
            v, s = k.kin_u_a_t(u, v, s, a, t)
            if th == x:
                th = h.protj_th(ux, V)
            prt()
