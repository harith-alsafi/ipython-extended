# Python 2.7
Author = "Harith Al-Safi"
Date = "15/4/2020"

# Modules:
import math
from turtle import *


# Intro
print('Helo, this is a triangle calculater by Harith Al-Safi')
print('Input 0 for the variable you want to calculate')
print('-------------------')
# Inputs
a = input('[a] Side a : ')
b = input('[b] Side b : ')
c = input('[c] Side c : ')
A = input('[A] Angle A: ')
B = input('[B] Angle B: ')
C = input('[C] Angle C: ')
Ar = input('[Ar] Area Ar: ')
print('-------------------')


# Angles additions
if A != 0 and B != 0 and C != 0:
    if A + B + C == 180:
        pass
    else:
        print('Angles dont add to 180')
        exit()

# functions and classes
# two sides and opposite angle of unknown side
class two_side_O:
    def side_a(self, a, b, c, A, B, C):
        del a
        del B
        del C
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        B = math.degrees(math.asin((math.sin(math.radians(A))*b)/a))
        C = 180 - (B + A)
        return a, b, c, A, B, C

    @classmethod
    def side_b(cls, a, b, c, A, B, C):
        del b
        del A
        del C
        b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
        A = math.degrees(math.asin((math.sin(math.radians(B))*a)/b))
        C = 180 - (B + A)
        return a, b, c, A, B, C

    @classmethod
    def side_c(cls, a, b, c, A, B, C):
        del c
        del A
        del B
        c = math.sqrt(a**2 + b**2 - 2 * b * a * math.cos(float(math.radians(C))))
        B = math.degrees(math.asin((math.sin(math.radians(C))*b)/c))
        A = 180 - (B + C)
        return a, b, c, A, B, C

# two sides without the opposite angle of unknown side:
class two_side:
    # Side a:
    def side_a_B(self, a, b, c, A, B, C): # side a is missing while angle B is given
        del a
        del C
        del A
        C = math.degrees(math.asin((math.sin(math.radians(B))*c)/b))
        A = 180 - (C + B)
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        return a, b, c, A, B, C

    @classmethod
    def side_a_C(cls, a, b, c, A, B, C): # side a is missing while angle C is given
        del a
        del B
        del A
        B = math.degrees(math.asin((math.sin(math.radians(C))*b)/c))
        A = 180 - (C + B)
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        return a, b, c, A, B, C

    # side b:
    @classmethod
    def side_b_A(cls, a, b, c, A, B, C): # side b is missing and angle A is given
        del b
        del B
        del C
        C = math.degrees(math.asin((math.sin(math.radians(A))*c)/a))
        B = 180 - (C + A)
        b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
        return a, b, c, A, B, C

    @classmethod
    def side_b_C(cls, a, b, c, A, B, C): # side b is missing and angle C is given
        del b
        del B
        del A
        A = math.degrees(math.asin((math.sin(math.radians(C))*a)/c))
        B = 180 - (C + A)
        b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
        return a, b, c, A, B, C

    # side c
    @classmethod
    def side_c_A(cls, a, b, c, A, B, C): # side c is missing and angle A is given
        del c
        del C
        del B
        B = math.degrees(math.asin((math.sin(math.radians(A))*b)/a))
        C = 180 - (B + A)
        c = math.sqrt(a**2 + b**2 - 2 * b * a * math.cos(float(math.radians(C))))
        return a, b, c, A, B, C

    @classmethod
    def side_c_B(cls, a, b, c, A, B, C):
        del c
        del C
        del A
        A = math.degrees(math.asin((math.sin(math.radians(B))*a)/b))
        C = 180 - (B + A)
        c = math.sqrt(a**2 + b**2 - 2 * b * a * math.cos(float(math.radians(C))))
        return a, b, c, A, B, C

# two angles with opposite side is known
class allangle:
    # angle C
    def angle_C_b(self, a, b, c, A, B, C): # angle C is missing and side b is given
        del C
        del a
        del c
        C = 180 - (A + B)
        a = b*(math.sin(math.radians(A)))/(math.sin(math.radians(B)))
        c = math.sqrt(a**2 + b**2 - 2 * b * a * math.cos(float(math.radians(C))))
        return a, b, c, A, B, C

    @classmethod
    def angle_C_a(cls, a, b, c, A, B, C): # angle C is missing and sice a is given
        del C
        del b
        del c
        C = 180 - (A + B)
        b = a*(math.sin(math.radians(B)))/(math.sin(math.radians(A)))
        c = math.sqrt(a**2 + b**2 - 2 * b * a * math.cos(float(math.radians(C))))
        return a, b, c, A, B, C

    @classmethod
    def angle_C_c(cls, a, b, c, A, B, C):
        del C
        del b
        del a
        C = 180 - (A + B)
        b = c*(math.sin(math.radians(B)))/(math.sin(math.radians(C)))
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        return a, b, c, A, B, C

    # angle B
    @classmethod
    def angle_B_a(cls, a, b, c, A, B, C): # angle B is missing and side a is given
        del B
        del b
        del c
        B = 180 - (A + C)
        c = a*(math.sin(math.radians(C)))/(math.sin(math.radians(A)))
        b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
        return a, b, c, A, B, C

    @classmethod
    def angle_B_c(cls, a, b, c, A, B, C): # angle B is missing and side c is given
        del B
        del b
        del a
        B = 180 - (A + C)
        b = c*(math.sin(math.radians(B)))/(math.sin(math.radians(C)))
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        return a, b, c, A, B, C

    @classmethod
    def angle_B_b(cls, a, b, c, A, B, C): # angle B is missing and side b is given
        del B
        del a
        del c
        B = 180 - (A + C)
        c = b*(math.sin(math.radians(C)))/(math.sin(math.radians(B)))
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        return a, b, c, A, B, C

    # angle A
    @classmethod
    def angle_A_a(cls, a, b, c, A, B, C): # angle C is missing and side a is given
        del A
        del b
        del c
        A = 180 - (C + B)
        b = a*(math.sin(math.radians(B)))/(math.sin(math.radians(A)))
        c = math.sqrt(b**2 + a**2 - 2 * b * a * math.cos(float(math.radians(C))))
        return a, b, c, A, B, C

    @classmethod
    def angle_A_b(cls, a, b, c, A, B, C): # angle C is missing and side b is given
        del A
        del a
        del c
        A = 180 - (C + B)
        c = b*(math.sin(math.radians(C)))/(math.sin(math.radians(B)))
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        return a, b, c, A, B, C

    @classmethod
    def angle_A_c(cls, a, b, c, A, B, C): # angle C is missing and side c is given
        del A
        del a
        del b
        A = 180 - (C + B)
        a = c*(math.sin(math.radians(A)))/(math.sin(math.radians(C)))
        b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
        return a, b, c, A, B, C

    # all angles
    @classmethod
    def angle_c(cls, a, b, c, A, B, C):
        del a
        del b
        a = c*(math.sin(math.radians(A)))/(math.sin(math.radians(C)))
        b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
        return a, b, c, A, B, C

    @classmethod
    def angle_b(cls, a, b, c, A, B, C):
        del a
        del c
        c = b*(math.sin(math.radians(C)))/(math.sin(math.radians(B)))
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        return a, b, c, A, B, C

    @classmethod
    def angle_a(cls, a, b, c, A, B, C):
        del b
        del c
        b = a*(math.sin(math.radians(B)))/(math.sin(math.radians(A)))
        c = math.sqrt(b**2 + a**2 - 2 * b * a * math.cos(float(math.radians(C))))
        return a, b, c, A, B, C

# All sides given
class allside:
    def allsides_a(self, a, b, c, A, B, C):
        del B
        del A
        del C
        A = math.degrees(math.acos((a**2-c**2-b**2)/(-2*c*b)))
        B = math.degrees(math.asin((math.sin(math.radians(A))*b)/a))
        C = 180 - (A + B)
        return a, b, c, A, B, C

    @classmethod
    def allside_b(cls, a, b, c, A, B, C):
        del A
        del B
        del C
        B = math.degrees(math.acos((b**2-c**2-a**2)/(-2*c*a)))
        A = math.degrees(math.asin((math.sin(math.radians(B))*a)/b))
        C = 180 - (A + B)
        return a, b, c, A, B, C

    @classmethod
    def allsdie_c(cls, a, b, c, A, B, C):
        del A
        del B
        del C
        C = math.degrees(math.acos((c**2-b**2-a**2)/(-2*b*a)))
        B = math.degrees(math.asin((math.sin(math.radians(C))*b)/c))
        A = 180 - (B + C)
        return a, b, c, A, B, C

    # one given angle
    @classmethod
    def allside_C(cls, a, b, c, A, B, C):
        del A
        del B
        B = math.degrees(math.asin((math.sin(math.radians(C))*b)/c))
        A = 180 - (B + C)
        return a, b, c, A, B, C

    @classmethod
    def allside_B(cls, a, b, c, A, B, C):
        del A
        del C
        C = math.degrees(math.asin((math.sin(math.radians(B))*c)/b))
        A = 180 - (B + C)
        return a, b, c, A, B, C

    @classmethod
    def allside_A(cls, a, b, c, A, B, C):
        del B
        del C
        B = math.degrees(math.asin((math.sin(math.radians(A))*b)/a))
        C = 180 - (A + B)
        return a, b, c, A, B, C

# Area
class area:
    def area_all(self, a, B, c, Ar):
        del Ar
        Ar = float(0.5) * a * c * math.sin(math.radians(B))
        return Ar

    # side b
    @classmethod
    def area_A_b(cls, a, b, c, A, B, C, Ar): # angle A
        del c
        del a
        del B
        del C
        c = float(Ar * 2)/float(b * math.sin(math.radians(A)))
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        B = math.degrees(math.asin((math.sin(math.radians(A))*b)/a))
        C = 180 - (A + B)
        return a, b, c, A, B, C, Ar

    @classmethod
    def area_C_b(cls, a, b, c, A, B, C, Ar):
        del a
        del c
        del A
        del B
        a = float(Ar * 2)/float(b * math.sin(math.radians(C)))
        c = math.sqrt(b**2 + a**2 - 2 * b * a * math.cos(float(math.radians(C))))
        B = math.degrees(math.asin((math.sin(math.radians(C))*b)/c))
        A = 180 - (C + B)
        return a, b, c, A, B, C, Ar

    # side c
    @classmethod
    def area_A_c(cls, a, b, c, A, B, C, Ar):
        del b
        del a
        del C
        del B
        b = float(Ar * 2)/float(c * math.sin(math.radians(A)))
        a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
        B = math.degrees(math.asin((math.sin(math.radians(A))*b)/a))
        C = 180 - (A + B)
        return a, b, c, A, B, C, Ar

    @classmethod
    def area_B_c(cls, a, b, c, A, B, C, Ar):
        del a
        del b
        del C
        del A
        a = float(Ar * 2)/float(c * math.sin(math.radians(B)))
        b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
        A = math.degrees(math.asin((math.sin(math.radians(B))*a)/b))
        C = 180 - (A + B)
        return a, b, c, A, B, C, Ar

    # side A
    @classmethod
    def area_B_a(cls, a, b, c, A, B, C, Ar):
        del c
        del b
        del A
        del C
        c = float(Ar * 2)/float(a * math.sin(math.radians(B)))
        b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
        A = math.degrees(math.asin((math.sin(math.radians(B))*a)/b))
        C = 180 - (A + B)
        return a, b, c, A, B, C, Ar

    @classmethod
    def area_C_a(cls, a, b, c, A, B, C, Ar):
        del b
        del c
        del A
        del B
        b = float(Ar * 2)/float(a * math.sin(math.radians(C)))
        c = math.sqrt(a**2 + b**2 - 2 * a * b * math.cos(float(math.radians(C))))
        A = math.degrees(math.asin((math.sin(math.radians(C))*a)/c))
        B = 180 - (C + A)
        return a, b, c, A, B, C, Ar

# class instances
y = two_side()
x = two_side_O()
z = allangle()
t = allside()
j = area()

#  print function
def prt():
    if A == 180 or B == 180 or C == 180:
        print('------------')
        print('Error with inputs')
        exit()
    else:
        print('Side a is: ' + str(a))
        print('Side b is: ' + str(b))
        print('Side c is: ' + str(c))
        print('Angle A is: ' +  str(A))
        print('Angle B is: ' + str(B))
        print('Angle C is: ' + str(C))
        print('Area Ar is: ' + str(Ar))

# Conditions
# 2 sides only
if b != 0 and c != 0 and a == 0 or c != 0 and a != 0 and b == 0 or b != 0 and a != 0 and c == 0:
    # only one angle is given
    if A == 0 and B ==0 or C == 0 and A == 0 or C ==0 and B == 0:
        if A != 0:
            if a == 0: # this concludes occurance of the angle of the unknown side
                a, b, c, A, B, C = x.side_a(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if b == 0:
                a, b, c, A, B, C = y.side_b_A(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if c == 0:
                a, b, c, A, B, C = y.side_c_A(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()

        if B != 0:
            if b == 0:
                a, b, c, A, B, C = x.side_b(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if a == 0:
                a, b, c, A, B, C = y.side_a_B(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if c == 0:
                a, b, c, A, B, C = y.side_c_B(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()

        if C != 0:
            if c == 0:
                a, b, c, A, B, C = x.side_c(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if a == 0:
                a, b, c, A, B, C = y.side_a_C(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if b == 0:
                a, b, c, A, B, C = y.side_b_C(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
    # 2 angles given
    if A != 0 and B != 0 or C != 0 and A != 0 or C != 0 and B != 0:
        if A == 0:
            del C
            A = 180 - (B + C)
            if a == 0:
                del a
                a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if b == 0:
                del b
                b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if c == 0:
                del c
                c = math.sqrt(a**2 + b**2 - 2 * b * a * math.cos(float(math.radians(C))))
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()

        if C == 0:
            del C
            C = 180 - (A + B)
            if a == 0:
                del a
                a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if b == 0:
                del b
                b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if c == 0:
                del c
                c = math.sqrt(a**2 + b**2 - 2 * b * a * math.cos(float(math.radians(C))))
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()

        if B == 0:
            del B
            B = 180 - (A + C)
            if a == 0:
                del a
                a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if b == 0:
                del b
                b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if c == 0:
                del c
                c = math.sqrt(a**2 + b**2 - 2 * b * a * math.cos(float(math.radians(C))))
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
    # all 3 angles
    if A != 0 and B != 0 and C != 0:
        if a == 0:
            del a
            a = math.sqrt(b**2 + c**2 - 2 * b * c * math.cos(float(math.radians(A))))
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()
        if b == 0:
            del b
            b = math.sqrt(a**2 + c**2 - 2 * a * c * math.cos(float(math.radians(B))))
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()
        if c == 0:
            del c
            c = math.sqrt(a**2 + b**2 - 2 * b * a * math.cos(float(math.radians(C))))
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()
    # no angle
    if A == 0 and B == 0 and C == 0:
        # area is given
        if Ar != 0:
            # a, b are given
            if a != 0 and b != 0 and c == 0:
                C = math.degrees(math.asin(float(Ar*2)/float(a*b)))
                a, b, c, A, B, C = x.side_c(a, b, c, A, B, C)
                prt()
            # if b, c are given
            if c != 0 and b != 0 and a == 0:
                A = math.degrees(math.asin(float(Ar*2)/float(c*b)))
                a, b, c, A, B, C = x.side_a(a, b, c, A, B, C)
                prt()
            # if c and a are given
            if c != 0 and a != 0 and b == 0:
                B = math.degrees(math.asin(float(Ar*2)/float(c*a)))
                a, b, c, A, B, C = x.side_b(a, b, c, A, B, C)
                prt()
# 1 side only
if a == 0 and b == 0 or c == 0 and b == 0 or a == 0 and c == 0: # 1 side only
    if B != 0 and C != 0 or A != 0 and B != 0 or A != 0 and C != 0: # 2 angles only
        if c != 0:
            if A == 0:
                a, b, c, A, B, C = z.angle_A_c(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if B == 0:
                a, b, c, A, B, C = z.angle_B_c(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if C == 0:
                a, b, c, A, B, C = z.angle_C_c(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()

        if a != 0:
            if A == 0:
                a, b, c, A, B, C = z.angle_A_a(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if B == 0:
                a, b, c, A, B, C = z.angle_B_a(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if C == 0:
                a, b, c, A, B, C = z.angle_C_a(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()

        if b != 0:
            if A == 0:
                a, b, c, A, B, C = z.angle_A_b(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if B == 0:
                a, b, c, A, B, C = z.angle_B_b(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if C == 0:
                a, b, c, A, B, C = z.angle_C_b(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()

    if B != 0 and C != 0 and A != 0: # 3 angles
        if c != 0 and a == 0 and b == 0:
            a, b, c, A, B, C = z.angle_c(a, b, c, A, B, C)
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()
        if a != 0 and b == 0 and c == 0:
            a, b, c, A, B, C = z.angle_a(a, b, c, A, B, C)
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()
        if b != 0 and a == 0 and c == 0:
            a, b, c, A, B, C = z.angle_b(a, b, c, A, B, C)
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()

    if B != 0 and A == 0 and C == 0  or C != 0 and B == 0 and A == 0 or A != 0 and B == 0 and C == 0: # 1angle
        if Ar != 0:
            if b != 0 and b == 0 and c == 0:
                if A != 0 and B == 0 and C == 0:
                    a, b, c, A, B, C, Ar = j.area_A_b(a, b, c, A, B, C, Ar)
                    prt()
                if B != 0 and A == 0 and C == 0:
                    print('Invalid inputs')
                if C != 0 and A == 0 and B == 0:
                    a, b, c, A, B, C, Ar = j.area_C_b(a, b, c, A, B, C, Ar)
                    prt()
            if c != 0 and a == 0 and b == 0:
                if C != 0 and B == 0 and A == 0:
                    print('Invalid inputs')
                if A != 0 and C == 0 and B == 0:
                    a, b, c, A, B, C, Ar = j.area_A_c(a, b, c, A, B, C, Ar)
                    prt()
                if B != 0 and A == 0 and C == 0:
                    a, b, c, A, B, C, Ar = j.area_B_c(a, b, c, A, B, C, Ar)
                    prt()
            if a != 0 and b == 0 and c == 0:
                if A != 0 and B == 0 and C == 0:
                    print('Invalid inputs')
                if B != 0 and A == 0 and C == 0:
                    a, b, c, A, B, C, Ar = j.area_B_a(a, b, c, A, B, C, Ar)
                    prt()
                if C != 0 and A == 0 and B == 0:
                    a, b, c, A, B, C, Ar = j.area_C_a(a, b, c, A, B, C, Ar)
                    prt()
        else:
            print('Invalid inputs')

# 3 sides
elif c != 0 and b !=0 and a != 0:
    if A == 0 and B == 0 and C == 0: # no angles
        if a > b > c or a > c > b:
            a, b, c, A, B, C = t.allsides_a(a, b, c, A, B, C)
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()
        if b > a > c or b > c > a:
            a, b, c, A, B, C = t.allside_b(a, b, c, A, B, C)
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()
        if c > a > b or c > b > a:
            a, b, c, A, B, C = t.allsdie_c(a, b, c, A, B, C)
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()

    if A != 0 or B != 0 or C != 0: # no angles
        if c**2 == b**2 + a**2 or a**2 == b**2 + c**2 or b**2 == c**2 + a**2:
            if a > b > c or a > c > b:
                del A
                A = 90
                if B == 0:
                    B = math.degrees(math.asin(float(b)/float(a)))
                    C = 90 - B
                    if Ar == 0:
                        Ar = j.area_all(a, B, c, Ar)
                        prt()
                    else:
                        prt()

                if C == 0:
                    C = math.degrees(math.asin(float(c)/float(a)))
                    B = 90 - C
                    if Ar == 0:
                        Ar = j.area_all(a, B, c, Ar)
                        prt()
                    else:
                        prt()

            if b > a > c or b > c > a:
                del B
                B = 90
                if A == 0:
                    A = math.degrees(math.asin(float(a)/float(b)))
                    C = 90 - A
                    if Ar == 0:
                        Ar = j.area_all(a, B, c, Ar)
                        prt()
                    else:
                        prt()
                if C == 0:
                    C = math.degrees(math.asin(float(c)/float(b)))
                    A = 90 - C
                    if Ar == 0:
                        Ar = j.area_all(a, B, c, Ar)
                        prt()
                    else:
                        prt()

            if c > a > b or c > b > a:
                del C
                C = 90
                if A == 0:
                    A = math.degrees(math.asin(float(a)/float(c)))
                    B = 90 - A
                    if Ar == 0:
                        Ar = j.area_all(a, B, c, Ar)
                        prt()
                    else:
                        prt()

                if B == 0:
                    B = math.degrees(math.asin(float(b)/float(c)))
                    A = 90 - B
                    if Ar == 0:
                        Ar = j.area_all(a, B, c, Ar)
                        prt()
                    else:
                        prt()

        else: # 2, 1, 3 angles
            if A == 0 and B == 0: # 1 angles only
                a, b, c, A, B, C = t.allside_C(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if A == 0 and C == 0:
                a, b, c, A, B, C = t.allside_B(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()
            if B == 0 and C == 0:
                a, b, c, A, B, C = t.allside_A(a, b, c, A, B, C)
                if Ar == 0:
                    Ar = j.area_all(a, B, c, Ar)
                    prt()
                else:
                    prt()

        if A == 0 and B != 0 and C != 0: # 2 angle
            A = 180 - (B + C)
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()
        if B == 0 and A != 0 and C != 0:
            B = 180 - (A + C)
            if Ar == 0:
                Ar = j.area_all(a, B, c, Ar)
                prt()
            else:
                prt()
    if C == 0 and B != 0 and A != 0: # 3 angles
        C = 180 - (A + B)
        if Ar == 0:
            Ar = j.area_all(a, B, c, Ar)
            prt()
        else:
            prt()
    # all sides equal
    if b == c == a:
        A = 60
        B = 60
        C = 60
        if Ar == 0:
            Ar = j.area_all(a, B, c, Ar)
            prt()
        else:
            prt()

# drawing
setworldcoordinates(-5, 100, 300, 200)
color("#FFFFFF" , '#1d1c24')
bgcolor("#1d1c24")
begin_fill()
screen = Screen()
screen.setup(600, 500)
pensize(5)
speed(3)

if c > b >= a or c > a >= b:
    setworldcoordinates(-20, -150, c*55, c*55)
    y = position()
    forward(c*50)
    left(180-B)
    forward(a*50)
    left(180-C)
    goto(y)
if a > b >= c or a > c >= b:
    setworldcoordinates(-20, -150, a*55, a*55)
    y = position()
    forward(a*50)
    left(180-B)
    forward(c*50)
    left(180-A)
    goto(y)
if b > a >= c or b >= c >= a:
    setworldcoordinates(-20, -150, b*55, b*55)
    y = position()
    forward(b*50)
    left(180-C)
    forward(a*50)
    left(180-B)
    goto(y)

fill()
done()