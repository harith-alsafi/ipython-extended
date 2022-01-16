# python3.6.9
# import matplotlib
# matplotlib.use("GTK3Agg")
from sympy import sqrt, solve, simplify, symbols, integrate, diff, solve_linear, Matrix, init_printing, I, logcombine, arg, linsolve, dsolve, pdsolve, fourier_series, Piecewise, limit, fps, series, summation, SeqFormula, nonlinsolve, expand_trig, trigsimp, expand_log, powsimp, powdenest, expand_power_base, factorial, binomial, factor, apart, cancel, collect, expand, lambdify, Eq, re, im, latex, pi, sin, cos, tan, cot, csc, sec, asin, acos, acot, atan, acsc, asec, sinh, cosh, tanh, coth, csch, sech, asinh, acosh, atanh, acoth, acsch, asech, exp, conjugate, oo, log
from numpy import eye, zeros, ones, linspace, array, arange, rad2deg, e, deg2rad, mgrid, meshgrid
from numpy.linalg import norm
from matplotlib.pyplot import hlines, vlines, xscale, yscale, plot, scatter, legend, gca, title, figure, xlim, ylim, quiver, xlabel, ylabel, grid, pause, draw, subplot, polar, stem
from sympy.plotting import plot_implicit
from sympy.physics import vector as vt # vectors
import sympy.physics.units as ut  # unit conversion
from scipy import signal
import numpy as np  # numeric
import scipy as sp  # other toolbox
import sympy as sm
from cmath import rect  # complex numbers
from IPython.terminal.embed import InteractiveShellEmbed
from IPython import get_ipython
import os as os
import mayavi.mlab as mb
import matplotlib.pyplot as plt  # plots
from mpl_toolkits.mplot3d import Axes3D
# from jupyterthemes import jtplot
# ipython
ipshell = InteractiveShellEmbed()
ipshell.dummy_mode = True
init_printing()
ipython = get_ipython()
# light = lambda: plt.style.use('default')
# dark = lambda: jtplot.style(theme='monokai')
# light()
plt.ion() # plots
# custom scripts
cwd = os.path.dirname(os.path.realpath(__file__))
cwd2 = '/media/harithalsafi/DATA/Media/Documents/Coding/Python/computational-science/ipython'
kinem = lambda: ipshell.run_cell('!python ' + cwd2 + '/kinematics.py')
trian = lambda: ipshell.run_cell('!python ' + cwd2 + '/triangle-calculater.py')
stats = lambda: ipshell.run_cell('!python3 ' + cwd2 + '/stats.py')
rr = lambda: ipython.magic('%run ' + cwd + '/pyth.py')
view = lambda: ipython.magic("%config InlineBackend.figure_format = 'retina'")
# complex
i = I
pie = np.pi
# ipython
ipshell = InteractiveShellEmbed()
ipshell.dummy_mode = True
init_printing()
ipython = get_ipython()
class cf:
	# delete maptplotlib plot
	@staticmethod
	def delete(a):
		a1 = a.pop(0)
		a1.remove()

	# log with any base log_n (a)
	@staticmethod
	def log(n, a):
	    return float(log(a, n))
	# complex numbers
	@staticmethod  #pol(2+3i)
	def pol(x):
		z = float(arg(x))
		l = float(abs(x))
		return l, z
	# log simplification
	@staticmethod
	def logsimp(a):
		return logcombine(a, force=True)
	# 2d 1st parametric
	@staticmethod
	def dif1(x, y):
		x1 = diff(x, t)
		y1 = diff(y, t)
		F = y1/x1
		return F
	# 2d 2nd parametric
	@staticmethod
	def dif2(x, y):
		x1 = diff(x, t)
		y1 = diff(y, t)
		F1 = diff(y1/x1, t)
		F = F1/x1
		return F
	# 2d implicit
	@staticmethod
	def impl(o):
		f = symbols('f')
		f = o
		F = -diff(f, x)/diff(f, y)
		return F
	# quartiles (quartiles of array z)
	@staticmethod
	def quart(z):
		MIN, Q1, Q2, Q3, MAX = np.percentile(z, [0, 25, 50, 75, 100], interpolation='midpoint')
		IQR = Q3 - Q1
		return MIN, Q1, Q2, Q3, IQR, MAX
	# sequences
	@staticmethod
	def seq(y, n1):
		return SeqFormula(y, (n, 0, oo)).coeff(n1)
	# dot product
	@staticmethod # returns the angle
	def dot(a, b):
		return np.arccos(np.dot(a, b)/(norm(a)*norm(b)))
	# cross product angle
	# finding intersection l1 = a+tb and L2 = c+sd
	@staticmethod
	def vectint(a, b, c, d):  # returns t, s, then point
		t, s = symbols('t s')
		x1 = a[0] + b[0] * t
		y1 = a[1] + b[1] * t
		x2 = c[0] + d[0] * s
		y2 = c[1] + d[1] * s
		if len(a) == 2 and len(b) == 2 and len(c) == 2 and len(d) == 2:
			t, s = linsolve([Eq(x1, x2), Eq(y1, y2)], [t, s])
			x = a[0] + b[0]*t
			y = a[1] + b[1]*t
			return t, s, x, y
		if len(a) == 3 and len(b) == 3 and len(c) == 3 and len(d) == 3:
			o = list(linsolve([Eq(x1, x2), Eq(y1, y2)], [t, s]))
			t = o[0][0]
			s = o[0][1]
			x = a[0] + b[0]*t
			y = a[1] + b[1]*t
			z = a[2] + b[2]*t
			return t, s, [x, y, z]
	# fourier
	@staticmethod # cs.fourier(2*pi, Q/pi*t, 0, pi, -Q/pi*t+2*Q, pi, 2*pi)
	def fourier(T, a, a1, a2, *args):
		if len(args) != 0:
			b = args[0]
			b1 = args[1]
			b2 = args[2]
			if len(args) > 3:
				c = args[3]
				c1 = args[4]
				c2 = args[5]
		n = symbols('n', nonzero = True, integer = True)
		w = (2*pi)/T
		if len(args) > 3:
			a_0 = 2/T*(integrate(a, (t, a1, a2))+integrate(b, (t, b1, b2))+integrate(c, (t, c1, c2)))
			a_n = simplify(2/T*(integrate(a*cos(n*t*w), (t, a1, a2))+integrate(b*cos(n*t*w), (t, b1, b2))+integrate(c*cos(n*t*w), (t, c1, c2))))
			b_n = simplify(2/T*(integrate(a*sin(n*t*w), (t, a1, a2))+integrate(b*sin(n*t*w), (t, b1, b2))+integrate(c*sin(n*t*w), (t, c1, c2))))
		if len(args) == 3:
		    a_0 = 2/T*(integrate(a, (t, a1, a2))+integrate(b, (t, b1, b2)))
		    a_n = simplify(2/T*(integrate(a*cos(n*t*w), (t, a1, a2))+integrate(b*cos(n*t*w), (t, b1, b2))))
		    b_n = simplify(2/T*(integrate(a*sin(n*t*w), (t, a1, a2))+integrate(b*sin(n*t*w), (t, b1, b2))))
		if len(args) == 0:
		    a_0 = 2/T*(integrate(a, (t, a1, a2)))
		    a_n = simplify(2/T*(integrate(a*cos(n*t*w), (t, a1, a2))))
		    b_n = simplify(2/T*(integrate(a*sin(n*t*w), (t, a1, a2))))
		return a_0, a_n, b_n
	# raduis of convergence
	@staticmethod  # j specifies which equation to use
	def limit(a, *args):
		x = symbols('x', positive = True)
		n = symbols('n', integer = True)
		a_n1 = a.subs(n, n+1)
		if len(args) == 0:
		    return limit(abs(a/a_n1), n, oo)
		else:
		    return limit(abs(a_n1/a), n, oo)
	# convert to binary with n-bits
	@staticmethod
	def de2bi(val, n):
		a = bin(val)[2:].zfill(n)
		return a
	# unit comversion
	@staticmethod
	def convert(a, b):
		return float(ut.convert_to(a, b).args[0])
	# unit base
	@staticmethod
	def convert_base(a):
		return ut.convert_to(a, [ut.second, ut.kilogram, ut.meter, ut.kelvin, ut.ampere, ut.moles, ut.candela])
	# Matrix to np.array()
	@staticmethod
	def np(a):
		return np.array(a).astype(np.float64)
	# checking matrix type
	@staticmethod
	def matcheck(a, tol=1e-8):
		if a.is_echelon == True:
			print('matrix is rref')
		if a.is_Identity == True:
			print("matrix is an identity matrix")
		if a.is_zero == True:
			return print("0 matrix")
		a = cf.np(a)
		am, an = np.shape(a)
		s = 0
		s1 = []
		if am == an:
			a_s = an+2*(an-1)  # num of non-zero elements
			for i in range(0, am):
				for j in range(0, an):
					if i == j:
						if i == am-1 and j == an-1:
							if a[i][j] != 0:
								s+=1
								s1.append(a[i][j])
								break
						if a[i][j] != 0:
							s += 1
							s1.append(a[i][j])
						if a[i][j+1] != 0:
							s += 1
							s1.append(a[i][j+1])
						if a[i+1][j] != 0:
							s += 1
							s1.append(a[i+1][j])
			if np.sum(a) == np.sum(s1) and s == a_s:
				print("matrix is tridiagonal (banded)")
		if np.all(np.abs(a-a.T) < tol) == True:
			print("matrix is symmetric")
		if (a.transpose() == -a).all() == True:
			print("matrix is skewsymmetric")
		if np.allclose(a, np.tril(a)) == True:
			print("matrix is lower triangular")
		if np.allclose(a, np.triu(a)) == True:
			print("matrix is upper triangular")
		if np.allclose(a, np.diag(np.diag(a))) == True:
			print("matrix is diagonal")
	# matrix cofactor
	@staticmethod
	def cofact(a):
		m, n = a.shape
		c = []
		for i in range(0, m):
			temp = []
			for j in range(0, n):
				temp.append(a.cofactor(i, j))
			c.append(temp)
			del(temp)
		return Matrix(c)
	#  large system of linear equations
	@staticmethod
	def solve_linear(a):
		r, c = a.shape
		x1, x2 = symbols('x_1 x_2')
		if r == 2:
			return solve_linear_system(a, x1, x2)
		if r == 3:
			x3 = symbols('x_3')
			return solve_linear_system(a, x1, x2, x3)
		if r == 4:
			x3, x4 = symbols('x_3 x_4')
			return solve_linear_system(a, x1, x2, x3, x4)
		if r == 5:
			x3, x4, x5 = symbols('x_3 x_4 x_5')
			return solve_linear_system(a, x1, x2, x3, x4, x5)
		if r == 6:
			x3, x4, x5, x6 = symbols('x_3 x_4 x_5 x_6')
			return solve_linear_system(a, x1, x2, x3, x4, x5, x6)
		if r == 7:
			x3, x4, x5, x6, x7 = symbols('x_3 x_4 x_5 x_6 x_7')
			return solve_linear_system(a, x1, x2, x3, x4, x5, x6, x7)
	# print array as table
	@staticmethod
	def prt(a):
		for line in a:
			print(*line)
	# linear dependece
	@staticmethod
	def linear_dependece(*args):
		a = np.row_stack(args)
		m, n = np.shape(a)
		b = np.column_stack([a, np.zeros(m)])
		A = cf.solve_linear(Matrix(b))
		c = 0
		for i in A:
			if A[i] != 0:
				c = 1
		if c == 0:
			print('Linear independent')
		if c == 1:
			print('linear dependent')
			return A
	# gram shmidt
	@staticmethod
	def gram_shmidt(a):
		if len(a) == 1:
			return Matrix(a)
		a = [Matrix(a[i]) for i in range(0, len(a))]
		orth = lambda u, v: ((u.T*v)[0]*u)/(u.T * u)[0]
		u = [a[0]]
		if len(a) >= 2:
			u.append(a[1]-orth(u[0], a[1]))
		if len(a) >= 3:
			u.append(a[2]-orth(u[0], a[2])-orth(u[1], a[2]))
		if len(a) >= 4:
			u.append(a[3]-orth(u[0], a[3])-orth(u[1], a[3])-orth(u[2], a[3]))
		uf = [u[i].normalized() for i in range(0, len(u))]
		return uf
	# projection
	@staticmethod
	def proj(a, b):
		b = Matrix(b)
		a = cf.gram_shmidt(a)
		v = []
		k = []
		if len(a) > 1:
			for i in a:
				v.append((b.T*i)[0]*i)
			for i in range(0, len(v)-1):
				k.append(v[i]+v[i+1])
				return k[0]
		if len(a) == 1:
			return (b.T*a[0])[0]*a[0]
	# 2d axis
	@staticmethod
	def axis2d():
		plt.axvline(x=0, color='k')
		plt.axhline(y=0, color='k')

	# hamming encode
	@staticmethod
	def encode_hamming(m):
		m = array(m)
		G = array(
			[[1, 1, 0, 1, 0, 0, 0],
			[0, 1, 1, 0, 1, 0, 0],
			[1, 1, 1, 0, 0, 1, 0],
			[1, 0, 1, 0, 0, 0, 1]])
		c = np.matmul(m, G)
		c = c%2
		return c
	# hamming decode
	@staticmethod
	def decode_hamming(r):
		r = array(r)
		s = array(
			[[0, 0, 0],
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1],
			[1, 1, 0],
			[0, 1, 1],
			[1, 1, 1],
			[1, 0, 1]])
		e = array(
			[[0, 0, 0, 0, 0, 0, 0],
			[1, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0],
			[0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 1]])
		H = array(
			[[1, 0, 0, 1, 0, 1, 1],
			[0, 1, 0, 1, 1, 1, 0],
			[0, 0, 1, 0, 1, 1, 1]])
		sr = np.matmul(r, H.T)
		sr = sr%2
		for i in range(0, len(s)):
			if np.all(sr == s[i]):
				er = e[i]
		c = np.add(r, er)
		c = c%2
		return c[3:len(c)]

	# linear regression
	@staticmethod
	def linear_regression(a, b):
		x1, x2, y1, x11 = symbols('x_1 x_2 y x')
		a0 = Matrix(a)
		b0 = Matrix(b)
		a1 = Matrix.ones(a0.shape[0], a0.shape[1])
		a2 = a1.row_join(a0)
		a3 = a2.T*a2
		b1 = a2.T*b0
		s = a3.row_join(b1)
		k = cf.solve_linear(s)
		c, m = k[x1], k[x2]
		x = np.linspace(-10, 10, 500)
		y = lambda x: m*x+c
		cf.axis2d()
		plt.scatter(a, b, color='r', label = 'Data')
		plt.plot(x, y(x), color='b', label = '$'+latex(Eq(y1, m*x11+c))+'$')
		plt.legend(loc = 'upper center', bbox_to_anchor=(0.9, 1.15), fancybox=True)
		plt.title('Linear regression')
		return y
	# legend
	@staticmethod
	def legend():
		h = plt.legend(loc = 'upper right')
		return h

	# 2d normal plots
	@staticmethod
	def plot(f, x1=[-10, 10]):
		x = symbols('x')
		f_symp = eval(f)
		f_lamb = lambdify(x, f_symp, modules='numpy')
		x2 = np.arange(x1[0], x1[1], 0.01)
		y = f_lamb(x2)
		p = plt.plot(x2, y, linewidth=2.5, label='$'+latex(f_symp)+'$')
		cf.axis2d()
		return p

	# point plots
	@staticmethod
	def point(a, b):
		if round(a, 2) == a:
			a = int(a)
		if round(b, 2) == b:
			b = int(b)
		h = plt.scatter(a, b, s=100, label='['+str(round(a, 2))+', '+str(round(b, 2))+']')
		return h

	# 2d parametric plots
	@staticmethod
	def param(f1, f2, t1=[-5, 5], l=1):
		t = symbols('t')
		f_1_symp, f_2_symp = eval(f1), eval(f2)
		f1 = lambdify(t, f_1_symp, modules='numpy')
		f2 = lambdify(t, f_2_symp, modules='numpy')
		t2 = np.arange(t1[0], t1[1], 0.01)
		x = f1(t2)
		y = f2(t2)
		p = plt.plot(x, y, linewidth=2.5, label='p-'+str(l))
		plt.title('$'+latex(f_1_symp)+'$'+', '+'$'+latex(f_2_symp)+'$', fontweight='bold')
		cf.axis2d()
		return p

	# 2d implicit
	@staticmethod
	def implicit(f, t=[-5, 5]):
		x, y = symbols('x y')
		plot_implicit(eval(f), (x, t[0], t[1]), (y, t[0], t[1]))

	# 2d vector field
	@staticmethod
	def field(u, v, x1=[-5, 5, 30], y1=[-5, 5, 30]):
		x, y = symbols('x y')
		u_symp, v_symp = eval(u), eval(v)
		u = lambdify([x, y], u_symp, modules='numpy')
		v = lambdify([x, y], v_symp, modules='numpy')
		X, Y = np.meshgrid(np.linspace(x1[0], x1[1], x1[2]), np.linspace(y1[0], y1[1], y1[2]))
		u1 = u(X, Y)
		v1 = v(X, Y)
		color_array = (u1-v1)*u1*v1
		p = plt.quiver(X, Y, u1, v1, color_array, alpha=0.9, headwidth=5, headlength=5)
		plt.title('$'+latex(u_symp)+'$'+', '+'$'+latex(v_symp)+'$', fontweight='bold')
		cf.axis2d()
		return p
	# tayor series
	@staticmethod
	def taylor(f, n, x0, x1=[-5, 5]):
		x = symbols('x')
		f_sym = eval(f)
		f_lamb = lambdify(x, f_sym, modules='numpy')
		f_taylor_sym = series(f_sym, x, x0, n=n).removeO()
		f_taylor_lamb = lambdify(x, f_taylor_sym, modules='numpy')
		x2 = np.arange(x1[0], x1[1], 0.01)
		y_original = f_lamb(x2)
		y_taylor = f_taylor_lamb(x2)
		p1 = plt.plot(x2, y_original, linewidth=2.5, label='$'+latex(f_sym)+'$')
		p2 = plt.plot(x2, y_taylor, linewidth=2.5, label='Series')
		plt.title('Taylor-series', fontweight='bold')
		cf.axis2d()
		cf.legend()
		return f_taylor_lamb

	# fourier series
	@staticmethod
	def fourier_plot(n, a, a1, a2, *args):
		args = list(args)
		x = symbols('x')
		j = a2
		a = eval(a)
		if len(args) == 0:
			f = Piecewise((a, (x >= a1) & (x <= a2)))
		if len(args) == 3:
			j = args[2]
			args[0] = eval(args[0])
			f = Piecewise((a, (x >= a1) & (x <= a2)), (args[0], (x >= args[1]) & (x <= args[2])))
		if len(args) == 6:
			j = args[5]
			args[0] = eval(args[0])
			args[3] = eval(args[3])
			f = Piecewise((a, (x >= a1) & (x <= a2)), (args[0], (x >= args[1]) & (x <= args[2])), (args[3], (x >= args[4]) & (x <= args[5])))
		x1 = np.arange(float(a1), float(j), 0.01)
		f_fourier_sym = fourier_series(f, (x, a1, j)).truncate(n)
		f_origin = lambdify(x, f, modules='numpy')
		f_fourier_lamb = lambdify(x, f_fourier_sym, modules='numpy')
		y_origin = f_origin(x1)
		y_fourier = f_fourier_lamb(x1)
		plt.plot(x1, y_origin, linewidth=2.5, label='Function')
		plt.plot(x1, y_fourier, linewidth=2.5, label='Fourier')
		plt.title('Fourier series', fontweight='bold')
		cf.legend()
		cf.axis2d()
		return f_fourier_lamb

	# contour
	@staticmethod
	def contour(f, x1=[-5, 5], y1=[-5, 5]):
		x, y = symbols('x y')
		f_sym = eval(f)
		f_lamb = lambdify([x, y], f_sym, modules='numpy')
		x2 , y2 = np.meshgrid(np.arange(x1[0], x1[1], 0.01), np.arange(y1[0], y1[1], 0.01))
		z = f_lamb(x2, y2)
		cont = plt.contour(x2, y2, z)
		plt.title('$'+latex(f_sym)+'$', fontweight='bold')
		cf.axis2d()
		return cont

	# argand diagram
	@staticmethod
	def argand(a, *args):
		z = a
		if len(args) == 1:
			z = rect(a, args[0])
		y = im(z)
		x = re(z)
		cf.point(x, y)
		cf.axis2d()
		plt.xlabel('Real-axis')
		plt.ylabel('Imaginary-axis')

	# grid
	@staticmethod
	def grid(a, b, c, n=0):
		if n == 0:
			x, y = np.mgrid[a:b:c, a:b:c]
			return x, y
		if n != 0:
			x, y, z = np.mgrid[a:b:c, a:b:c, a:b:c]
			return x, y, z

	# 3d figure
	@staticmethod
	def figure3d(n):
		fig = mb.figure(figure=n, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(550, 500))
		return fig

	# 3d axis
	@staticmethod
	def axis3d(w=0.05, r=[-100, 100]):
		t = np.arange(r[0], r[1], 0.1)
		mb.plot3d(t, 0*t, 0*t, color=(0, 0, 0), tube_radius=w)
		mb.text(np.max(t), 0, 'x', z=0, width=0.01)
		mb.plot3d(0*t, t, 0*t, color=(0, 0, 0), tube_radius=w)
		mb.text(0, np.max(t), 'y', z=0, width=0.01)
		mb.plot3d(0*t, 0*t, t, color=(0, 0, 0), tube_radius=w)
		mb.text(0, 0, 'z', z=np.max(t), width=0.01)
		mb.orientation_axes()

	# 3d plot with mayavi
	@staticmethod
	def plot3d(f, t=[-5, 5, 0.1], n=1):
		x, y = symbols('x y')
		f = lambdify([x, y], eval(f), modules='numpy')
		x, y = cf.grid(t[0], t[1], t[2])
		fig = cf.figure3d(n)
		mb.surf(x, y, f)
		mb.colorbar()
		cf.axis3d()
		mb.view(distance=np.max(f(x, y))*1.5)

	# 3d plot
	@staticmethod
	def field3d(u, v, w, t=[-1.5, 1.5, 0.1], n=1):
		x, y, z = symbols('x y z')
		us, vs, ws = u, v, w
		u = lambdify([x, y, z], eval(u), modules='numpy')
		v = lambdify([x, y, z], eval(v), modules='numpy')
		w = lambdify([x, y, z], eval(w), modules='numpy')
		x, y, z = cf.grid(t[0], t[1], t[2], n=1)
		fig = cf.figure3d(n)
		mb.quiver3d(x, y, z, u(x, y, z), v(x, y, z), w(x, y, z))
		mb.axes(color=(0, 0, 0))
		mb.vectorbar(orientation='vertical')
		cf.axis3d(w=0.0003)
		mb.view(distance=t[1]*5)
		mb.text(0, 0.965, 'u='+us, width=0.2)
		mb.text(0, 0.93, 'v='+vs, width=0.2)
		mb.text(0, 0.90, 'w='+ws, width=0.2)

	# 3d line
	@staticmethod
	def line3d(a, b, t=[-5, 5], n=1): # n for figure
		t1 = np.linspace(t[0], t[1], 200)
		x = a[0]+b[0]*t1
		y = a[1]+b[1]*t1
		z = a[2]+b[2]*t1
		s = np.random.uniform(0, 1, 3)
		j = np.random.uniform(-1, 1, 3)
		fig = cf.figure3d(n)
		mb.plot3d(x, y, z, color=tuple(s), tube_radius=0.08)
		cf.axis3d()
		mb.view(distance=t[1]*5)
		mb.text(np.median(x)+j[0], np.median(y)+j[1], str(a)+str(b), width=0.4, color=tuple(s), z=np.median(z)+j[2])

	# 3d point
	def point3d(a, b, c, n=1, w=0.2):
		fig = cf.figure3d(n)
		s = np.random.uniform(0, 1, 3)
		mb.points3d(a, b, c, scale_factor=w, color=tuple(s))
		mb.view(distance=max([a, b, c])*4)
		mb.text(a+0.1, b+0.1, str([a, b, c]), width=w, color=tuple(s), z=c+0.1)

	# 3d implicit
	@staticmethod
	def implicit3d(f, t=[-3, 3, 0.01j], n=1):
		x, y, z = symbols('x y z')
		f = lambdify([x, y, z], eval(f), modules='numpy')
		x, y, z = np.mgrid[t[0]:t[1]:(1/t[2]), t[0]:t[1]:(1/t[2]), t[0]:t[1]:(1/t[2])]
		s = np.random.uniform(0, 1, re(1/t[2]))
		fig = cf.figure3d(n)
		cf.axis3d(w=0.1)
		mb.contour3d(f(x, y, z), colormap='rainbow', contours=[0])

	# 3d parametric
	def parametric3d(x, y, z, d=[-2*pie, 2*pie, 0.01j], n=1, v=1):
		if v == 1:
			t = symbols('t')
			x = lambdify(t, eval(x), modules='numpy')
			y = lambdify(t, eval(y), modules='numpy')
			z = lambdify(t, eval(z), modules='numpy')
			t = np.linspace(d[0], d[1], int(abs(1/d[2])))
			fig = cf.figure3d(n)
			mb.plot3d(x(t), y(t), z(t), t, tube_radius=0.2)
			cf.axis3d(w=0.03)
			mb.view(distance=d[1]*2)
		if v == 2:
			t, s = symbols('t s')
			x = lambdify([t, s], eval(x), modules='numpy')
			y = lambdify([t, s], eval(y), modules='numpy')
			z = lambdify([t, s], eval(z), modules='numpy')
			t, s = np.mgrid[d[0]:d[1]:(1/d[2]), d[0]:d[1]:(1/d[2])]
			fig = cf.figure3d(n)
			mb.mesh(x(t, s), y(t, s), z(t, s), tube_radius=0.2)
			cf.axis3d(w=0.03)
			mb.view(distance=d[1]*2)

	# xplane
	@staticmethod
	def xplane(o, t=[-10, 10, 0.1], n=1):
		y, z = cf.grid(t[0], t[1], t[2])
		x = o*np.ones(np.shape(y))
		fig = cf.figure3d(n)
		mb.mesh(x, y, z, tube_radius=0.2)

	# yplane
	@staticmethod
	def yplane(o, t=[-10, 10, 0.1], n=1):
		x, z = cf.grid(t[0], t[1], t[2])
		y = o*np.ones(np.shape(x))
		fig = cf.figure3d(n)
		mb.mesh(x, y, z, tube_radius=0.2)

	# zplane
	@staticmethod
	def zplane(o, t=[-10, 10, 0.1], n=1):
		x, y = cf.grid(t[0], t[1], t[2])
		z = o*np.ones(np.shape(x))
		fig = cf.figure3d(n)
		mb.mesh(x, y, z, tube_radius=0.2)

	# vector
	@staticmethod
	def vector(a, b, c, n=1):
		fig = cf.figure3d(n)
		s = np.random.uniform(0, 1, 3)
		cf.axis3d(w=0.03)
		mb.quiver3d(a, b, c, color=tuple(s), line_width=3.0, scale_factor=5.0)
		mb.text(a*5, b*5, str([a, b, c]), width=0.2, color=tuple(s), z=c*5)
		mb.view(distance=max([a, b, c])*50)



	# 3d plots with plotly
	# @staticmethod
	# def plot_3d(f, x1=[-5, 5], y1=[-5, 5]):
	# 	x, y = symbols('x y')
	# 	f = lambdify([x, y], eval(f), modules='numpy')
	# 	x2, y2 = np.arange(x1[0], x1[1], 0.05), np.arange(y1[0], y1[1], 0.05)
	# 	X, Y = np.meshgrid(x2, y2)
	# 	z = f(X, Y)
	# 	fig = go.Figure(data=[go.Surface(z=z, x=X, y=Y)])
	# 	fig.show()
	# 	return fig;



	# 3d line
	# 3d vector field

	# # figures
	# @staticmethod
	# def figure(n):
	# 	fig.append(plt.figure(n))

	# # 3d implicit
	# @staticmethod
	# def plot_implicit3(fn, n, bbox=(-2.5,2.5)):
	# 	''' create a plot of an implicit function
	# 	fn  ...implicit function (plot where fn==0)
	# 	bbox ..the x,y,and z limits of plotted interval'''
	# 	xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
	# 	cf.figure(n)
	# 	ax = fig[n].add_subplot(111, projection='3d')
	# 	A = np.linspace(xmin, xmax, 100) # resolution of the contour
	# 	B = np.linspace(xmin, xmax, 20) # number of slices
	# 	A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

	# 	for z in B: # plot contours in the XY plane
	# 		X,Y = A1,A2
	# 		Z = fn(X,Y,z)
	# 		cset = ax.contour(X, Y, Z+z, [z], zdir='z', colors='r')
	# 		# [z] defines the only level to plot for this contour for this value of z

	# 	for y in B: # plot contours in the XZ plane
	# 		X,Z = A1,A2
	# 		Y = fn(X,y,Z)
	# 		cset = ax.contour(X, Y+y, Z, [y], zdir='y', colors='b')

	# 	for x in B: # plot contours in the YZ plane
	# 		Y,Z = A1,A2
	# 		X = fn(x,Y,Z)
	# 		cset = ax.contour(X+x, Y, Z, [x], zdir='x')

	# 	# must set plot limits because the contour will likely extend
	# 	# way beyond the displayed level.  Otherwise matplotlib extends the plot limits
	# 	# to encompass all values in the contour.
	# 	ax.set_zlim3d(zmin,zmax)
	# 	ax.set_xlim3d(xmin,xmax)
	# 	ax.set_ylim3d(ymin,ymax)

	# 	plt.show()

	# # figure title
	# @staticmethod
	# def title(s, n):
	# 	fig[n].suptitle(s)
