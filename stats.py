# python 3.6
Author = 'Harith Al-safi'
Date = "5/5/2020"

# modules
from numpy import percentile, std, var, argmax, bincount, unique, array, mean, linspace
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression

# 1 var vs 2 vars
print('Hello everyone, this is a statistical calculation tool by Harith Al-Safi')
print('If you chose two variables, then the x and the y must have same number of elements ')
u = int(input('Enter 1 for one variable, 2 for two variables: '))
print('--------')

# 1 var inputs
if u == 1:
    z = list(eval(input('Enter your values: ')))
    print('----------')

# 2 var inputs
elif u == 2:
    x = list(eval(input('Enter your x values: ')))
    y = list(eval(input('Enter you y values: ')))
    print('---------')

# mode
def modez(readList):
    numCount={}
    highestNum=0
    for i in readList:
        if i in numCount.keys(): numCount[i] += 1
        else: numCount[i] = 1
    for i in numCount.keys():
        if numCount[i] > highestNum:
            highestNum=numCount[i]
            mode=i
    if highestNum != 1:
        return mode
    elif highestNum == 1:
        return False

def uncert(h):
    h = sorted(h)
    n = len(h)
    q = (h[n-1]-h[0])/2
    return q


# calculations
def calc(p):
    p_unc = uncert(p)
    p_mode = modez(p)
    p_mean = mean(p)
    p_std = std(p)
    p_var = var(p)
    p_min, p_q1, p_q2, p_q3, p_max = percentile(p, [0, 25, 50, 75, 100], interpolation='midpoint')
    p_IQR = p_q3 - p_q1
    print('Mean is: ' + str(p_mean))
    if p_mode == False:
        print('No mode is found')
    else:
        print('Mode is: ' + str(p_mode))
    print('Uncertainity is: ' + str(p_unc))
    print('Minimum is: ' + str(p_min))
    print('Q1 is: ' + str(p_q1))
    print('Median is: ' + str(p_q2))
    print('Q3 is: ' + str(p_q3))
    print('Maximum is: ' + str(p_max))
    print('Standard diviation is: ' + str(p_std))
    print('Variance is: ' + str(p_var))

# 1 var calculations
if u == 1:
    calc(z)
    uncert(z)
    plot.boxplot(z, vert=False)
    plot.show()

# linear regression
def liner(x1, c1, a):
    y1 = a*x1 + c1
    return y1

# 2 var calculations
if u == 2:
    print('X calculations:')
    print('-------------')
    calc(x)
    print('-------------')
    print('Y calculations:')
    print('-------------')
    calc(y)
    x = array([x]).reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    a = model.coef_
    c1 = model.intercept_
    x1 = linspace(-4, 4, 211)
    y1 = liner(x1, c1, a)
    print('-------------')
    print('Value of r squared: ' + str(r_sq))
    plot.figure()
    plot.subplot(311)
    plot.boxplot(x, vert=False)
    plot.subplot(312)
    plot.boxplot(y, vert=False)
    ax = plot.subplot(313)
    ax.plot(x1, y1, label='y = %s*x +' %a + str(c1))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=3)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    plot.show()



