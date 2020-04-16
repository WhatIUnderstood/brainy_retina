import numpy as np
import matplotlib.pyplot as plt

#"CurcioConeDensities  This table contains cone densities at various
# locations in the visual field, derived from Curcio, C. A., Sloan, K.
# R., Kalina, R. E., & Hendrickson, A. E. (1990). Human photoreceptor
# topography. J Comp Neurol, 292(4), 497-523. The table is in the form
# of six lists {mm, deg, temporal, superior, nasal, inferior] and the
# last four lists are cone densities (cones/mm^2) as a function of
# eccentricity in mm along four meridians. Meridians are defined in the
# visual field.";
curcio_cone_densities = np.array([[0., 0.04975, 0.0995, 0.14925, 0.199,
                                                                      0.2985, 0.398, 0.4975, 0.597, 0.6965, 0.796,
                                                                      0.8955000000000001, 0.995, 1.99, 2.985, 3.98, 4.975, 5.97,
                                                                      6.965, 7.96, 8.955, 9.95, 10.945, 11.94, 12.935, 13.93,
                                                                      14.925, 15.92, 16.915, 17.91, 18.905, 19.9, 20.895,
                                                                      21.89],
                                                                     [0., 0.178571, 0.357143, 0.535714, 0.714286,
                                                                      1.07143, 1.42857, 1.78571, 2.14286, 2.5, 2.85714, 3.21429,
                                                                      3.57143, 7.14286, 10.7143, 14.2857, 17.8571, 21.4286, 25.,
                                                                      28.5714, 32.1429, 35.7143, 39.2813, 42.85, 46.64, 50.7889,
                                                                      54.8951, 58.7683, 62.9145, 67.6747, 72.6892, 77.9739,
                                                                      84.3846, 91.3462],
                                                                     [196890., 154441.42857142858,
                                                                      115399.42857142857, 92991.85714285714, 74232.71428571429,
                                                                      53431.57142857143, 42975.42857142857, 36493.,
                                                                      31530.428571428572, 26109.285714285714, 23872.,
                                                                      22338.85714285714, 20961.85714285714, 12521.857142857143,
                                                                      8848.685714285715, 0, 7178.65, 6650.95, 6408.516666666666,
                                                                      6232.1, 5959.316666666667, 5609.033333333333, 5347.65,
                                                                      5139.233333333334, 4989.066666666667, 4886.45,
                                                                      4806.566666666667, 4726.116666666667, 4720.883333333333,
                                                                      4716.716666666666, 4846.25, 5233.88, 4906.425,
                                                                      5190.65],
                                                                     [196890., 140595.2857142857, 105145.57142857143,
                                                                      83181.85714285714, 65106., 46900.142857142855,
                                                                      35696.42857142857, 28034.14285714286, 24346.,
                                                                      21231.85714285714, 19248.714285714286, 17452.714285714286,
                                                                      15863., 10101.785714285714, 7580.6, 5911.05,
                                                                      5293.966666666666, 4868., 4575.133333333333,
                                                                      4366.833333333333, 4232.016666666666, 4228.88, 4068.72,
                                                                      3917.36, 3871.26, 3847.34, 3777.76, 3727.66, 3767.225,
                                                                      3722.8, 3983.133333333333, 3227.2, 0, 0],
                                                                     [196890.,
                                                                      162414.2857142857, 121619.28571428571, 98423.57142857143,
                                                                      80004.42857142857, 57712.142857142855, 45420.28571428572,
                                                                      38380.42857142857, 34223.28571428572, 29027.428571428572,
                                                                      24144.285714285714, 21165.14285714286, 19698.571428571428,
                                                                      11703.014285714287, 9118.114285714286, 7030.228571428571,
                                                                      5785.533333333333, 5103.25, 4779.866666666667,
                                                                      4476.766666666666, 4208.116666666667, 3887.75, 3763.48,
                                                                      3538.8, 3435.95, 3368.3, 3335.5, 3175.525, 3069.575,
                                                                      3358.98, 0, 0, 0, 0],
                                                                     [196890., 145724.2857142857,
                                                                      107324.57142857143, 85645.28571428571, 68785.28571428571,
                                                                      50905.142857142855, 39422.28571428572, 32397.14285714286,
                                                                      28194.428571428572, 24597., 21911.428571428572,
                                                                      19204.714285714286, 16993.428571428572, 10430.,
                                                                      7649.742857142856, 6223.628571428571, 5544.733333333334,
                                                                      4990.9, 4700.22, 4426.32, 4196.24, 4029.24, 3861.24,
                                                                      3722.32, 3673.42, 3713.52, 3700.9, 3675.64, 3673.14,
                                                                      3736.075, 4539.2, 0, 0, 0]]);

# Degree of the fitting polynomial
x_log = np.log(curcio_cone_densities[1][0:-4]+1.0)
y_log = np.log(curcio_cone_densities[4][0:-4])
deg = 2
f = np.polyfit(x_log, y_log, deg)
# f[2]+= 0.09
#f[1]+= 0.05

def cone_approx_log(xlog):
    return sum([ f[d]*xlog**(deg-d) for d in range(len(f))])

def cone_approx(x):
    xlog = np.log(x+1.0)#0.19
    return np.exp(cone_approx_log(xlog))

def best1approx(x):
    return np.exp(-0.72465853 * np.log(x + 0.17) + 10.90944352 )

def best2approx(x):
    return np.exp(0.18203247 * np.log(x + 1.0)**2 + -1.74195991 * np.log(x + 1.0) + 12.18370016 )


#cone_approx = lambda x, i=i: f[deg-i-1]*x**i for i in range(deg)

print f
print [f[0],f[1]+0.15*f[0]]
print cone_approx(0)

plt.figure()
plt.plot(x_log,y_log, linewidth=2)
#plt.plot(curcio_cone_densities[1],[ f[3] + f[2]*x + f[1]*x**2 + f[0]*x**3 for x in curcio_cone_densities[1]])
plt.plot(x_log,[cone_approx_log(i) for i in x_log], linewidth=2)
plt.ylabel('cones')
plt.title('cone polynomial approximation log field')

plt.figure()
plt.plot(curcio_cone_densities[1],curcio_cone_densities[4], linewidth=2)
#plt.plot(curcio_cone_densities[1],[ f[3] + f[2]*x + f[1]*x**2 + f[0]*x**3 for x in curcio_cone_densities[1]])
#plt.plot(np.arange(0, 80, 0.1),[cone_approx(i) for i in np.arange(0, 80, 0.1)], linewidth=2)
plt.plot(np.arange(0, 80, 0.1),[best2approx(i) for i in np.arange(0, 80, 0.1)], linewidth=2)
plt.ylabel('cones')
plt.title('cone polynomial approximation')
plt.xscale('log')
plt.yscale('log')

plt.show()

