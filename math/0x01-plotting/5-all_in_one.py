#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, p = plt.subplots(3, 2)
p[0, 0].plot(y0, color="red")
p[0, 1].scatter(x1, y1, color="magenta")
p[0, 1].set_title("Men's Height vs Weight")
p[0, 1].set(xlabel="Height (in)")
p[0, 1].set(ylabel="Weight (lbs)")
p[1, 0].set_title("Exponential Decay of C-14")
p[1, 0].plot(x2, y2)
p[1, 0].set(xlabel="Time (years)")
p[1, 0].set(ylabel="Fraction Remaining")
p[1, 0].set(yscale='log')
p[1, 1].set_title("Exponential Decay of Radioactive Elements")
p[1, 1].set(xlabel="Time (years)")
p[1, 1].set(ylabel="Fraction Remaining")
p[1, 1].set_xlim(0, 20000)
p[1, 1].set_ylim(0, 1)
p[1, 1].plot(x3, y31, label="C-14", color="red", linestyle="--")
p[1, 1].plot(x3, y32, label="Ra-226", color="green")
p[1, 1].legend(loc="upper right")
fig.delaxes(p[2, 1])
p[2, 0].hist(student_grades, edgecolor='black', range=(0, 100), bins=10)
p[2, 0].set_title("Project A")
p[2, 0].set(xlabel="Grades")
p[2, 0].set(ylabel="Number of Students")
p[2, 0].set(xticks=range(0, 101, 10))
fig.suptitle('All in One')
fig.tight_layout()
plt.rcParams.update({'font.size': 0.694})
plt.show()
