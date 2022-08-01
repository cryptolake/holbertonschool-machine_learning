#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, edgecolor='black', range=(0, 100), bins=10)
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.xticks(range(0, 101, 10))
plt.show()
