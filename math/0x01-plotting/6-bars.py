#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

names = ['Farrah', 'Fred', 'Felicia']
plt.bar(names, fruit[0], color='red')
plt.bar(names, fruit[1], bottom=fruit[0], color='yellow')
plt.bar(names, fruit[2], bottom=fruit[1] + fruit[0], color='#ff8000')
plt.bar(names, fruit[3], bottom=fruit[2] + fruit[1] + fruit[0], color='#ffe5b4')
plt.legend(["Apples", "bananas", "oranges", "peaches"])
plt.show()
