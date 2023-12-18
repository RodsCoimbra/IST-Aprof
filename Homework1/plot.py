# %%
import matplotlib.pyplot as plt
from numpy import linspace

plt.scatter(1, 1, c='r', marker='o')
plt.scatter(-1, 1, c='g', marker='o')
plt.scatter(1, -1, c='g', marker='o')
plt.scatter(-1, -1, c='r', marker='o')
plt.axvline(x=0, c="black")
plt.axhline(y=0, c="black")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
x = linspace(-2, 2, 10)
y = -x - 1
plt.plot(x, y, c='g', label="A")
plt.plot(x, y + 2, c='y', label="B")
plt.legend()
plt.show()

# %%
