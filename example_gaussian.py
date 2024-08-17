# This script provides an example of how to use the SCSA class.
# Copyright (C) 2024  Georgios Is. Detorakis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import matplotlib.pylab as plt

from scsa import SCSA


def g(x, mu=0.75, sigma=0.1):
    scale = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    return scale * np.exp(-(x - mu)**2 / (2.0 * sigma**2))


if __name__ == "__main__":
    N = 100
    a, b = 0, 2

    Dx = (b - a) / (N - 1)
    X = np.linspace(a, b, N)
    Y = g(X)
    Y0 = Y.copy()

    test_scsa = SCSA(M=100, Dx=Dx)
    Un, yhat = test_scsa.estimateSCSA(Y, chi=450, x=X)

    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    for i in range(6):
        if i == 0:
            ax = fig.add_subplot(2, 3, 1)
            ax.plot(Y0, label="original", lw=2)
            ax.plot(yhat, label="reconstruction", lw=2)
            ax.set_xlabel(r"$x$", fontsize=25)
            ax.set_xticks([0, 30, 60])
            ax.set_xticklabels(['0', '7.5', '15'], fontsize=16, weight='bold')
            ax.set_ylabel(r"$y(x)$", fontsize=25)
            ticks = ax.get_yticks()
            ax.set_yticks(ticks)
            ax.set_yticklabels(np.round(ticks, 2), fontsize=16, weight='bold')
            ax.legend()
        else:
            ax = fig.add_subplot(2, 3, i+1)
            ax.plot(Un[:, i-1]**2, 'k', lw=2)
            ax.set_ylabel(r"$\psi_i^2(x)$", fontsize=25, weight='bold')
            ax.set_xticks([])
            ticks = ax.get_yticks()
            ax.set_yticks(ticks)
            ax.set_yticklabels(np.round(ticks, 2), fontsize=14, weight='bold')
            ax.set_xticks([0, 30, 60])
            ax.set_xticklabels(['0', '7.5', '15'], fontsize=14, weight='bold')
            ax.set_xlabel(r"$x$", fontsize=25)

    plt.show()
