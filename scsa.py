# This script provides a Python class that implements the Semi-Classical Signal
# Analysis (SCSA) algorithm proposed by
#    [1] Laleg-Kirati, Taous-Meriem, Emmanuelle Crépeau, and Michel Sorine.
#        "Semi-classical signal analysis."
#         Mathematics of Control, signals, and Systems 25 (2013): 37-61.
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

from scipy.sparse import spdiags
from scipy.integrate import simpson


class SCSA:
    """ Main class that implements the Semi-Classical Signal Analysis
    algorithm proposed in [1].
    """
    def __init__(self,
                 M=100,
                 Dx=0.1,
                 is_sym_neg_def=True,
                 rtol=1e-05,
                 atol=1e-08):
        """ Constructor of the SCSA class.

        @param M (int)  Number of discretization points (steps)
        @param Dx (float)  Discretization step
        @param is_sym_neg_def (bool)  Diagnostics for matrix D2
        @param rtol (float) Relative tolerance for matrix diagnostics
        @param atol (float) Absolute tolerance for matrix diagnostics

        """
        self.M = M
        self.Dx = Dx

        # Construct the second order differentiation matrix
        self.D2 = self.buildDifferentiationMatrix()

        # Check if D2 is symmetric and definite negative
        if is_sym_neg_def is True:
            symmetry = np.allclose(self.D2, self.D2.T, rtol=rtol, atol=atol)
            print(f"D2 Symmetric: {symmetry}")
            negative_definite = np.all(np.linalg.eigvals(self.D2) < 0)
            print(f"D2 Negative Definite: {negative_definite}")

    def buildDifferentiationMatrix(self):
        """ This method constructs a second order differentiation matrix for
        discretizing the Schrondiger operator. The matrix is used in solving
        the eigenvalue problem [1].

        @param Void

        @note For more details on how to construct the matrix D2 please see [1]

        @return The second order differentiation matrix that discretizes the
        Schrodinger operator
        """
        Delta = (2.0 * np.pi) / self.M

        dk = np.kron(range(self.M-1, 0, -1), np.ones((self.M, 1)))
        if self.M % 2 == 0:
            DL = (-(-1)**(dk) * 0.5 / (np.sin(dk * Delta * 0.5)**2))
            DR = (-(-1)**(-dk) * 0.5 / (np.sin(-dk * Delta * 0.5)**2))
            Dd = (-np.pi**2 / (3 * Delta**2)) - 1.0 / 6.0*np.ones((self.M, 1))
        else:
            DL = (-0.5 * (-1)**(dk) / (np.sin(dk * Delta * 0.5)
                  * np.tan(dk * Delta * 0.5)))
            DR = (-0.5 * (-1)**(-dk) / (np.sin(dk * Delta * 0.5)
                  * np.tan(dk * Delta * 0.5)))
            Dd = (-np.pi**2 / (3 * Delta**2)) - 1.0 / 12.0*np.ones((self.M, 1))

        Dtmp = np.hstack([DL, Dd, DR])
        diags = np.array([i for i in range(-(self.M-1), 1)]
                         + [i for i in range(self.M-1, 0, -1)])
        D = np.squeeze(np.asarray(spdiags(Dtmp.T,
                                          diags,
                                          m=self.M,
                                          n=self.M).todense()))
        D *= (Delta**2 / self.Dx**2)
        return D

    def estimateSCSA(self, y, chi=20, x=None):
        """ This function implements the basic algorithm for estimating the
        SCSA of the input signal y for a given χ (chi).

        @param y (ndarray)  Input signal
        @param chi (float)  Parameter chi of the algorithm (see [1])
        @param x (ndarray)  Discretization points (e.g., [0, 1/n, 2/n, ...])

        @note The entire algorithm is presented in [1]. Before starting this
        implementation, you are advised to read [1].

        @return The eigenfunctions U (ndarray) and the reconstruction, yhat
        (ndarray), of the input signal y
        """
        if x is None:
            x = np.linspace(0, len(y), len(y))
        y_min = y.min()
        y -= y_min

        # Solve the eigenvalues-eigenfunctions problem
        A = -self.D2 - chi * np.diag(y)
        v, u = np.linalg.eig(A)

        # Identify all negative eigenvalues
        idx = np.where(v <= 0)[0]
        V = v[idx]
        K = np.sqrt(-V)

        # Take all the eigenfunctions corresponding to K
        U = u[:, idx]
        # Normalize eigenfunctions
        Un = np.zeros_like(U)
        for i in range(U.shape[1]):
            Un[:, i] = U[:, i] / np.sqrt(simpson(U[:, i]**2, x=x))

        self.Nx = len(V)
        yhat = (4.0 / chi) * np.sum(Un[:, :self.Nx]**2 * K[:self.Nx], axis=1)

        return Un, yhat
