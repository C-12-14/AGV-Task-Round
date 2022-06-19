import numpy as np
import matplotlib.pyplot as plt
import csv

t = 1

sig_x_est, sig_y_est, sig_vx_est, sig_vy_est = np.array([0.25, 0.25, 0.1, 0.1]) * 20

sig_x_mea, sig_y_mea, sig_vx_mea, sig_vy_mea = np.array([0.1, 0.1, 1, 1]) * 40


def predict(A, x, y, vx, vy):

    X = np.array([[x], [y], [vx], [vy]])

    return np.dot(A, X)


def main():
    data = []
    xi = 0
    yi = 0
    with open("kalman-filter/kalmann.txt") as f:
        lines = f.readlines()
        xi, yi = [float(x) for x in lines[0].split(",")]
        data = [[float(x) for x in line.split(",")] for line in lines[1:]]
        data = np.array(data)

    P = np.array(
        [
            [sig_x_est**2, 0, 0, 0],
            [0, sig_y_est**2, 0, 0],
            [0, 0, sig_vx_est**2, 0],
            [0, 0, 0, sig_vy_est**2],
        ]
    )

    A = np.array(
        [
            [1, 0, t, 0],
            [0, 1, 0, t],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    R = np.array(
        [
            [sig_x_mea**2, 0, 0, 0],
            [0, sig_y_mea**2, 0, 0],
            [0, 0, sig_vx_mea**2, 0],
            [0, 0, 0, sig_vy_mea**2],
        ]
    )
    X = np.diagflat([xi, yi, 0, 0])

    x_kal = [xi]
    y_kal = [yi]
    x_mea = [xi]
    y_mea = [yi]
    with open("kalman-filter/kalmann_est.txt", "w") as wf:
        for x, y, vx, vy in data:
            X = predict(A, X[0][0], X[1][0], X[2][0], X[3][0])
            P = np.diag(np.diag(A @ P @ A.T))

            H = np.identity(4)

            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            Y = H @ np.array([[x], [y], [vx], [vy]])
            X = X + K @ (Y - H @ X)

            P = np.diag(np.diag((np.identity(4) - K @ H) @ P))
            x_kal.append(X[0][0])
            y_kal.append(X[1][0])
            x_mea.append(x)
            y_mea.append(y)

            wf.write(
                f"{X[0][0]} , {X[1][0]} , {X[2][0]} , {X[3][0]} , {P[0][0]} , {P[1][1]} , {P[2][2]} , {P[3][3]}\n"
            )

        plt.plot(x_kal, y_kal)
        plt.plot(x_mea, y_mea, alpha=0.5)
        plt.show()


main()
