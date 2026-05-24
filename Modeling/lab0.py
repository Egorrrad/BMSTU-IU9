import matplotlib.pyplot as plt
import numpy as np

v0 = 75.0
angle_deg = 40.0
r = 0.06
rho_iron = 7800.0
rho_air = 1.293
C = 0.15
g = 9.81

alpha = np.radians(angle_deg)
S = np.pi * r ** 2
m = (4 / 3) * np.pi * r ** 3 * rho_iron
beta = (C * S * rho_air) / 2

dist_galileo = (v0 ** 2 * np.sin(2 * alpha)) / g
x_gal = np.linspace(0, dist_galileo, 100)

y_gal = x_gal * np.tan(alpha) - (g * x_gal ** 2) / (2 * v0 ** 2 * np.cos(alpha) ** 2)


def equations_of_motion(state):
    x, y, u, w = state
    v = np.sqrt(u ** 2 + w ** 2)
    du_dt = -(beta * u * v) / m
    dw_dt = -g - (beta * w * v) / m
    return np.array([u, w, du_dt, dw_dt])


dt = 0.01

state = np.array([0.0, 0.0, v0 * np.cos(alpha), v0 * np.sin(alpha)])
trajectory = [state.copy()]

while state[1] >= 0:
    k1 = equations_of_motion(state)
    k2 = equations_of_motion(state + (dt / 2) * k1)
    k3 = equations_of_motion(state + (dt / 2) * k2)
    k4 = equations_of_motion(state + dt * k3)
    state += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    trajectory.append(state.copy())

p1 = trajectory[-2]
p2 = trajectory[-1]
fraction = p1[1] / (p1[1] - p2[1])
x_final_newton = p1[0] + fraction * (p2[0] - p1[0])

traj_np = np.array(trajectory)

print(f"Дальность (Галилей): {dist_galileo:.4f} м")
print(f"Дальность (Ньютон):  {x_final_newton:.4f} м")
print(f"Потеря дальности:    {dist_galileo - x_final_newton:.4f} м")

plt.figure(figsize=(10, 5))
plt.plot(x_gal, y_gal, '--r', label='Модель Галилея')
plt.plot(traj_np[:, 0], traj_np[:, 1], 'b', label='Модель Ньютона')
plt.axhline(0, color='black', lw=1)
plt.title(f"Траектория (Вариант 4): v0={v0} м/с, r={r} м, {angle_deg}°")
plt.xlabel("x (м)"), plt.ylabel("y (м)")
plt.legend(), plt.grid(True, alpha=0.3)
plt.show()
