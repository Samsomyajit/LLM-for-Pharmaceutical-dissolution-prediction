import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# ----------------- 基础物性计算函数 -----------------
def calculate_D(temp_C, viscosity_Pas, r_molecule):
    """斯托克斯-爱因斯坦方程计算扩散系数"""
    R = 8.314  # 气体常数(J/mol/K)
    T = temp_C + 273.15
    return (R * T) / (6 * np.pi * viscosity_Pas * r_molecule * 6.022e23)  # m²/s


def calculate_Sh(r_particle, D, velocity, visc_fluid=0.00089, rho_fluid=1000):
    """Ranz-Marshall方程计算Sherwood数"""
    Re = 2 * r_particle * velocity * rho_fluid / visc_fluid  # 雷诺数
    Sc = visc_fluid / (rho_fluid * D)  # 施密特数
    return 2 + 0.6 * np.sqrt(Re) * (Sc) ** (1 / 3)


def calculate_solubility(temp_C, delta_h=25e3, Tm=150, logP=2.5):
    """Yalkowsky溶解度方程计算平衡溶解度 (mol/m³)"""
    R = 8.314  # 添加气体常数定义
    T = temp_C + 273.15
    S_ideal = 10**(0.8 - logP)
    S_corr = S_ideal * np.exp(-delta_h/R * (1/T - 1/(Tm+273.15)))
    return S_corr * 1e3



# ----------------- 主模型函数 -----------------
def dissolution_model(y, t, params):
    M = max(y[0], 0)
    r = (3 * M / (4 * np.pi * params['density'])) ** (1 / 3) if M > 1e-18 else 0

    # 动态计算物性参数
    D = calculate_D(params['temp'], params['viscosity'], params['r_mol'])
    Sh = calculate_Sh(r, D, params['velocity'])
    h = 2 * r / Sh if Sh > 1e-6 else 1e-6

    # 计算溶解度（新增的关键步骤）
    Cs_molar = calculate_solubility(params['temp'], params['delta_h'],
                                    params['Tm'], params['logP'])
    Cs = Cs_molar * params['MW']  # 转换为kg/m³

    # 溶出速率计算
    Ct = M / params['V']
    concentration_driving = max(Cs - Ct, 0)
    S = 4 * np.pi * r ** 2 if r > 1e-9 else 0
    dMdt = -S * D * concentration_driving / h if r > 1e-9 else 0

    return [dMdt]


# ----------------- 参数配置 -----------------
params = {
    'r0': 36e-6,  # 初始粒子半径(m) 外部实验参数
    'temp': 37.0,  # 温度(°C) 外部实验参数
    'V': 0.0005,  # 介质体积(m³) 外部实验参数
    'velocity': 0.02,  # 流体速度(m/s) 外部实验参数
    # velocity=0.02 m/s 对应 约50 RPM（假设桨叶直径25 mm且修正系数0.3）。

    'viscosity': 0.00089,  # 介质粘度(Pa·s) 温度变化
    'r_mol': 0.5e-9,  # 分子半径(m) 材料内在性质
    'MW': 0.206,  # 分子量(kg/mol) 材料内在性质
    'density': 1030.0,  # 密度(kg/m³) 温度变化
    'delta_h': 25e3,  # 熔解焓(J/mol) 材料内在性质
    'Tm': 76.0,  # 熔点(°C) 材料内在性质
    'logP': 3.7  # 辛醇/水分配系数 材料内在性质
}

# ----------------- 执行计算 -----------------
initial_M = 4 / 3 * np.pi * (params['r0'] ** 3) * params['density']
t = np.linspace(0, 5400, 1000)  # 0-1小时
solution = odeint(dissolution_model, [initial_M], t, args=(params,))
M_dissolved = initial_M - solution[:, 0]

# ----------------- 结果可视化 -----------------
plt.figure(figsize=(10, 6))
plt.plot(t / 60, 100 * M_dissolved / initial_M, linewidth=2)
plt.title('Theoretical Dissolution Profile')
plt.xlabel('Time (minutes)')
plt.ylabel('Dissolved (%)')
plt.grid(True)
plt.show()

# 提取并格式化数据
X = np.round(t / 60, 2)        # 时间(分钟)
Y = np.round(100 * M_dissolved / initial_M, 2)  # 溶出百分比

# 输出前10个数据点验证
print("Time(min)\tDissolved(%)")
for x, y in zip(X[:10], Y[:10]):
    print(f"{x*100:.2f}\t\t{y*100:.2f}")

# 完整数据矩阵 (1000×2 array)
dissolution_data = np.column_stack((X, Y))
