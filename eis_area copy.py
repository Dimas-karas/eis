# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 22:31:49 2025

@author: otser
"""

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

import os
import re
from impedance import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from impedance.models.circuits import CustomCircuit
from impedance.visualization import plot_nyquist
from tkinter import Tk, filedialog
import pandas as pd

root = Tk()
root.attributes('-topmost', True)
root.withdraw()
folder_path = filedialog.askdirectory(parent=root, title="Select folder with .cor files")
root.update()


freq_min = 10   
freq_max = 30000


circuit_str = 'R0-p(CPE1,R1)'
params = [1.5, 0.0012371, 1, 1000000]
fitted_results = []

filenames = [f for f in os.listdir(folder_path) if f.endswith('.z60')]

spectra = []

for file in filenames:
    # Извлечение потенциала из имени
    match = re.search(r'eis_([-+]?\d*\.\d+|\d+)', file)
    if match:
        potential = float(match.group(1))
    else:
        raise ValueError(f'Не удалось извлечь потенциал из имени файла: {file}')

    filepath = os.path.join(folder_path, file)
    frequencies, Z = preprocessing.readZPlot(filepath)
    
    frequencies, Z = preprocessing.cropFrequencies(frequencies, Z, freq_min, freq_max)
    
    frequencies, Z = preprocessing.ignoreBelowX(frequencies, Z)

    spectra.append((potential, frequencies, Z))

"""
for potential, frequencies, Z in spectra:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'EIS @ {potential:.2f} V', fontsize=14)

    axs[0].plot(np.real(Z), -np.imag(Z), 'o-')
    axs[0].set_xlabel('Re(Z), Ω')
    axs[0].set_ylabel('-Im(Z), Ω')
    axs[0].set_title('Nyquist')
    axs[0].grid(True)

    axs[1].set_title('Bode')
    axs[1].set_xlabel('log10(f), Hz')
    axs[1].grid(True)

    axs[1].plot(np.log10(frequencies), np.abs(Z), 'o-', label='|Z|')

    phase = np.angle(Z, deg=True)
    axs[1].twinx().plot(np.log10(frequencies), phase, 's--', color='orange', label='Phase')

    plt.tight_layout()
    plt.show()
"""
    
spectra.sort(key=lambda x: x[0])

for potential, frequencies, Z in spectra:
    # Копия схемы с текущим приближением
    circuit = CustomCircuit(initial_guess=params, circuit=circuit_str)
    
    # Пфит
    try:
        circuit.fit(frequencies, Z)
    except Exception as e:
        print(f'Ошибка при аппроксимации потенциала {potential:.2f} В: {e}')
        continue

    Z_fit = circuit.predict(frequencies)

    mse = np.mean(np.abs(Z_fit - Z)**2)
    nrmse = np.sqrt(mse) / np.mean(np.abs(Z))
    fitted_results.append((potential, *circuit.parameters_, mse, nrmse))

    params = list(circuit.parameters_)
    params[3]=1000
    Z_fit = circuit.predict(frequencies)

    fig, ax = plt.subplots()
    plot_nyquist(Z, fmt='o', scale=1, ax=ax)
    plot_nyquist(Z_fit, fmt='-', scale=1, ax=ax)

    plt.legend(['Data', 'Fit'])
    plt.show()
    print(potential, circuit)
    
results_array = np.array(fitted_results)

E_ref = results_array[:, 0]
Rs = results_array[:, 1]
Q = results_array[:, 2]
n = results_array[:, 3]
Rp = results_array[:, 4]
mse = results_array[:, 5]
nrmse = results_array[:, 6]
E_rhe = E_ref + 0.9276


# Расчёт C по формуле
R_sum = 1/Rs + 1/Rp
C = (Q * R_sum**(n - 1))**(1/n)*1e6

plt.figure(figsize=(6, 4))
plt.plot(E_rhe, C, 'o-', color='teal')  # перевод в мкФ
plt.xlabel('E vs RHE (V)')
plt.ylabel('C (μF)')
plt.yscale('log')
plt.title('')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(E_rhe, nrmse * 100, 's--r')  # в процентах
plt.xlabel('Potential vs RHE (V)')
plt.ylabel('RMSE (%)')
plt.title('Fit quality vs Potential')
plt.grid(True)
plt.tight_layout()
plt.show()
df = pd.DataFrame({"E_RHE": E_rhe, "C": C})
output_file = os.path.join(folder_path, "output.csv")
df.to_csv(output_file, index=False)

    