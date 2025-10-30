import os
import re
import math
from impedance import preprocessing
import numpy as np
from impedance.models.circuits import CustomCircuit
from tkinter import Tk, filedialog
import ctypes
from typing import Optional
import originpro as op
from origin_decorator import interact_with_origin


try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


def get_folder_path() -> str:
    root = Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder_path = filedialog.askdirectory(parent=root, title="Select folder with .z60 files")
    root.update()
    return folder_path

def get_save_project_path() -> str:
    root = Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    filepath = filedialog.asksaveasfilename(
        parent=root,
        title="Выберите, куда сохранить Origin-проект",
        defaultextension=".opju",
        filetypes=[("Origin Project", "*.opju")],
        initialfile="EIS_Results.opju",
    )
    root.update()
    return filepath or ""


def extract_potential(filename: str) -> float:
    """
    Извлекает потенциал из имени файла, ищет [число]V или [число]mV и возвращает значение в вольтах.
    Учитывает десятичные точки и знаки +/-. Например, 'eis_-0.5V.z60' вернет -0.5, 'eis_250mV.z60' вернет 0.25.
    """
    match = re.search(r'([-+]?\d+(?:\.\d+)?)(m?V)', filename)
    if not match:
        raise ValueError(f"Не удалось извлечь потенциал из имени файла: {filename}")
    value, unit = match.groups()
    value = float(value)
    if unit == 'mV':
        value /= 1000
    return value


def process_eis_data(
        circuit_str: str = 'R0-p(CPE1,R1)',
        initial_params: dict[str, float | int] = {
            "R0": 1,
            "Q": 0.001,
            "n": 1, 
            "R1": 1_000_000,
        },
        freq_min: int = 0,
        freq_max: Optional[int] = None,
):
    folder_path = get_folder_path()
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.z60')]
    assert len(filenames) > 0, "В выбранной папке нет файлов с расширением .z60"
    spectra = []

    for file in filenames:
        potential = extract_potential(file)
        filepath = os.path.join(folder_path, file)
        frequencies, Z = preprocessing.readZPlot(filepath)
        frequencies, Z = preprocessing.cropFrequencies(frequencies, Z, freq_min, freq_max,)
        frequencies, Z = preprocessing.ignoreBelowX(frequencies, Z)
        spectra.append((potential, frequencies, Z))
    spectra.sort(key=lambda x: x[0])

    circuit = CustomCircuit(circuit=circuit_str, initial_guess=initial_params.values())
    results_origin = {}
    results_raw = []

    for potential, frequencies, Z in spectra:
        try:
            circuit.fit(frequencies, Z)
        except Exception as e:
            print(f'Ошибка при аппроксимации потенциала {potential:.2f} В: {e}')
            continue
        Z_fit = circuit.predict(frequencies)
        mse = np.mean(np.abs(Z_fit - Z)**2)
        nrmse = np.sqrt(mse) / np.mean(np.abs(Z))
        results_origin[potential] = {
            "params_names": list(initial_params.keys()) + ["NRMSE"],
            "params_values": circuit.parameters_.tolist() + [float(nrmse)],
            "params_confs": circuit.conf_.tolist(),
            "frequencies": frequencies,
            "Z_exp_real": Z.real,
            "Z_exp_imag": Z.imag,
            "Z_fit_real": Z_fit.real,
            "Z_fit_imag": Z_fit.imag,
        }
        results_raw.append((potential, *circuit.parameters_, mse, nrmse))
        circuit.initial_guess = list(circuit.parameters_)
        print(potential, circuit)

    return {"origin": results_origin, "raw": np.array(results_raw)}



def _eval_extra_value(formula: str, names: list[str], values: list[float]):
    """Вычисляет дополнительный параметр по формуле в безопасном окружении."""
    ctx = {name: val for name, val in zip(names, values)}
    ctx["np"] = np
    ctx["math"] = math
    try:
        return eval(formula, {"__builtins__": {}}, ctx)
    except Exception:
        return None


@interact_with_origin
def export_to_origin(
    results_origin: dict[str, dict],
    formula: str | None = None,
    formula_name: str = "C",
):
    project_path = get_save_project_path()
    if not project_path:
        print("Сохранение отменено.")
        return None

    op.new(asksave=False)
    wb = op.new_book(type='w', lname='EIS Results')

    potentials = sorted(results_origin.keys())
    if not potentials:
        raise ValueError("Пустой словарь results_origin — нечего экспортировать.")

    first = results_origin[potentials[0]]
    param_names = list(first["params_names"])
    param_confs = first.get("params_confs", [])

    # формируем список колонок в порядке: Potential, P1, P1_err, P2, P2_err, ..., NRMSE, [extra]
    paired_columns: list[tuple[str, str | None]] = []
    for i, pname in enumerate(param_names):
        if pname == "NRMSE":
            paired_columns.append((pname, None))
        else:
            paired_columns.append((pname, f"{pname}_err"))

    has_formula = bool(formula)
    if has_formula:
        paired_columns.append((formula_name, None))

    total_cols = 1  # Potential
    for _, err_name in paired_columns:
        total_cols += 1
        if err_name is not None:
            total_cols += 1

    wks_params = wb.add_sheet(name='Parameters', active=True)
    wks_params.cols = total_cols

    # колонка потенциалов
    wks_params.from_list(0, potentials, lname="Potential", units="V")

    col_index = 1
    for pname, err_name in paired_columns:
        if pname == formula_name and has_formula:
            col_data = []
            for p in potentials:
                rd = results_origin[p]
                vals = rd["params_values"]
                val = _eval_extra_value(formula, param_names, vals)
                col_data.append(val)
            wks_params.from_list(col_index, col_data, lname=pname)
            col_index += 1
            continue

        col_vals = []
        col_errs = []
        for p in potentials:
            rd = results_origin[p]
            vals = rd["params_values"]
            confs = rd.get("params_confs", [])
            try:
                idx = param_names.index(pname)
            except ValueError:
                col_vals.append(None)
                if err_name:
                    col_errs.append(None)
                continue

            col_vals.append(vals[idx])

            if err_name is not None:
                if idx < len(confs):
                    col_errs.append(confs[idx])
                else:
                    col_errs.append(None)

        wks_params.from_list(col_index, col_vals, lname=pname)
        col_index += 1

        if err_name is not None:
            wks_params.from_list(col_index, col_errs, lname=err_name)
            col_index += 1

    # листы с данными и графики Nyquist
    for p in potentials:
        rd = results_origin[p]
        wks = wb.add_sheet(name=f"{p:+.3f} V", active=False)
        freqs = rd["frequencies"]
        zexp_r = rd["Z_exp_real"]
        zexp_i = rd["Z_exp_imag"]
        zfit_r = rd["Z_fit_real"]
        zfit_i = rd["Z_fit_imag"]

        neg_zexp_i = [-val for val in zexp_i]
        neg_zfit_i = [-val for val in zfit_i]

        # f, Z'exp, Z''exp, Z'fit, Z''fit, -Z''exp, -Z''fit
        wks.cols = 7
        wks.from_list(0, freqs, lname="f", units="Hz")
        wks.from_list(1, zexp_r, lname="Z' (exp)", units="Ohm")
        wks.from_list(2, neg_zexp_i, lname="-Z'' (exp)", units="Ohm")
        wks.from_list(3, zfit_r, lname="Z' (fit)", units="Ohm")
        wks.from_list(4, neg_zfit_i, lname="-Z'' (fit)", units="Ohm")

        # график Nyquist для этого потенциала
        g = op.new_graph(template='scatter', lname=f"Nyquist {p:+.3f} V")
        gl = g[0]
        # экспериментальные точки
        gl.add_plot(wks, coly=2, colx=1, type='scatter')
        # аппроксимация линией
        gl.add_plot(wks, coly=4, colx=3, type='line')
        gl.rescale()
        gl.label('Legend').text = f"{p:+.3f} V"
        
    # скрыть все графики с Найквистом из окна проекта, чтобы не захламлять пространство
    op.lt_exec("doc -e P {win -ch 1;}")

    save_path = os.path.normpath(project_path)
    ok = op.save(save_path)
    if not ok:
        raise IOError(f"Origin не смог сохранить проект в '{save_path}'")
    return save_path
