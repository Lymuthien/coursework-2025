import os
import sys
import subprocess
import glob
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

BIN_DEFAULT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "cpp",
        "gd_bench" + (".exe" if os.name == "nt" else ""),
    )
)
CSV_DEFAULT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results.csv")
)
PER_DIR_DEFAULT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "per_iter")
)

HINTS = {
    "binary": "Исполняемый файл бенчмарка C++ (gd_bench).",
    "label": "Метка прогона/процессора. Попадёт в CSV и заголовки графиков.",
    "backend": "Режим выполнения: cpu — один поток, openmp — многопоточный CPU, gpu — встроенный gpu",
    "n_samples": "N — количество наблюдений (строк выборки).",
    "n_features": "D — количество признаков (столбцов).",
    "iters": "T — число итераций (игнорируется, если задана целевая длительность).",
    "lr": "η — шаг обучения градиентного спуска (>0).",
    "threads": "Потоки OpenMP (0=авто, -1=свип по потокам).",
    "seed": "Seed генератора случайных чисел.",
    "csv": "Файл агрегированных результатов (results.csv).",
    "per_iter": "Базовое имя per-iter файла; создаются *_runK.csv.",
    "target_sec": "Целевая длительность прогона (сек). 0 — не использовать.",
    "pilot_iters": "Пилот — короткий прогон для оценки времени одной итерации.",
    "runs": "Количество повторов теста (усреднение).",
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GD Benchmark GUI")
        if os.name == "nt":
            try:
                self.state("zoomed")
            except:
                pass
        else:
            self.attributes("-zoomed", True)

        self.style_setup()
        self.build()

    def style_setup(self):
        style = ttk.Style(self)
        try:
            style.theme_use("vista" if os.name == "nt" else "clam")
        except:
            pass

        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Hint.TLabel", foreground="#555555", font=("Segoe UI", 9))
        style.configure("Error.TLabel", foreground="#c62828", font=("Segoe UI", 9))

    def build(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Gradient Descent Benchmark", style="Header.TLabel").pack(
            anchor="w", pady=(0, 10)
        )

        self.status = ttk.Label(root, text="", style="Error.TLabel")
        self.status.pack(anchor="w", pady=(0, 8))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        self.set_form(body)
        self.set_result_panel(body)

    def set_form(self, parent):
        self.form = ttk.Frame(parent)
        self.form.pack(side="top", fill="x", pady=(0, 10))

        self.vars = {}
        self.errs = {}
        self.hints = {}

        fields = [
            ("binary", "Бинарник C++", BIN_DEFAULT),
            ("label", "Метка прогона", "CPU"),
            ("backend", "Backend", "openmp"),
            ("n_samples", "N (строк)", "20000"),
            ("n_features", "D (признаков)", "128"),
            ("iters", "Итераций (игнор при target)", "50"),
            ("lr", "Learning rate η", "1e-3"),
            ("threads", "Потоки (0=авто, -1=график)", "0"),
            ("seed", "Seed", "42"),
            ("csv", "results.csv", CSV_DEFAULT),
            ("per_iter", "per-iter база", os.path.join(PER_DIR_DEFAULT, "session.csv")),
            ("target_sec", "Цель (сек)", "0"),
            ("pilot_iters", "Пилот (итераций)", "5"),
            ("runs", "Повторов", "1"),
        ]

        self.form.columnconfigure(1, weight=1)

        row = 0
        for key, label_text, default in fields:
            ttk.Label(self.form, text=label_text).grid(
                row=row, column=0, sticky="w", padx=6, pady=4
            )

            var = tk.StringVar(value=str(default))
            self.vars[key] = var

            if key == "backend":
                frame = ttk.Frame(self.form)
                frame.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
                combo = ttk.Combobox(
                    frame,
                    textvariable=var,
                    values=["openmp", "cpu", "gpu"],
                    state="readonly",
                    width=12,
                    font=("Segoe UI", 10),
                )
                combo.pack(side="left")
                ttk.Label(
                    frame,
                    text="← cpu / openmp / gpu",
                    foreground="#1976d2",
                    font=("Segoe UI", 9, "italic"),
                ).pack(side="left", padx=(8, 0))
            else:
                ent = ttk.Entry(
                    self.form, textvariable=var, width=48, font=("Consolas", 10)
                )
                ent.grid(row=row, column=1, sticky="ew", padx=6, pady=4)

            err = ttk.Label(self.form, text="", style="Error.TLabel")
            err.grid(row=row, column=2, sticky="w", padx=6)

            hint = ttk.Label(self.form, text=HINTS.get(key, ""), style="Hint.TLabel")
            hint.grid(row=row, column=3, sticky="w", padx=6)

            self.errs[key] = err
            self.hints[key] = hint
            row += 1

        btns = ttk.Frame(self.form)
        btns.grid(row=row, column=0, columnspan=4, sticky="ew", pady=12)
        btns.columnconfigure(3, weight=1)

        ttk.Button(btns, text="Бинарный файл…", command=self.choose_bin).grid(
            row=0, column=0, padx=4
        )
        ttk.Button(btns, text="results.csv…", command=self.choose_csv).grid(
            row=0, column=1, padx=4
        )
        ttk.Button(btns, text="per-iter…", command=self.choose_per).grid(
            row=0, column=2, padx=4
        )
        ttk.Button(btns, text="Запустить", command=self.on_run, style="TButton").grid(
            row=0, column=3, sticky="e", padx=4
        )

    def set_result_panel(self, parent):
        panel = ttk.Frame(parent)
        panel.pack(side="top", fill="both", expand=True)

        left = tk.Text(panel, width=68, height=20, font=("Consolas", 10), bg="#fafafa")
        left.pack(side="left", fill="y", padx=(0, 8))

        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=panel)
        canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

        self.left_text = left
        self.fig = fig
        self.ax = ax
        self.canvas = canvas

    def choose_bin(self):
        fn = filedialog.askopenfilename(
            title="Выберите gd_bench.exe",
            filetypes=[("Executable", "*.exe *.out"), ("All", "*")],
        )
        if fn:
            self.vars["binary"].set(fn)

    def choose_csv(self):
        fn = filedialog.asksaveasfilename(
            title="results.csv", defaultextension=".csv", filetypes=[("CSV", "*.csv")]
        )
        if fn:
            self.vars["csv"].set(fn)

    def choose_per(self):
        fn = filedialog.asksaveasfilename(
            title="Базовое имя per-iter",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if fn:
            self.vars["per_iter"].set(fn)

    def set_status(self, msg):
        self.status.config(text=msg)
        self.update_idletasks()

    def _ok_int(self, name, positive=True, allow_zero=False, allow_minus_one=False):
        v = self.vars[name].get().strip()
        if not v:
            return True
        try:
            iv = int(v)
            if allow_minus_one and iv == -1:
                return True
            if not positive and iv < 0 and not allow_zero:
                raise ValueError("Отрицательное недопустимо")
            if positive and iv <= 0:
                raise ValueError("Должно быть >0")
            self.errs[name].config(text="")
            return True
        except ValueError as e:
            self.errs[name].config(text=str(e))
            return False

    def _ok_float_pos(self, name):
        v = self.vars[name].get().strip()
        if not v:
            return True
        try:
            fv = float(v)
            if fv <= 0:
                raise ValueError("Должно быть >0")
            self.errs[name].config(text="")
            return True
        except ValueError as e:
            self.errs[name].config(text="Некорректное число")
            return False

    def validate(self):
        ok = True
        for key in ("binary", "label", "csv", "per_iter"):
            if not self.vars[key].get().strip():
                self.errs[key].config(text="Обязательно")
                ok = False
            else:
                self.errs[key].config(text="")

        for name in ["n_samples", "n_features", "iters", "seed", "pilot_iters", "runs"]:
            ok &= self._ok_int(name)
        ok &= self._ok_float_pos("lr")
        ok &= self._ok_int(
            "threads", positive=False, allow_zero=True, allow_minus_one=True
        )
        ok &= self._ok_int("target_sec", positive=False, allow_zero=True)

        return ok

    def run_gd(self, threads_override=None):
        bin_path = self.vars["binary"].get().strip()
        if not os.path.isfile(bin_path):
            self.set_status("Бинарный файл не найден")
            return False

        args = []

        def add(key, flag):
            v = self.vars[key].get().strip()
            if v:
                args.extend([flag, v])

        for key, flag in [
            ("n_samples", "--n-samples"),
            ("n_features", "--n-features"),
            ("iters", "--iters"),
            ("lr", "--lr"),
            ("seed", "--seed"),
            ("label", "--label"),
            ("backend", "--backend"),
            ("target_sec", "--target-sec"),
            ("pilot_iters", "--pilot-iters"),
            ("runs", "--runs"),
        ]:
            add(key, flag)

        # CSV и per-iter
        csv_path = self.vars["csv"].get().strip()
        per_path = self.vars["per_iter"].get().strip()
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(per_path), exist_ok=True)

        args += ["--csv", csv_path]
        args += ["--per-iter-log", per_path]

        threads_val = (
            threads_override
            if threads_override is not None
            else self.vars["threads"].get().strip()
        )
        if threads_val and int(threads_val) != 0:
            args += ["--threads", str(threads_val)]

        cmd = [bin_path] + args

        env = os.environ.copy()
        if threads_val and int(threads_val) > 0:
            env["OMP_NUM_THREADS"] = str(threads_val)

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=600,
            )
            if proc.returncode != 0:
                self.set_status(f"Ошибка: {proc.stderr.strip()[-200:]}")
                return False
            self.set_status("Готово")
            return True
        except Exception as e:
            self.set_status(f"Исключение: {e}")
            return False

    def sweep_threads(self):
        max_t = (os.cpu_count() or 16) + 2
        return [1, 2, 4, 6, 8, 12, 16, 20, 24][:max_t]

    def run_sweep_threads(self):
        vals = self.sweep_threads()
        any_ok = False
        for t in vals:
            self.set_status(f"Запуск threads={t} …")
            self.update_idletasks()
            ok = self.run_gd(threads_override=t)
            any_ok = any_ok or ok
        if any_ok:
            self.set_status("Свип завершён")
        self.update_results(sweep=True)

    def update_results(self, sweep=False):
        lines = []
        csv_path = self.vars["csv"].get().strip()
        try:
            df = pd.read_csv(csv_path)
            last = df.iloc[-1]
            for col in df.columns:
                lines.append(f"{col}: {last[col]}")
        except Exception as e:
            lines.append(f"Не удалось открыть {csv_path}: {e}")

        self.ax.clear()

        if sweep:
            try:
                df = pd.read_csv(csv_path)
                g = df.groupby("threads")["time_ms_total"].mean().reset_index()
                self.ax.plot(g["threads"], g["time_ms_total"], "o-", color="#1976d2")
                self.ax.set_xlabel("Потоков")
                self.ax.set_ylabel("Время всего, мс")
                self.ax.set_title("Зависимость времени от числа потоков")
                self.ax.grid(True)
            except:
                self.ax.text(
                    0.5,
                    0.5,
                    "Нет данных для графика",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                )
        else:
            base = os.path.splitext(self.vars["per_iter"].get().strip())[0]
            files = sorted(
                glob.glob(base + "_run*.csv"), key=os.path.getmtime, reverse=True
            )
            try:
                runs = int(self.vars["runs"].get() or 1)
            except:
                runs = 1
            files = files[:runs]

            series = []
            for f in files:
                try:
                    d = pd.read_csv(f)
                    if "iter_idx" in d.columns and "time_ms" in d.columns:
                        series.append(d.set_index("iter_idx")["time_ms"])
                except:
                    continue

            if series:
                concat = pd.concat(series, axis=1)
                median = concat.median(axis=1)
                self.ax.plot(median.index, median.values, ".-", color="#2e7d32")
                self.ax.set_xlabel("Итерация")
                self.ax.set_ylabel("Время итерации, мс")
                self.ax.set_title(f"Per-iter время (медиана по {len(series)} запускам)")
                self.ax.grid(True)
                lines.append(
                    f"Медиана per-iter: {np.median([s.mean() for s in series]):.2f} мс"
                )
            else:
                self.ax.text(
                    0.5,
                    0.5,
                    "per-iter файлы не найдены",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                )

        self.left_text.delete("1.0", tk.END)
        self.left_text.insert(tk.END, "\n".join(lines))
        self.canvas.draw()

    def on_run(self):
        if not self.validate():
            self.set_status("Исправьте ошибки в параметрах")
            return

        if self.vars["threads"].get().strip() == "-1":
            self.run_sweep_threads()
        else:
            if self.run_gd():
                self.update_results(sweep=False)


if __name__ == "__main__":
    App().mainloop()
