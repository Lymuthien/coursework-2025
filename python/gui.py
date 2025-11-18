import os, sys, subprocess, glob
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
    "n_samples": "N — количество наблюдений (строк выборки).",
    "n_features": "D — количество признаков (столбцов).",
    "iters": "T — число итераций (игнорируется, если задана целевая длительность).",
    "lr": "η — шаг обучения градиентного спуска (>0).",
    "threads": "Число потоков (0=авто, -1=график).",
    "seed": "Seed генератора случайных чисел.",
    "csv": "Файл агрегированных результатов (results.csv).",
    "per_iter": "Базовое имя per-iter файла; создаются *_runK.csv.",
    "target_sec": "Целевая длительность (сек). 0 — не использовать.",
    "pilot_iters": "Пилот — короткий предварительный прогон (k итераций) для оценки времени одной итерации.",
    "runs": "Повторы теста (усреднение времён).",
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Benchmark GUI")
        try:
            if os.name == "nt":
                self.state("zoomed")
            else:
                self.attributes("-zoomed", True)
        except Exception:
            self.attributes("-fullscreen", True)
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
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Hint.TLabel", foreground="#666666", font=("Segoe UI", 9))
        style.configure("Error.TLabel", foreground="#c62828", font=("Segoe UI", 9))

    def build(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Gradial descent algorithm", style="Header.TLabel").pack(
            anchor="w", pady=(0, 6)
        )

        self.status = ttk.Label(root, text="", style="Error.TLabel")
        self.status.pack(anchor="w", pady=(0, 6))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)

        row = self.set_form(body)
        buttons = self.set_buttons(row)
        self.set_result_panel(body)

    def set_form(self, body):
        self.form = ttk.Frame(body)
        self.form.pack(side="top", fill="x")

        self.vars = {}
        self.errs = {}
        self.hints = {}
        fields = [
            ("binary", "Бинарник C++", BIN_DEFAULT),
            ("label", "Метка CPU/прогона", "CPU"),
            ("n_samples", "N (>0)", "20000"),
            ("n_features", "D (>0)", "128"),
            ("iters", "T (>0, игнор при target)", "50"),
            ("lr", "η (>0)", "1e-3"),
            ("threads", "Потоки (0=авто, -1=график)", "0"),
            ("seed", "Seed", "42"),
            ("csv", "results.csv", CSV_DEFAULT),
            ("per_iter", "per-iter база", os.path.join(PER_DIR_DEFAULT, "session.csv")),
            ("target_sec", "Цель (сек, >=0)", "0"),
            ("pilot_iters", "Пилот (k>0)", "5"),
            ("runs", "Повторы (>=1)", "1"),
        ]

        self.form.columnconfigure(1, weight=1)

        r = 0
        for key, label, default in fields:
            ttk.Label(self.form, text=label).grid(
                row=r, column=0, sticky="w", padx=6, pady=4
            )

            var = tk.StringVar(value=str(default))
            self.vars[key] = var

            ent = ttk.Entry(self.form, textvariable=var, width=48)
            ent.grid(row=r, column=1, sticky="ew", padx=6, pady=4)

            err = ttk.Label(self.form, text="", style="Error.TLabel")
            err.grid(row=r, column=2, sticky="w", padx=6)

            hint = ttk.Label(self.form, text=HINTS.get(key, ""), style="Hint.TLabel")
            hint.grid(row=r, column=3, sticky="w", padx=6)

            self.errs[key] = err
            self.hints[key] = hint
            r += 1

        return r

    def set_result_panel(self, body):
        self.panel = ttk.Frame(body)
        self.panel.pack(side="top", fill="both", expand=True, pady=(8, 0))
        right = ttk.Frame(self.panel)
        right.pack(fill="both", expand=True)

        self.left = tk.Text(right, width=58, font=("Consolas", 10))
        self.left.pack(side="left", fill="y", padx=(0, 8), pady=6)

        self.fig = plt.Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True, pady=6)

    def set_buttons(self, row):
        btns = ttk.Frame(self.form)
        btns.grid(row=row, column=0, columnspan=4, sticky="ew", pady=8)
        ttk.Button(btns, text="Выбрать бинарник…", command=self.choose_bin).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Выбрать results.csv…", command=self.choose_csv).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Выбрать per-iter…", command=self.choose_per).pack(
            side="left", padx=4
        )
        ttk.Button(btns, text="Запустить", command=self.on_run).pack(
            side="right", padx=4
        )

        return btns

    def set_status(self, msg):
        self.status.config(text=msg)

    def choose_bin(self):
        try:
            fn = filedialog.askopenfilename(title="gd_bench(.exe)")
            if fn:
                self.vars["binary"].set(fn)
        except Exception as e:
            self.set_status(f"Error: {e}")

    def choose_csv(self):
        try:
            fn = filedialog.asksaveasfilename(
                title="results.csv", defaultextension=".csv"
            )
            if fn:
                self.vars["csv"].set(fn)
        except Exception as e:
            self.set_status(f"Error: {e}")

    def choose_per(self):
        try:
            fn = filedialog.asksaveasfilename(
                title="per-iter base", defaultextension=".csv"
            )
            if fn:
                self.vars["per_iter"].set(fn)
        except Exception as e:
            self.set_status(f"Error: {e}")

    # validation
    def _ok_int(self, name, positive=True, allow_zero=False, allow_minus_one=False):
        v = self.vars[name].get().strip()
        if not v:
            self.errs[name].config(text="Empty. Default value is enable.")
            return True
        try:
            iv = int(v)
            if allow_minus_one and iv == -1:
                return True
            if allow_zero and iv < 0:
                raise ValueError("Number must be non-negative.")
            if positive and iv <= 0:
                raise ValueError("Number must be positive.")
            self.errs[name].config(text="")
            return True
        except ValueError as e:
            self.errs[name].config(text=str(e))
            return False

    def _ok_float_pos(self, name):
        v = self.vars[name].get().strip()
        if not v:
            self.errs[name].config(text="Empty. Default value is enable.")
            return True
        try:
            fv = float(v)
            if fv <= 0:
                raise ValueError("Number must be positive.")
            self.errs[name].config(text="")
            return True
        except ValueError as e:
            self.errs[name].config(text=str(e))
            return False

    def validate(self):
        ok = True

        for k in ["binary", "label", "csv", "per_iter"]:
            if not self.vars[k].get().strip():
                self.errs[k].config(text="Enter value")
                ok = False
            else:
                self.errs[k].config(text="")

        for name in ["n_samples", "n_features", "iters", "seed", "pilot_iters", "runs"]:
            ok &= self._ok_int(name)
        ok &= self._ok_float_pos("lr")
        ok &= self._ok_int(
            "threads", positive=False, allow_zero=True, allow_minus_one=True
        )
        ok &= self._ok_int("target_sec", positive=False, allow_zero=True)

        return ok

    # run single
    def run_gd(self, threads_override=None):
        bin_path = self.vars["binary"].get().strip()
        if not os.path.exists(bin_path):
            self.set_status("Binary file not found")
            return False

        args = []

        def add(n, flag):
            v = self.vars[n].get().strip()
            if v:
                args.extend([flag, v])

        for n, flag in [
            ("n_samples", "--n-samples"),
            ("n_features", "--n-features"),
            ("iters", "--iters"),
            ("lr", "--lr"),
            ("seed", "--seed"),
            ("label", "--label"),
            ("target_sec", "--target-sec"),
            ("pilot_iters", "--pilot-iters"),
            ("runs", "--runs"),
        ]:
            add(n, flag)

        def check_path(path, name):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            except Exception as e:
                self.set_status(f"Error {name} path: {e}")
                return False
            return True

        csv_path = self.vars["csv"].get().strip()
        if not check_path(csv_path, "csv"):
            return False
        args.extend(["--csv", csv_path])

        per = self.vars["per_iter"].get().strip()
        if not check_path(per, "per-iter"):
            return False
        args.extend(["--per-iter-log", per])

        threads_value = (
            threads_override
            if threads_override is not None
            else self.vars["threads"].get().strip()
        )
        if int(threads_value) == 0:
            args.extend(["--threads", "0"])

        cmd = [bin_path] + args

        env = os.environ.copy()
        try:
            num_threads = int(threads_value)
            if num_threads > 0:
                env['OMP_NUM_THREADS'] = str(num_threads)
        except:
            pass

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                env=env
            )
            self.set_status("")
            return True
        except subprocess.CalledProcessError as e:
            self.set_status(f"Run error: {e.stderr or e.stdout}")
            return False
        except Exception as e:
            self.set_status(f"Unexpected error: {e}")
            return False

    def update_left_and_plot(self, sweep_mode=False):
        lines = []
        csv_path = self.vars["csv"].get().strip()
        try:
            df = pd.read_csv(csv_path)
            if sweep_mode:
                tail = df.tail(min(200, len(df)))
                lines.append("Last results:")
                for rec in tail.tail(6).to_dict("records"):
                    lines.append(str(rec))
            else:
                last = df.tail(1).to_dict("records")[0]
                for k in [
                    "date",
                    "lang",
                    "cpu_label",
                    "n_samples",
                    "n_features",
                    "iters",
                    "lr",
                    "threads",
                    "time_ms_total",
                    "time_ms_per_iter",
                    "loss_final",
                    "runs",
                ]:
                    if k in last:
                        lines.append(f"{k}: {last[k]}")
        except Exception as e:
            lines.append(f"Cant read results.csv: {e}")

        base = os.path.splitext(self.vars["per_iter"].get().strip())[0]
        files = glob.glob(base + "_run*.csv")

        self.ax.clear()

        if sweep_mode:
            try:
                df = pd.read_csv(csv_path)
                tail = df.tail(min(16, len(df)))
                if "threads" in tail.columns and "time_ms_total" in tail.columns:
                    g = (
                        tail.groupby("threads", as_index=False)["time_ms_total"]
                        .mean()
                        .sort_values("threads")
                    )
                    self.ax.plot(g["threads"], g["time_ms_total"], marker="o")
                    self.ax.set_title("Total time by threads")
                    self.ax.set_xlabel("Threads")
                    self.ax.set_ylabel("Total time, ms")
                    self.ax.grid(True)
                else:
                    self.ax.text(0.5, 0.5, "No data for threads→time", ha="center")
            except Exception as e:
                self.ax.text(0.5, 0.5, f"Error: {e}", ha="center")
        else:
            try:
                runs_needed = int(self.vars["runs"].get())
            except:
                try:
                    df_main = pd.read_csv(csv_path)
                    runs_needed = int(
                        df_main.tail(1).to_dict("records")[0].get("runs", 1)
                    )
                except:
                    runs_needed = 1

            if not files:
                self.ax.text(0.5, 0.5, "There is no per-iter file", ha="center")
            else:
                selected = sorted(files, key=os.path.getmtime, reverse=True)[
                    :runs_needed
                ]
                series_list = []
                used_files = []
                means = []
                for p in selected:
                    try:
                        dfi = pd.read_csv(p)
                        if "iter_idx" in dfi.columns and "time_ms" in dfi.columns:
                            s = dfi.set_index("iter_idx")["time_ms"]
                            means.append(s.mean())
                            series_list.append(s)
                            used_files.append(p)
                        else:
                            continue
                    except Exception:
                        continue

                if not series_list:
                    self.ax.text(
                        0.5,
                        0.5,
                        "per-iter CSVs don't have required columns (iter_idx, time_ms)",
                        ha="center",
                    )
                else:
                    df_concat = pd.concat(series_list, axis=1)
                    mean_series = df_concat.median(axis=1, skipna=True).sort_index()
                    self.ax.plot(
                        mean_series.index, mean_series.values, marker=".", linewidth=1
                    )
                    self.ax.set_title(
                        f"Per-iteration time (mean over {len(used_files)} run(s))"
                    )
                    self.ax.set_xlabel("Iteration")
                    self.ax.set_ylabel("ms")
                    self.ax.grid(True)
                    lines.append(
                        f"Averaged per-iter from {len(used_files)} files (requested {runs_needed})"
                    )
                    lines.append(f"Median per-iter time: {np.median(means)}")

        self.left.delete("1.0", tk.END)
        self.left.insert(tk.END, "\n".join(lines))
        self.canvas.draw()

    def sweep_threads_values(self):
        max_t = os.cpu_count() or 16
        vals = []
        for t in range(1, max_t + 1):
            vals.append(t)

        return sorted(set(vals))

    def run_sweep_threads(self):
        vals = self.sweep_threads_values()
        if not vals:
            self.set_status("Can not define threads")
            return
        ok_any = False
        for t in vals:
            self.set_status(f"Running for threads={t}…")
            self.update_idletasks()
            ok = self.run_gd(threads_override=t)
            ok_any = ok_any or ok
            if not ok:
                self.errs["threads"].config(text=f"Error for threads={t}")
        if ok_any:
            self.set_status("Done")
        self.update_left_and_plot(sweep_mode=True)

    def on_run(self):
        if not self.validate():
            self.set_status("Incorrect values")
            return
        if self.vars["threads"].get().strip() == "-1":
            self.run_sweep_threads()
        else:
            ok = self.run_gd()
            if ok:
                self.update_left_and_plot(sweep_mode=False)


if __name__ == "__main__":
    App().mainloop()