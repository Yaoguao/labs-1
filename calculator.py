# pip install numpy scipy matplotlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
import os
import io
import base64

class Calculator:
    def __init__(self):
        # все параметры системы (словарь)
        self.parameters = self.generate_system_parameters()
        # временные точки от 0 до 1 с шагом 0.01
        self.time_points = np.linspace(0, 1, 100)
        # переменная для решения системы
        self.solution = None

    # генерация словаря случайных значений параметров системы   
    def generate_system_parameters(self):
        parameters = {}
        
        # параметры X1-X11 (от 0.01 до 1.00)
        for i in range(1, 12):
            param_name = f"X{i}"
            parameters[param_name] = round(random.uniform(0.01, 1.0), 2)
            parameters[f"{param_name}_min"] = round(random.uniform(0.01, parameters[param_name]), 2)
            parameters[f"{param_name}_max"] = round(random.uniform(parameters[param_name], 1.0), 2)
        
        # факторы внешней среды F1-F7
        factors = {
            'F1': (10, 1000),  # общее количество выброшенных при аварии химически опасных веществ на объекте
            'F2': (10, 500),   # количество персонала на химически опасном объекте
            'F3': (1, 20),     # скорость ветра, м/с
            'F4': (-20, 40),   # температура воздуха, °C
            'F5': (0.1, 10),   # время до начала оповещения, ч
            'F6': (100, 100000),  # численность населения
            'F7': (1, 100),    # количество убежищ
        }
        
        for factor_name, (min_val, max_val) in factors.items():
            if factor_name in ['F1', 'F2', 'F6', 'F7']:
                parameters[factor_name] = random.randint(min_val, max_val)
            else:
                parameters[factor_name] = round(random.uniform(min_val, max_val), 2)
        
        return parameters
    
    # Константы для функций f1-f11
    def get_k_constants(self):
        return {
            'k1+': 0.0186,
            'k1-': 0.01,
            'k2+': 0.085,
            'k2-': -0.95,
            'k3+': -3.58e-6,
            'k4+': 0.9,
            'k4-': -0.0322,
            'k5+': 0.0145,
            'k6+': 0.002,
            'k6-': -0.606,
            'k7+': 0.95,
            'k8+': 0.817,
            'k8-': -0.42,
            'k9+': 1.4,
            'k10+': 0.052,
            'k11+': 375.1,
            'k11-': -0.0018,
        }
    
    # Вычисление функций f1+ - f11-
    def f1_plus(self, X2, F1):
        k = self.get_k_constants()['k1+']
        return k * X2 * (F1 ** 0.8)
    
    def f1_minus(self, X9, X10, F3, F4):
        k = self.get_k_constants()['k1-']
        # Избегаем деления на ноль
        denominator = max(F3 * F4, 1e-6)
        return k * X9 * X10 / denominator
    
    def f2_plus(self, X3, X7, X8, F1, F5):
        k = self.get_k_constants()['k2+']
        # Избегаем деления на ноль
        F5_safe = max(F5, 1e-6)
        return k * X3 * X7 * X8 * (F1 ** 0.8) / F5_safe
    
    def f2_minus(self, X9, X10):
        k = self.get_k_constants()['k2-']
        return k * X9 * X10
    
    def f3_plus(self, X1, F1, F3, F4):
        k = self.get_k_constants()['k3+']
        # Избегаем деления на ноль
        X1_safe = max(abs(X1), 1e-6)
        return k * (F1 ** 0.8) * F3 * F4 / (X1_safe ** 2.9)
    
    def f4_plus(self, X1):
        k = self.get_k_constants()['k4+']
        return k * X1
    
    def f4_minus(self, F1, F3, F4):
        k = self.get_k_constants()['k4-']
        return k * (F1 ** 0.8) * F3 * F4    
    def f5_plus(self, X1, F2):
        k = self.get_k_constants()['k5+']
        # Избегаем деления на ноль
        X1_safe = max(abs(X1), 1e-6)
        return k * F2 / (X1_safe ** 2.1)
    
    def f6_plus(self, X2, F5, F6):
        k = self.get_k_constants()['k6+']
        return k * X2 * F5 * F6
    
    def f6_minus(self, X4, X10, X11, F7):
        k = self.get_k_constants()['k6-']
        # Избегаем деления на ноль
        denominator = max(X10 * X11 * F7, 1e-6)
        return k * X4 / denominator
    
    def f7_plus(self, X5, X6, X10):
        k = self.get_k_constants()['k7+']
        sum_x5_x6 = max(X5 + X6, 1e-6)
        return k * (sum_x5_x6 ** 0.6) * X10
    
    def f8_plus(self, X3):
        k = self.get_k_constants()['k8+']
        return k * X3
    
    def f8_minus(self, X9, X10):
        k = self.get_k_constants()['k8-']
        return k * X9 * X10
    
    def f9_plus(self, X3, X8):
        k = self.get_k_constants()['k9+']
        product = max(X3 * X8, 1e-6)
        return k * (product ** 0.2)
    
    def f10_plus(self, X3, F6):
        k = self.get_k_constants()['k10+']
        return k * np.sqrt(X3 * F6)
    
    def f11_plus(self, X10, F6, F7):
        k = self.get_k_constants()['k11+']
        # Избегаем деления на ноль
        F6_safe = max(F6, 1e-6)
        return k * X10 * F7 / F6_safe
    
    def f11_minus(self, F5):
        k = self.get_k_constants()['k11-']
        # Избегаем деления на ноль
        F5_safe = max(F5, 1e-6)
        return k / F5_safe
    
    # система дифференциальных уравнений
    def system_equations(self, t, X):
        try:
            dXdt = np.zeros(11)
            params = self.parameters
            
            # ограничиваем значения X от X_min до X_max для каждого параметра
            for i in range(11):
                X_min = params.get(f"X{i+1}_min", 0.01)
                X_max = params.get(f"X{i+1}_max", 1.0)
                X[i] = np.clip(X[i], X_min, X_max)
            
            # Получаем факторы
            F1 = params.get('F1', 100)
            F2 = params.get('F2', 50)
            F3 = params.get('F3', 5)
            F4 = params.get('F4', 20)
            F5 = params.get('F5', 1)
            F6 = params.get('F6', 1000)
            F7 = params.get('F7', 10)
            
            # Получаем максимальные значения для нормализации
            X_max = [params.get(f"X{i}_max", 1.0) for i in range(1, 12)]
            
            # dX1/dt = (1/X1_max) * (f1+ - f1-)
            dXdt[0] = (1.0 / max(X_max[0], 1e-6)) * (
                self.f1_plus(X[1], F1) - 
                self.f1_minus(X[8], X[9], F3, F4)
            )
            
            # dX2/dt = (1/X2_max) * (f2+ - f2-)
            dXdt[1] = (1.0 / max(X_max[1], 1e-6)) * (
                self.f2_plus(X[2], X[6], X[7], F1, F5) - 
                self.f2_minus(X[8], X[9])
            )
            
            # dX3/dt = (1/X3_max) * f3+
            dXdt[2] = (1.0 / max(X_max[2], 1e-6)) * (
                self.f3_plus(X[0], F1, F3, F4)
            )
            
            # dX4/dt = (1/X4_max) * (f4+ - f4-)
            dXdt[3] = (1.0 / max(X_max[3], 1e-6)) * (
                self.f4_plus(X[0]) - 
                self.f4_minus(F1, F3, F4)
            )
            
            # dX5/dt = (1/X5_max) * f5+
            dXdt[4] = (1.0 / max(X_max[4], 1e-6)) * (
                self.f5_plus(X[0], F2)
            )
            
            # dX6/dt = (1/X6_max) * (f6+ - f6-)
            dXdt[5] = (1.0 / max(X_max[5], 1e-6)) * (
                self.f6_plus(X[1], F5, F6) - 
                self.f6_minus(X[3], X[9], X[10], F7)
            )
            
            # dX7/dt = (1/X7_max) * f7+
            dXdt[6] = (1.0 / max(X_max[6], 1e-6)) * (
                self.f7_plus(X[4], X[5], X[9])
            )
            
            # dX8/dt = (1/X8_max) * (f8+ - f8-)
            # Примечание: в формуле указано f8+(X5, X6, X11, X13), но X13 нет в списке параметров
            # Используем f8+ = k8+ * X3 и f8- = k8- * X9 * X10
            dXdt[7] = (1.0 / max(X_max[7], 1e-6)) * (
                self.f8_plus(X[2]) - 
                self.f8_minus(X[8], X[9])
            )
            
            # dX9/dt = (1/X9_max) * (f9+ - f8-)
            # Примечание: в формуле указано f9+ - f9-, но f9- не определена, используем f8-
            dXdt[8] = (1.0 / max(X_max[8], 1e-6)) * (
                self.f9_plus(X[2], X[7])
            )
            
            # dX10/dt = (1/X10_max) * f10+
            dXdt[9] = (1.0 / max(X_max[9], 1e-6)) * (
                self.f10_plus(X[2], F6)
            )
            
            # dX11/dt = (1/X11_max) * (f11+ - f11-)
            dXdt[10] = (1.0 / max(X_max[10], 1e-6)) * (
                self.f11_plus(X[9], F6, F7) - 
                self.f11_minus(F5)
            )
            
            # ограничиваем производные для стабильности
            dXdt = np.clip(dXdt, -0.5, 0.5)
            
            return dXdt
            
        except Exception as e:
            print(f"Ошибка в вычислении производных: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(11)
    
    # решение системы дифференциальных уравнений
    def solve_system(self):
        try:
            # начальные условия из параметров, список значений Х1-Х11 на момент времени 0
            X0 = [self.parameters[f"X{i}"] for i in range(1, 12)]
            
            # решение системы
            self.solution = solve_ivp(
                self.system_equations, 
                [0, 1], 
                X0, 
                t_eval=self.time_points, 
                method='RK45',
                rtol=1e-3,
                atol=1e-3
            )
            
            # ограничиваем значения от X_min до X_max для каждого параметра
            for i in range(11):
                X_min = self.parameters.get(f"X{i+1}_min", 0.01)
                X_max = self.parameters.get(f"X{i+1}_max", 1.0)
                self.solution.y[i] = np.clip(self.solution.y[i], X_min, X_max)
            
            return self.solution
            
        except Exception as e:
            print(f"Ошибка при решении системы: {e}")
            import traceback
            traceback.print_exc()
            # Возвращаем фиктивное решение для отладки
            self.solution = type('obj', (object,), {
                't': self.time_points,
                'y': np.random.uniform(0, 1, (11, len(self.time_points)))
            })
            return self.solution
        
    # отрисовка общего графика изменений Х1-Х11 по времени (все на одном графике)
    def plot_time_series(self):
        if self.solution is None:
            self.solve_system()        
        fig = plt.figure(figsize=(12, 8))        
        for i in range(11):
            plt.plot(self.solution.t, self.solution.y[i], label=f'X{i+1}', linewidth=2)        
        plt.xlabel('Время', fontsize=12)
        plt.ylabel('Значение параметра', fontsize=12)
        plt.title('Изменение параметров X1-X11 по времени (общий график)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Устанавливаем общий диапазон для оси Y на основе всех min и max
        all_mins = [self.parameters.get(f"X{i+1}_min", 0.01) for i in range(11)]
        all_maxs = [self.parameters.get(f"X{i+1}_max", 1.0) for i in range(11)]
        global_min = min(all_mins)
        global_max = max(all_maxs)
        y_range = global_max - global_min
        y_padding = y_range * 0.1 if y_range > 0 else 0.1
        plt.ylim(global_min - y_padding, global_max + y_padding)
        
        plt.tight_layout()

        # сохраняем в буфер
        time_series_buffer = io.BytesIO()
        plt.savefig(time_series_buffer, format='png', dpi=100, bbox_inches='tight')
        time_series_buffer.seek(0)
        time_series_b64 = base64.b64encode(time_series_buffer.getvalue()).decode()
        plt.close(fig)
        return time_series_b64
    
    # отрисовка детальных графиков изменений Х1-Х11 по времени (отдельный subplot для каждого)
    def plot_time_series_detailed(self):
        if self.solution is None:
            self.solve_system()
        
        # Создаем subplot для каждого параметра, чтобы каждый имел свой масштаб
        fig, axes = plt.subplots(11, 1, figsize=(14, 20))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Получаем min и max для всех параметров
        all_mins = []
        all_maxs = []
        for i in range(11):
            X_min = self.parameters.get(f"X{i+1}_min", 0.01)
            X_max = self.parameters.get(f"X{i+1}_max", 1.0)
            all_mins.append(X_min)
            all_maxs.append(X_max)
        
        # Рисуем каждый параметр на своем subplot с реальными значениями
        for i in range(11):
            ax = axes[i]
            X_min = all_mins[i]
            X_max = all_maxs[i]
            
            # Рисуем основную линию с реальными значениями
            ax.plot(self.solution.t, self.solution.y[i], label=f'X{i+1}', linewidth=2, color=f'C{i}')
            
            # Рисуем линии min и max
            ax.axhline(y=X_min, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Min' if i == 0 else '')
            ax.axhline(y=X_max, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Max' if i == 0 else '')
            
            # Устанавливаем пределы осей с небольшим запасом
            y_range = X_max - X_min
            y_padding = y_range * 0.1 if y_range > 0 else 0.1
            ax.set_ylim(X_min - y_padding, X_max + y_padding)
            
            # Настройки осей
            ax.set_ylabel(f'X{i+1}\n[{X_min:.2f}-{X_max:.2f}]', fontsize=9, rotation=0, labelpad=30)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # Подписи времени только на последнем графике
            if i == 10:
                ax.set_xlabel('Время', fontsize=12)
            else:
                ax.set_xticklabels([])
        
        plt.suptitle('Изменение параметров X1-X11 по времени (детальный вид)\n(Красная линия - минимум, Зеленая - максимум)', 
                     fontsize=14, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # сохраняем в буфер
        time_series_detailed_buffer = io.BytesIO()
        plt.savefig(time_series_detailed_buffer, format='png', dpi=100, bbox_inches='tight')
        time_series_detailed_buffer.seek(0)
        time_series_detailed_b64 = base64.b64encode(time_series_detailed_buffer.getvalue()).decode()
        plt.close(fig)
        return time_series_detailed_b64
    
    # отрисовка 5 лепестковых диаграмм
    def plot_radar_charts(self):
        if self.solution is None:
            self.solve_system()
        
        time_points = [0, 0.25, 0.5, 0.75, 1]
        time_indices = [np.abs(self.solution.t - t).argmin() for t in time_points]
        categories = [f'X{i+1}' for i in range(11)]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
        axes = axes.flatten()
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
        
        # Получаем min и max для нормализации
        X_mins = [self.parameters.get(f"X{i+1}_min", 0.01) for i in range(11)]
        X_maxs = [self.parameters.get(f"X{i+1}_max", 1.0) for i in range(11)]
        
        for i, (t_idx, ax) in enumerate(zip(time_indices, axes)):
            if i >= len(time_points):
                break
            
            # Получаем реальные значения
            real_values = self.solution.y[:, t_idx]
            
            # Нормализуем значения для отображения (0-1), но сохраняем реальные для меток
            normalized_values = []
            for j in range(11):
                X_min = X_mins[j]
                X_max = X_maxs[j]
                X_range = X_max - X_min
                if X_range > 0:
                    # Нормализуем: (value - min) / (max - min)
                    normalized = (real_values[j] - X_min) / X_range
                else:
                    normalized = 0.5
                normalized_values.append(normalized)
            
            normalized_values += normalized_values[:1]
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(angles, normalized_values, color=colors[i], linewidth=2, linestyle='solid')
            ax.fill(angles, normalized_values, color=colors[i], alpha=0.25)
            ax.set_xticks(angles[:-1])
            
            # Создаем метки с реальными значениями
            labels = []
            for j in range(11):
                labels.append(f'X{j+1}\n({real_values[j]:.2f})')
            ax.set_xticklabels(labels, fontsize=7)
            
            ax.set_rlabel_position(0)
            # Показываем нормализованные значения на радиальной оси
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
            ax.set_ylim(0, 1)
            ax.set_title(f'Время t = {self.solution.t[t_idx]:.2f}', size=11, color=colors[i], pad=10)
        
        for i in range(len(time_points), len(axes)):
            fig.delaxes(axes[i])
        plt.suptitle('Лепестковые диаграммы параметров X1-X11 (нормализованные относительно min-max)', fontsize=14)
        plt.tight_layout()
        radar_buffer = io.BytesIO()
        plt.savefig(radar_buffer, format='png', dpi=100, bbox_inches='tight')
        radar_buffer.seek(0)
        radar_b64 = base64.b64encode(radar_buffer.getvalue()).decode()
        plt.close(fig)
        return radar_b64
    
    # построение всех графиков
    def plot_all_results(self):
        self.solve_system()
        time_series_b64 = self.plot_time_series()
        time_series_detailed_b64 = self.plot_time_series_detailed()
        radar_b64 = self.plot_radar_charts()
        return [time_series_b64, time_series_detailed_b64, radar_b64]
