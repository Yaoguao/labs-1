# pip install flask
from flask import Flask, request, render_template, jsonify

from calculator import Calculator


app = Flask(__name__)

CALCULATOR = None
CALCULATOR2 = None
CALCULATOR3 = None
CALCULATOR4 = None  # Добавляем переменную для калькулятора 4

# Главная страница
@app.route("/")
def main():
    return render_template("main.html")


# инициализация калькулятора со случайными значениями
def init_calculator():
    global CALCULATOR
    CALCULATOR = Calculator()
    return CALCULATOR.parameters


# обновление параметров калькулятора из данных формы
def update_calculator_from_form(form_data):
    global CALCULATOR
    # Инициализируем калькулятор, если он еще не создан
    if CALCULATOR is None:
        init_calculator()

    CALCULATOR.parameters = {}
    for key, value in form_data.items():
        if value and value != "":
            try:
                if "." in str(value):
                    CALCULATOR.parameters[key] = float(value)
                else:
                    CALCULATOR.parameters[key] = int(value)
            except ValueError:
                CALCULATOR.parameters[key] = value
    
    # Убедимся, что все необходимые X параметры присутствуют
    for i in range(1, 12):
        param_name = f"X{i}"
        if param_name not in CALCULATOR.parameters:
            # Если параметр отсутствует, установим значение по умолчанию
            CALCULATOR.parameters[param_name] = 0.5
        
        # Убедимся, что присутствуют min/max значения
        min_name = f"{param_name}_min"
        max_name = f"{param_name}_max"
        
        if min_name not in CALCULATOR.parameters:
            CALCULATOR.parameters[min_name] = 0.01
        
        if max_name not in CALCULATOR.parameters:
            CALCULATOR.parameters[max_name] = 1.0
    
    # Убедимся, что присутствуют факторы F1-F7
    for i in range(1, 8):
        factor_name = f"F{i}"
        if factor_name not in CALCULATOR.parameters:
            # Значения по умолчанию для факторов
            defaults = {'F1': 100, 'F2': 50, 'F3': 5, 'F4': 20, 'F5': 1, 'F6': 1000, 'F7': 10}
            CALCULATOR.parameters[factor_name] = defaults.get(factor_name, 1)
    
    return CALCULATOR.parameters


# главная страница и внедрение модели калькулятора
@app.route("/lab1")
def index():
    global CALCULATOR
    if CALCULATOR is None:
        parameters = init_calculator()
    else:
        parameters = CALCULATOR.parameters
    return render_template("index.html", parameters=parameters)


# очистка всех значений
@app.route("/clear", methods=["POST"])
def clear_values():
    global CALCULATOR
    CALCULATOR = None
    return jsonify({"status": "success"})


# случайные значения
@app.route("/random", methods=["POST"])
def random_values():
    parameters = init_calculator()
    return jsonify({"status": "success", "parameters": parameters})


# построение диаграмм
@app.route("/plot", methods=["POST"])
def plot_diagrams():
    global CALCULATOR
    form_data = request.json

    # проверка на заполненность всех полей
    empty_fields = []
    for key, value in form_data.items():
        if value == "":
            empty_fields.append(key)
    if empty_fields:
        return jsonify(
            {
                "status": "error",
                "message": "Не все поля заполнены",
                "empty_fields": empty_fields,
            }
        )

    update_calculator_from_form(form_data)

    # создаем картинки диаграмм и выводим их
    try:
        time_series_b64, time_series_detailed_b64, radar_b64 = CALCULATOR.plot_all_results()
        return jsonify(
            {
                "status": "success",
                "time_series": time_series_b64,
                "time_series_detailed": time_series_detailed_b64,
                "radar_charts": radar_b64,
            }
        )
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Ошибка при построении графиков: {str(e)}"}
        )


# инициализация калькулятора2 со случайными значениями
def init_calculator2():
    global CALCULATOR2
    CALCULATOR2 = Calculator2()
    return CALCULATOR2.parameters


# обновление параметров калькулятора2 из данных формы
def update_calculator2_from_form(form_data):
    global CALCULATOR2
    # Инициализируем калькулятор, если он еще не создан
    if CALCULATOR2 is None:
        init_calculator2()

    CALCULATOR2.parameters = {}
    for key, value in form_data.items():
        if value and value != "":
            try:
                if "." in value:
                    CALCULATOR2.parameters[key] = float(value)
                else:
                    CALCULATOR2.parameters[key] = int(value)
            except ValueError:
                CALCULATOR2.parameters[key] = value

    # Убедимся, что все необходимые X параметры присутствуют
    for i in range(1, 19):
        param_name = f"X{i}"
        if param_name not in CALCULATOR2.parameters:
            # Если параметр отсутствует, установим значение по умолчанию
            min_val, max_val, _ = CALCULATOR2.pR[param_name]
            CALCULATOR2.parameters[param_name] = (min_val + max_val) / 2

    # Также убедимся, что присутствуют min/max значения
    for i in range(1, 19):
        param_name = f"X{i}"
        min_name = f"{param_name}_min"
        max_name = f"{param_name}_max"

        if min_name not in CALCULATOR2.parameters:
            min_val, max_val, _ = CALCULATOR2.pR[param_name]
            CALCULATOR2.parameters[min_name] = min_val

        if max_name not in CALCULATOR2.parameters:
            min_val, max_val, _ = CALCULATOR2.pR[param_name]
            CALCULATOR2.parameters[max_name] = max_val

    # Обновляем нормализованные параметры
    CALCULATOR2.parameters_norm = CALCULATOR2._get_normalized(CALCULATOR2.parameters)
    return CALCULATOR2.parameters


# главная страница и внедрение модели калькулятора2
@app.route("/lab2")
def index2():
    global CALCULATOR2
    if CALCULATOR2 is None:
        parameters = init_calculator2()
    else:
        parameters = CALCULATOR2.parameters
    return render_template("index2.html", parameters=parameters)


# очистка всех значений
@app.route("/clear2", methods=["POST"])
def clear_values2():
    global CALCULATOR2
    CALCULATOR2 = None
    return jsonify({"status": "success"})


# случайные значения
@app.route("/random2", methods=["POST"])
def random_values2():
    parameters = init_calculator2()
    return jsonify({"status": "success", "parameters": parameters})


# построение диаграмм
@app.route("/plot2", methods=["POST"])
def plot_diagrams2():
    global CALCULATOR2
    form_data = request.json

    # проверка на заполненность всех полей
    empty_fields = []
    for key, value in form_data.items():
        if value == "":
            empty_fields.append(key)
    if empty_fields:
        return jsonify(
            {
                "status": "error",
                "message": "Не все поля заполнены",
                "empty_fields": empty_fields,
            }
        )

    # обновляем калькулятор из формы
    update_calculator2_from_form(form_data)

    # создаем картинки диаграмм и выводим их
    try:
        # Пересчитываем решение системы с новыми параметрами
        CALCULATOR2.solve_system()

        time_series_b64 = CALCULATOR2.plot_time_series()
        radar_b64 = CALCULATOR2.plot_radar_charts()

        return jsonify(
            {
                "status": "success",
                "time_series": time_series_b64,
                "radar_charts": radar_b64,
            }
        )
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Ошибка при построении графиков: {str(e)}"}
        )


def init_calculator3():
    global CALCULATOR3
    CALCULATOR3 = Calculator3()
    return CALCULATOR3.parameters

def update_calculator3_from_form(form_data):
    global CALCULATOR3
    if CALCULATOR3 is None:
        init_calculator3()

    CALCULATOR3.parameters = {}
    for key, value in form_data.items():
        if value and value != "":
            try:
                if "." in value:
                    CALCULATOR3.parameters[key] = float(value)
                else:
                    CALCULATOR3.parameters[key] = int(value)
            except ValueError:
                CALCULATOR3.parameters[key] = value

    # Убедимся, что все необходимые K параметры присутствуют
    for i in range(1, 16):
        param_name = f"K{i}"
        if param_name not in CALCULATOR3.parameters:
            min_val, max_val = CALCULATOR3.pR[param_name]
            CALCULATOR3.parameters[param_name] = (min_val + max_val) / 2

    # Также убедимся, что присутствуют max значения
    for i in range(1, 16):
        param_name = f"K{i}"
        max_name = f"{param_name}_max"
        if max_name not in CALCULATOR3.parameters:
            min_val, max_val = CALCULATOR3.pR[param_name]
            CALCULATOR3.parameters[max_name] = max_val

    # Убедимся, что коэффициенты возмущений R1-R4 присутствуют
    for i in range(1, 5):
        for coef in ['a3', 'a2', 'a1', 'a0']:
            param_name = f"R{i}_{coef}"
            if param_name not in CALCULATOR3.parameters:
                # Значения по умолчанию для коэффициентов
                default_values = {'a3': 0.0, 'a2': 0.0, 'a1': 0.5, 'a0': 0.2}
                CALCULATOR3.parameters[param_name] = default_values[coef]

    # Обновляем нормализованные параметры
    CALCULATOR3.parameters_norm = CALCULATOR3._get_normalized(CALCULATOR3.parameters)
    return CALCULATOR3.parameters

@app.route("/lab3")
def index3():
    global CALCULATOR3
    if CALCULATOR3 is None:
        parameters = init_calculator3()
    else:
        parameters = CALCULATOR3.parameters

    names = CALCULATOR3.names if CALCULATOR3 else {}
    return render_template("index3.html", parameters=parameters, names=names)

@app.route("/clear3", methods=["POST"])
def clear_values3():
    global CALCULATOR3
    CALCULATOR3 = None
    return jsonify({"status": "success"})

@app.route("/random3", methods=["POST"])
def random_values3():
    parameters = init_calculator3()
    return jsonify({"status": "success", "parameters": parameters})

@app.route("/plot3", methods=["POST"])
def plot_diagrams3():
    global CALCULATOR3
    form_data = request.json

    empty_fields = []
    for key, value in form_data.items():
        if value == "":
            empty_fields.append(key)
    if empty_fields:
        return jsonify({
            "status": "error",
            "message": "Не все поля заполнены",
            "empty_fields": empty_fields,
        })

    update_calculator3_from_form(form_data)

    try:
        CALCULATOR3.solve_system()
        time_series_b64 = CALCULATOR3.plot_time_series()
        radar_b64 = CALCULATOR3.plot_radar_charts()
        disturbances_b64 = CALCULATOR3.plot_disturbances()  # Новый график

        return jsonify({
            "status": "success",
            "time_series": time_series_b64,
            "radar_charts": radar_b64,
            "disturbances": disturbances_b64  # Добавляем в ответ
        })
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Ошибка при построении графиков: {str(e)}"}
        )


# ========== КАЛЬКУЛЯТОР 4: Эффективность ПО хранилища данных ==========

def init_calculator4():
    global CALCULATOR4
    CALCULATOR4 = Calculator4()
    return CALCULATOR4.parameters

def update_calculator4_from_form(form_data):
    global CALCULATOR4
    if CALCULATOR4 is None:
        init_calculator4()

    CALCULATOR4.parameters = {}
    for key, value in form_data.items():
        if value and value != "":
            try:
                if "." in value:
                    CALCULATOR4.parameters[key] = float(value)
                else:
                    CALCULATOR4.parameters[key] = int(value)
            except ValueError:
                CALCULATOR4.parameters[key] = value

    # Убедимся, что все необходимые X параметры присутствуют
    for i in range(1, 15):
        param_name = f"X{i}"
        if param_name not in CALCULATOR4.parameters:
            min_val, max_val = CALCULATOR4.pR[param_name]
            CALCULATOR4.parameters[param_name] = (min_val + max_val) / 2

    # Также убедимся, что присутствуют max значения
    for i in range(1, 15):
        param_name = f"X{i}"
        max_name = f"{param_name}_max"
        if max_name not in CALCULATOR4.parameters:
            min_val, max_val = CALCULATOR4.pR[param_name]
            CALCULATOR4.parameters[max_name] = max_val

    # Убедимся, что коэффициенты полиномов f1-f160 присутствуют
    for i in range(1, 161):
        for coef in ['a3', 'a2', 'a1', 'a0']:
            param_name = f"f{i}_{coef}"
            if param_name not in CALCULATOR4.parameters:
                min_val, max_val = CALCULATOR4.coeff_range[coef]
                CALCULATOR4.parameters[param_name] = (min_val + max_val) / 2

    # Убедимся, что коэффициенты возмущений ζ1-ζ5 присутствуют
    for i in range(1, 6):
        for coef in ['a3', 'a2', 'a1', 'a0']:
            param_name = f"zeta{i}_{coef}"
            if param_name not in CALCULATOR4.parameters:
                min_val, max_val = CALCULATOR4.coeff_range[coef]
                CALCULATOR4.parameters[param_name] = (min_val + max_val) / 2

    # Обновляем нормализованные параметры
    CALCULATOR4.parameters_norm = CALCULATOR4._get_normalized(CALCULATOR4.parameters)
    return CALCULATOR4.parameters

@app.route("/lab4")
def index4():
    global CALCULATOR4
    if CALCULATOR4 is None:
        parameters = init_calculator4()
    else:
        parameters = CALCULATOR4.parameters

    names = CALCULATOR4.names if CALCULATOR4 else {}
    disturbance_names = CALCULATOR4.disturbance_names if CALCULATOR4 else {}
    return render_template("index4.html",
                           parameters=parameters,
                           names=names,
                           disturbance_names=disturbance_names)

@app.route("/clear4", methods=["POST"])
def clear_values4():
    global CALCULATOR4
    CALCULATOR4 = None
    return jsonify({"status": "success"})

@app.route("/random4", methods=["POST"])
def random_values4():
    parameters = init_calculator4()
    return jsonify({"status": "success", "parameters": parameters})

@app.route("/plot4", methods=["POST"])
def plot_diagrams4():
    global CALCULATOR4
    form_data = request.json

    empty_fields = []
    for key, value in form_data.items():
        if value == "":
            empty_fields.append(key)
    if empty_fields:
        return jsonify({
            "status": "error",
            "message": "Не все поля заполнены",
            "empty_fields": empty_fields,
        })

    update_calculator4_from_form(form_data)

    try:
        CALCULATOR4.solve_system()
        time_series_b64 = CALCULATOR4.plot_time_series()
        radar_b64 = CALCULATOR4.plot_radar_charts()
        # disturbances_b64 = CALCULATOR4.plot_disturbances()  # Опционально

        return jsonify({
            "status": "success",
            "time_series": time_series_b64,
            "radar_charts": radar_b64,
            # "disturbances": disturbances_b64  # Если нужно
        })
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Ошибка при построении графиков: {str(e)}"}
        )


if __name__ == "__main__":
    app.run(port=8000, debug=True)