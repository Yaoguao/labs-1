# pip install flask
from flask import Flask, request, render_template, jsonify

from calculator import Calculator


app = Flask(__name__)

CALCULATOR = None


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


# Главная страница - lab1 (index.html)
@app.route("/")
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


if __name__ == "__main__":
    app.run(port=8000, debug=True)