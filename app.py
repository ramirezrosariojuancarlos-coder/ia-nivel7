from flask import Flask, jsonify, request
from flask_cors import CORS
import math, os, json

app = Flask(__name__)
CORS(app)

ARCHIVO = "memoria.json"

def sigmoide(x):
    return 1 / (1 + math.exp(-x))

def cargar():
    if os.path.exists(ARCHIVO):
        try:
            with open(ARCHIVO,'r') as f:
                return json.load(f)
        except: pass
    return {"pesos": [0.05, -0.05, 0.3], "bias": 0, "total": 6}

mem = cargar()

@app.route('/')
def home():
    return jsonify({"status": "IA CEREBRO ACTIVO - SIN FALLAS", "aprendidos": mem["total"], "pesos": mem["pesos"]})

@app.route('/predecir')
def predecir():
    ataque = float(request.args.get('ataque', 75))
    defensa = float(request.args.get('defensa', 40))
    localia = float(request.args.get('localia', 1))
    w = mem["pesos"]
    z = w[0]*ataque + w[1]*defensa + w[2]*localia + mem["bias"]
    prob = sigmoide(z) * 100
    pred = "Gana Local" if prob > 50 else "No Gana Local"
    return jsonify({"prediccion": pred, "confianza": f"{prob:.1f}%"})

@app.route('/aprender', methods=['POST'])
def aprender():
    global mem
    data = request.get_json()
    ataque = float(data['ataque'])
    defensa = float(data['defensa'])
    localia = float(data['localia'])
    real = int(data['resultado_real'])

    w = mem["pesos"]
    z = w[0]*ataque + w[1]*defensa + w[2]*localia + mem["bias"]
    prob = sigmoide(z)
    error = real - prob

    lr = 0.01
    w[0] += lr * error * ataque
    w[1] += lr * error * defensa
    w[2] += lr * error * localia
    mem["bias"] += lr * error
    mem["total"] += 1
    mem["pesos"] = w

    with open(ARCHIVO,'w') as f:
        json.dump(mem, f)

    return jsonify({"mensaje": "IA MEJORADA", "nuevo_peso": w, "total": mem["total"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
