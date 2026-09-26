from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import SGDClassifier
import os, json

app = Flask(__name__)
CORS(app)

ARCHIVO = "memoria.json"

# Cargar o crear cerebro
def cargar_cerebro():
    if os.path.exists(ARCHIVO):
        try:
            with open(ARCHIVO, 'r') as f:
                d = json.load(f)
                return np.array(d['X']), np.array(d['y'])
        except:
            pass
    X = np.array([[80,30,1],[70,60,1],[40,80,0],[30,90,0],[90,20,1],[60,40,1]])
    y = np.array([1,1,0,0,1,1])
    return X, y

def guardar_cerebro(X, y):
    with open(ARCHIVO, 'w') as f:
        json.dump({'X': X.tolist(), 'y': y.tolist()}, f)

X_train, y_train = cargar_cerebro()
modelo = SGDClassifier(loss='log_loss')
modelo.fit(X_train, y_train)

@app.route('/')
def home():
    return jsonify({
        "status": "IA CEREBRO ACTIVO",
        "cerebro": "funcionando",
        "partidos_aprendidos": len(y_train),
        "precision": f"{modelo.score(X_train, y_train)*100:.2f}%"
    })

@app.route('/predecir')
def predecir():
    try:
        ataque = float(request.args.get('ataque', 75))
        defensa = float(request.args.get('defensa', 40))
        localia = float(request.args.get('localia', 1))
        datos = np.array([[ataque, defensa, localia]])
        prob = modelo.predict_proba(datos)[0]
        idx = np.argmax(prob)
        res = "Gana Local" if modelo.predict(datos)[0]==1 else "No Gana Local"
        return jsonify({
            "prediccion": res,
            "confianza": f"{max(prob)*100:.1f}%",
            "probabilidades": {"no_gana": f"{prob[0]*100:.1f}%", "gana": f"{prob[1]*100:.1f}%"}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/aprender', methods=['POST'])
def aprender():
    global X_train, y_train, modelo
    try:
        data = request.get_json()
        nuevo = [float(data['ataque']), float(data['defensa']), float(data['localia'])]
        real = int(data['resultado_real'])

        X_train = np.vstack([X_train, nuevo])
        y_train = np.append(y_train, real)
        modelo.fit(X_train, y_train)
        guardar_cerebro(X_train, y_train)

        return jsonify({
            "mensaje": "CEREBRO MEJORADO",
            "total_datos": len(y_train),
            "nueva_precision": f"{modelo.score(X_train, y_train)*100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/reset', methods=['POST'])
def reset():
    global X_train, y_train, modelo
    if os.path.exists(ARCHIVO):
        os.remove(ARCHIVO)
    X_train, y_train = cargar_cerebro()
    modelo.fit(X_train, y_train)
    return jsonify({"mensaje": "Cerebro reseteado"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
