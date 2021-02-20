from flask import Flask, render_template, request, redirect, jsonify
import predictor
import os

app = Flask(__name__)

# ------------------------- POST ROUTES ----------------------------
@app.route('/guess/person-type', methods=['POST'])
def post_person_type():
    json_data = request.get_json(force=True)
    print(json_data)
    clima = json_data['clima']
    ambiente = json_data['ambiente']
    agua = json_data['agua']
    zona = json_data['zona']
    distancia = json_data['distancia']

    response = {
        "tipo": predictor.guessPersonType(clima=clima, ambiente=ambiente, agua=agua, zona=zona, distancia=distancia)
    }

    return jsonify(response)

if __name__ == "__main__":
    # get port from environment, default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)