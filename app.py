from flask import Flask, render_template, request, redirect
import predictor
import os

app = Flask(__name__)

# ------------------------- POST ROUTES ----------------------------
@app.route('/guess/person-type', methods=['POST'])
def post_style_grade_sex():
    sex = request.form['sex']
    style = request.form['style']
    grade = str(float(request.form['grade']))

    return predictor.guessHeadquarter(style=style, grade=grade, sex=sex)

if __name__ == "__main__":
    # get port from environment, default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)