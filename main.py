from flask import Flask, send_file, render_template, request
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import csv
from numpy import array

app = Flask(__name__)

fig, ax=plt.subplots(figsize=(10, 10))
ax=sns.set_style(style="darkgrid")


@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')

@app.route('/data', methods=['GET', 'POST'])
def data():

    if request.method == 'POST':
        f = request.form['csvfile']
        x_arr, y_arr = np.array(list()), np.array(list())
        x_arr = []
        y_arr = []
        with open(f, 'r') as file:
            csvfile = csv.reader(file)
            for line in csvfile:
                t = ''.join(line).split(';')
                x_arr = np.append(x_arr, t[0])
                y_arr = np.append(y_arr, t[1])

                x_arr_int = x_arr.astype('f')
                y_arr_int=y_arr.astype('f')

            data_graph = pd.DataFrame({"X": x_arr_int, "Y": y_arr_int})

            sns.lineplot(x="X", y="Y", data=data_graph)


            return render_template('data.html')




@app.route('/visualize')
def visualize():
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='img/png')


if __name__ == '__main__':
    app.run(debug=True)
