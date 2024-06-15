import pickle

import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pandas.core.indexes.numeric':
            from pandas import Index
            return Index
        return super().find_class(module, name)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return CustomUnpickler(f).load()


popular_df = load_pickle('popular_df.pkl')
df = load_pickle('df.pkl')
pt = load_pickle('pt.pkl')
similarity_scores = load_pickle('similarity_scores.pkl')
model = joblib.load('xgboost_model.joblib')


@app.route('/')
def index():
    return render_template('index.html',
                           name=list(popular_df['name'].values),
                           ram=list(popular_df['ram'].values),
                           image=list(popular_df['img_link'].values),
                           price=list(popular_df['price(in Rs.)'].values),
                           rating=list(popular_df['rating'].values),
                           storage=list(popular_df['storage'].values),
                           os=list(popular_df['os'].values),
                           processor=list(popular_df['processor'].values),
                           display_size=list(popular_df['display(in inch)'].values))


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_laptop', methods=['POST'])
def recommend():
    user_input = request.form.get('laptop')

    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = df[df['name'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('name')['name'].values))
        item.extend(list(temp_df.drop_duplicates('name')['ram'].values))
        item.extend(list(temp_df.drop_duplicates('name')['img_link'].values))
        item.extend(list(temp_df.drop_duplicates('name')['price(in Rs.)'].values))
        item.extend(list(temp_df.drop_duplicates('name')['no_of_ratings'].values))
        item.extend(list(temp_df.drop_duplicates('name')['rating'].values))
        item.extend(list(temp_df.drop_duplicates('name')['storage'].values))
        item.extend(list(temp_df.drop_duplicates('name')['os'].values))
        item.extend(list(temp_df.drop_duplicates('name')['processor'].values))
        item.extend(list(temp_df.drop_duplicates('name')['display(in inch)'].values))

        data.append(item)
    return render_template('recommend.html', data=data)


@app.route("/price_predc")
def prediction_ui():
    return render_template("price_predc.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        company_name = request.form["company_nm"]
        hp, acer, aus, dell, lenevo, others = [0] * 6
        if company_name == 'hp':
            hp = 1
        elif company_name == 'acer':
            acer = 1
        elif company_name == 'aus':
            aus = 1
        elif company_name == 'dell':
            dell = 1
        elif company_name == 'lenevo':
            lenevo = 1
        elif company_name == 'others':
            others = 1

        laptop_type = request.form["company_nm"]
        lp_type = 0 if laptop_type == 'ultrabookp' else 1 if laptop_type == 'notebook' else 2

        ram = request.form["ram"]
        weight = float(request.form["weight"])
        memory = request.form["memory"]
        memory_type = request.form["memory_typ"]
        m_type = 0 if memory_type == 'ssd' else 1

        cpu_mdl_name = request.form["cpumdl"]
        intel_core_i5, intel_core_i7, intel_core_i3, intel_celeron_series_process, amd_processor, intel_other_processor = [
                                                                                                                              0] * 6
        if cpu_mdl_name == 'intel_core_i5':
            intel_core_i5 = 1
        elif cpu_mdl_name == 'intel_core_i7':
            intel_core_i7 = 1
        elif cpu_mdl_name == 'intel_core_i3':
            intel_core_i3 = 1
        elif cpu_mdl_name == 'intel_celeron_series_process':
            intel_celeron_series_process = 1
        elif cpu_mdl_name == 'amd_processor':
            amd_processor = 1
        else:
            intel_other_processor = 1

        cpu_ghz = float(request.form["cpu_ghz"])
        screen_type = request.form["scrn_typ"]
        scr_type = 0 if screen_type == 'ips' else 1

        touch_display = request.form["touch_dsply"]
        touc_dis = 1 if touch_display == 'yes' else 0

        screen_res = request.form["scrn_res"]
        x_re, y_rez = map(int, screen_res.split('x'))
        screen_size = float(request.form["scrn_size"])
        ppi = (((x_re ** 2) + (y_rez ** 2)) ** 0.5 / screen_size)

        gpu_brand = request.form["gpu_brand"]
        gpuBrnd = 0 if gpu_brand == 'intel' else 1 if gpu_brand == 'amd' else 2

        os = request.form["os"]
        os_nm = 1 if os == 'window' else 0

        inputdt = [[hp, acer, aus, dell, lenevo, others, lp_type, ram, weight, memory, m_type,
                    intel_core_i5, intel_core_i7, intel_core_i3, intel_celeron_series_process, amd_processor,
                    intel_other_processor, cpu_ghz, scr_type, touc_dis, ppi, gpuBrnd, os_nm]]

        features_array = np.array(inputdt, dtype=float)
        prediction = model.predict(features_array)[0]

        return render_template('price_predc.html', prediction_text="Your Laptop price is Rs. {}".format(prediction))

    return render_template("price_predc.html")


if __name__ == '__main__':
    app.run(debug=True)
