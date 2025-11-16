from flask import Flask, render_template, request
import requests
import pandas as pd
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
from datetime import datetime, timedelta

app = Flask(__name__)
matplotlib.use('Agg')


@app.route('/', methods=['GET', 'POST'])
def home():
    search_done = False
    if request.method == "POST":
        city = request.form['city']
        try:
            current_data = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=4cd7dd228d8dee9c35573d80ba13bea6&units=metric")
            data_for_current_temp = current_data.json()
            city_name = data_for_current_temp['name']
        except KeyError:
            return render_template('404_error.html')
        else:
            current_temp = round(data_for_current_temp['main']['temp'])
            feels_like = round(data_for_current_temp['main']['feels_like'])
            temp_min = round(data_for_current_temp['main']['temp_min'])
            temp_max = round(data_for_current_temp['main']['temp_max'])
            humidity = round(data_for_current_temp['main']['humidity'])
            country = data_for_current_temp['sys']['country']
            description = data_for_current_temp['weather'][0]['description']
            search_done = True
            return render_template('index.html', city=city_name, current_temp=current_temp, temp_max=temp_max,
                                   temp_min=temp_min, description=description, feels_like=feels_like, country=country,
                                   status=search_done, humidity=humidity)
    return render_template("index.html", status=search_done)


@app.route('/predict-weather', methods=['GET', 'POST'])
def prediction():
    predict_status = False
    search_done = False
    if request.method == "POST":
        city = request.form['city']
        try:
            current_data = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=4cd7dd228d8dee9c35573d80ba13bea6&units=metric")
            data_for_current_temp = current_data.json()
            LAT = data_for_current_temp['coord']['lat']
            LON = data_for_current_temp['coord']['lon']
        except KeyError:
            return render_template('404_error.html')
        else:
            parameters = {
                "lat": LAT,
                "lon": LON,
                "appid": "4cd7dd228d8dee9c35573d80ba13bea6"
            }
            response = requests.get("https://api.openweathermap.org/data/2.5/onecall", params=parameters)
            response.raise_for_status()
            data = response.json()

            temperature, humidity, hours = [], [], []
            for i in range(48):
                hourly_data = data['hourly'][i]
                hours.append(i)
                temperature.append(hourly_data['temp'] - 273)
                humidity.append(hourly_data['humidity'])

            reversed_hour = hours[::-1]

            df = pd.DataFrame({'hours': reversed_hour, 'temp': temperature, 'hum': humidity})
            df.to_csv('static/csv/weather_data.csv')

            data = pd.read_csv("static/csv/weather_data.csv", index_col='hours').dropna()
            weather_data = data['temp']
            hum_data = data['hum']

            warnings.filterwarnings("ignore")

            weather_fit = auto_arima(weather_data, trace=True, suppress_warnings=True)
            weather_param = weather_fit.get_params().get("order")

            hum_fit = auto_arima(hum_data, trace=True, suppress_warnings=True)
            hum_param = hum_fit.get_params().get("order")

            model_temp = ARIMA(weather_data, order=weather_param).fit()
            model_hum = ARIMA(hum_data, order=hum_param).fit()

            future_times = [datetime.now() + timedelta(hours=i) for i in range(5)]
            future_labels = [x.strftime("%H:%M") for x in map(lambda dt: dt.time(), future_times)]

            weather_pred = model_temp.predict(start=48, end=52, typ='levels')
            weather_pred.index = future_labels
            temp_list = list(weather_pred.round(1))

            hum_pred = model_hum.predict(start=48, end=52, typ='levels')
            hum_pred.index = future_labels
            hum_list = list(hum_pred.round(1))

            current_temp = round(data_for_current_temp['main']['temp'])
            feels_like = round(data_for_current_temp['main']['feels_like'], 1)
            temp_min = round(data_for_current_temp['main']['temp_min'], 1)
            temp_max = round(data_for_current_temp['main']['temp_max'], 1)
            humidity_now = round(data_for_current_temp['main']['humidity'], 1)
            country = data_for_current_temp['sys']['country']
            description = data_for_current_temp['weather'][0]['description']
            city_name = data_for_current_temp['name']
            predict_status = True
            search_done = True

            tlabels, tvalues = future_labels, temp_list
            hlabels, hvalues = future_labels, hum_list

            return render_template("index.html", predicted_temp=weather_pred, predicted_humidity=hum_pred,
                                   predict_status=predict_status, status=search_done,
                                   temperature_1=temp_list[0], temperature_2=temp_list[1], temperature_3=temp_list[2],
                                   temperature_4=temp_list[3], temperature_5=temp_list[4],
                                   humidity_1=hum_list[0], humidity_2=hum_list[1], humidity_3=hum_list[2],
                                   humidity_4=hum_list[3], humidity_5=hum_list[4],
                                   city=city_name, current_temp=current_temp, temp_max=temp_max,
                                   temp_min=temp_min, description=description, feels_like=feels_like,
                                   country=country, humidity=humidity_now,
                                   tlabels=tlabels, tvalues=tvalues, hlabels=hlabels, hvalues=hvalues)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
