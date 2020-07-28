# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:53:22 2020

@author: Harsh Anand
"""
from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL
app = Flask(__name__)


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'train'

mysql = MySQL(app)

@app.route("/")
def view_template():
    return render_template('test.html')

@app.route('/data4', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        user_data = request.form
        store = user_data["store_nbr"]
        item = user_data["item_nbr"]
        season = user_data["season"]
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM walmart WHERE store_nbr = %s AND item_nbr = %s AND season = %s", (store, item, season))
        data=cur.fetchall()
        mysql.connection.commit()
        cur.close()
        print(len(data))
        if len(data)>0:
            for row in data:
                result = row[2]
                print(result)
        else:
            result=0
    return jsonify(msg=str(result))



if __name__ == '__main__':
    app.run(debug=False,port=8000)