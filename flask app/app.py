from flask import Flask, render_template, request, redirect, session, flash
from flask_bootstrap import Bootstrap
from flask_mysqldb import MySQL
import numpy as np
import yaml
import math
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model
import pandas as pd
from sqlalchemy import create_engine
import pymysql


app=Flask(__name__)
Bootstrap(app)
mysql = MySQL(app)

db = yaml.load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

app.config['SECRET_KEY'] = 'secret'

# split into input (X) and output (Y) variables
def prepare_data(timeseries_data, n_features):
        	X, y =[],[]
        	for i in range(len(timeseries_data)):
        		# find the end of this pattern
        		end_ix = i + n_features
        		# check if we are beyond the sequence
        		if end_ix > len(timeseries_data)-1:
        			break
        		# gather input and output parts of the pattern
        		seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        		X.append(seq_x)
        		y.append(seq_y)
        	return np.array(X), np.array(y)

# demonstrate prediction for next n days
def logic(j):
    data = timeseries_data.astype(float)
    x_input=data[-100:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    lst_output=[]
    i=0
    while(i<j):
    
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            #print(x_input)
            x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.append(yhat[0][0])
            i=i+1
        else:
            x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i=i+1
 
    return lst_output        
        
#EOQ Model
class EOQ_Model:
            def __init__(self, demand=0, order=0, holding=0, cost=0, lead=0, planned_shortage=False, shortage_cost=0):
                self.demand = demand
                self.order = order
                self.holding = holding
                self.cost = cost
                self.lead = lead
                self.planned_shortage = planned_shortage
                self.shortage_cost = shortage_cost
                
                
            def optimal_order_quantity(self, d=None, o=None, h=None, s=None):
                '''
                Calculate the order qunatity
                
                d = demand
                o = ordering cost
                h = holding cost
                returns = reorder optimal quantity
                
                '''
                if d is None:
                    d = self.demand       
                if o is None:
                    o = self.order
                if h is None: 
                    h = self.holding
                if s is None:
                    s = self.shortage_cost
                    
                if self.planned_shortage:
                    return math.sqrt((2*d*o)/h) * math.sqrt(self.shortage_cost/(self.shortage_cost + self.holding))
                else:
                    return math.sqrt((2*d*o)/h)
                
                
            def reorder_point(self, d=None, lt=None):
                '''
                Calculates the reorder point with no planned shortages.
        
                d: total demand
                l: lead time
                returns: reorder point
                '''
                if d is None:
                    d = self.demand
                if lt is None:
                    lt = self.lead
                return d * lt
            
                
            def optimal_cycle_time(self, d=None, o=None, h=None, s=None):
                '''
                Calculates the optimal cycle time.
                
                d: total demand
                o: ordering cost
                h: holding cost
                returns: reorder point
                '''
                
                if d is None:
                    d = self.demand 
                if o is None:
                    o = self.order
                if h is None: 
                    h = self.holding
                if s is None:
                    s = self.shortage_cost
                    
                if self.planned_shortage:
                    return math.sqrt((2*o)/(h*d)) * math.sqrt((self.shortage_cost + self.holding)/self.shortage_cost)
                else:
                    return math.sqrt((2*o)/(h*d))
                
            
            def complete_calculations(self):
                '''Calculates and prints the main 2 metrics: order quantity, optimal cycle time
                
                :returns: tuple of metrics
                :rtype: tuple of length 2
                '''
                global q
                global t
                q = self.optimal_order_quantity()
                t = self.optimal_cycle_time()
                q = round(q)
                t = round(t, 3)
                print("Optimal Order Quantity (q*): {} units".format(q))
                print("Optimal Cycle Time (t*): {}".format(t))        
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/register/', methods=['GET', 'POST'])
def register():
        if request.method == 'POST':
         userDetails = request.form
         if userDetails['password'] != userDetails['confirm_password']:
            flash('Passwords do not match! Try again.', 'danger')
            return render_template('register.html')
         cur = mysql.connection.cursor()
         cur.execute("INSERT INTO user(first_name, last_name, age, user_name, email, password) "\
         "VALUES(%s,%s,%s,%s,%s,%s)",(userDetails['first_name'], userDetails['last_name'], \
         userDetails['age'], userDetails['username'], userDetails['email'], userDetails['password']))
         mysql.connection.commit()
         cur.close()
         flash('Registration successful! Please login.', 'success')
         return redirect('/login')
        return render_template('register.html')

@app.route('/login/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        userDetails = request.form
        username = userDetails['username']
        cur = mysql.connection.cursor()
        resultValue = cur.execute("SELECT * FROM user WHERE user_name = %s", ([username]))
        if resultValue > 0:
            user = cur.fetchone()
            if userDetails['password'] == user['password']:
                session['login'] = True
                session['firstName'] = user['first_name']
                session['lastName'] = user['last_name']
                session['userName'] = user['user_name']
                flash('Welcome ' + session['firstName'] +'! You have been successfully logged in', 'success')
            else:
                cur.close()
                flash('Password does not match', 'danger')
                return render_template('login.html')
        else:
            cur.close()
            flash('User not found', 'danger')
            return render_template('login.html')
        cur.close()
        return redirect('/')    
    return render_template('login.html')

@app.route('/0251/', methods=['GET', 'POST'])
def p0251():
    return render_template('0251.html')

@app.route('/product0251/', methods=['GET', 'POST'])
def product0251():
    if request.method == 'POST':
        global timeseries_data, n_steps, n_features, n_steps, n_seq, model, j, o, c, h, s, list2
        # load model
        model = load_model('model0251.h5')

        # summarize model.
        model.summary()

        # load dataset
        db_connection = 'mysql+pymysql://root:root@localhost/product'
        conn = create_engine(db_connection)
        df = pd.read_sql("select product_demand from product_0251", conn)
        
        # define input sequence
        timeseries_data = df.to_numpy()

        # choose a number of time steps
        n_steps = 100

        # split into samples
        X, y = prepare_data(timeseries_data, n_steps)
        
        # choose a number of time steps
        n_steps = 100

        # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
        n_features = 1
        n_seq = 1
        n_steps = 100
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
 
        #Evaluate model
        model.evaluate(X, y, verbose=0)
        
        # demonstrate prediction for next n days
        value= request.form
        j= value['daysprediction']   #number of days to be predicted for demand
        j=int(j)
        predictions = logic(j)
        
        list2 = [1 if i == 0 else i for i in predictions]
        print(list2)
        
        #EOQ model values
        l = value['eoqprediction']             #l = day on which EOQ is applied
        l = int(l)
        d = list2[l]
        o = value['ordercost']
        o = int(o)
        c = value['productcost']
        c = int(c)
        h = value   ['holdingcost']
        h = int(h)
        s=0.8
        
        eoqmodel = EOQ_Model(demand=d, order=o, cost=c, holding=h, planned_shortage=False, shortage_cost=s)
        eoqmodel.complete_calculations()
        
        flash("Result ready", 'success')
        return render_template('product_0251.html', demandPredict ='Demand predicted for mentioned days: {}'.format(predictions),Day ='Day selected for order analysis: {}'.format(l), QO ='Quantity to be ordered: {}'.format(q), CT = 'Order Cycle: {}'.format(t))    
    return render_template('product_0251.html')

@app.route('/product0251_add/', methods=['GET', 'POST'])
def product0251_add():
    if request.method == 'POST':
        data = request.form
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO product_0251(Date, product_demand) VALUES(%s, %s)", (data['date'], data['demand']))
        mysql.connection.commit()
        cur.close()
        flash("Result ready", 'success')    
    return render_template('product_0251_add.html')    

@app.route('/0338/', methods=['GET', 'POST'])
def p0338():
    return render_template('0338.html')

@app.route('/product0338/', methods=['GET', 'POST'])
def product0338():
    if request.method == 'POST':
        global timeseries_data, n_steps, n_features, n_steps, n_seq, model, j, o, c, h, s, list2
        # load model
        model = load_model('model0338.h5')
       
        # summarize model.
        model.summary()
        
        # load dataset
        db_connection = 'mysql+pymysql://root:root@localhost/product'
        conn = create_engine(db_connection)
        df = pd.read_sql("select product_demand from product_0338", conn)
                
        # define input sequence
        timeseries_data = df.to_numpy()
      
        # choose a number of time steps
        n_steps = 100
        
        # split into samples
        X, y = prepare_data(timeseries_data, n_steps)
        
        # choose a number of time steps
        n_steps = 100
        
        # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
        n_features = 1
        n_seq = 1
        n_steps = 100
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
        
        #Evaluet Model
        model.evaluate(X, y, verbose=0)
        
        # demonstrate prediction for next n days
        values= request.form
        j= values['daysprediction']   #number of days to be predicted for demand
        j=int(j)
        predictions = logic(j)

        list2 = [1 if i == 0 else i for i in predictions]
        print(list2)
        
        #Eoq values
        l = values['eoqprediction']             #l = day on which EOQ is applied
        l = int(l)
        d = list2[l]
        o = values['ordercost']
        o = int(o)
        c = values['productcost']
        c = int(c)
        h = values['holdingcost']
        h = int(h)
        s=0.8
        
        eoqmodel = EOQ_Model(demand=d, order=o, cost=c, holding=h, planned_shortage=False, shortage_cost=s)
        eoqmodel.complete_calculations()
        
        flash("Result ready", 'success')
        return render_template('product_0338.html', demandPredict ='Demand predicted for mentioned days: {}'.format(predictions),Day ='Day selected for order analysis: {}'.format(l), QO ='Quantity to be ordered: {}'.format(q), CT = 'Order Cycle: {}'.format(t))    
    return render_template('product_0338.html')

@app.route('/product0338_add/', methods=['GET', 'POST'])
def product0338_add():
    if request.method == 'POST':
        data = request.form
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO product_0338(Date, product_demand) VALUES(%s, %s)", (data['date'], data['demand']))
        mysql.connection.commit()
        cur.close()
        flash("Result ready", 'success')    
    return render_template('product_0338_add.html')

@app.route('/1770/', methods=['GET', 'POST'])
def p1770():
    return render_template('1770.html')

@app.route('/product1770/', methods=['GET', 'POST'])
def product1770():
    if request.method == 'POST':
        global timeseries_data, n_steps, n_features, n_steps, n_seq, model, j, o, c, h, s, list2
        # load model
        model = load_model('model1770.h5')
       
        # summarize model.
        model.summary()
        
        # load dataset
        db_connection = 'mysql+pymysql://root:root@localhost/product'
        conn = create_engine(db_connection)
        df = pd.read_sql("select product_demand from product_1770", conn)
 
        # define input sequence
        timeseries_data = df.to_numpy()
       
        # choose a number of time steps
        n_steps = 100
        
        # split into samples
        X, y = prepare_data(timeseries_data, n_steps)
        
        # choose a number of time steps
        n_steps = 100
        
        # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
        n_features = 1
        n_seq = 1
        n_steps = 100
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
        
        #Evaluate model
        model.evaluate(X, y, verbose=0)
        
        # demonstrate prediction for next n days
        values= request.form
        j= values['daysprediction']   #number of days to be predicted for demand
        j=int(j)
        predictions = logic(j)
        
        list2 = [1 if i == 0 else i for i in predictions]
        print(list2)
        
        #EOQ values
        l = values['eoqprediction']             #l = day on which EOQ is applied
        l = int(l)
        d = list2[l]
        o = values['ordercost']
        o = int(o)
        c = values['productcost']
        c = int(c)
        h = values['holdingcost']
        h = int(h)
        s=0.8
        
        eoqmodel = EOQ_Model(demand=d, order=o, cost=c, holding=h, planned_shortage=False, shortage_cost=s)
        eoqmodel.complete_calculations()
        
        flash("Result ready", 'success')
        return render_template('product_1770.html', demandPredict ='Demand predicted for mentioned days: {}'.format(predictions),Day ='Day selected for order analysis: {}'.format(l), QO ='Quantity to be ordered: {}'.format(q), CT = 'Order Cycle: {}'.format(t))    
    return render_template('product_1770.html')

@app.route('/product1770_add/', methods=['GET', 'POST'])
def product1770_add():
    if request.method == 'POST':
        data = request.form
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO product_1770(Date, product_demand) VALUES(%s, %s)", (data['date'], data['demand']))
        mysql.connection.commit()
        cur.close()
        flash("Result ready", 'success')    
    return render_template('product_1770_add.html')

@app.route('/2038/', methods=['GET', 'POST'])
def p2038():
    return render_template('2038.html')

@app.route('/product2038/', methods=['GET', 'POST'])
def product2038():
    if request.method == 'POST':
        global timeseries_data, n_steps, n_features, n_steps, n_seq, model, j, o, c, h, s, list2
        # load model
        model = load_model('model2038.h5')
       
        # summarize model.
        model.summary()
        
        # load dataset
        db_connection = 'mysql+pymysql://root:root@localhost/product'
        conn = create_engine(db_connection)
        df = pd.read_sql("select product_demand from product_2038", conn)

        # define input sequence
        global timeseries_data  
        timeseries_data = df.to_numpy()
        
        # choose a number of time steps
        n_steps = 100
        
        # split into samples
        X, y = prepare_data(timeseries_data, n_steps)
        
        # choose a number of time steps
        n_steps = 100
        
        # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
        n_features = 1
        n_seq = 1
        n_steps = 100
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
        
        #Evaluate Model
        model.evaluate(X, y, verbose=0)
        
        # demonstrate prediction for next n days
        values= request.form
        j= values['daysprediction']   #number of days to be predicted for demand
        j=int(j)
        predictions = logic(j)
        
        list2 = [1 if i == 0 else i for i in predictions]
        print(list2)
        
        #EOQ Values
        l = values['eoqprediction']             #l = day on which EOQ is applied
        l = int(l)
        d = list2[l]
        o = values['ordercost']
        o = int(o)
        c = values['productcost']
        c = int(c)
        h = values['holdingcost']
        h = int(h)
        s=0.8
        
        eoqmodel = EOQ_Model(demand=d, order=o, cost=c, holding=h, planned_shortage=False, shortage_cost=s)
        eoqmodel.complete_calculations()
                  
        flash("Result ready", 'success')
        return render_template('product_2038.html', demandPredict ='Demand predicted for mentioned days: {}'.format(predictions),Day ='Day selected for order analysis: {}'.format(l), QO ='Quantity to be ordered: {}'.format(q), CT = 'Order Cycle: {}'.format(t))    
    return render_template('product_2038.html')

@app.route('/product2038_add/', methods=['GET', 'POST'])
def product2038_add():
    if request.method == 'POST':
        data = request.form
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO product_2038(Date, product_demand) VALUES(%s, %s)", (data['date'], data['demand']))
        mysql.connection.commit()
        cur.close()
        flash("Result ready", 'success')    
    return render_template('product_2038_add.html')

@app.route("/analysis", methods=["GET"])
def analysis():

    # Generate demand plot    
    global a, day_new, day_pred
    a = j-1
    day_new=np.arange(1,j)
    day_pred=np.arange(j,j+j)
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Demand Prediction")
    axis.set_xlabel("Number of days")
    axis.set_ylabel("Demand")
    axis.grid()
    axis.plot(day_new, timeseries_data[-a:] , "ro-")
    axis.plot(day_pred, list2 , "bo-")
    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    
    # Generate order plot 
    list3=[]
    for x in list2:
        d = x
        eoqmodel = EOQ_Model(demand=d, order=o, cost=c, holding=h, planned_shortage=False, shortage_cost=s)
        eoqmodel.complete_calculations()
        list3.append(q)
            
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    day_new1 = np.arange(0,j)
    axis.set_title("Order Prediction")
    axis.set_xlabel("Number of days")
    axis.set_ylabel("Order quantity")
    axis.grid()
    axis.plot(day_new1, list3 , "ro-")
    
    # Convert plot to PNG image
    pngImage1 = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage1)  
    
    # Encode PNG image to base64 string
    pngImageB64String1 = "data:image/png;base64,"
    pngImageB64String1 += base64.b64encode(pngImage1.getvalue()).decode('utf8')
    
    # Generate time plot 
    list4=[]
    for x in list2:
        d = x
        eoqmodel = EOQ_Model(demand=d, order=o, cost=c, holding=h, planned_shortage=False, shortage_cost=s)
        eoqmodel.complete_calculations()
        list4.append(t)
            
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Order Time Cycle")
    axis.set_xlabel("Number of days")
    axis.set_ylabel("Order Cycle")
    axis.grid()
    axis.plot(day_new1, list4 , "ro-")
    
    # Convert plot to PNG image
    pngImage2 = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage2)  
    
    # Encode PNG image to base64 string
    pngImageB64String2 = "data:image/png;base64,"
    pngImageB64String2 += base64.b64encode(pngImage2.getvalue()).decode('utf8')
    
    return render_template("image.html", image=pngImageB64String, image1=pngImageB64String1, image2=pngImageB64String2)


@app.route('/logout/')
def logout():
    session.clear()
    flash("You have been logged out", 'info')
    return render_template('logout.html')

if __name__=='__main__':
    app.run()    