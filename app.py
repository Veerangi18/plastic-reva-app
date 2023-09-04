import onnxruntime as ort
from flask import request, Flask, jsonify, render_template,session,redirect,url_for,flash
# import session
import bcrypt
from waitress import serve
from PIL import Image
import pandas as pd
import plotly.express as px 
import pandas as pd
import numpy as np
import exifread
import sqlite3
from fractions import Fraction


# use geopy to get location from lat and lon
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="object-detection-app")


app = Flask(__name__)

# main_path = os.path.dirname(os.path.realpath(__file__))

# db_s = SQLAlchemy(app)

app.secret_key = 'secret_key'

import sqlite3

# Function to create the database tables
def create_database_tables(db_path):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Create the User table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password VARCHAR(100) NOT NULL
        )
    """)

    # Create the object_detection_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS object_detection_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            object_type TEXT,
            probability REAL,
            latitude REAL,
            longitude REAL,
            FOREIGN KEY (user_id) REFERENCES user (id)
        )
    """)

    # Commit the changes and close the connection
    connection.commit()
    connection.close()

    print("Database tables created successfully.")

# Example usage:
db_path = "REVA.db"
create_database_tables(db_path)




"""

#######################################################################################################################################################################333
THIS SECTION CONTAIN OBJECT DETECTION RELATED CODE
"""

# Location of the image taken fuctions

def get_image_geolocation(file):
    # Function to extract geolocation from image metadata using exifread library
    file.seek(0)
    tags = exifread.process_file(file)

    latitude_ref = tags.get("GPS GPSLatitudeRef")
    latitude = tags.get("GPS GPSLatitude")
    longitude_ref = tags.get("GPS GPSLongitudeRef")
    longitude = tags.get("GPS GPSLongitude")

    if latitude_ref and latitude and longitude_ref and longitude:
        latitude = parse_exif_gps_value(latitude)
        longitude = parse_exif_gps_value(longitude)

        # Convert the latitude and longitude from degrees, minutes, seconds to decimal degrees
        latitude_dec = convert_dms_to_dd(latitude)
        longitude_dec = convert_dms_to_dd(longitude)

        # Adjust the sign of latitude and longitude based on their reference
        if latitude_ref.values == "S":
            latitude_dec = -latitude_dec
        if longitude_ref.values == "W":
            longitude_dec = -longitude_dec

        return {"latitude": latitude_dec, "longitude": longitude_dec}
    else:
        raise ValueError("Geolocation data not found in image metadata.")


def parse_exif_gps_value(value):
    # Helper function to parse EXIF GPS coordinates in the format "[x, y, z]"
    parts = str(value).replace("[", "").replace("]", "").split(", ")
    degrees = float(parts[0])
    minutes_frac = Fraction(parts[1])
    seconds_frac = Fraction(parts[2])
    
    # Convert fractions to float
    minutes = minutes_frac.numerator / minutes_frac.denominator
    seconds = seconds_frac.numerator / seconds_frac.denominator
    
    return degrees, minutes, seconds

def convert_dms_to_dd(dms):
    # Function to convert degrees, minutes, seconds to decimal degrees
    degrees, minutes, seconds = dms
    dd = degrees + minutes / 60.0 + seconds / 3600.0
    return dd




# IOU and YOLOv8 classes

def iou(box1,box2):
    """
    Function calculates "Intersection-over-union" coefficient for specified two boxes
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
    :param box1: First box in format: [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format: [x1,y1,x2,y2,object_class,probability]
    :return: Intersection over union ratio as a float number
    """
    return intersection(box1,box2)/union(box1,box2)


def union(box1,box2):
    """
    Function calculates union area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of the boxes union as a
Detect, Defend, Restore: Our Mission to Safeguard Nature from Plastic Menace. float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)


def intersection(box1,box2):
    """
    Function calculates intersection area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of intersection of the boxes as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)


# Array of YOLOv8 class labels
yolo_classes = ["0"]


# function 3
def process_output(output, img_width, img_height):
    """
    Function used to convert RAW output from YOLOv8 to an array
    of detected objects. Each object contain the bounding box of
    this object, the type of object and the probability
    :param output: Raw output of YOLOv8 network which is an array of shape (1,84,8400)
    :param img_width: The width of original image
    :param img_height: The height of original image
    :return: Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
    """
    output = output[0].astype(float)
    output = output.transpose()

    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < 0.2:
            continue
        class_id = row[4:].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 2176 * img_width
        y1 = (yc - h/2) / 2176 * img_height
        x2 = (xc + w/2) / 2176 * img_width
        y2 = (yc + h/2) / 2176 * img_height
        boxes.append([x1, y1, x2, y2, label, prob])

    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < 0.3] 

    return result


# function 3
def run_model(input):
    """
    Function used to pass provided input tensor to
    YOLOv8 neural network and return result
    :param input: Numpy array in a shape (3,width,height)
    :return: Raw output of YOLOv8 network as an array of shape (1,84,8400)
    """
    model = ort.InferenceSession("Phase2_TeamAtlanticModel.onnx")
    outputs = model.run(["output0"], {"images":input})
    return outputs[0]


# function 2
def prepare_input(buf):
    """
    Function used to convert input image to tensor,
    required as an input to YOLOv8 object detection
    network.
    :param buf: Uploaded file input stream
    :return: Numpy array in a shape (3,width,height) where 3 is number of color channels
    """
    img = Image.open(buf)
    img_width, img_height = img.size
    img = img.resize((2176, 2176))
    img = img.convert("RGB")
    input = np.array(img) / 255.0
    input = input.transpose(2, 0, 1)
    input = input.reshape(1, 3, 2176, 2176)
    return input.astype(np.float32), img_width, img_height


# Function 1
def detect_objects_on_image(stream):
    input, img_width, img_height = prepare_input(stream)
    output = run_model(input)
    return process_output(output, img_width, img_height)

"""

#######################################################################################################################################################################333
Service Based Function
"""


def fetch_lat_lon_from_db_1(filename):
    connection = None  # Initialize the connection variable outside the try block

    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        user = session.get("user")
        # Fetch one of the latitudes and longitudes for the given filename from the database as per user
        query = "SELECT latitude, longitude FROM object_detection_data WHERE user_id = ? AND filename = ? LIMIT 1"
        cursor.execute(query, (user, filename))
        result = cursor.fetchone()

        return result

    except sqlite3.Error as error:
        print("Error fetching data from the database:", error)

    finally:
        if connection:
            connection.close()

    # Handle the case where an error occurred
    return None  # You might want to return an appropriate value or raise an exception here


def fetch_lat_lon_from_db():
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch unique filenames and their count from the database as per user
    user = session.get("user")
    cursor.execute("SELECT filename, COUNT(filename) FROM object_detection_data WHERE user_id = ? GROUP BY filename", (user,))
    filenames_data = cursor.fetchall()

    # Fetch unique latitudes and longitudes from the database as per user
    cursor.execute( "SELECT latitude, longitude FROM object_detection_data WHERE user_id = ? GROUP BY latitude, longitude", (user,))
    lat_lon_data = cursor.fetchall()

    # Close the database connection
    conn.close()

    return filenames_data, lat_lon_data



def Bubble_map(db_name):
    # get the data from the sqlite database 
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Fetch unique filenames and their count with their unique lat and long from the database as per user
    cursor.execute("SELECT filename, COUNT(filename), latitude, longitude FROM object_detection_data WHERE user_id = ? GROUP BY filename, latitude, longitude", (session.get("user"),))
    rows = cursor.fetchall()

    # Create a dataframe from the rows
    df = pd.DataFrame(rows, columns=['filename', 'Plastic_count', 'latitude', 'longitude'])
    # Mapbox plot
    mapbox_fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', size='Plastic_count',
                            color='Plastic_count', color_continuous_scale='plasma',
                            zoom=18, mapbox_style='open-street-map')
    mapbox_fig.update_traces(hovertemplate='<b>%{text}</b><br>' +
                                    'Plastic Count: %{marker.size:,}<br>' +
                                    'Latitude: %{lat}<br>' +
                                    'Longitude: %{lon}<br>',
                        text=df['filename'])
   
    # Bar plot
    bar_fig = px.bar(df, x='filename', y='Plastic_count', color='Plastic_count', color_continuous_scale='plasma')
    # add filename to the hover data
    bar_fig.update_traces(hovertemplate='<b>%{text}</b><br>' +
                                    'Plastic Count: %{y:,}<br>',
                        text=df['filename'])
    # line plot

  
    mapbox_plot_div = mapbox_fig.to_html(full_html=False)
    bar_plot_div = bar_fig.to_html(full_html=False)
    # dist_plot_div = dist_fig.to_html(full_html=False)


    return mapbox_plot_div, bar_plot_div

"""
#######################################################################################################################################################################333
THIS SECTION CONTAIN DATA BASE RELATED CODE

"""
# Function to add a user to the database

def get_user_by_email(email):
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Fetch the user by email
        query = "SELECT * FROM User WHERE email = ?"
        cursor.execute(query, (email,))
        user = cursor.fetchone()

        return user

    except sqlite3.Error as error:
        print("Error fetching user by email:", error)

    finally:
        if connection:
            connection.close()

    # Handle the case where an error occurred
    return None  # You might want to return an appropriate value or raise an exception here


def add_user(email, password, name):
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO user (name, email, password) VALUES (?, ?, ?)",
            (name, email, password)
        )
        connection.commit()
        return True
    except sqlite3.Error as e:
        print("Error adding user:", e)
        return False
    finally:
        if connection:
            connection.close()

# Function to add object detection data to the database
def add_object_detection_data(user_id, filename, x1, y1, x2, y2, object_type, probability, latitude, longitude):
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO object_detection_data (user_id, filename, x1, y1, x2, y2, object_type, probability, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, filename, x1, y1, x2, y2, object_type, probability, latitude, longitude)
        )
        connection.commit()
        return True
    except sqlite3.Error as e:
        print("Error adding object detection data:", e)
        return False
    finally:
        if connection:
            connection.close()

# Function to get object detection data for a user
def get_object_detection_data(user_id):
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT * FROM object_detection_data WHERE user_id = ?",
            (user_id,)
        )
        return cursor.fetchall()
    except sqlite3.Error as e:
        print("Error fetching object detection data:", e)
        return []
    finally:
        if connection:
            connection.close()

# Function to get a user by user_id
def get_user(user_id):
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute(
            "SELECT * FROM user WHERE id = ?",
            (user_id,)
        )
        return cursor.fetchone()
    except sqlite3.Error as e:
        print("Error fetching user:", e)
        return None
    finally:
        if connection:
            connection.close()



"""

#######################################################################################################################################################################333
FLASK ROUTES
"""


# ################## Login Register Function ##############################
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Handle registration request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Add the new user to the database
        if add_user(email, hashed_password, name):
            return redirect('/login')
        else:
            flash('Error registering user. Please try again.')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Retrieve the user from the database
        user = get_user_by_email(email)

        if user and bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'):
            session['user'] = user[0]
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        user_id = session['user']
        user = get_user(user_id)

        if user:
            return render_template('index.html', user=user)
        else:
            flash('User not found')
            return redirect('/login')
    else:
        return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))



@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """

    # with open("templates/index.html") as file:
    #     return file.read()
    return render_template("home.html")


# create different route for the database
@app.route("/database", methods=["GET"])
def database():
    """
    Handler of /database GET endpoint
    Retrieves data from the SQLite database and returns it as a JSON array
    :return: a JSON array of objects containing bounding boxes in format [[x1, y1, x2, y2, object_type, probability], ...]
    """
    # Fetch data from the database
    filenames_data, lat_lon_data = fetch_lat_lon_from_db()

    d_b = {
        "filenames": filenames_data,
        "lat_lon": lat_lon_data
    }

    return jsonify(d_b)

@app.route("/get_lat_lon/<filename>")
def get_lat_lon(filename):
    print("Received filename:", filename)
    lat_lon = fetch_lat_lon_from_db_1(filename)
    print("Latitude and Longitude:", lat_lon)
    return jsonify(lat_lon)

@app.route("/get_location/<lat>/<lon>")
def get_location(lat, lon):
    print("Received lat and lon:", lat, lon)
    location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
    print("Location:", location)

    if location:
        location_data = {
            "address": location.address,
            "country": location.raw.get("address", {}).get("country"),
            "postcode": location.raw.get("address", {}).get("postcode")
        }
    else:
        location_data = {
            "address": "Location data not found",
            "country": "Unknown",
            "postcode": "Unknown"
        }

    return jsonify(location_data)

# create route to get plastic count for a filename
@app.route("/get_plastic_count/<filename>")
def get_plastic_count(filename):
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch unique filenames and their count from the database as per user
    user = session.get("user")
    cursor.execute("SELECT COUNT(filename) FROM object_detection_data WHERE user_id = ? AND filename = ?", (user, filename))
    plastic_count = cursor.fetchone()[0]

    # Close the database connection
    conn.close()

    return jsonify(plastic_count)



@app.route("/db_data")
def db_data():
    # Fetch data from the database
    filenames_data, lat_lon_data = fetch_lat_lon_from_db()

    # get total number of plastic 
    total_plastic = 0
    for i in range(len(filenames_data)):
        total_plastic += filenames_data[i][1]

    d_b = {
        "filenames": filenames_data,
        "lat_lon": lat_lon_data,
        "total_plastic": total_plastic
    }

    return jsonify(d_b)

@app.route("/db")
def db():
    return render_template("db.html")

@app.route("/visualize")
def bubblemap():
    mapbox, bar = Bubble_map(db_path)
    return render_template('visualize.html', mapbox_plot_div=mapbox, bar_plot_div=bar)

@app.route("/locate")
def locate():
    return render_template("locate.html")


@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/login_reg")
def login_reg():
    return render_template("login.html")



@app.route("/detect", methods=["POST"])
def detect():
    # Get the user ID associated with the session (you may need to implement user authentication and session handling)
    user_id = session.get("user")

    if user_id is None:
        return jsonify({"error": "User not authenticated"})

    buf = request.files["image_file"]
    filename = buf.filename
    print(filename)
    boxes = detect_objects_on_image(buf.stream)

    # Get geolocation from the image metadata
    geolocation = get_image_geolocation(buf)


    # Save the detected objects to the database
    for box in boxes:
        x1, y1, x2, y2, object_type, probability = box
        add_object_detection_data(user_id, filename, x1, y1, x2, y2, object_type, probability, geolocation["latitude"], geolocation["longitude"])

    return jsonify(boxes)


@app.route("/about")
def about():
    # redirect to about us id of  home page
    return redirect(url_for('home', _anchor='AboutUs'))

@app.route("/contact")
def contact():
    # redirect to contact us id of  home page
    return redirect(url_for('home', _anchor='Contact'))

@app.route("/team")
def team():
    # redirect to team id of  home page
    return redirect(url_for('home', _anchor='team'))

@app.route("/Services")
def Services():
    # redirect to services id of  home page
    return redirect(url_for('home', _anchor='Services'))

@app.route("/Testimonials")
def Testimonials():
    # redirect to testimonials id of  home page
    return redirect(url_for('home', _anchor='Testimonials'))

@app.route("/log_about")
def log_about():
    # redirect to about us id of  home page
    return redirect(url_for('dashboard', _anchor='AboutUs'))

@app.route("/log_contact")
def log_contact():
    # redirect to contact us id of  home page
    return redirect(url_for('dashboard', _anchor='Contact'))

@app.route("/log_team")
def log_team():
    # redirect to team id of  home page
    return redirect(url_for('dashboard', _anchor='team'))

@app.route("/log_Services")
def log_Services():
    # redirect to services id of  home page
    return redirect(url_for('dashboard', _anchor='Services'))

@app.route("/log_Testimonials")
def log_Testimonials():
    # redirect to testimonials id of  home page
    return redirect(url_for('dashboard', _anchor='Testimonials'))

def main():
    app.run(debug=True,port = 8080)

if __name__ == "__main__":
    main()
    # serve(app, host='




