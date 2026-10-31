import os
import json
import sqlite3
from contextlib import contextmanager
from functools import wraps

from flask import Flask, flash, redirect, render_template, request, session, url_for
import pandas as pd
import joblib
from werkzeug.security import check_password_hash, generate_password_hash


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32)
DATABASE_PATH = os.path.join(app.root_path, "users.db")


@contextmanager
def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_name TEXT NOT NULL DEFAULT 'Unknown',
                input_values TEXT NOT NULL DEFAULT '{}',
                result TEXT NOT NULL,
                probability REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        history_columns = {
            column["name"]
            for column in connection.execute(
                "PRAGMA table_info(prediction_history)"
            )
        }
        if "patient_name" not in history_columns:
            connection.execute(
                """
                ALTER TABLE prediction_history
                ADD COLUMN patient_name TEXT NOT NULL DEFAULT 'Unknown'
                """
            )
        if "input_values" not in history_columns:
            connection.execute(
                """
                ALTER TABLE prediction_history
                ADD COLUMN input_values TEXT NOT NULL DEFAULT '{}'
                """
            )


initialize_database()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("model_joblib_heart.pkl")
DISEASE_THRESHOLD = 50
INPUT_FIELD_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type (cp)",
    "trestbps": "Resting Blood Pressure (trestbps)",
    "chol": "Cholesterol (chol)",
    "fbs": "Fasting Blood Sugar (fbs)",
    "restecg": "Resting ECG (restecg)",
    "thalach": "Max Heart Rate (thalach)",
    "exang": "Exercise Induced Angina (exang)",
    "oldpeak": "Oldpeak",
    "slope": "ST Segment Slope (slope)",
    "ca": "Major Vessels (ca)",
    "thal": "Thalassemia (thal)",
}


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
@login_required
def home():
    history_total, recent_predictions = get_user_history(
        session["user_id"], limit=3
    )
    return render_template(
        "index.html",
        history_total=history_total,
        recent_predictions=recent_predictions,
    )


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/how-it-works")
@login_required
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/contact")
@login_required
def contact():
    return render_template("contact.html")


def get_user_history(user_id, limit=None):
    with get_db_connection() as connection:
        history_total = connection.execute(
            "SELECT COUNT(*) AS total FROM prediction_history WHERE user_id = ?",
            (user_id,),
        ).fetchone()["total"]
        query = """
            SELECT id, patient_name, input_values, result, probability, created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
        """
        parameters = (user_id,)
        if limit is not None:
            query += " LIMIT ?"
            parameters += (limit,)
        predictions = connection.execute(query, parameters).fetchall()
    predictions = [
        {
            **dict(prediction),
            "input_values": json.loads(prediction["input_values"]),
        }
        for prediction in predictions
    ]
    return history_total, predictions


@app.route("/history")
@login_required
def history():
    history_total, predictions = get_user_history(session["user_id"])
    return render_template(
        "history.html",
        history_total=history_total,
        predictions=predictions,
        input_field_labels=INPUT_FIELD_LABELS,
    )


@app.route("/history/<int:prediction_id>/delete", methods=["POST"])
@login_required
def delete_prediction(prediction_id):
    with get_db_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM prediction_history WHERE id = ? AND user_id = ?",
            (prediction_id, session["user_id"]),
        )

    if cursor.rowcount:
        flash("Prediction history entry deleted.", "success")
    else:
        flash("Prediction history entry was not found.", "error")
    return redirect(url_for("history"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(username) < 3:
            flash("Username must be at least 3 characters long.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        else:
            try:
                with get_db_connection() as connection:
                    cursor = connection.execute(
                        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                        (username, generate_password_hash(password)),
                    )
                session.clear()
                session["user_id"] = cursor.lastrowid
                session["username"] = username
                return redirect(url_for("home"))
            except sqlite3.IntegrityError:
                flash("That username is already in use.", "error")

    return render_template("auth.html", page="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db_connection() as connection:
            user = connection.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "error")
        else:
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))

    return render_template("auth.html", page="login")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    try:

        # ----------------------------------
        # Get values from HTML form
        # ----------------------------------

        patient_name = request.form.get("patient_name", "").strip()
        if not patient_name or len(patient_name) > 100:
            flash("Enter a patient name of 1 to 100 characters.", "error")
            return redirect(url_for("home"))

        age = int(request.form["age"])
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecg"])
        thalach = int(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])

        input_values = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        # ----------------------------------
        # Create DataFrame
        # ----------------------------------

        input_data = pd.DataFrame({

            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal]

        })


        # ----------------------------------
        # Prediction
        # ----------------------------------

        # ----------------------------------
        # Probability
        # ----------------------------------

        probability = model.predict_proba(input_data)[0]

        disease_class_index = list(model.classes_).index(1)
        disease_probability = probability[disease_class_index] * 100


        # ----------------------------------
        # Result
        # ----------------------------------

        if disease_probability >= DISEASE_THRESHOLD:

            result = "Heart Disease Detected"

            message = (
                "The model predicts a possibility "
                "of heart disease."
            )

        else:

            result = "No Heart Disease"

            message = (
                "The model predicts no heart disease."
            )

        probability_percent = round(disease_probability, 2)
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO prediction_history (
                    user_id, patient_name, input_values, result, probability
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session["user_id"],
                    patient_name,
                    json.dumps(input_values),
                    result,
                    probability_percent,
                ),
            )

        history_total, recent_predictions = get_user_history(
            session["user_id"], limit=3
        )


        # ----------------------------------
        # Send result to UI
        # ----------------------------------

        return render_template(
            "index.html",
            result=result,
            message=message,
            probability=probability_percent,
            history_total=history_total,
            recent_predictions=recent_predictions,
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )