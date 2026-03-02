import os
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from config import get_config
from utils.logger import setup_logger
from manager import register_manager_routes
from user import register_user_routes

load_dotenv()

app = Flask(__name__)
config_class = get_config(os.environ.get("FLASK_ENV", "development"))
app.config.from_object(config_class)

logger = setup_logger(__name__)
db = SQLAlchemy(app)

UPLOAD_FOLDER = app.config.get("UPLOAD_FOLDER", "uploads")
DATASET_FOLDER = app.config.get("DATASET_FOLDER", "datasets")
VECTOR_DB_FOLDER = app.config.get("VECTOR_DB_FOLDER", "vector_db")

for folder in [UPLOAD_FOLDER, DATASET_FOLDER, VECTOR_DB_FOLDER]:
	if not os.path.exists(folder):
		os.makedirs(folder, exist_ok=True)


class User(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	username = db.Column(db.String(80), unique=True, nullable=False, index=True)
	email = db.Column(db.String(120), unique=True, nullable=False, index=True)
	password_hash = db.Column(db.String(255), nullable=False)
	role = db.Column(db.String(20), nullable=False, default="user")
	created_at = db.Column(db.DateTime, default=datetime.utcnow)
	last_login = db.Column(db.DateTime)
	is_active = db.Column(db.Boolean, default=True)

	def set_password(self, password):
		self.password_hash = generate_password_hash(password)

	def check_password(self, password):
		return check_password_hash(self.password_hash, password)


class Review(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	restaurant = db.Column(db.String(200), index=True, nullable=False)
	reviewer = db.Column(db.String(200), default="anonymous")
	text = db.Column(db.Text, nullable=False)
	rating = db.Column(db.Float, nullable=True)
	sentiment = db.Column(db.String(50), index=True)
	score = db.Column(db.Float)
	keywords = db.Column(db.String(500))
	categories = db.Column(db.String(500))
	source_file = db.Column(db.String(100), index=True)
	created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


with app.app_context():
	db.create_all()
	logger.info("Database tables created/verified")


def login_required(func):
	def wrapper(*args, **kwargs):
		if "user_id" not in session:
			flash("Please login to access this page.", "warning")
			return redirect(url_for("login"))
		return func(*args, **kwargs)

	wrapper.__name__ = func.__name__
	return wrapper


def manager_required(func):
	def wrapper(*args, **kwargs):
		if "user_id" not in session:
			flash("Please login to access this page.", "warning")
			return redirect(url_for("login"))
		user = db.session.get(User, session["user_id"])
		if not user or user.role != "manager":
			flash("Manager access required.", "danger")
			return redirect(url_for("user_dashboard"))
		return func(*args, **kwargs)

	wrapper.__name__ = func.__name__
	return wrapper


@app.route("/")
def index():
	if "user_id" not in session:
		return redirect(url_for("login"))

	user = db.session.get(User, session["user_id"])
	if not user:
		session.clear()
		return redirect(url_for("login"))

	if user.role == "manager":
		return redirect(url_for("manager_dashboard"))
	return redirect(url_for("user_dashboard"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
	if "user_id" in session:
		return redirect(url_for("index"))

	if request.method == "POST":
		username = request.form.get("username", "").strip()
		email = request.form.get("email", "").strip()
		password = request.form.get("password", "")
		confirm_password = request.form.get("confirm_password", "")
		role = request.form.get("role", "user")

		if not username or not email or not password:
			flash("All fields are required.", "danger")
			return render_template("signup.html")
		if len(username) < 3:
			flash("Username must be at least 3 characters long.", "danger")
			return render_template("signup.html")
		if len(password) < 6:
			flash("Password must be at least 6 characters long.", "danger")
			return render_template("signup.html")
		if password != confirm_password:
			flash("Passwords do not match.", "danger")
			return render_template("signup.html")
		if role not in ["user", "manager"]:
			flash("Invalid role selected.", "danger")
			return render_template("signup.html")

		if User.query.filter_by(username=username).first():
			flash("Username already exists.", "danger")
			return render_template("signup.html")
		if User.query.filter_by(email=email).first():
			flash("Email already registered.", "danger")
			return render_template("signup.html")

		user = User(username=username, email=email, role=role)
		user.set_password(password)
		db.session.add(user)
		db.session.commit()

		flash("Account created successfully. Please login.", "success")
		return redirect(url_for("login"))

	return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
	if "user_id" in session:
		return redirect(url_for("index"))

	if request.method == "POST":
		username = request.form.get("username", "").strip()
		password = request.form.get("password", "")

		if not username or not password:
			flash("Username and password are required.", "danger")
			return render_template("login.html")

		user = User.query.filter_by(username=username).first()
		if not user or not user.check_password(password):
			flash("Invalid username or password.", "danger")
			return render_template("login.html")

		if not user.is_active:
			flash("Your account is inactive.", "danger")
			return render_template("login.html")

		session["user_id"] = user.id
		session["username"] = user.username
		session["role"] = user.role
		session.permanent = True

		user.last_login = datetime.utcnow()
		db.session.commit()

		if user.role == "manager":
			return redirect(url_for("manager_dashboard"))
		return redirect(url_for("user_dashboard"))

	return render_template("login.html")


@app.route("/logout")
def logout():
	session.clear()
	flash("Logged out successfully.", "info")
	return redirect(url_for("login"))


register_manager_routes(app, db, User, Review, manager_required)
register_user_routes(app, db, User, Review, login_required)


if __name__ == "__main__":
	debug_mode = app.config.get("DEBUG", False)
	app.run(debug=debug_mode, use_reloader=False, host="0.0.0.0", port=5000)
