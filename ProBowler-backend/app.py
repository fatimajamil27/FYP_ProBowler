import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from model.analyze import analyze_bowling

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
bcrypt = Bcrypt(app)

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["probowler"]
users = db["users"]

@app.route('/')
def home():
    return jsonify({"message": "✅ ProBowler Flask backend is running!"})

# ------------------ REGISTER ------------------
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "message": "All fields are required."}), 400

    if users.find_one({"username": username}):
        return jsonify({"success": False, "message": "Username already exists."}), 400


    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    users.insert_one({
        "username": username,
        "password": hashed_pw,
        "createdAt": datetime.utcnow()
    })

    return jsonify({"success": True, "message": "User registered successfully."}), 201


# ------------------ LOGIN ------------------
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "message": "All fields are required."}), 400

    user = users.find_one({"username": username})
    if not user or not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"success": False, "message": "Invalid username or password."}), 401

    return jsonify({"success": True, "message": "Login successful.", "username": username}), 200


# ------------------ VIDEO UPLOAD ------------------
@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({"success": False, "message": "No video uploaded."}), 400

    video = request.files['video']
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Generate unique filename
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{video.filename}"
    video_path = os.path.join('uploads', filename)
    video.save(video_path)

    # Analyze bowling video
    result = analyze_bowling(video_path)
    return jsonify({"success": True, "result": result}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
