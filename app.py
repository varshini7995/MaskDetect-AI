import os
import uuid
from datetime import datetime

from flask import (
    Flask,
    render_template,
    Response,
    jsonify,
    request,
    redirect,
    url_for,
    session,
    flash
)

from werkzeug.utils import secure_filename

from detect_live import generate_frames, live_stats

import cv2
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = 'mask_detect_secret_key_99'


# ============================================================
# UPLOAD FOLDERS
# ============================================================

UPLOAD_FOLDER = 'static/uploads'
PROFILE_FOLDER = 'static/uploads/profiles'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROFILE_FOLDER'] = PROFILE_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROFILE_FOLDER, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model("models/model.h5")


# ============================================================
# DATABASES
# ============================================================

# User information is stored dynamically.
# There is NO fixed Varshini account anymore.

users_db = {}


# Detection history
detection_history = []


# ============================================================
# HOME
# ============================================================

@app.route('/')
def home():

    return render_template('index.html')




@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/features')
def features_page():
    return render_template('features.html')

@app.route('/how-it-works')
def how_it_works_page():
    return render_template('how_it_works.html')
# ============================================================
# LOGIN
# ============================================================

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form.get(
            'email',
            ''
        ).strip().lower()

        password = request.form.get(
            'password',
            ''
        )


        # ----------------------------------------------------
        # CHECK ACCOUNT
        # ----------------------------------------------------

        if email not in users_db:

            flash(
                "Account does not exist. Please sign up first!",
                "warning"
            )

            return redirect(
                url_for('signup')
            )


        # ----------------------------------------------------
        # CHECK PASSWORD
        # ----------------------------------------------------

        if users_db[email]['password'] != password:

            flash(
                "Incorrect password. Please try again.",
                "danger"
            )

            return redirect(
                url_for('login')
            )


        # ----------------------------------------------------
        # LOGIN SUCCESS
        # ----------------------------------------------------

        user = users_db[email]

        session['user_email'] = email

        session['user_name'] = user['name']

        session['profile_image'] = user.get(
            'profile_image'
        )


        return redirect(
            url_for('detect_hub')
        )


    return render_template(
        'login.html'
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':

        name = request.form.get(
            'name',
            ''
        ).strip()

        email = request.form.get(
            'email',
            ''
        ).strip().lower()

        password = request.form.get(
            'password',
            ''
        )


        # ----------------------------------------------------
        # CHECK EMPTY FIELDS
        # ----------------------------------------------------

        if not name or not email or not password:

            flash(
                "Please fill in all required fields.",
                "warning"
            )

            return redirect(
                url_for('signup')
            )


        # ----------------------------------------------------
        # CHECK EXISTING ACCOUNT
        # ----------------------------------------------------

        if email in users_db:

            flash(
                "Account already exists! Please login instead.",
                "info"
            )

            return redirect(
                url_for('login')
            )


        # ----------------------------------------------------
        # CREATE USER
        # ----------------------------------------------------

        users_db[email] = {

            "name": name,

            "password": password,

            "profile_image": None

        }


        # ----------------------------------------------------
        # SUCCESS MESSAGE
        # ----------------------------------------------------

        flash(
            "Account created successfully! Please login.",
            "success"
        )


        return redirect(
            url_for('login')
        )


    return render_template(
        'signup.html'
    )


# ============================================================
# GOOGLE LOGIN
# ============================================================

@app.route('/auth/google')
def auth_google():

    # --------------------------------------------------------
    # DEMO GOOGLE LOGIN
    #
    # This is only a placeholder because actual Google OAuth
    # has not been configured.
    # --------------------------------------------------------

    google_email = "googleuser@gmail.com"

    google_name = "Google User"


    # Create the demo user if it doesn't exist

    if google_email not in users_db:

        users_db[google_email] = {

            "name": google_name,

            "password": "oauth_user",

            "profile_image": None

        }


    # Login

    user = users_db[google_email]

    session['user_email'] = google_email

    session['user_name'] = user['name']

    session['profile_image'] = user.get(
        'profile_image'
    )


    flash(
        "Successfully logged in with Google!",
        "success"
    )


    return redirect(
        url_for('detect_hub')
    )


# ============================================================
# FORGOT PASSWORD
# ============================================================

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():

    if request.method == 'POST':

        email = request.form.get(
            'email',
            ''
        ).strip().lower()

        new_password = request.form.get(
            'password',
            ''
        )

        confirm_password = request.form.get(
            'confirm_password',
            ''
        )


        # ----------------------------------------------------
        # CHECK EMAIL
        # ----------------------------------------------------

        if email not in users_db:

            flash(
                "Email address not found in our system. Please sign up!",
                "danger"
            )

            return redirect(
                url_for('signup')
            )


        # ----------------------------------------------------
        # CHECK PASSWORDS
        # ----------------------------------------------------

        if new_password != confirm_password:

            flash(
                "Passwords do not match. Please try again.",
                "danger"
            )

            return redirect(
                url_for('forgot_password')
            )


        # ----------------------------------------------------
        # RESET PASSWORD
        # ----------------------------------------------------

        users_db[email]['password'] = new_password


        flash(
            "Password successfully reset! You can now login with your new password.",
            "success"
        )


        return redirect(
            url_for('login')
        )


    return render_template(
        'forgot_password.html'
    )


# ============================================================
# DETECTION HUB
# ============================================================

@app.route('/detect', methods=['GET', 'POST'])
def detect_hub():

    # --------------------------------------------------------
    # LOGIN REQUIRED
    # --------------------------------------------------------

    if 'user_email' not in session:

        return redirect(
            url_for('login')
        )


    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    if request.method == 'POST':

        if 'file' not in request.files:

            return redirect(
                request.url
            )


        file = request.files['file']


        if file.filename == '':

            return redirect(
                request.url
            )


        if file:

            # ------------------------------------------------
            # SECURE FILE NAME
            # ------------------------------------------------

            original_filename = secure_filename(
                file.filename
            )


            ext = os.path.splitext(
                original_filename
            )[1]


            # ------------------------------------------------
            # UNIQUE FILE NAME
            # ------------------------------------------------

            unique_filename = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:8]}"
                f"{ext}"
            )


            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                unique_filename
            )


            # ------------------------------------------------
            # SAVE IMAGE
            # ------------------------------------------------

            file.save(filepath)


            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            image = cv2.imread(
                filepath
            )


            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )


            resized = cv2.resize(
                image_rgb,
                (224, 224)
            )


            x = img_to_array(
                resized
            )


            x = preprocess_input(
                x
            )


            x = np.expand_dims(
                x,
                axis=0
            )


            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            preds = model.predict(
                x
            )[0]


            mask, withoutMask = preds


            is_mask = mask > withoutMask


            confidence = float(
                max(
                    mask,
                    withoutMask
                ) * 100
            )


            result_label = (
                "Mask"
                if is_mask
                else
                "No Mask"
            )


            face_status = (
                "Yes (With Mask)"
                if is_mask
                else
                "Yes (Without Mask)"
            )


            # ------------------------------------------------
            # IMAGE PATH
            # ------------------------------------------------

            image_path = (
                f'uploads/{unique_filename}'
            )


            # ------------------------------------------------
            # SAVE HISTORY
            # ------------------------------------------------

            history_item = {

                "id": len(detection_history) + 1,

                "image": image_path,

                "result": result_label,

                "confidence": round(
                    confidence,
                    2
                ),

                "date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            }


            detection_history.insert(
                0,
                history_item
            )


            # ------------------------------------------------
            # RETURN DETECTION RESULT
            # ------------------------------------------------

            return render_template(
                'detect.html',

                uploaded_image=url_for(
                    'static',
                    filename=image_path
                ),

                result_label=result_label,

                confidence=round(
                    confidence,
                    2
                ),

                face_detected=face_status
            )


    return render_template(
        'detect.html'
    )


# ============================================================
# LIVE CAMERA
# ============================================================

@app.route('/live')
def live_page():

    if 'user_email' not in session:

        return redirect(
            url_for('login')
        )


    return render_template(
        'live.html'
    )


# ============================================================
# HISTORY
# ============================================================

@app.route('/history')
def history_page():

    if 'user_email' not in session:

        return redirect(
            url_for('login')
        )


    return render_template(
        'history.html',
        history=detection_history
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route('/clear_history')
def clear_history():

    detection_history.clear()


    return redirect(
        url_for('history_page')
    )


# ============================================================
# PROFILE
# ============================================================

@app.route('/profile', methods=['GET', 'POST'])
def profile_page():

    # --------------------------------------------------------
    # LOGIN REQUIRED
    # --------------------------------------------------------

    if 'user_email' not in session:

        return redirect(
            url_for('login')
        )


    # --------------------------------------------------------
    # GET CURRENT USER
    # --------------------------------------------------------

    email = session.get(
        'user_email'
    )


    if email not in users_db:

        session.clear()

        return redirect(
            url_for('login')
        )


    user = users_db[email]


    # ========================================================
    # PROFILE PHOTO UPLOAD
    # ========================================================

    if request.method == 'POST':

        if 'profile_image' not in request.files:

            flash(
                "Please select a profile image.",
                "warning"
            )

            return redirect(
                url_for('profile_page')
            )


        file = request.files[
            'profile_image'
        ]


        if file.filename == '':

            return redirect(
                url_for('profile_page')
            )


        # ----------------------------------------------------
        # ALLOWED IMAGE TYPES
        # ----------------------------------------------------

        allowed_extensions = {

            'png',
            'jpg',
            'jpeg',
            'webp'

        }


        extension = file.filename.rsplit(
            '.',
            1
        )[-1].lower()


        if extension not in allowed_extensions:

            flash(
                "Please select a PNG, JPG, JPEG or WEBP image.",
                "danger"
            )

            return redirect(
                url_for('profile_page')
            )


        # ----------------------------------------------------
        # CREATE UNIQUE PROFILE IMAGE NAME
        # ----------------------------------------------------

        unique_filename = (
            f"profile_{uuid.uuid4().hex}.{extension}"
        )


        profile_path = os.path.join(

            app.config[
                'PROFILE_FOLDER'
            ],

            unique_filename

        )


        # ----------------------------------------------------
        # SAVE PROFILE IMAGE
        # ----------------------------------------------------

        file.save(
            profile_path
        )


        relative_path = (
            f"uploads/profiles/{unique_filename}"
        )


        # ----------------------------------------------------
        # SAVE IMAGE FOR THIS USER
        # ----------------------------------------------------

        users_db[email][
            'profile_image'
        ] = relative_path


        session[
            'profile_image'
        ] = relative_path


        flash(
            "Profile picture updated successfully!",
            "success"
        )


        return redirect(
            url_for('profile_page')
        )


    # ========================================================
    # USER INFORMATION
    # ========================================================

    name = user.get(
        'name',
        ''
    )


    profile_image = user.get(
        'profile_image'
    )


    # ========================================================
    # DETECTION STATISTICS
    # ========================================================

    total_scans = len(
        detection_history
    )


    mask_detected = sum(

        1

        for item in detection_history

        if item['result'] == 'Mask'

    )


    no_mask = sum(

        1

        for item in detection_history

        if item['result'] == 'No Mask'

    )


    # ========================================================
    # RENDER PROFILE
    # ========================================================

    return render_template(

        'profile.html',

        name=name,

        email=email,

        profile_image=profile_image,

        total_scans=total_scans,

        mask_detected=mask_detected,

        no_mask=no_mask

    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route('/video_feed')
def video_feed():

    return Response(

        generate_frames(),

        mimetype='multipart/x-mixed-replace; boundary=frame'

    )


# ============================================================
# LIVE STATS
# ============================================================

@app.route('/stats')
def stats():

    return jsonify(
        live_stats
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route('/logout')
def logout():

    session.clear()

    return redirect(
        url_for('login')
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == '__main__':

    app.run(
        debug=True
    )