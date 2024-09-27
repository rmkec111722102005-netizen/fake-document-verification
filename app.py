
from flask import Flask, request, render_template
import mysql.connector
from PyPDF2 import PdfReader
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import os
import cv2
import numpy as np
import PyPDF2
import logging
import re
import io

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="certificatedb2"
    )

# Initialize Flask app
app = Flask(__name__,static_folder='static')

# Set upload folder path
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Setup logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

# Global variable to store the Aadhaar holder name for comparison
aadhaar_holder_name = ""

@app.route('/')
def index():
    return render_template('index.html')  # Create an index.html for the form

# Aadhaar verification route
@app.route('/upload_aadhaar', methods=['POST'])
def upload_aadhaar():
    global aadhaar_holder_name
    aadhaar_number = request.form['aadhaar_number']
    name = request.form['name']
    dob = request.form['dob']
    gender = request.form['gender']

    aadhaar_holder_name = name.lower().strip()
    
    # Normalize the input DOB
    normalized_input_dob = normalize_date(dob)
    if not normalized_input_dob:
        logging.error(f"Failed to normalize input DOB: {dob}")
        return "Invalid date format for DOB. Use DD/MM/YYYY", 400

    if 'aadhaar_certificate' not in request.files:
        return "No Aadhaar file uploaded", 400

    aadhaar_file = request.files['aadhaar_certificate']

    if aadhaar_file.filename == '':
        return "No selected Aadhaar file", 400

    if aadhaar_file and aadhaar_file.filename.endswith('.pdf'):
        aadhaar_filepath = os.path.join(app.config['UPLOAD_FOLDER'], aadhaar_file.filename)
        aadhaar_file.save(aadhaar_filepath)

        pdf_text = ""

        try:
            with open(aadhaar_filepath, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    pdf_text += page.extract_text() or ""

            if not pdf_text.strip():
                logging.error("Empty text extracted from PDF")
                return "Could not extract text from the uploaded Aadhaar PDF", 400

        except Exception as e:
            logging.error(f"Error processing Aadhaar PDF file: {e}")
            return f"Error processing Aadhaar PDF file: {e}", 500

        pdf_text = pdf_text.lower().strip()
        aadhaar_number = aadhaar_number.lower().strip()
        name = name.lower().strip()
        gender = gender.lower().strip()
        
        # Extract and normalize DOB from PDF
        pdf_dob = extract_dob_from_pdf(pdf_text)

        logging.info("Comparing Aadhaar details:")
        logging.info(f"Aadhaar Number: {aadhaar_number}")
        logging.info(f"Aadhaar Number in PDF: {aadhaar_number in pdf_text}")
        logging.info(f"Name: {name}")
        logging.info(f"Name in PDF: {name in pdf_text}")
        logging.info(f"Date of Birth (Input): {normalized_input_dob}")
        logging.info(f"Date of Birth (PDF): {pdf_dob}")
        logging.info(f"Gender: {gender}")
        logging.info(f"Gender in PDF: {gender in pdf_text}")
        logging.info(f"Extracted PDF text: {pdf_text[:500]}...")  # Log first 500 characters of PDF text

        mismatch_reasons = []
        if aadhaar_number not in pdf_text:
            mismatch_reasons.append("Aadhaar number not found in PDF")
        if name not in pdf_text:
            mismatch_reasons.append("Name not found in PDF")
        if normalized_input_dob != pdf_dob:
            mismatch_reasons.append("Date of Birth mismatch")
        if gender not in pdf_text:
            mismatch_reasons.append("Gender not found in PDF")

        if not mismatch_reasons:
            
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO aadhaar (aadhaar_number, name, dob, gender, aadhaar_certificate)
                    VALUES (%s, %s, %s, %s, %s)
                """, (aadhaar_number, name, dob, gender, aadhaar_filepath))
                conn.commit()
            except mysql.connector.Error as err:
                logging.error(f"Database error: {err}")
                return f"Database error: {err}", 500
            finally:
                cursor.close()
                conn.close()
            return "Aadhaar details match successfully! Proceed to Smart Ration Card verification."
              
        
        else:
            logging.error(f"Aadhaar details mismatch reasons: {', '.join(mismatch_reasons)}")
            return f"Aadhaar details do not match! Reasons: {', '.join(mismatch_reasons)}", 400

    return "Invalid Aadhaar file format. Only PDFs are allowed", 400
def extract_dob_from_pdf(pdf_text):
    """Extract and normalize DOB from PDF text"""
    # Look for date patterns in the text
    date_patterns = [
        r'\b(\d{2}/\d{2}/\d{4})\b',  # DD/MM/YYYY
        r'\b(\d{2}-\d{2}-\d{4})\b',  # DD-MM-YYYY
        r'\b(\d{4}/\d{2}/\d{2})\b',  # YYYY/MM/DD
        r'\b(\d{4}-\d{2}-\d{2})\b' ,
         r'\b(\d{1,2} [A-Za-z]{3} \d{4})\b',   # D MMM YYYY or DD MMM YYYY
        r'\b(\d{1,2} [A-Za-z]+ \d{4})\b',     # D Month YYYY or DD Month YYYY
        r'\b([A-Za-z]+ \d{1,2}, \d{4})\b'  
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, pdf_text)
        if match:
            # Attempt to normalize the found date
            normalized_date = normalize_date(match.group(1))
            if normalized_date:
                return normalized_date
    
    return None
def normalize_date(date_string):
   date_formats = [
        "%d/%m/%Y",  # 01/01/1990
        "%d-%m-%Y",  # 01-01-1990
        "%Y/%m/%d",  # 1990/01/01
        "%Y-%m-%d",  # 1990-01-01
        "%d %b %Y",  # 01 Jan 1990
        "%d %B %Y",  # 01 January 1990
        "%B %d, %Y"  # January 01, 1990
    ]
    
   for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_string, fmt)
            return date_obj.strftime("%d-%m-%Y")
        except ValueError:
            continue
    
   return None

def detect_face(image):
    # Load OpenCV face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Extract the face from the image (use the first detected face)
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            return face
    
    return None

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files or 'pdf' not in request.files:
        return "Both image and PDF must be uploaded", 400

    image_file = request.files['image']
    pdf_file = request.files['pdf']

    if image_file.filename == '' or pdf_file.filename == '':
        return "No selected file", 400

    if image_file and pdf_file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)

        image_file.save(image_path)
        pdf_file.save(pdf_path)

        # Extract the first image from the PDF
        pdf_image_path = extract_image_from_pdf(pdf_path)

        if pdf_image_path is None:
            return "No images found in the PDF", 400

        match_result = compare_images(image_path, pdf_image_path)

        if match_result:
            return "Images match successfully!"
        else:
            return "Images do not match!", 400

    return "Invalid file format. Only image files and PDF are allowed", 400

def extract_image_from_pdf(pdf_path):
    """Extract the first image from a PDF and save it as a temporary image file."""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        # Attempt to extract images
        images = page.images
        if images:
            # Assuming we take the first image
            image = images[0]
            image_bytes = image.data
            image_name = 'extracted_image.png'  # Save it as PNG
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

            with open(image_path, 'wb') as img_file:
                img_file.write(image_bytes)

            return image_path
    return None

def compare_images(image1_path, image2_path):
    """Compare two images and return True if they match, False otherwise."""
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        return False
    
    face1 = detect_face(image1)
    face2 = detect_face(image2)

    if face1 is None or face2 is None:
        return False

    # Resize faces to a standard size
    face1 = cv2.resize(face1, (300, 300))
    face2 = cv2.resize(face2, (300, 300))

    # Resize images to a standard size
    image1 = cv2.resize(image1, (520, 520))
    image2 = cv2.resize(image2, (520, 520))
    

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images
    (score, diff) = ssim(gray1, gray2, full=True)

    # The SSIM score ranges from -1 to 1, where 1 indicates perfect similarity
    similarity_threshold = 0.5  # Adjust based on your needs

    return score > similarity_threshold

# Ration Card verification route
@app.route('/upload_ration_card', methods=['POST'])
def upload_ration_card():
    global aadhaar_holder_name  # Get the Aadhaar holder name for comparison
    # Get the form data for Smart Ration Card
    ration_card_number = request.form['ration_card_number']
    ration_card_name = request.form['ration_card_name']
    ration_card_gender = request.form['ration_card_gender']

    # Check if a file was uploaded
    if 'ration_card_certificate' not in request.files:
        return "No Ration Card file uploaded", 400

    ration_card_file = request.files['ration_card_certificate']

    if ration_card_file.filename == '':
        return "No selected Ration Card file", 400

    if ration_card_file and ration_card_file.filename.endswith('.pdf'):
        ration_card_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ration_card_file.filename)
        ration_card_file.save(ration_card_filepath)

        pdf_text = ""

        try:
            # Extract text from the Ration Card PDF file
            with open(ration_card_filepath, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    pdf_text += page.extract_text() or ""

            if not pdf_text.strip():
                return "Could not extract text from the uploaded Ration Card PDF", 400

        except Exception as e:
            return f"Error processing Ration Card PDF file: {e}", 500

        # Normalize the extracted text for comparison
        pdf_text = pdf_text.lower().strip()
        ration_card_number = ration_card_number.lower().strip()
        ration_card_name = ration_card_name.lower().strip()
        ration_card_gender = ration_card_gender.lower().strip()

        # Logging comparison details
        logging.debug("Comparing Ration Card details:")
        logging.debug(f"Ration Card Number: {ration_card_number}, in PDF text: {ration_card_number in pdf_text}")
        logging.debug(f"Aadhaar holder name: {aadhaar_holder_name}, in Ration Card PDF text: {aadhaar_holder_name in pdf_text}")
        logging.debug(f"Ration Card Gender: {ration_card_gender}, in PDF text: {ration_card_gender in pdf_text}")
        logging.debug(f"Aadhaar holder name: {aadhaar_holder_name}, in Ration Card PDF text: {aadhaar_holder_name in pdf_text}")
        
        if (ration_card_number in pdf_text and
            ration_card_name in pdf_text and
            ration_card_gender in pdf_text):
           
            # Store Ration Card details in the database
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO smart_card (ration_card_number, ration_card_name, ration_Card_gender, ration_card_certificate)
                    VALUES (%s, %s, %s, %s)
                """, (ration_card_number,ration_card_name, ration_card_gender, ration_card_filepath))
                conn.commit()
            except mysql.connector.Error as err:
                logging.error(f"Database error: {err}")
                return f"Database error: {err}", 500
            finally:
                cursor.close()
                conn.close()
            if aadhaar_holder_name in pdf_text:
                   return "Ration Card details and Aadhaar holder's name match successfully and stored in database!"
        else:
            return "Ration Card details or Aadhaar holder's name do not match!", 400

    return "Invalid Ration Card file format. Only PDFs are allowed", 400

@app.route('/upload_tc', methods=['POST'])
def upload_tc():
    # Get the form data for Transfer Certificate
    student_name = request.form['student_name']
    father_name = request.form['father_name']
    dob = request.form['dob']
    last_class = request.form['class']
    school_name = request.form['school_name']
    admission_date = request.form['admission_date']
    reason = request.form['reason']

    # Check if a file was uploaded
    if 'tc_certificate' not in request.files:
        return "No Transfer Certificate file uploaded", 400

    uploaded_file = request.files['tc_certificate']

    if uploaded_file.filename == '':
        return "No selected Transfer Certificate file", 400

    if uploaded_file and uploaded_file.filename.endswith('.pdf'):
        file_path = os.path.join("uploads", uploaded_file.filename)
        uploaded_file.save(file_path)

        extracted_text = ""

        try:
            # Extract text from the TC PDF file
            with open(file_path, 'rb') as pdf_file:
                reader = PdfReader(pdf_file)
                for page in reader.pages:
                    extracted_text += page.extract_text() or ""

            if not extracted_text.strip():
                return "Could not extract text from the uploaded Transfer Certificate PDF", 400

        except Exception as e:
            return f"Error processing Transfer Certificate PDF file: {e}", 500

        # Normalize the extracted text for comparison
        extracted_text = extracted_text.lower().strip()
        student_name = student_name.lower().strip()
        father_name = father_name.lower().strip()
        dob = dob.lower().strip()
        last_class = last_class.lower().strip()
        school_name = school_name.lower().strip()
        admission_date = admission_date.lower().strip()
        reason = reason.lower().strip()

        # Logging comparison details
        logging.debug("Comparing Transfer Certificate details:")
        logging.debug(f"Student Name: {student_name}, in PDF text: {student_name in extracted_text}")
        logging.debug(f"Father Name: {father_name}, in PDF text: {father_name in extracted_text}")
        logging.debug(f"DOB: {dob}, in PDF text: {dob in extracted_text}")
        logging.debug(f"Last Class: {last_class}, in PDF text: {last_class in extracted_text}")
        logging.debug(f"School Name: {school_name}, in PDF text: {school_name in extracted_text}")
        logging.debug(f"Admission Date: {admission_date}, in PDF text: {admission_date in extracted_text}")
        logging.debug(f"Reason: {reason}, in PDF text: {reason in extracted_text}")

        # Compare TC details with the extracted PDF text
        if (student_name in extracted_text and
            father_name in extracted_text and
            dob in extracted_text and
            last_class in extracted_text and
            school_name in extracted_text and
            admission_date in extracted_text and
            reason in extracted_text):
            
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO transfer_certificate 
                    (student_name, father_name, dob, class, school_name, admission_date, reason, tc_certificate)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (student_name, father_name, dob, last_class, school_name, admission_date, reason, file_path))
                conn.commit()
            except mysql.connector.Error as err:
                logging.error(f"Database error: {err}")
                return f"Database error: {err}", 500
            finally:
                cursor.close()
                conn.close()
            return "Transfer certificate verified successfully"
        else:
            return "Mismatch Found", 400

    return "Invalid Transfer Certificate file format. Only PDFs are allowed", 400

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
    return text

# New route to view PDF files
@app.route('/view_pdf/<doc_type>/<doc_id>', methods=['GET'])
def view_pdf(doc_type, doc_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if doc_type == 'aadhaar':
            cursor.execute("SELECT file_content FROM aadhaar WHERE id = %s", (doc_id,))
        elif doc_type == 'smart_card':
            cursor.execute("SELECT file_content FROM smart_card WHERE id = %s", (doc_id,))
        elif doc_type == 'transfer_certificate':
            cursor.execute("SELECT file_content FROM transfer_certificate WHERE id = %s", (doc_id,))
        else:
            return "Invalid document type", 400

        result = cursor.fetchone()
        if result:
            pdf_content = result[0]
            return send_file(io.BytesIO(pdf_content),
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'{doc_type}_{doc_id}.pdf'
            )
        else:
            return "Document not found", 404
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        return f"Database error: {err}", 500
    finally:
        cursor.close()
        conn.close()
        


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False,port=5002)