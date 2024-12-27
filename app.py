from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Hello from Auxilium!"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

from flask import Flask, request, send_file, jsonify, send_from_directory, session
import openai
import PyPDF2
import io
import os
import json
from datetime import datetime
import uuid  # Add this for generating unique IDs
from supabase import create_client  # Add this for Supabase
import requests
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, LayoutError
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTLine
from reportlab.lib.colors import black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import json
from supabase import create_client, Client
from functools import wraps
from flask import session, redirect, url_for
import os
from flask_cors import CORS
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import time
# Conditional import for jobspy
try:
    # from jobspy import scrape_jobs
    JOBSPY_AVAILABLE = False
except ImportError:
    JOBSPY_AVAILABLE = False
    print("Warning: jobspy not available, job search functionality will be limited")
# import pandas as pd
import pdfkit
import razorpay
import hmac
import hashlib
import base64
import tempfile
from elevenlabs import ElevenLabs
import os

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='../dist', static_url_path='')
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "https://s-5212-anushk97s-projects.vercel.app"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Configure session
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Error handling
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error (you might want to use a proper logging system)
    print(f"Unhandled Exception: {str(e)}")
    # Return JSON instead of HTML for HTTP errors
    return jsonify(error=str(e)), 500

@app.errorhandler(500)
def handle_500_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": str(error),
        "status": 500
    }), 500

@app.errorhandler(404)
def handle_404_error(error):
    return jsonify({
        "error": "Not found",
        "message": str(error),
        "status": 404
    }), 404

# API Keys and Configuration
openai.api_key = os.getenv('OPENAI_API_KEY')
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase_service_key = os.getenv('SUPABASE_SERVICE_KEY')
razorpay_key_id = os.getenv('RAZORPAY_KEY_ID')
razorpay_key_secret = os.getenv('RAZORPAY_KEY_SECRET')
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')

# Initialize the default Supabase client
supabase = create_client(supabase_url, supabase_key)

# Initialize Razorpay client
razorpay_client = razorpay.Client(
    auth=(razorpay_key_id, razorpay_key_secret)
)

def get_supabase_client():
    """Get a Supabase client with the current user's session"""
    try:
        if 'access_token' in session:
            # Create client with user's access token
            client = create_client(supabase_url, supabase_key)
            client.postgrest.auth(session['access_token'])
            return client
        return supabase
    except Exception as e:
        print(f"Error creating Supabase client: {str(e)}")
        return supabase

def get_storage_client():
    """Get a Supabase client with service role for storage operations"""
    try:
        # Create client with service role key
        client = create_client(supabase_url, supabase_service_key)
        return client
    except Exception as e:
        print(f"Error creating storage client: {str(e)}")
        return None

def get_admin_client():
    """Get a Supabase client with admin privileges"""
    return create_client(
        supabase_url,
        supabase_service_key  # Use service role key for admin operations
    )

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Authentication APIs
@app.route('/api/auth/check', methods=['GET'])
def check_auth():
    if 'user_id' in session and 'user_email' in session:
        return jsonify({
            "authenticated": True,
            "user": {
                "id": session['user_id'],
                "email": session['user_email']
            }
        }), 200
    return jsonify({"authenticated": False}), 401

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    try:
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "email_confirm": True  # Enable email confirmation
        })
        
        if auth_response.user:
            user_id = auth_response.user.id
            user_data = {
                'id': user_id,
                'email': email,
                'is_premium': False
            }
            get_supabase_client().table('users').insert(user_data).execute()
            
            return jsonify({
                "success": True,
                "message": "Please check your email to verify your account",
                "user": {
                    "id": user_id,
                    "email": email
                }
            }), 200
        else:
            raise Exception("User creation failed")
    except Exception as e:
        return jsonify({"error": "Signup failed"}), 400

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Missing email or password"}), 400
            
        email = data['email']
        password = data['password']
        
        # Get Supabase client
        supabase = get_supabase_client()
        if not supabase:
            return jsonify({"error": "Failed to initialize Supabase client"}), 500
            
        try:
            # Attempt login with Supabase
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if not response or not response.user:
                return jsonify({"error": "Invalid credentials"}), 401
                
            # Set session data
            session['user_id'] = response.user.id
            session['access_token'] = response.session.access_token
            session['refresh_token'] = response.session.refresh_token
            
            return jsonify({
                "message": "Login successful",
                "user": {
                    "id": response.user.id,
                    "email": response.user.email
                }
            })
            
        except Exception as e:
            print(f"Supabase login error: {str(e)}")
            return jsonify({"error": "Authentication failed", "details": str(e)}), 401
            
    except Exception as e:
        print(f"Login route error: {str(e)}")
        return jsonify({"error": "Server error", "details": str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True}), 200

@app.route('/api/subscribe')
def subscribe():
    return jsonify({"message": "Subscribe page"}), 200

@app.route('/api/create-subscription', methods=['POST'])
@login_required
def create_subscription():
    try:
        user_id = session.get('user_id')
        
        # Create a Razorpay Order
        subscription_data = {
            'amount': 399,  # ₹399 in paise (Razorpay expects amount in paise)
            'currency': 'USD',
            'payment_capture': 1,
            'notes': {
                'user_id': user_id,
                'package': 'Pro Subscription'
            }
        }
        
        order = razorpay_client.order.create(data=subscription_data)
        
        # Store order details in Supabase
        supabase.table('payment_history').insert({
            'user_id': user_id,
            'order_id': order['id'],
            'amount': order['amount'],
            'currency': order['currency'],
            'status': 'created'
        }).execute()
        
        return jsonify({
            'key_id': razorpay_key_id,
            'order_id': order['id'],
            'amount': order['amount'],
            'currency': order['currency']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Resume Generation API
@app.route('/api/generate', methods=['POST'])
@app.route('/api/resume/generate', methods=['POST'])  # Keep both routes for backward compatibility
@login_required
def generate():
    try:
        user_id = session.get('user_id')
        
        # Check resume generation limit
        can_generate, error_message = check_resume_limit(user_id)
        if not can_generate:
            return jsonify({'error': error_message}), 403
            
        modification_level = request.form.get('modification_level', 'aggressive')
        action = request.form.get('action', 'resume_only')
        template_id = request.form.get('template', 'modern')  # Default to modern template
        custom_instructions = request.form.get('custom_instructions')
        
        resume_file = request.files['resume']
        original_filename = resume_file.filename
        
        # Get job description
        job_description = ""
        if request.form.get('job_url'):
            job_url = request.form['job_url']
            job_description = requests.get(job_url).text
        elif request.form.get('job_description'):
            job_description = request.form['job_description']
        else:
            return jsonify({"error": "Please provide either a job URL or job description"}), 400
        
        # Extract and process resume
        resume_text = extract_resume_text(resume_file)
        tailored_resume = structure_resume_with_openai(resume_text, job_description, modification_level, custom_instructions)
        # Apply template styling
        # styled_resume = format_resume_with_template(tailored_resume, template_id)
        
        # Set base filename based on action
        base_filename = 'tailored_resume.pdf' if action == 'resume_only' else 'resume_and_cover_letter.pdf'
        
        if action == 'resume_only':
            pdf_buffer = create_formatted_pdf(tailored_resume)
        else:
            cover_letter = generate_cover_letter(tailored_resume, job_description)
            resume_pdf = create_formatted_pdf(tailored_resume)
            cover_letter_pdf = create_cover_letter_pdf(cover_letter)
            
            merger = PyPDF2.PdfMerger()
            merger.append(resume_pdf)
            merger.append(cover_letter_pdf)
            
            pdf_buffer = BytesIO()
            merger.write(pdf_buffer)
            pdf_buffer.seek(0)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{session['user_id']}_{timestamp}_{base_filename}"
        storage_path = f"resumes/{unique_filename}"
        
        # Upload to Supabase storage
        storage = get_storage_client()
        if not storage:
            return jsonify({"error": "Failed to initialize storage client"}), 500
            
        try:
            # Get the PDF content
            pdf_content = pdf_buffer.getvalue()
            
            # Upload to storage using service role client
            response = upload_to_storage(pdf_content, storage_path)
            if not response:
                return jsonify({"error": "Failed to upload resume to storage"}), 500
            
            # Save to history
            save_resume_history(
                user_id=session['user_id'],
                original_filename=original_filename,
                generated_filename=storage_path,
                job_description=job_description
            )
            
            # Reset buffer position for sending file
            pdf_buffer.seek(0)
            
            
            
            # # Convert HTML to PDF
            # pdf_buffer = BytesIO()
            # pdfkit.from_string(styled_resume, pdf_buffer)
            # pdf_buffer.seek(0)
            
            return send_file(
                pdf_buffer,
                download_name=base_filename,
                as_attachment=True,
                mimetype='application/pdf'
            )
            
        except Exception as storage_error:
            print(f"Storage error: {str(storage_error)}")
            return jsonify({"error": "Failed to save resume to storage"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def upload_to_storage(file_content, path):
    """Upload file directly to Supabase storage"""
    try:
        # Get storage client with service role
        storage = get_storage_client()
        if not storage:
            print("Failed to get storage client")
            return False
            
        try:
            # Upload file directly to the bucket using service role
            response = storage.storage.from_('resumes').upload(
                path=path,
                file=file_content,
                file_options={"contentType": "application/pdf"}
            )
            
            print(f"Upload response: {response}")  # Debug log
            return True
                
        except Exception as upload_error:
            print(f"Upload error: {str(upload_error)}")
            if hasattr(upload_error, 'response'):
                print(f"Upload error response: {upload_error.response.text if hasattr(upload_error.response, 'text') else 'No response text'}")
            return False
            
    except Exception as e:
        print(f"Error in upload_to_storage: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Error response: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
        return False

def save_resume_history(user_id, original_filename, generated_filename, job_description, pdf_buffer=None):
    print(f"Debug - user_id: {user_id}")
    print(f"Debug - original_filename: {original_filename}")
    
    try:
        # Ensure all required fields are present and of correct type
        if not all([user_id, original_filename, generated_filename]):
            print("Debug - missing required fields")
            return False
            
        # Extract job title for resume name
        resume_name = extract_job_title(job_description)
        if not resume_name:
            resume_name = original_filename
            
        # Create a unique filename for storage with user folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        storage_path = f"{user_id}/{timestamp}_{generated_filename}"  # Store in user-specific folder
            
        data = {
            "user_id": str(user_id),  # Ensure UUID is string
            "original_filename": str(resume_name),  # Use the extracted job title or original filename
            "generated_filename": str(storage_path),  # Store the full storage path
            "job_description": str(job_description[:1000]) if job_description else "",
            "created_at": datetime.utcnow().isoformat()  # Use UTC time and ISO format
        }
        print(f"Debug - data to insert: {data}")
        
        try:
            # Get the current user's access token from the session
            access_token = session.get('access_token')
            if not access_token:
                print("Debug - no access token found in session")
                return False
                
            # Use user client for database operations
            db_client = create_client(supabase_url, supabase_key)
            db_client.postgrest.auth(access_token)
            
            # Try to insert with explicit error handling
            result = db_client.table('resume_history').insert(data).execute()
            print(f"Debug - insert result: {result.data}")
            return True
            
        except Exception as insert_error:
            print(f"Debug - insert error type: {type(insert_error)}")
            print(f"Debug - insert error message: {str(insert_error)}")
            if hasattr(insert_error, 'response'):
                print(f"Debug - insert error response: {insert_error.response.text if hasattr(insert_error.response, 'text') else 'No response text'}")
            return False
            
    except Exception as e:
        print(f"Debug - error type: {type(e)}")
        print(f"Debug - error message: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Debug - error response: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
        return False

def refresh_token_if_needed():
    try:
        # Get stored tokens
        access_token = session.get('access_token')
        refresh_token = session.get('refresh_token')
        
        if not access_token or not refresh_token:
            return False
            
        # Create client and attempt to refresh
        client = create_client(supabase_url, supabase_key)
        refresh_response = client.auth.refresh_session(refresh_token)
        
        if refresh_response:
            # Update session with new tokens
            new_access_token = refresh_response.session.access_token
            new_refresh_token = refresh_response.session.refresh_token
            session['access_token'] = new_access_token
            session['refresh_token'] = new_refresh_token
            return True
            
    except Exception as e:
        print(f"Error refreshing token: {str(e)}")
        return False
    
    return False

@app.route('/api/resume/latest', methods=['GET'])
@login_required
def get_latest_resume():
    try:
        # Get the current user's access token
        access_token = session.get('access_token')
        if not access_token:
            return jsonify({"error": "Authentication required"}), 401
            
        # Use user client for database operations
        db_client = create_client(supabase_url, supabase_key)
        db_client.postgrest.auth(access_token)
        
        try:
            response = db_client.table('resume_history') \
                .select('id,original_filename,generated_filename,created_at') \
                .eq('user_id', session['user_id']) \
                .order('created_at', desc=True) \
                .limit(1) \
                .execute()
        except Exception as db_error:
            if 'JWT expired' in str(db_error):
                # Try to refresh the token
                if refresh_token_if_needed():
                    # Retry with new token
                    db_client.postgrest.auth(session['access_token'])
                    response = db_client.table('resume_history') \
                        .select('id,original_filename,generated_filename,created_at') \
                        .eq('user_id', session['user_id']) \
                        .order('created_at', desc=True) \
                        .limit(1) \
                        .execute()
                else:
                    return jsonify({"error": "Session expired. Please log in again."}), 401
            else:
                raise db_error

        if not response.data or len(response.data) == 0:
            return jsonify({"success": False, "message": "No resume found"}), 404

        latest_resume = response.data[0]
        print(f"Found latest resume: {latest_resume}")
        
        # Get the file content from storage using service role client
        storage_filename = latest_resume['generated_filename']
        print(f"Attempting to download file: {storage_filename}")
        
        storage_client = get_storage_client()
        if not storage_client:
            return jsonify({"error": "Storage client initialization failed"}), 500
        
        try:
            # Download file from storage using service role client
            response = storage_client.storage.from_('resumes').download(storage_filename)
            
            if not response:
                return jsonify({"error": "Failed to download file from storage"}), 500
            
            # Create BytesIO object from the response content
            file_buffer = BytesIO(response)
            
            # Get the original filename from the path
            filename = os.path.basename(storage_filename)
            
            return jsonify({
                "success": True,
                "resume": {
                    "id": latest_resume['id'],
                    "filename": latest_resume['original_filename'],
                    "content": file_buffer.getvalue().decode('latin-1')  # Use latin-1 to handle binary data
                }
            }), 200
        except Exception as storage_error:
            print(f"Error accessing storage: {storage_error}")
            if 'Bucket not found' in str(storage_error):
                return jsonify({
                    "error": "Storage not configured",
                    "message": "Please create a 'resumes' bucket in your Supabase dashboard"
                }), 400
            return jsonify({"success": False, "message": f"Failed to download resume: {str(storage_error)}"}), 500
            
    except Exception as e:
        print(f"Error fetching latest resume: {str(e)}")
        return jsonify({"error": "Failed to retrieve resume"}), 500

def truncate_text(text, max_chars=3000):
    """Truncate text to stay within token limits"""
    return text[:max_chars] + ("..." if len(text) > max_chars else "")

def tailor_resume(structured_resume, job_description):
    """Tailor the structured resume to the job description"""
    system_prompt = """You are an expert ATS resume optimizer. Your task is to:
    1. Analyze the job requirements carefully
    2. Identify key skills and experiences that match
    3. Rewrite bullet points to highlight relevant achievements. Feel free to expand on the achievements or make up new one to come across as a strong candidate for the job including relevant keywords
    4. Use industry-specific keywords from the {job_description}
    5. Maintain truthfulness and accuracy of all information
    6. Ensure all modifications enhance ATS compatibility"""

    prompt = f"""Given this job description and structured resume, create a better version:

    JOB DESCRIPTION:
    {job_description}

    CURRENT RESUME:
    {json.dumps(structured_resume, indent=2)}

    Optimize the resume while maintaining the same JSON structure. Focus on:
    - Matching keywords and skills
    - Quantifying and elaborating on achievements adding more content and relevant keywords
    - Make up new achievements to come across as a strong candidate for the job including relevant keywords
    - Make sure the resume is effective and enhanced
    - Using action verbs
    - Maintaining ATS compatibility"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        tailored_resume = json.loads(response.choices[0].message.content)
        return tailored_resume
    except Exception as e:
        print(f"Error tailoring resume: {e}")
        return structured_resume

def analyze_pdf_formatting(pdf_file):
    """Analyze the formatting of the original PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    page_size = (
        float(pdf_reader.pages[0].mediabox.width),
        float(pdf_reader.pages[0].mediabox.height)
    )
    
    pdf_file.seek(0)
    
    formatting = {
        'fonts': set(),
        'font_sizes': set(),
        'alignments': [],
        'margins': {'left': float('inf'), 'right': 0, 'top': float('inf'), 'bottom': 0},
        'sections': [],
        'line_spacing': set(),
        'page_size': page_size
    }
    
    for page_layout in extract_pages(pdf_file):
        last_y = None
        
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                x0, y0, x1, y1 = element.bbox
                
                formatting['margins']['left'] = min(formatting['margins']['left'], x0)
                formatting['margins']['right'] = max(formatting['margins']['right'], x1)
                formatting['margins']['top'] = min(formatting['margins']['top'], y1)
                formatting['margins']['bottom'] = max(formatting['margins']['bottom'], y0)
                
                if last_y is not None:
                    line_spacing = abs(last_y - y0)
                    if line_spacing > 0:
                        formatting['line_spacing'].add(round(line_spacing, 1))
                last_y = y0
                
                for obj in element._objs:
                    if isinstance(obj, LTChar):
                        formatting['fonts'].add(obj.fontname)
                        formatting['font_sizes'].add(round(obj.size, 1))
                
                formatting['sections'].append({
                    'text': element.get_text().strip(),
                    'bbox': element.bbox,
                    'font_size': next((obj.size for obj in element._objs if isinstance(obj, LTChar)), 11),
                    'is_bold': any('Bold' in obj.fontname for obj in element._objs if isinstance(obj, LTChar))
                })
    
    formatting['font_sizes'] = sorted(formatting['font_sizes'])
    formatting['line_spacing'] = sorted(formatting['line_spacing'])
    
    return formatting

def create_styles_from_formatting(formatting):
    """Create reportlab styles based on the analyzed formatting"""
    styles = {}
    
    # Get the most common font size for body text
    body_size = sorted(formatting['font_sizes'])[len(formatting['font_sizes'])//2] if formatting['font_sizes'] else 11
    
    # Create base style
    styles['Normal'] = ParagraphStyle(
        'Normal',
        fontSize=body_size,
        leading=body_size * 1.2,
        spaceBefore=6,
        spaceAfter=6,
        alignment=TA_LEFT,
        wordWrap='CJK',
    )
    
    # Create header style (for name)
    styles['Header'] = ParagraphStyle(
        'Header',
        parent=styles['Normal'],
        fontSize=body_size * 1.5,
        leading=body_size * 1.8,
        alignment=TA_LEFT,
        spaceBefore=12,
        spaceAfter=12,
        textTransform='uppercase'
    )
    
    # Create contact info style
    styles['Contact'] = ParagraphStyle(
        'Contact',
        parent=styles['Normal'],
        fontSize=body_size,
        alignment=TA_LEFT,
        spaceAfter=12
    )
    
    # Create section header style
    styles['Section'] = ParagraphStyle(
        'Section',
        parent=styles['Normal'],
        fontSize=body_size * 1.2,
        leading=body_size * 1.4,
        spaceBefore=16,
        spaceAfter=8,
        textTransform='uppercase',
        bold=True
    )
    
    # Create bullet style
    styles['Bullet'] = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=body_size,
        leading=body_size * 1.2,
        leftIndent=20,
        firstLineIndent=-15,
        spaceBefore=3,
        spaceAfter=3
    )
    
    return styles

def extract_resume_text(pdf_file):
    """Extract raw text from the PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()
    
    return resume_text

def get_modification_prompt(level):
    """Get the appropriate prompt based on modification level"""
    prompts = {
        'conservative': """Enhance the resume while preserving its original content and structure:
        1. Match keywords from job description naturally:
           - Use exact phrases from job requirements
           - Keep original meaning but use industry-standard terms
           - Maintain authenticity of experiences
        2. Enhance existing achievements:
           - Add specific numbers and metrics (e.g., "increased efficiency by 40%")
           - Use strong action verbs (e.g., "spearheaded" instead of "led")
           - Highlight outcomes and impact
        3. Prioritize relevant skills:
           - Move most relevant experiences to top of each section
           - Emphasize technologies mentioned in job posting
           - Keep all original content but reorganize for relevance""",
        
        'balanced': """Optimize the resume with targeted modifications:
        1. Enhance achievements with specifics:
           - Add concrete metrics (e.g., "managed $500K budget")
           - Include team sizes and project scopes
           - Quantify impact and results
        2. Align experiences with job requirements:
           - Highlight relevant projects and responsibilities
           - Add technical details matching job needs
           - Emphasize transferable skills
        3. Optimize skills and technologies:
           - Prioritize skills mentioned in job posting
           - Add relevant technical expertise from background
           - Group skills by job requirements
        4. Strengthen professional summary:
           - Focus on job-specific qualifications
           - Include key achievements
           - Use industry keywords strategically""",
        
        'aggressive': """Transform the resume for maximum job alignment:
        1. Reframe experiences with detailed context:
           - Add detailed project details
           - Include specific technologies and methodologies
           - Quantify all possible achievements
        2. Expand technical implementations:
           - Detail architecture decisions
           - Specify tools and frameworks used
           - Include performance improvements
        3. Highlight leadership and impact:
           - Add team management details
           - Include budget responsibilities
           - Specify business impact with metrics
        4. Maximize keyword optimization:
           - Use exact terms from job posting
           - Include industry-specific certifications
           - Add relevant methodologies
        5. Focus on achievements:
           - Add specific success metrics
           - Include project outcomes and elaborate on impact. If not impact is given then make one.
           - Highlight innovations and optimizations
           - Keep each achievement 3-4 sentences long"""
    }
    return prompts.get(level, prompts['aggressive'])

def structure_resume_with_openai(resume_text, job_description, modification_level, custom_instructions=None):
    """Use OpenAI to structure and analyze the resume text with specified modification level"""
    modification_prompt = get_modification_prompt(modification_level)
    
    base_system_prompt = f"""You are an expert ATS-optimized resume writer. {modification_prompt}
    
    Your task is to structure and optimize the resume content according to these guidelines.
    IMPORTANT: Your response must be a valid JSON object with the exact structure provided."""

    if custom_instructions:
        system_prompt = base_system_prompt + f"\n\nAdditional Custom Instructions:\n{custom_instructions}"
    else:
        system_prompt = base_system_prompt

    base_prompt = f"""Use the {modification_prompt} to optimize the resume according to the job description.
    
    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {job_description}

    Format the cover letter with:
    1. Proper greeting
    2. 3-4 main paragraphs
    3. Professional closing
    4. If the company name is not known, do NOT mention company name
    
    Focus on matching the candidate's most relevant experiences with the job requirements.
    Do NOT include any header information, dates, addresses, or contact details."""

    if custom_instructions:
        prompt = base_prompt + f"\n\nCUSTOM INSTRUCTIONS:\n{custom_instructions}"
    else:
        prompt = base_prompt

    prompt += """

    Return a valid JSON object with exactly this structure (do not modify the structure):
    {
        "contact_info": {
            "name": "string",
            "email": "string",
            "phone": "string",
            "location": "string",
            "linkedin": "string"
        },
        "professional_summary": "string",
        "core_competencies": ["string"],
        "work_experience": [
            {
                "title": "string",
                "company": "string",
                "dates": "string",
                "location": "string",
                "achievements": ["string"]
            }
        ],
        "education": [
            {
                "degree": "string",
                "institution": "string",
                "dates": "string",
                "details": "string"
            }
        ],
        "technical_skills": {
            "languages": ["string"],
            "frameworks": ["string"],
            "tools": ["string"],
            "other": ["string"]
        }
    }

    IMPORTANT: Ensure your response is a valid JSON object. Do not include any explanations or text outside the JSON object."""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            # response_format={ "type": "json_object" }  # Ensure JSON response,
            max_tokens=10000
        )
        
        response_text = response.choices[0].message.content
        print(f"Debug - Response text: {response_text}")
        try:
            structured_resume = json.loads(response_text)
            return structured_resume
        except json.JSONDecodeError as je:
            print(f"JSON Decode Error: {je}")
            # Return a default structure if JSON parsing fails
            return {
                "contact_info": {
                    "name": "",
                    "email": "",
                    "phone": "",
                    "location": "",
                    "linkedin": ""
                },
                "professional_summary": "Error processing resume",
                "core_competencies": [],
                "work_experience": [],
                "education": [],
                "technical_skills": {
                    "languages": [],
                    "frameworks": [],
                    "tools": [],
                    "other": []
                }
            }
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return {"error": str(e)}

def create_formatted_pdf(tailored_resume):
    """Create a well-formatted PDF using reportlab"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Add custom styles
    styles.add(ParagraphStyle(
        name='Name',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=TA_CENTER
    ))
    
    styles.add(ParagraphStyle(
        name='Section',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12,
        textTransform='uppercase',
        bold=True
    ))
    
    styles.add(ParagraphStyle(
        name='Content',
        parent=styles['Normal'],
        fontSize=10,
        leading=14
    ))

    styles.add(ParagraphStyle(
        name='JobTitle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceBefore=8,
        bold=True
    ))

    styles.add(ParagraphStyle(
        name='BulletPoint',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=20,
        firstLineIndent=-15
    ))
    
    try:
        # Contact Information
        if 'contact_info' in tailored_resume:
            contact = tailored_resume['contact_info']
            story.append(Paragraph(contact.get('name', ''), styles['Name']))
            
            contact_text = []
            if contact.get('email'): contact_text.append(contact['email'])
            if contact.get('phone'): contact_text.append(contact['phone'])
            if contact.get('location'): contact_text.append(contact['location'])
            if contact.get('linkedin'): contact_text.append(contact['linkedin'])
            
            story.append(Paragraph(' • '.join(contact_text), styles['Content']))
            story.append(Paragraph('<br/>', styles['Content']))

        # Professional Summary
        if tailored_resume.get('professional_summary'):
            story.append(Paragraph('PROFESSIONAL SUMMARY', styles['Section']))
            story.append(Paragraph(tailored_resume['professional_summary'], styles['Content']))
            story.append(Paragraph('<br/>', styles['Content']))

        # Core Competencies
        if tailored_resume.get('core_competencies'):
            story.append(Paragraph('CORE COMPETENCIES', styles['Section']))
            competencies = ' • '.join(tailored_resume['core_competencies'])
            story.append(Paragraph(competencies, styles['Content']))
            story.append(Paragraph('<br/>', styles['Content']))

        # Work Experience
        if tailored_resume.get('work_experience'):
            story.append(Paragraph('PROFESSIONAL EXPERIENCE', styles['Section']))
            for job in tailored_resume['work_experience']:
                # Job title and company
                title_line = f"{job.get('title', '')} - {job.get('company', '')}"
                story.append(Paragraph(title_line, styles['JobTitle']))
                
                # Dates and location
                details_line = []
                if job.get('dates'): details_line.append(job['dates'])
                if job.get('location'): details_line.append(job['location'])
                if details_line:
                    story.append(Paragraph(' • '.join(details_line), styles['Content']))
                
                # Achievements/bullet points
                for achievement in job.get('achievements', []):
                    bullet_text = f"• {achievement}"
                    story.append(Paragraph(bullet_text, styles['BulletPoint']))
                
                story.append(Paragraph('<br/>', styles['Content']))

        # Technical Skills
        if tailored_resume.get('technical_skills'):
            story.append(Paragraph('TECHNICAL SKILLS', styles['Section']))
            skills = tailored_resume['technical_skills']
            for category, items in skills.items():
                if items:
                    skill_text = f"<b>{category.title()}:</b> {', '.join(items)}"
                    story.append(Paragraph(skill_text, styles['Content']))
            story.append(Paragraph('<br/>', styles['Content']))

        # Education
        if tailored_resume.get('education'):
            story.append(Paragraph('EDUCATION', styles['Section']))
            for edu in tailored_resume['education']:
                edu_text = f"<b>{edu.get('degree', '')}</b>"
                if edu.get('institution'): edu_text += f" - {edu['institution']}"
                story.append(Paragraph(edu_text, styles['Content']))
                if edu.get('dates'): 
                    story.append(Paragraph(edu['dates'], styles['Content']))
                if edu.get('details'):
                    story.append(Paragraph(edu['details'], styles['Content']))
                story.append(Paragraph('<br/>', styles['Content']))

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        print(f"Error creating PDF: {e}")
        raise Exception("Failed to generate PDF. Please try again.")

def truncate_job_description(job_description, max_tokens=4000):
    """Truncate job description to fit within token limits while preserving key information"""
    system_prompt = """You are a job description analyzer. Extract and summarize the most important information from this job description, focusing on:
    1. Key responsibilities
    2. Required skills and qualifications
    3. Technical requirements
    Keep only the most relevant information within the token limit."""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Using 3.5 for summarization to save tokens
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this job description in a concise way:\n\n{job_description}"}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        # If API call fails, fall back to simple truncation
        return job_description[:8000] + "..."

def generate_cover_letter(resume_info, job_description):
    """Generate a tailored cover letter based on resume and job description"""
    system_prompt = """You are an expert cover letter writer. Your task is to:
    1. Create a compelling cover letter that matches the job requirements
    2. Use the candidate's experience from their resume to demonstrate fit
    3. Keep the tone professional but personable
    4. Focus on specific achievements that relate to the job
    5. Keep the length to one page (around 300-400 words)
    6. Structure it in clear paragraphs
    7. Start directly with 'Dear Hiring Manager,' without any header information
    8. Do NOT include candidate's contact information or date
    9. Do NOT include company address or other header formatting"""

    prompt = f"""Create a professional cover letter based on this candidate's resume and the job description:

    RESUME INFO:
    {json.dumps(resume_info, indent=2)}

    JOB DESCRIPTION:
    {job_description}

    Format the cover letter with:
    1. Proper greeting
    2. 3-4 main paragraphs
    3. Professional closing
    4. If the company name is not known, do NOT mention company name
    
    Focus on matching the candidate's most relevant experiences with the job requirements.
    Do NOT include any header information, dates, addresses, or contact details."""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating cover letter: {e}")
        return None

def create_cover_letter_pdf(cover_letter_text):
    """Create a PDF of the cover letter"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=1*inch,
        rightMargin=1*inch,
        topMargin=1*inch,
        bottomMargin=1*inch
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Add custom styles
    styles.add(ParagraphStyle(
        name='CoverLetter',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceBefore=6,
        spaceAfter=6
    ))
    
    # Split the cover letter into paragraphs
    paragraphs = cover_letter_text.split('\n\n')
    
    # Add each paragraph to the story
    for paragraph in paragraphs:
        if paragraph.strip():
            story.append(Paragraph(paragraph.strip(), styles['CoverLetter']))
            story.append(Paragraph('<br/>', styles['CoverLetter']))
    
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error creating cover letter PDF: {e}")
        raise Exception("Failed to generate cover letter PDF.")

def extract_skills(job_descriptions):
    try:
        # Combine all job descriptions for analysis
        combined_text = " ".join(job_descriptions)
        
        # Use OpenAI to extract key skills with improved prompt
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert job market analyst specializing in identifying technical skills, tools, and technologies. Focus on extracting specific, actionable skills that can be used for course recommendations."},
                {"role": "user", "content": f"""Analyze these job descriptions and extract key technical skills, tools, and technologies. 
                Focus on:
                1. Programming languages
                2. Frameworks and libraries
                3. Tools and platforms
                4. Technical concepts and methodologies
                
                Return only a comma-separated list of the most relevant skills: {combined_text}"""}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Process and clean the extracted skills
        content = response.choices[0].message.content.strip()
        skills = [skill.strip() for skill in content.split(',') if skill.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = [x for x in skills if not (x.lower() in seen or seen.add(x.lower()))]
        
        return unique_skills[:5]  # Return top 5 most relevant skills
    except Exception as e:
        print(f"Error extracting skills: {str(e)}")
        return []

def search_courses(skill):
    try:
        courses = []
        
        # Search Coursera with improved parameters
        coursera_api_url = "https://api.coursera.org/api/courses.v1"
        params = {
            'q': 'search',
            'query': f"{skill} programming development",  # Improved search query
            'fields': 'name,description,photoUrl,slug,primaryLanguages,workload,rating',
            'limit': 10,  # Request more courses for better filtering
            'includes': 'primaryLanguages'
        }
        
        # Add rate limiting
        time.sleep(0.5)  # Basic rate limiting
        
        response = requests.get(coursera_api_url, params=params, timeout=10)
        if response.ok:
            coursera_data = response.json()
            coursera_courses = []
            
            # Improved course filtering
            for course in coursera_data.get('elements', []):
                if len(coursera_courses) >= 2:
                    break
                    
                # Check language and ensure course has a description
                languages = course.get('primaryLanguages', [])
                if (not languages or 'en' in languages) and course.get('description'):
                    coursera_courses.append({
                        'platform': 'Coursera',
                        'title': course['name'],
                        'description': course.get('description', '')[:200] + '...' if len(course.get('description', '')) > 200 else course.get('description', ''),
                        'url': f"https://www.coursera.org/learn/{course['slug']}",
                        'thumbnail': course.get('photoUrl', ''),
                        'workload': course.get('workload', 'Flexible'),
                        'rating': course.get('rating', 'Not rated')
                    })
            
            courses.extend(coursera_courses)
        
        # Only add YouTube courses if needed and YouTube API is available
        youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        if youtube_api_key and len(courses) < 4:
            youtube_url = 'https://www.googleapis.com/youtube/v3/search'
            params = {
                'key': youtube_api_key,
                'q': f"{skill} programming tutorial course",
                'part': 'snippet',
                'maxResults': 3,
                'type': 'video',
                'relevanceLanguage': 'en',
                'videoDuration': 'long',  # Prefer longer, more comprehensive videos
                'order': 'rating'  # Sort by rating
            }
            
            # Add rate limiting
            time.sleep(0.5)  # Basic rate limiting
            
            response = requests.get(youtube_url, params=params, timeout=10)
            if response.ok:
                data = response.json()
                youtube_courses = []
                
                for item in data.get('items', []):
                    if len(youtube_courses) >= 2:
                        break
                        
                    # Only add videos with proper titles and descriptions
                    title = item['snippet']['title']
                    description = item['snippet']['description']
                    
                    if title and description and not any(word in title.lower() for word in ['spam', 'hack', 'cheat']):
                        youtube_courses.append({
                            'platform': 'YouTube',
                            'title': title,
                            'description': description[:200] + '...' if len(description) > 200 else description,
                            'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                            'thumbnail': item['snippet']['thumbnails']['medium']['url']
                        })
                
                courses.extend(youtube_courses)
        
        return courses
    except requests.RequestException as e:
        print(f"Network error in search_courses: {str(e)}")
        return []
    except Exception as e:
        print(f"Error searching courses: {str(e)}")
        return []

@app.route('/api/courses/recommend', methods=['GET'])
def recommend_courses():
    try:
        user_id = session.get('user_id')
        access_token = session.get('access_token')
        
        if not user_id or not access_token:
            return jsonify({"error": "Not authenticated"}), 401

        # Get user's job descriptions from history
        client = create_client(supabase_url, supabase_key)
        client.postgrest.auth(access_token)
        
        response = client.table('resume_history').select('job_description').eq('user_id', user_id).execute()
        job_descriptions = [item['job_description'] for item in response.data if item['job_description']]

        if not job_descriptions:
            return jsonify({
                "message": "No job descriptions found. Please upload a resume with a job description to get course recommendations.",
                "recommendations": []
            }), 200

        # Extract skills from job descriptions
        skills = extract_skills(job_descriptions)
        
        if not skills:
            return jsonify({
                "message": "No relevant skills found in your job descriptions. Try updating your job descriptions with more technical details.",
                "recommendations": []
            }), 200
        
        # Search for courses for each skill
        all_courses = []
        for skill in skills[:3]:  # Limit to top 3 skills
            skill_courses = search_courses(skill)
            if skill_courses:
                all_courses.append({
                    'skill': skill,
                    'courses': skill_courses
                })

        if not all_courses:
            return jsonify({
                "message": "No courses found at the moment. Please try again later.",
                "skills": skills,
                "recommendations": []
            }), 200

        return jsonify({
            "message": "Successfully found course recommendations based on your job descriptions.",
            "skills": skills,
            "recommendations": all_courses
        }), 200

    except Exception as e:
        print(f"Error in recommend_courses: {str(e)}")
        return jsonify({
            "error": "Failed to get course recommendations. Please try again later.",
            "details": str(e)
        }), 500

@app.route('/api/resume/rename', methods=['POST'])
def rename_resume():
    try:
        user_id = session.get('user_id')
        access_token = session.get('access_token')
        
        print(f"Debug - rename_resume - user_id: {user_id}")
        
        if not user_id or not access_token:
            print("Debug - rename_resume - missing authentication")
            return jsonify({"error": "Not authenticated"}), 401

        data = request.get_json()
        if not data:
            print("Debug - rename_resume - no data provided")
            return jsonify({"error": "No data provided"}), 400
            
        resume_id = data.get('resume_id')
        new_name = data.get('new_name')
        
        # Ensure the new name has a .pdf extension
        if not new_name.lower().endswith('.pdf'):
            new_name = f"{new_name}.pdf"
        
        print(f"Debug - rename_resume - resume_id: {resume_id}, new_name: {new_name}")
        
        if not resume_id or not new_name:
            print("Debug - rename_resume - missing required fields")
            return jsonify({"error": "Missing resume_id or new_name"}), 400
            
        # Create Supabase client and set auth token
        client = create_client(supabase_url, supabase_key)
        client.postgrest.auth(access_token)
        
        # First verify the resume exists and belongs to the user
        verify_response = client.table('resume_history').select('*').eq('id', resume_id).eq('user_id', user_id).execute()
        print(f"Debug - rename_resume - verify response: {verify_response.data}")
        
        if not verify_response.data:
            print("Debug - rename_resume - resume not found or unauthorized")
            return jsonify({"error": "Resume not found or unauthorized"}), 404
            
        original_record = verify_response.data[0]
        
        try:
            # Update using a simple query with both conditions in a single filter
            data = {
                'original_filename': new_name,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            print(f"Debug - rename_resume - attempting update with data: {data}")
            
            response = client.table('resume_history').update(data).filter('id', 'eq', resume_id).filter('user_id', 'eq', user_id).execute()
            print(f"Debug - rename_resume - update response: {response.data}")
            
            if response.data:
                updated_record = response.data[0]
                print(f"Debug - rename_resume - update successful: {updated_record}")
                return jsonify({
                    "message": "Resume renamed successfully",
                    "data": updated_record
                }), 200
            else:
                # Fallback: if update succeeded but returned no data, return the original record with new name
                original_record['original_filename'] = new_name
                print(f"Debug - rename_resume - using fallback response: {original_record}")
                return jsonify({
                    "message": "Resume renamed successfully",
                    "data": original_record
                }), 200
                
        except Exception as update_error:
            print(f"Debug - rename_resume - update error: {str(update_error)}")
            return jsonify({"error": f"Database update failed: {str(update_error)}"}), 500
        
    except Exception as e:
        print(f"Error renaming resume: {str(e)}")
        return jsonify({"error": f"Failed to rename resume: {str(e)}"}), 500

@app.route('/api/resume/history', methods=['GET'])
@login_required
def get_resume_history():
    """Get resume history for the current user"""
    try:
        # Get user ID from session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'No user ID in session'}), 401
            
        # Get user's access token
        access_token = session.get('access_token')
        if not access_token:
            return jsonify({'error': 'No access token found'}), 401
        
        # Create database client with user token
        db_client = create_client(supabase_url, supabase_key)
        db_client.postgrest.auth(access_token)
        
        # Query resume history for the user
        result = db_client.table('resume_history')\
            .select('*')\
            .eq('user_id', str(user_id))\
            .order('created_at', desc=True)\
            .execute()
            
        # Format the response
        history = []
        for item in result.data:
            history.append({
                'id': item['id'],
                'originalFilename': item['original_filename'],
                'generatedFilename': item['generated_filename'],
                'jobDescription': item['job_description'],
                'createdAt': item['created_at']
            })
            
        return jsonify({'history': history}), 200
        
    except Exception as e:
        print(f"Error fetching resume history: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Error response: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
        return jsonify({'error': 'Failed to fetch resume history'}), 500

@app.route('/api/resume/download/<resume_id>', methods=['GET'])
@login_required
def download_resume(resume_id):
    """Download a resume from history"""
    try:
        # Get user ID from session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'No user ID in session'}), 401
            
        # Get user's access token
        access_token = session.get('access_token')
        if not access_token:
            return jsonify({'error': 'No access token found'}), 401
            
        # Create database client with user token
        db_client = create_client(supabase_url, supabase_key)
        db_client.postgrest.auth(access_token)
        
        # Query resume history for the specific resume
        result = db_client.table('resume_history')\
            .select('*')\
            .eq('id', resume_id)\
            .eq('user_id', str(user_id))\
            .single()\
            .execute()
            
        if not result.data:
            return jsonify({'error': 'Resume not found'}), 404
            
        resume_data = result.data
        storage_path = resume_data['generated_filename']
        print(f"Original storage path: {storage_path}")
        
        # Get storage client
        storage = get_storage_client()
        if not storage:
            return jsonify({'error': 'Failed to initialize storage client'}), 500
        
        # Download file from storage
        try:
            # List files in the resumes bucket recursively to debug
            try:
                files = storage.storage.from_('resumes').list()
                print(f"Files in resumes bucket: {files}")
            except Exception as list_error:
                print(f"Error listing files: {list_error}")
            
            # Extract just the filename from the full path
            filename = os.path.basename(storage_path)
            
            # The actual path in storage should match what we use in generate()
            storage_path = f"resumes/{filename}"
            print(f"Final storage path: {storage_path}")
            
            # Download the file directly from storage bucket
            response = storage.storage.from_('resumes').download(storage_path)
            
            if not response:
                return jsonify({'error': 'Failed to download file from storage'}), 500
            
            # Create BytesIO object from the response content
            file_buffer = BytesIO(response)
            
            return send_file(
                file_buffer,
                download_name=filename,
                as_attachment=True,
                mimetype='application/pdf'
            )
            
        except Exception as storage_error:
            print(f"Storage error details: {storage_error}")
            if hasattr(storage_error, 'response'):
                print(f"Storage error response: {storage_error.response.text if hasattr(storage_error.response, 'text') else 'No response text'}")
            return jsonify({'error': 'Failed to download resume from storage'}), 500
            
    except Exception as e:
        print(f"Error downloading resume: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Error response: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
        return jsonify({'error': 'Failed to download resume'}), 500

def extract_job_title(job_description):
    """Extract a job title from the job description using OpenAI."""
    try:
        if not job_description:
            return None
            
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a job title classifier. Extract or infer the most likely job title from the job description. Return only the job title, nothing else."},
                {"role": "user", "content": f"Extract the most appropriate job title from this description. Return only the title, for example 'Software Engineer' or 'Data Scientist': {job_description}"}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        job_title = response.choices[0].message.content.strip()
        # Clean and format the job title
        job_title = job_title.replace("'", "").replace('"', "")
        return f"{job_title} Resume"
    except Exception as e:
        print(f"Error extracting job title: {str(e)}")
        return None

# Voice Assistant APIs
@app.route('/api/voice/chat', methods=['POST'])
def voice_chat():
    try:
        data = request.get_json()
        text_input = data.get('text', '')
        audio_data = data.get('audio')  # Base64 encoded audio data
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User not authenticated'
            }), 401
        
        if not text_input and not audio_data:
            return jsonify({
                'success': False,
                'error': 'No input provided'
            }), 400
            
        # If audio data is provided, transcribe it first
        if audio_data:
            audio_bytes = base64.b64decode(audio_data)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio.flush()
                
                with open(temp_audio.name, 'rb') as audio_file:
                    transcript = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                text_input = transcript.text

        # Get user ID from session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'No user ID in session'}), 401
            
        # Get user's access token
        access_token = session.get('access_token')
        if not access_token:
            return jsonify({'error': 'No access token found'}), 401
        
        # Initialize Supabase client
        client = create_client(supabase_url, supabase_key)
        client.postgrest.auth(access_token)
        
        # Get user's resume data and job description from resume_history
        resume_response = client.table('resume_history').select(
            'original_filename, job_description, created_at, generated_filename'
        ).eq('user_id', user_id).order('created_at', desc=True).limit(1).execute()

        resume_data = None
        job_description = None
        resume_content = None
        
        if resume_response.data:
            latest_resume = resume_response.data[0]
            resume_data = {
                'filename': latest_resume['original_filename'],
                'created_at': latest_resume['created_at'],
                'storage_path': latest_resume.get('generated_filename')
            }
            job_description = latest_resume['job_description']
            
            # Fetch resume content from storage if storage_path exists
            if resume_data['storage_path']:
                try:
                    storage_path = resume_data['storage_path']
                    print(f"Original storage path: {storage_path}")
                    
                    # Get storage client with service role for storage operations
                    storage_client = get_storage_client()
                    
                    # List files in the bucket for debugging
                    try:
                        files = storage_client.storage.from_('resumes').list()
                        print(f"Files in resumes bucket: {files}")
                    except Exception as list_error:
                        print(f"Error listing files: {list_error}")
                    
                    # Extract just the filename and construct storage path
                    filename = os.path.basename(storage_path)
                    storage_path = f"resumes/{filename}"
                    print(f"Final storage path: {storage_path}")
                    
                    # Download the file from storage
                    response = storage_client.storage.from_('resumes').download(storage_path)
                    
                    if response:
                        # Create BytesIO object and extract text
                        file_buffer = BytesIO(response)
                        
                        # Extract text based on file type
                        if storage_path.lower().endswith('.pdf'):
                            from pdfminer.high_level import extract_text
                            resume_content = extract_text(file_buffer)
                        else:
                            # For text files
                            resume_content = file_buffer.read().decode('utf-8')
                            
                        print("Resume content extracted successfully")
                    else:
                        print("Failed to download resume from storage")
                        
                except Exception as e:
                    print(f"Error fetching resume content: {str(e)}")
                    if hasattr(e, 'response'):
                        print(f"Storage error response: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")

        # print('JD', job_description)
        
        # Prepare context for GPT
        system_message = """You are Auxilium's AI voice assistant, designed to help users with their job search journey. 
        You can assist with:
        - Resume writing and improvement tips
        - Interview preparation and common questions
        - Job search strategies
        - Career advice and industry insights
        
Here is the context about the user:"""

        if resume_data:
            system_message += f"\n\nUser's Latest Resume:\nFilename: {resume_data['filename']}\nUploaded: {resume_data['created_at']}"
            
            if resume_content:
                system_message += f"\n\nResume Content:\n{resume_content}"
            
        if job_description:
            system_message += f"\n\nUser's Job Description:\n{job_description}"
            
        system_message += "\n\nPlease use this information to provide personalized assistance while answering the user's questions. If they ask about their resume or desired job role, refer to the specific details provided above. You can reference specific parts of their resume when giving advice."
        
        # Get response from GPT
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": text_input}
            ],
            temperature=0.7,
            max_tokens=250
        )
        
        response_text = response.choices[0].message.content
        
        
        client = ElevenLabs(
        api_key= os.getenv('ELEVENLABS_API_KEY'),
        )
        
        audio = client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            output_format="mp3_44100_128",
            text=response_text,
            model_id="eleven_multilingual_v2",
        )
        
        
        # Convert response to speech using ElevenLabs
        # voice = Voice(
        #     voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        #     settings=VoiceSettings(stability=0.5, similarity_boost=0.75)
        # )
        
        # audio = eleven_generate(
        #     text=response_text,
        #     voice=voice,
        #     model="eleven_monolingual_v1"
        # )

        audio_bytes = bytes()
        for chunk in audio:
            audio_bytes += chunk
            
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Convert audio to base64
        # audio_base64 = base64.b64encode(audio).decode('utf-8')
        
        return jsonify({
            'success': True,
            'text': response_text,
            'audio': audio_base64
        })
        
    except Exception as e:
        print(f"Error in voice_chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/jobs/recommended', methods=['GET'])
def get_recommended_jobs():
    try:
        # Get user ID and access token from session
        user_id = session.get('user_id')
        access_token = session.get('access_token')
        
        if not user_id or not access_token:
            return jsonify({'error': 'User not authenticated'}), 401
        
        # Initialize Supabase client
        client = create_client(supabase_url, supabase_key)
        client.postgrest.auth(access_token)
        
        # Get job descriptions from resume history
        response = client.table('resume_history').select('job_description').eq('user_id', user_id).execute()
        job_descriptions = [item['job_description'] for item in response.data if item['job_description']]

        if not job_descriptions:
            return jsonify({
                "message": "No job descriptions found. Please upload a resume with a job description to get job recommendations.",
                "jobs": []
            }), 200

        # Extract job titles from all job descriptions
        positions = []
        for jd in job_descriptions:
            position = extract_job_title(jd)
            if position:
                # Clean up the position (remove 'Resume' suffix and clean spaces)
                position = position.replace(' Resume', '').strip()
                positions.append(position)

        if not positions:
            return jsonify({
                "message": "Could not determine job positions from your descriptions. Please try uploading a different resume.",
                "jobs": []
            }), 200

        # Default locations
        locations = {'London, UK', 'Remote'}  # Include Remote as an option

        # Get recommended jobs
        job_list = []  # Initialize job list
        seen_urls = set()  # Track URLs to avoid duplicates

        for position in positions[:3]:  # Limit to last 3 positions
            for location in locations:
                try:
                    # URL encode the position and location
                    encoded_position = requests.utils.quote(position)
                    encoded_location = requests.utils.quote(location)
                    
                    # Use JobSpy with more focused parameters and better error handling
                    search_params = {
                        'site_name': ['linkedin'],  # Focus on LinkedIn for more reliable results
                        'search_term': position,
                        'location': location,
                        'results_wanted': 5,
                        'country_indeed': 'usa'  # Add country parameter for better results
                    }
                    
                    # Get jobs using JobSpy
                    scraped_jobs = scrape_jobs(**search_params)
                    
                    # Convert DataFrame to list of dictionaries if not empty
                    if scraped_jobs is not None and not scraped_jobs.empty:
                        jobs_list = scraped_jobs.to_dict('records')
                        for job in jobs_list:
                            try:
                                # Skip if we've seen this URL before
                                job_url = str(job.get('url', ''))
                                if job_url in seen_urls:
                                    continue
                                seen_urls.add(job_url)
                                
                                job_entry = {
                                    'title': str(job.get('title', '')),
                                    'company': str(job.get('company', '')),
                                    'location': str(job.get('location', '')),
                                    'date_posted': str(job.get('date_posted', '')),
                                    'salary': str(job.get('salary', '')),
                                    'link': job_url,
                                    'source': 'LinkedIn',
                                    'relevance_score': calculate_relevance_score(
                                        title=str(job.get('title', '')),
                                        description=str(job.get('description', '')),
                                        search_position=position
                                    )
                                }
                                
                                # Only add non-empty jobs with required fields
                                if job_entry['title'] and job_entry['company'] and job_entry['link']:
                                    job_list.append(job_entry)
                            except Exception as job_error:
                                print(f"Error processing job entry: {str(job_error)}")
                                continue
                    
                except Exception as e:
                    print(f"Error scraping jobs for {position} in {location}: {str(e)}")
                    continue
                
                # Add delay between requests to avoid rate limiting
                time.sleep(2)
        
        # Sort by relevance score and limit results
        job_list.sort(key=lambda x: x['relevance_score'], reverse=True)
        final_jobs = job_list[:20]

        return jsonify({
            'jobs': final_jobs,
            'total': len(final_jobs),
            'positions_found': positions
        })

    except Exception as e:
        print(f"Error in get_recommended_jobs: {str(e)}")
        return jsonify({'error': 'Failed to get recommended jobs'}), 500

def calculate_relevance_score(title, description, search_position):
    """Calculate a relevance score for a job based on title and description match."""
    try:
        # Convert everything to lowercase for comparison
        title = title.lower()
        description = description.lower()
        search_position = search_position.lower()
        
        # Initialize score
        score = 0
        
        # Title exact match gives highest score
        if search_position in title:
            score += 10
        
        # Partial title matches
        title_words = set(title.split())
        search_words = set(search_position.split())
        matching_words = title_words.intersection(search_words)
        score += len(matching_words) * 2
        
        # Description matches
        if search_position in description:
            score += 5
        
        # Normalize score to be between 0 and 1
        return min(score / 20.0, 1.0)
        
    except Exception as e:
        print(f"Error calculating relevance score: {str(e)}")
        return 0.0

@app.route('/api/search-jobs', methods=['POST'])
def search_jobs():
    if not JOBSPY_AVAILABLE:
        return jsonify({
            "error": "Job search functionality is not available in this environment",
            "message": "This feature is disabled in the production environment"
        }), 503
    
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            return jsonify({'error': 'No search prompt provided'}), 400

        # Use OpenAI to extract job position and location
        try:
            messages = [
                {"role": "system", "content": "Extract the job position and location from the user's search query. Respond in JSON format with 'position' and 'location' fields."},
                {"role": "user", "content": prompt}
            ]
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            parsed_info = json.loads(response.choices[0].message.content)
            position = parsed_info.get('position', '')
            location = parsed_info.get('location', '')
            
            if not position:
                return jsonify({'error': 'Could not determine job position from the search query'}), 400
                
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return jsonify({'error': 'Failed to process search query'}), 500

        # Use mock job search function
        results = mock_job_search(prompt, location)
        
        return jsonify(results)
    except Exception as e:
        print(f"Error in search_jobs: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

def mock_job_search(search_query, location=None):
    return {
        "jobs": [
            {
                "title": "Software Engineer",
                "company": "Example Corp",
                "location": "Remote",
                "description": "This is a mock job posting while the job search feature is under maintenance.",
                "url": "#"
            }
        ],
        "message": "Job search functionality is currently under maintenance. Please try again later."
    }

def verify_razorpay_signature(body, signature, key_secret):
    expected_signature = hmac.new(
        key_secret.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_signature, signature)

@app.route('/api/razorpay-webhook', methods=['POST'])
def razorpay_webhook():
    try:
        # Verify webhook signature
        webhook_signature = request.headers.get('X-Razorpay-Signature')
        if not webhook_signature:
            return jsonify({'error': 'Missing signature'}), 400
            
        webhook_body = request.get_data()
        
        if not verify_razorpay_signature(
            webhook_body,
            webhook_signature,
            razorpay_key_secret
        ):
            return jsonify({'error': 'Invalid signature'}), 400

        # Parse webhook data
        webhook_data = request.get_json()
        event = webhook_data.get('event')
        
        if event == 'payment.captured':
            payment = webhook_data.get('payload', {}).get('payment', {})
            order_id = payment.get('order_id')
            payment_id = payment.get('id')
            
            if not order_id or not payment_id:
                return jsonify({'error': 'Missing order_id or payment_id'}), 400
            
            # Get order details from Supabase
            result = supabase.table('payment_history').select('*').eq('order_id', order_id).execute()
            if not result.data:
                return jsonify({'error': 'Order not found'}), 404
                
            order = result.data[0]
            user_id = order['user_id']
            
            try:
                # Update payment status
                supabase.table('payment_history').update({
                    'status': 'completed',
                    'payment_id': payment_id,
                    'completed_at': datetime.utcnow().isoformat()
                }).eq('order_id', order_id).execute()
                
                # Update user subscription status
                supabase.table('user_subscriptions').upsert({
                    'user_id': user_id,
                    'status': 'active',
                    'plan': 'pro',
                    'start_date': datetime.utcnow().isoformat(),
                    'end_date': (datetime.utcnow() + timedelta(days=30)).isoformat()
                }).execute()
            except Exception as db_error:
                print(f"Database error in webhook: {str(db_error)}")
                # Don't expose internal errors to Razorpay
                return jsonify({'status': 'error', 'message': 'Database update failed'}), 500
            
        return jsonify({'status': 'success'}), 200
        
    except Exception as e:
        print(f"Webhook error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/subscription/status', methods=['GET'])
def get_subscription_status():
    try:
        user_id = session.get('user_id')
        
        # If user is not logged in, return inactive status
        if not user_id:
            return jsonify({
                'status': 'inactive',
                'isSubscribed': False,
                'remainingGenerations': 0
            })
        
        # Get user's subscription status
        subscription_result = supabase.table('user_subscriptions').select('*').eq('user_id', user_id).execute()
        
        # If user has an active subscription, they have unlimited access
        if subscription_result.data:
            subscription = subscription_result.data[0]
            end_date = datetime.fromisoformat(subscription['end_date'].replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            
            if end_date > current_time:
                return jsonify({
                    'status': 'active',
                    'isSubscribed': True,
                    'type': 'paid',
                    'remainingGenerations': None  # Unlimited for paid users
                })
        
        # For free users, check resume generation count
        resume_count_result = supabase.table('resume_history').select('count').eq('user_id', user_id).execute()
        resume_count = len(resume_count_result.data) if resume_count_result.data else 0
        remaining_generations = max(5 - resume_count, 0)
        
        return jsonify({
            'status': 'free',
            'isSubscribed': remaining_generations > 0,
            'type': 'free',
            'remainingGenerations': remaining_generations
        })
            
    except Exception as e:
        print(f"Error checking subscription status: {str(e)}")
        return jsonify({
            'error': 'Failed to check subscription status',
            'isSubscribed': False,
            'remainingGenerations': 0
        }), 500

@app.route('/api/payment/history', methods=['GET'])
@login_required
def get_payment_history():
    try:
        user_id = session.get('user_id')
        
        # Get user's payment history
        result = supabase.table('payment_history').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        
        return jsonify({'payments': result.data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify-payment', methods=['POST'])
@login_required
def verify_payment():
    try:
        data = request.get_json()
        payment_id = data.get('razorpay_payment_id')
        order_id = data.get('razorpay_order_id')
        signature = data.get('razorpay_signature')
        
        if not all([payment_id, order_id, signature]):
            return jsonify({'error': 'Missing payment details'}), 400

        # Verify payment signature
        params_dict = {
            'razorpay_payment_id': payment_id,
            'razorpay_order_id': order_id,
            'razorpay_signature': signature
        }
        
        try:
            razorpay_client.utility.verify_payment_signature(params_dict)
        except Exception as e:
            return jsonify({'error': 'Invalid payment signature'}), 400

        user_id = session.get('user_id')
        current_time = datetime.now(timezone.utc)
        
        # Update payment status
        supabase.table('payment_history').update({
            'status': 'completed',
            'payment_id': payment_id,
            'completed_at': current_time.isoformat()
        }).eq('order_id', order_id).execute()
        
        # Update user subscription status
        supabase.table('user_subscriptions').upsert({
            'user_id': user_id,
            'status': 'active',
            'plan': 'pro',
            'start_date': current_time.isoformat(),
            'end_date': (current_time + timedelta(days=30)).isoformat()
        }).execute()
        
        return jsonify({
            'status': 'success',
            'message': 'Payment verified and subscription activated'
        })
        
    except Exception as e:
        print(f"Payment verification error: {str(e)}")
        return jsonify({'error': 'Failed to verify payment'}), 500

def check_resume_limit(user_id):
    try:
        # First check if user is pro
        subscription_result = supabase.table('user_subscriptions').select('*').eq('user_id', user_id).execute()
        
        if subscription_result.data:
            subscription = subscription_result.data[0]
            end_date = datetime.fromisoformat(subscription['end_date'].replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            
            if subscription['status'] == 'active' and end_date > current_time:
                return True, None  # Pro users have unlimited generations
        
        # If not pro, check generation count
        result = supabase.table('resume_history').select('count').eq('user_id', user_id).execute()
        count = len(result.data) if result.data else 0
        
        if count >= 5:
            return False, "You have reached the free limit of 5 resume generations. Please upgrade to Pro for unlimited generations."
            
        return True, None
            
    except Exception as e:
        print(f"Error checking resume limit: {str(e)}")
        return False, "Error checking resume limit"

@app.route('/api/resume/limit', methods=['GET'])
@login_required
def get_resume_limit():
    try:
        user_id = session.get('user_id')
        result = supabase.table('resume_history').select('count').eq('user_id', user_id).execute()
        count = len(result.data) if result.data else 0
        
        # Check if user is pro
        subscription_result = supabase.table('user_subscriptions').select('*').eq('user_id', user_id).execute()
        is_pro = False
        
        if subscription_result.data:
            subscription = subscription_result.data[0]
            end_date = datetime.fromisoformat(subscription['end_date'].replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            is_pro = subscription['status'] == 'active' and end_date > current_time
        
        return jsonify({
            'count': count,
            'limit': None if is_pro else 5,
            'is_pro': is_pro
        })
        
    except Exception as e:
        print(f"Error getting resume limit: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/prep/chat', methods=['POST'])
def chat_with_prep_bot():
    try:
        user_id = session.get('user_id')
        access_token = session.get('access_token')
        
        if not user_id or not access_token:
            return jsonify({'error': 'User not authenticated'}), 401
            
        # Get the message from request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        user_message = data['message']
        is_first_message = data.get('isFirstMessage', False)
        
        # Get user's job descriptions from history
        client = create_client(supabase_url, supabase_key)
        client.postgrest.auth(access_token)
        
        response = client.table('resume_history').select('job_description').eq('user_id', user_id).execute()
        job_descriptions = [item['job_description'] for item in response.data if item['job_description']]

        if not job_descriptions:
            return jsonify({
                "error": "No job descriptions found. Please upload a resume first."
            }), 400

        # Get the latest job title
        latest_job_title = extract_job_title(job_descriptions[-1])
        
        if not latest_job_title:
            return jsonify({
                "error": "Could not determine job position. Please upload a resume with a clear job title."
            }), 400
            
        # Construct the system message based on the job title
        system_message = f"""You are an expert interviewer and mentor for {latest_job_title} positions. 
        Your role is to help prepare candidates for interviews by:
        1. Asking relevant technical questions
        2. Providing coding challenges when appropriate
        3. Recommending study materials and resources
        4. Giving feedback on answers
        5. Explaining complex concepts clearly
        
        If this is the first message, introduce yourself and ask a relevant technical question to start the conversation.
        Otherwise, evaluate the candidate's response and continue the conversation naturally."""

        # If it's the first message, generate an introduction and first question
        if is_first_message:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Let's start the interview preparation."}
            ]
        else:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

        # Get response from OpenAI
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        bot_response = response.choices[0].message.content

        return jsonify({
            'response': bot_response,
            'position': latest_job_title
        })

    except Exception as e:
        print(f"Error in chat_with_prep_bot: {str(e)}")
        return jsonify({'error': 'Failed to process chat message'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True)
