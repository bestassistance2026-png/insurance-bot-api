import os
import time
import tempfile
import shutil
import logging
import sqlite3
import base64
import re
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.header import Header
from email.utils import formataddr, formatdate

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# استيراد مكتبات الصور
try:
    from PIL import Image
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# استيراد الذكاء الاصطناعي
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# -------------------- إعدادات التسجيل --------------------
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- الإعدادات الخاصة بك --------------------
EMAIL_ADDRESS = 'amerba2011@gmail.com'      # 👈 ضع إيميلك هنا
EMAIL_PASSWORD = 'nqiy hlqa kvfw ppom'        # 👈 ضع كلمة مرور التطبيقات هنا
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
FALLBACK_EMAIL = "support@company.com"

# -------------------- قاموس الشركات --------------------
INSURANCE_COMPANIES = {
    "آروب للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["3.jpg"], "emails": ["AShaar@best-assistance.net.sy"]},
    "المتحدة للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["1.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "الاسلامية للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["9.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "العربية للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["7.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "الكويتية للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["6.png", "13.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "اورينت للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["11.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "العقيلة للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["10.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "أدير للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["12.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "الوطنية للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["2.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "السورية العربية للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["8.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "الثقة للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["5.png"], "emails": ["AShaar@best-assistance.net.sy"]},
    "سولدارتي للتأمين": {"display_name_ar": "", "display_name_en": "", "logo_files": ["4.png"], "emails": ["AShaar@best-assistance.net.sy"]},
}

EMAIL_RECIPIENTS = {'btn1': ['ChronicPM@best-assistance.net.sy'], 'btn2': ['BANetwork@best-assistance.net.sy'], 'btn3': ['customerservice@best-assistance.net.sy'], 'btn4': ['customerservice@best-assistance.net.sy']}

# -------------------- قاعدة البيانات --------------------
def init_db():
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS requests 
                 (conv_id TEXT PRIMARY KEY, service_id TEXT, insured_id TEXT, user_name TEXT, company_name TEXT, timestamp REAL)''')
    conn.commit(); conn.close()

# -------------------- الذكاء الاصطناعي (CLIP و SIFT) --------------------
CLIP_MODEL = None; CLIP_PROCESSOR = None; OCR_READER = None

POSITIVE_PROMPTS = ["a health insurance card", "a medical insurance document", "a plastic ID card with text and logo", "صورة بطاقة تأمين صحي", "مستند تأمين طبي", "بطاقة تأمين", "بطاقة هوية"]
NEGATIVE_PROMPTS = ["a car engine", "a laptop screen showing a document", "a landscape photo", "a person selfie", "a building", "a food plate", "a pet animal", "a calculator", "a wooden table", "a wall", "محرك سيارة", "شاشة لابتوب", "منظر طبيعي", "صورة شخصية", "مبنى", "طبق طعام", "آلة حاسبة", "طاولة خشبية", "جدار", "bathroom", "interior design", "حمام", "تصميم داخلي"]
OCR_KEYWORDS = ["insured", "policy", "individual", "المؤمن له", "تأمين", "شركة", "بطاقة", "am", "amb", "dc", "ih", "in", "dv"]

SIFT_CACHE = {}
LOGO_TO_COMPANY = {}
MIN_INLIERS = 25

def init_ai_models():
    global CLIP_MODEL, CLIP_PROCESSOR
    if CLIP_AVAILABLE:
        try:
            logger.info("🧠 جاري تحميل CLIP...")
            CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            CLIP_MODEL.eval()
            logger.info("✅ CLIP جاهز")
        except Exception as e: logger.error(f"❌ فشل تحميل CLIP: {e}")

def load_ocr():
    global OCR_READER
    if OCR_AVAILABLE and OCR_READER is None:
        try:
            logger.info("🔍 جاري تحميل EasyOCR (قد يستغرق دقيقة)...")
            OCR_READER = easyocr.Reader(['ar', 'en'], gpu=False)
            logger.info("✅ EasyOCR جاهز")
        except Exception as e: logger.error(f"❌ فشل تحميل EasyOCR: {e}")

def build_logo_index():
    global SIFT_CACHE, LOGO_TO_COMPANY
    if not CV_AVAILABLE: return
    try: detector = cv2.SIFT_create()
    except: detector = cv2.ORB_create(nfeatures=1000)
    total = 0
    for comp_key, comp_data in INSURANCE_COMPANIES.items():
        for logo_file in comp_data.get('logo_files', []):
            logo_path = os.path.join("logos", logo_file)
            if not os.path.exists(logo_path): continue
            img = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            kp, des = detector.detectAndCompute(img, None)
            if des is not None:
                SIFT_CACHE[logo_file] = {'keypoints': kp, 'descriptors': des}
                LOGO_TO_COMPANY[logo_file] = comp_key
                total += 1
    logger.info(f"📚 تم فهرسة {total} شعار")

def identify_company_by_logo(image_path):
    if not CV_AVAILABLE or not SIFT_CACHE: return None, None
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None, None
        try: detector, norm = cv2.SIFT_create(), cv2.NORM_L2
        except: detector, norm = cv2.ORB_create(nfeatures=1000), cv2.NORM_HAMMING
        card_kp, card_des = detector.detectAndCompute(img, None)
        if card_des is None or len(card_kp) < 10: return None, None
        company_best = {}
        matcher = cv2.BFMatcher(norm)
        for logo_file, logo_data in SIFT_CACHE.items():
            comp_key = LOGO_TO_COMPANY.get(logo_file)
            try: raw = matcher.knnMatch(logo_data['descriptors'], card_des, k=2)
            except: continue
            good = [pair[0] for pair in raw if len(pair) >= 2 and pair[0].distance < 0.75 * pair[1].distance]
            if len(good) < 8: continue
            src = np.float32([logo_data['keypoints'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([card_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if M is not None and mask is not None else 0
            if inliers > company_best.get(comp_key, 0): company_best[comp_key] = inliers
        if not company_best: return None, None
        sorted_scores = sorted(company_best.items(), key=lambda x: -x[1])
        best_comp, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
        if best_score < MIN_INLIERS or (second_score > 0 and best_score < second_score * 1.3): return None, None
        return best_comp, INSURANCE_COMPANIES[best_comp]
    except Exception as e:
        logger.error(f"خطأ في SIFT: {e}")
        return None, None

def check_image_is_card(image_path):
    if CLIP_MODEL is None: return True
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = CLIP_PROCESSOR(text=POSITIVE_PROMPTS + NEGATIVE_PROMPTS, images=image, return_tensors="pt", padding=True)
        with torch.no_grad(): probs = CLIP_MODEL(**inputs).logits_per_image.softmax(dim=1)[0]
        pos_score = float(torch.sum(probs[:len(POSITIVE_PROMPTS)]))
        neg_score = float(torch.sum(probs[len(POSITIVE_PROMPTS):]))
        logger.info(f"⚖️ CLIP -> بطاقة: {pos_score:.2%} | غير بطاقة: {neg_score:.2%}")
        return pos_score > neg_score
    except: return True

def check_text_is_card(image_path):
    load_ocr() # تحميل OCR فقط عند الحاجة
    if OCR_READER is None: return False
    try:
        text = " ".join(OCR_READER.readtext(image_path, detail=0)).lower()
        return any(kw in text for kw in OCR_KEYWORDS)
    except: return False

# -------------------- إرسال الإيميل --------------------
def send_email(recipients, subject, user_name, service_name, insured_id, company_name, messages, attachments):
    try:
        if isinstance(recipients, str): recipients = [recipients]
        seen = set(); unique_recipients = [r for r in recipients if r and not (r in seen or seen.add(r))]
        if not unique_recipients: return False, "لا توجد إيميلات"
        
        msg = MIMEMultipart()
        msg['From'] = formataddr((str(Header('خدمة التأمين الصحي', 'utf-8')), EMAIL_ADDRESS))
        msg['To'] = ', '.join(unique_recipients)
        msg['Subject'] = str(Header(f"insured_id={insured_id} - طلب {service_name}" + (f" - {company_name}" if company_name else ""), 'utf-8'))
        
        html = f'<html dir="rtl"><body><h2>طلب جديد</h2><p><b>المستخدم:</b> {user_name}</p><p><b>الخدمة:</b> {service_name}</p><p><b>الشركة:</b> {company_name}</p><div style="background:#eee;padding:10px;">{"<br>".join(messages)}</div></body></html>'
        msg.attach(MIMEText(html, 'html', 'utf-8'))
        
        if attachments:
            for f_path in attachments:
                if f_path and os.path.exists(f_path):
                    with open(f_path, 'rb') as f:
                        part = MIMEApplication(f.read(), _subtype='pdf' if f_path.endswith('.pdf') else 'jpeg')
                        part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(f_path)}"')
                        msg.attach(part)
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30)
        server.starttls(context=ssl.create_default_context())
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg); server.quit()
        return True, "تم الإرسال"
    except Exception as e: return False, str(e)

# -------------------- الخادم (FastAPI) --------------------
app = FastAPI(title="Insurance Bot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event():
    init_db()
    build_logo_index()
    init_ai_models() # تحميل CLIP فقط لضمان سرعة الخادم

@app.get("/")
async def root(): return {"message": "Insurance Bot API is running."}

# 1. فحص صورة البطاقة
@app.post("/api/validate_card")
async def api_validate_card(file: UploadFile = File(...)):
    temp_path = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time())}.jpg")
    with open(temp_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    
    if not check_image_is_card(temp_path):
        os.remove(temp_path)
        return {"status": "error", "message": "الصورة لا تبدو كبطاقة تأمين."}
        
    company_key, company_data = identify_company_by_logo(temp_path)
    
    if company_key and company_data:
        os.remove(temp_path)
        return {"status": "success", "company_key": company_key, "company_name": company_data.get('display_name_ar')}
    
    # إذا فشل SIFT، نجرب OCR
    if check_text_is_card(temp_path):
        os.remove(temp_path)
        return {"status": "success", "company_key": None, "company_name": "غير معروفة"}
        
    os.remove(temp_path)
    return {"status": "error", "message": "لم نتمكن من قراءة البطاقة."}

# 2. إرسال الطلب النهائي
@app.post("/api/submit_request")
async def api_submit_request(
    service_id: str = Form(...),
    insured_id: str = Form(...),
    user_name: str = Form(...),
    company_key: str = Form(""),
    company_name: str = Form(""),
    message: str = Form(""),
    file: UploadFile = File(None)
):
    # جمع الإيميلات
    service_emails = EMAIL_RECIPIENTS.get(service_id, [])
    company_emails = INSURANCE_COMPANIES.get(company_key, {}).get('emails', []) if company_key else []
    seen = set()
    recipients = [e for e in service_emails + company_emails if e and not (e in seen or seen.add(e))]
    if not recipients: recipients = [FALLBACK_EMAIL]
    
    # حفظ الملف مؤقتاً
    attachments = []
    if file:
        temp_path = os.path.join(tempfile.gettempdir(), f"req_{int(time.time())}_{file.filename}")
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        attachments.append(temp_path)
    
    # إرسال الإيميل
    messages = [message] if message else ["لا توجد رسائل نصية"]
    success, msg = send_email(recipients, "طلب", user_name, service_id, insured_id, company_name, messages, attachments)
    
    # تنظيف
    for f in attachments:
        if os.path.exists(f): os.remove(f)
        
    if success: return {"status": "success", "message": "تم الإرسال بنجاح"}
    return {"status": "error", "message": msg}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
