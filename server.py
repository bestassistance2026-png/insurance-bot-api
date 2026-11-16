from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# إعداد التسجيل
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء التطبيق
app = FastAPI(title="Insurance Bot API - Step 1")

# السماح للتطبيق بالاتصال (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌟 الرابط الوحيد حالياً: للاختبار
@app.get("/api/test")
async def test_endpoint():
    logger.info("✅ تم استدعاء رابط الاختبار")
    return {
        "status": "success",
        "message": "مرحباً! الخادم يعمل بنجاح 🎉",
        "version": "1.0"
    }

# الصفحة الرئيسية (لتأكيد أن الخادم يعمل عند فتح الرابط في المتصفح)
@app.get("/")
async def root():
    return {"message": "Insurance Bot API is running. Go to /api/test to test."}

if __name__ == "__main__":
    logger.info("🚀 جاري تشغيل الخادم على المنفذ 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)