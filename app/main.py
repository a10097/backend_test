from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from passlib.context import CryptContext
from datetime import datetime, timedelta
from email.message import EmailMessage
import random
import redis
import aiosmtplib
import jwt
import os
import mysql.connector
from dotenv import load_dotenv


#  수정된 부분: recommendation_api 라우터 추가
from app.recommendation_api import router as recommendation_router




# Redis 클라이언트 설정
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# 환경변수 로드
load_dotenv()

app = FastAPI()

# 모델 정의
class EmailRequest(BaseModel):
    email: EmailStr

class CodeVerifyRequest(BaseModel):
    email: EmailStr
    code: str

class PasswordResetRequest(BaseModel):
    email: EmailStr
    code: str
    new_password: str

class UserCreate(BaseModel):
    nickname: str
    email: str
    password: str

class UserProfile(BaseModel):
    height_cm: int
    weight_kg: int
    birth_year: int
    gender: str = Field(pattern="^(M|F)$")
    pal_value: float


#  수정된 부분: 라우터 통합 (추천 기능 추가)
app.include_router(recommendation_router, prefix="/recommendation")



# 이메일 전송 함수
async def send_email(to_email: str, code: str):
    provider = os.getenv("DEFAULT_EMAIL_PROVIDER", "gmail").lower()

    if provider == "naver":
        smtp_host = os.getenv("NAVER_HOST")
        smtp_port = int(os.getenv("NAVER_PORT"))
        smtp_user = os.getenv("NAVER_USER")
        smtp_pass = os.getenv("NAVER_PASS")
        from_email = smtp_user
    else:
        smtp_host = os.getenv("GMAIL_HOST")
        smtp_port = int(os.getenv("GMAIL_PORT"))
        smtp_user = os.getenv("GMAIL_USER")
        smtp_pass = os.getenv("GMAIL_PASS")
        from_email = smtp_user

    message = EmailMessage()
    message["From"] = f"살뜰링 <{from_email}>"
    message["To"] = to_email
    message["Subject"] = "비밀번호 재설정 인증 코드"
    message.set_content(f"""
    요청하신 인증 코드는 다음과 같습니다:

     인증 코드: {code}

     10분 이내에 입력해주세요.
    """)

    await aiosmtplib.send(
        message,
        hostname=smtp_host,
        port=smtp_port,
        start_tls=True,
        username=smtp_user,
        password=smtp_pass,
    )

# 유틸 함수
def generate_code():
    return str(random.randint(100000, 999999))

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def calculate_tdee(height_cm: int, weight_kg: int, birth_year: int, gender: str, pal_value: float) -> int:
    age = datetime.now().year - birth_year
    if gender == 'M':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif gender == 'F':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 0
    return round(bmr * pal_value)

def calculate_macros(tdee: int) -> tuple[int, int]:
    protein_g = round((tdee * 0.2) / 4)
    fat_g = round((tdee * 0.3) / 9)
    return protein_g, fat_g

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB 연결 함수
def get_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='1509',
        database='test'
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

@app.post("/register")
def register(user: UserCreate):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        raise HTTPException(status_code=400, detail="이미 존재하는 이메일입니다.")

    hashed_pw = hash_password(user.password)
    now = datetime.now()
    cursor.execute("""
        INSERT INTO users (nickname, email, password_hash, created_at)
        VALUES (%s, %s, %s, %s)
    """, (user.nickname, user.email, hashed_pw, now))
    conn.commit()
    cursor.close()
    conn.close()
    return {"msg": "회원가입의 성공했습니다."}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (form_data.username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="이메일 또는 비밀번호가 틀렸습니다.")

    token_data = {"sub": user["email"], "id": user["id"]}
    token = create_access_token(token_data)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/profile")
def get_profile(Authorization: str = Header(...)):
    token = Authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT nickname, email FROM users WHERE email = %s", (user_email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return user

@app.get("/profile-info")
def get_profile_info(Authorization: str = Header(...)):
    token = Authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM users WHERE email = %s", (user_email,))
    result = cursor.fetchone()
    if not result:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    user_id = result["id"]
    cursor.execute("SELECT height_cm, weight_kg, birth_year, gender, pal_value, tdee FROM user_profiles WHERE user_id = %s", (user_id,))
    profile = cursor.fetchone()
    cursor.close()
    conn.close()

    return profile if profile else {}

@app.post("/profile-info")
def save_profile_info(profile: UserProfile, Authorization: str = Header(...)):
    # 1. 사용자 인증
    token = Authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    # 2. 사용자 ID 조회
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (user_email,))
    result = cursor.fetchone()
    if not result:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    user_id = result[0]
    now = datetime.now()

    # 3. TDEE 및 영양소 계산
    tdee = calculate_tdee(
        profile.height_cm,
        profile.weight_kg,
        profile.birth_year,
        profile.gender,
        profile.pal_value
    )
    target_protein_g, target_fat_g = calculate_macros(tdee)

    # 4. DB 저장 (있으면 UPDATE)
    cursor.execute("""
        INSERT INTO user_profiles (
            user_id, height_cm, weight_kg, birth_year, gender, pal_value,
            tdee, target_protein_total_g, target_fat_total_g, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            height_cm = VALUES(height_cm),
            weight_kg = VALUES(weight_kg),
            birth_year = VALUES(birth_year),
            gender = VALUES(gender),
            pal_value = VALUES(pal_value),
            tdee = VALUES(tdee),
            target_protein_total_g = VALUES(target_protein_total_g),
            target_fat_total_g = VALUES(target_fat_total_g),
            created_at = VALUES(created_at)
    """, (
        user_id,
        profile.height_cm,
        profile.weight_kg,
        profile.birth_year,
        profile.gender,
        profile.pal_value,
        tdee,
        target_protein_g,
        target_fat_g,
        now
    ))

    # 5. 마무리
    conn.commit()
    cursor.close()
    conn.close()

    return {
        "msg": f"프로필 저장 완료 (TDEE: {tdee} kcal, 단백질: {target_protein_g}g, 지방: {target_fat_g}g)"
    }


@app.post("/send-verification-code")
async def send_verification_code(req: EmailRequest):
    code = generate_code()
    redis_client.setex(f"verify:{req.email}", 600, code)
    await send_email(req.email, code)
    return {"msg": "인증 코드가 이메일로 전송되었습니다."}

@app.post("/verify-code")
def verify_code(req: CodeVerifyRequest):
    saved_code = redis_client.get(f"verify:{req.email}")
    if not saved_code:
        raise HTTPException(status_code=400, detail="인증 코드가 만료되었거나 존재하지 않습니다.")
    if saved_code != req.code:
        raise HTTPException(status_code=400, detail="인증 코드가 일치하지 않습니다.")
    return {"msg": "인증 코드가 확인되었습니다."}

@app.post("/reset-password")
def reset_password(req: PasswordResetRequest):
    saved_code = redis_client.get(f"verify:{req.email}")
    if not saved_code or saved_code != req.code:
        raise HTTPException(status_code=400, detail="인증 코드가 올바르지 않습니다.")

    conn = get_connection()
    cursor = conn.cursor()
    hashed_pw = pwd_context.hash(req.new_password)
    cursor.execute("UPDATE users SET password_hash = %s WHERE email = %s", (hashed_pw, req.email))
    conn.commit()
    cursor.close()
    conn.close()

    redis_client.delete(f"verify:{req.email}")
    return {"msg": "비밀번호가 성공적으로 재설정되었습니다!"}
@app.post("/profile-info")
def save_profile_info(profile: UserProfile, Authorization: str = Header(...)):
    # 1. 사용자 인증 (JWT 토큰 디코드)
    token = Authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    # 2. 사용자 ID 조회
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (user_email,))
    result = cursor.fetchone()
    if not result:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    
    user_id = result[0]
    now = datetime.now()

    # 3. TDEE 및 영양소 계산
    tdee = calculate_tdee(
        profile.height_cm,
        profile.weight_kg,
        profile.birth_year,
        profile.gender,
        profile.pal_value
    )
    target_protein_g, target_fat_g = calculate_macros(tdee)

    # 4. DB 저장 (있으면 업데이트)
    cursor.execute("""
        INSERT INTO user_profiles (
            user_id, height_cm, weight_kg, birth_year, gender, pal_value,
            tdee, target_protein_total_g, target_fat_total_g, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            height_cm = VALUES(height_cm),
            weight_kg = VALUES(weight_kg),
            birth_year = VALUES(birth_year),
            gender = VALUES(gender),
            pal_value = VALUES(pal_value),
            tdee = VALUES(tdee),
            target_protein_total_g = VALUES(target_protein_total_g),
            target_fat_total_g = VALUES(target_fat_total_g),
            created_at = VALUES(created_at)
    """, (
        user_id,
        profile.height_cm,
        profile.weight_kg,
        profile.birth_year,
        profile.gender,
        profile.pal_value,
        tdee,
        target_protein_g,
        target_fat_g,
        now
    ))

    # 5. 정리 및 응답
    conn.commit()
    cursor.close()
    conn.close()

    return {
        "msg": f"프로필 저장 완료 (TDEE: {tdee} kcal, 단백질: {target_protein_g}g, 지방: {target_fat_g}g)"
    }
