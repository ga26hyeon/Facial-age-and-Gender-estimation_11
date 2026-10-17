from pathlib import Path
import shutil, tempfile
from typing import Annotated
import cv2, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT=Path(__file__).parent; MODEL_DIR=ROOT/'models'; STATIC_DIR=ROOT/'static'
app=FastAPI(title='Local Face Estimator',version='1.0.0'); app.mount('/static',StaticFiles(directory=STATIC_DIR),name='static')
AGE_BUCKETS=['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60+']; GENDER_LABELS=['여성','남성']; ALLOWED_TYPES={'image/jpeg','image/png','image/webp'}; MAX_BYTES=10*1024*1024
def load_models():
    required=[MODEL_DIR/'face_deploy.prototxt',MODEL_DIR/'res10_300x300_ssd_iter_140000.caffemodel',MODEL_DIR/'age_deploy.prototxt',MODEL_DIR/'age_net.caffemodel',MODEL_DIR/'gender_deploy.prototxt',MODEL_DIR/'gender_net.caffemodel']; missing=[p.name for p in required if not p.exists()]
    if missing: raise RuntimeError('모델 파일이 없습니다: '+', '.join(missing))
    load_dir=Path(tempfile.gettempdir())/'face_estimator_models'; load_dir.mkdir(exist_ok=True)
    for source in required: shutil.copy2(source,load_dir/source.name)
    return {'face':cv2.dnn.readNet(str(load_dir/required[1].name),str(load_dir/required[0].name)),'age':cv2.dnn.readNet(str(load_dir/required[3].name),str(load_dir/required[2].name)),'gender':cv2.dnn.readNet(str(load_dir/required[5].name),str(load_dir/required[4].name))}
MODELS=None
@app.on_event('startup')
def startup():
    global MODELS; MODELS=load_models()
@app.get('/')
def index(): return FileResponse(STATIC_DIR/'index.html')
@app.get('/api/health')
def health(): return {'ok':MODELS is not None,'model':'OpenCV DNN Age/Gender (CPU)'}
@app.post('/api/analyze')
async def analyze(file:Annotated[UploadFile,File(...)]):
    if file.content_type not in ALLOWED_TYPES: raise HTTPException(415,'JPG, PNG, WEBP 이미지 파일만 업로드할 수 있습니다.')
    data=await file.read()
    if not data or len(data)>MAX_BYTES: raise HTTPException(413,'이미지는 비어 있지 않아야 하며 10MB 이하여야 합니다.')
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise HTTPException(400,'이미지를 읽을 수 없습니다. 파일 형식을 확인하세요.')
    h,w=image.shape[:2]; blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104,117,123)); MODELS['face'].setInput(blob); detections=MODELS['face'].forward(); faces=[]
    for i in range(detections.shape[2]):
        confidence=float(detections[0,0,i,2])
        if confidence<0.55: continue
        x1,y1,x2,y2=[int(v) for v in (detections[0,0,i,3:7]*np.array([w,h,w,h])).astype(int)]; x1,y1,x2,y2=max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)
        if x2-x1<40 or y2-y1<40: continue
        face=image[y1:y2,x1:x2]; fb=cv2.dnn.blobFromImage(face,1.0,(227,227),(78.426,87.769,114.896),swapRB=False); MODELS['gender'].setInput(fb); gs=MODELS['gender'].forward()[0]; MODELS['age'].setInput(fb); ass=MODELS['age'].forward()[0]; gi=int(np.argmax(gs)); ai=int(np.argmax(ass))
        faces.append({'box':{'x':x1,'y':y1,'width':x2-x1,'height':y2-y1},'confidence':round(confidence*100,1),'gender':GENDER_LABELS[gi],'gender_confidence':round(float(gs[gi])*100,1),'age':AGE_BUCKETS[ai],'age_confidence':round(float(ass[ai])*100,1)})
    if not faces: raise HTTPException(422,'얼굴을 찾지 못했습니다. 정면에 가까운 얼굴 사진을 사용해 보세요.')
    return {'faces':faces,'image':{'width':w,'height':h},'notice':'모델 결과는 추정치이며 실제 나이·성별을 보장하지 않습니다.'}
