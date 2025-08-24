# server/app_server.py
import tempfile, subprocess, os, sys, base64
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import tensorflow_hub as hub
import numpy as np
import librosa
import requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 上线请收紧
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 资源定位 ----------
def resource_path(name: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / name)
    return str(Path(__file__).parent / name)

LABELS_PATH = resource_path("yamnetclassmap.csv")
_model = None
_labels = None

def get_model():
    global _model, _labels
    if _model is None:
        _model = hub.load("https://tfhub.dev/google/yamnet/1")
    if _labels is None:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            _labels = [ln.strip().split(",")[2] for ln in f.readlines()[1:]]
    return _model, _labels

def infer_wav_top1(path: str) -> str:
    y, sr = librosa.load(path, sr=16000, mono=True)
    model, labels = get_model()
    scores, _, _ = model(y)
    mean = np.mean(scores.numpy(), axis=0)
    return labels[int(mean.argmax())]

def rec_from_label(lbl: str) -> str:
    s = lbl.lower()
    if any(x in s for x in ["subway","metro","underground","public transport","train","station","platform"]):
        return "推荐：只展示大纲"
    if any(x in s for x in [
        "street","traffic","vehicle","car","bus","truck","motorcycle","horn",
        "outdoor","crowd","speech","talking","conversation",
        "applause","clap","clapping","cheering","laughter",
        "music","instrument","singing","tv","radio",
        "construction","drill","hammer","ambulance","fire engine","siren",
        "dog","dogs","bird","birds"
    ]):
        return "推荐：推送音频界面"
    if any(x in s for x in ["silence","quiet","typing","keyboard","mouse click","paper","indoor","room","office"]):
        return "推荐：详细完整界面"
    return "推荐：详细完整界面"

# ---------- 用户状态 ----------
user_latest: Dict[str, str] = {}

@app.get("/health")
def health(user_id: Optional[str] = None):
    return {"ok": True, "user_id": user_id, "recommend": user_latest.get(user_id)}

@app.get("/get_latest_recommend")
def get_latest_recommend(user_id: str = Query(...)):
    return {"recommend": user_latest.get(user_id, "暂无")}

# ---------- 调试 echo ----------
@app.websocket("/ws_echo")
async def ws_echo(ws: WebSocket):
    await ws.accept()
    print("[WS] echo connected")
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                print("[WS] echo disconnected")
                break
            if msg.get("text") is not None:
                await ws.send_text(msg["text"])
            elif msg.get("bytes") is not None:
                await ws.send_text(f"got {len(msg['bytes'])} bytes")
    except WebSocketDisconnect:
        print("[WS] echo disconnected (exception)")

# ---------- WebSocket：接收“完整段文件”并识别 ----------
@app.websocket("/stream_audio")
async def stream_audio(ws: WebSocket):
    await ws.accept()
    uid = ws.query_params.get("user_id") or "anon"
    print(f"[WS] connected: {uid}")
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                print(f"[WS] disconnected: {uid}")
                break

            data = msg.get("bytes")
            if not data:
                continue

            # 先按 .ogg 保存；ffmpeg 会自动探测（即使是 webm）
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                f.write(data)
                in_path = f.name
            wav_path = in_path + ".wav"

            # 直接无格式参数交给 ffmpeg 自动识别（完整段才可靠）
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", wav_path]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print("ffmpeg stderr:", r.stderr)     # 如果还错，会打印详细原因
                try: os.remove(in_path)
                except: pass
                continue

            try:
                label = infer_wav_top1(wav_path)
                user_latest[uid] = rec_from_label(label)
                print(f"[WS] {uid} -> {label}")
            except Exception as e:
                print("[WS] infer error:", e)
            finally:
                for p in (in_path, wav_path):
                    try: os.remove(p)
                    except: pass
    except WebSocketDisconnect:
        print(f"[WS] disconnected (exception): {uid}")

# ---------- 图片识别 ----------
API_KEY = "8tysemSM2znDUagY7NYuyfQj"
SECRET_KEY = "Cnk4aYkrmBIBeiMQOCzEmWIJOdhcAdnG"

def get_baidu_access_token(api_key, secret_key):
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type":"client_credentials","client_id":api_key,"client_secret":secret_key}
    return requests.post(url, params=params).json()["access_token"]

@app.post("/recognize_image_scene")
async def recognize_image_scene(user_id: Optional[str] = None, file: UploadFile = File(...)):
    img = await file.read()
    token = get_baidu_access_token(API_KEY, SECRET_KEY)
    data = {"image": base64.b64encode(img).decode()}
    res = requests.post(
        "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general",
        params={"access_token": token},
        data=data,
        headers={"Content-Type":"application/x-www-form-urlencoded"}
    ).json()
    kws = [it["keyword"] for it in res.get("result", [])]
    rec = "推荐：详细完整界面"
    for kw in kws:
        if any(w in kw for w in ["地铁","地铁站","公共交通","站台","列车"]): rec = "推荐：只展示大纲"; break
        if any(w in kw for w in ["街道","马路","户外","公园","道路","行人","小区"]): rec = "推荐：推送音频界面"
        if any(w in kw for w in ["家","房间","卧室","客厅","书桌","办公室","显示器","电脑"]): rec = "推荐：详细完整界面"
    if user_id: user_latest[user_id] = rec
    return {"scene_keywords": kws, "recommend": rec}