import threading
import time
from fastapi import FastAPI, UploadFile, File
import requests
import base64
import sounddevice as sd
import soundfile as sf
import tensorflow_hub as hub
import librosa
import numpy as np

# ========== 1. 推荐内容全局变量 ==========
current_recommend = "暂无推荐"

# ========== 2. 声音自动采集和识别 ==========
labels_path = 'yamnetclassmap.csv'


def record_audio(filename, duration=20, fs=16000):
    print(f"开始录音（{duration}秒）...")
    data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, data, fs)
    print(f"录音完成：{filename}")


def load_labels(label_file):
    with open(label_file, 'r') as lf:
        lines = lf.readlines()
        return [line.strip().split(',')[2] for line in lines[1:]]


def yamnet_scene_recognize(audio_file):
    wav, sr = librosa.load(audio_file, sr=16000, mono=True)
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    scores, embeddings, spectrogram = model(wav)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)
    class_names = load_labels(labels_path)
    top_N = 5
    top_N_idx = mean_scores.argsort()[-top_N:][::-1]
    top_results = [(class_names[i], mean_scores[i]) for i in top_N_idx]
    print("Top-5 识别结果：", top_results)
    return class_names[top_N_idx[0]]


def recommend_content_audio(scene_label):
    scene_label = scene_label.lower()
    if any(x in scene_label for x in [
        "subway", "metro", "underground", "public transport", "train", "rail", "railroad", "station",
        "platform", "carriage", "wagon", "engine", "mass transit", "transportation noise", "announcement"
    ]):
        return "推荐：只展示大纲"
    if any(x in scene_label.lower() for x in [
        "street", "road", "sidewalk", "traffic", "vehicle", "car", "bus", "truck", "motorcycle",
        "horn", "outdoor", "park", "plaza", "square", "crosswalk", "bicycle", "footsteps", "crowd",
        "speech", "talking", "conversation", "shout", "yelling", "dog", "bird", "birds", "wind", "nature",
        "children playing", "skateboard", "ambulance", "fire engine", "siren", "animal", "animals"
    ]):
        return "推荐：推送音频界面"
    if any(x in scene_label for x in [
        "silence", "quiet", "writing", "pen", "pencil", "typing", "keyboard", "mouse click", "paper",
        "desk", "furniture", "indoor", "room", "home", "living room", "office", "paper rustling",
        "flipping pages", "human breath", "cough", "clearing throat"
    ]):
        return "推荐：详细完整界面"
    return "推荐：详细完整界面"


def auto_audio_task():
    global current_recommend
    while True:
        record_audio("env_audio.wav", duration=10)
        scene = yamnet_scene_recognize("env_audio.wav")
        current_recommend = recommend_content_audio(scene)
        print("【声音自动识别】推荐内容：", current_recommend)
        time.sleep(30)  # 7分钟


# ========== 3. 图片识别相关 ==========
API_KEY = "8tysemSM2znDUagY7NYuyfQj"
SECRET_KEY = "Cnk4aYkrmBIBeiMQOCzEmWIJOdhcAdnG"


def get_baidu_access_token(api_key, secret_key):
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    res = requests.post(url, params=params)
    return res.json()["access_token"]


def baidu_scene_classify(image_path, access_token):
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    params = {"access_token": access_token}
    data = {"image": img_data}
    res = requests.post(url, params=params, data=data, headers=headers)
    return res.json()


def recommend_content(keywords):
    for kw in keywords:
        if any(word in kw for word in [
            "地铁", "地铁站", "轨道", "车厢", "地铁口", "公共交通", "站台", "列车", "交通工具"
        ]):
            return "推荐：只展示大纲"
        if any(word in kw for word in [
            "家", "房间", "卧室", "客厅", "书桌", "书房", "办公桌", "办公室", "宿舍", "桌面", "书架",
            "电脑", "台式电脑", "笔记本", "笔记本电脑", "显示器", "显示器屏幕"
        ]):
            return "推荐：详细完整界面"
        if any(word in kw for word in [
            "街道", "马路", "户外", "公路", "人行道", "广场", "道路", "步行", "路口", "行人", "人行横道", "公园", "小区"
        ]):
            return "推荐：推送音频界面"
    return "推荐：只展示大纲"


# ========== 4. FastAPI接口 ==========
app = FastAPI()


@app.post("/recognize_image_scene")
async def recognize_image_scene(file: UploadFile = File(...)):
    with open("temp_img.jpg", "wb") as f:
        f.write(await file.read())
    token = get_baidu_access_token(API_KEY, SECRET_KEY)
    result = baidu_scene_classify("temp_img.jpg", token)
    keywords = [item['keyword'] for item in result.get('result', [])]
    recommend = recommend_content(keywords)
    global current_recommend
    current_recommend = recommend
    return {"scene_keywords": keywords, "recommend": recommend}


@app.get("/get_latest_recommend")
def get_latest_recommend():
    return {"recommend": current_recommend}


# ========== 5. 主程序启动 ==========
if __name__ == "__main__":
    import uvicorn
    t = threading.Thread(target=auto_audio_task, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
