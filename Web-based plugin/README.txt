环境：
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

requirements：

fastapi
uvicorn
websockets
aiofiles
requests
numpy
librosa
tensorflow_hub
soundfile
pydub
ffmpeg-python
python-multipart
需要安装ffmpeg

启动：
cd server
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app_server:app --host 0.0.0.0 --port 8080 --ws websockets --reload
检查：
curl "http://127.0.0.1:8080/health"
最新推荐：
curl "http://127.0.0.1:8080/get_latest_recommend?user_id=TEST"
本地接口地址：
WebSocket: ws://127.0.0.1:8080/stream_audio?user_id=TEST
HTTP: http://127.0.0.1:8080/get_latest_recommend?user_id=TEST
服务器部署后：
WebSocket: wss://地址/stream_audio?user_id=TEST
HTTP: https://地址/get_latest_recommend?user_id=TEST
