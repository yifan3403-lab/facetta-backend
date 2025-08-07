# context awareness

安装依赖

```bash
pip install -r requirements.txt

# 运行
python context-awareness/main.py

uvicorn context-awareness.main:app --reload --host 0.0.0.0 --port 8000
```

## 接口

### 音频环境自动识别（后台线程自动循环）

通过本地麦克风采集环境音，调用 YAMNet 自动识别场景标签，根据结果实时更新推荐内容。

### 图片场景识别

1. POST /recognize_image_scene
2. 参数：上传图片（表单字段名 file，类型为 UploadFile）
3. 返回：{ "scene_keywords": ["标签 1", "标签 2", ...],"recommend": "推荐内容"}
4. GET 返回{ "recommend": "推荐内容"}

## 说明与注意事项

- 需要在 main.py 代码中配置好 百度智能云 API_KEY 和 SECRET_KEY，可前往百度智能云控制台获取。
- 运行环境需有 本地麦克风硬件支持（用于环境音自动采集），建议先本地测试音频采集功能。
- yamnetclassmap.csv 文件需放置在与主程序同级目录下，内容为 YAMNet 分类标签映射表。
- 推荐在 Linux/macOS/Windows 下均可运行，但部分依赖（如 sounddevice）在服务器/无声环境需注意权限或驱动配置。
- 如需自定义内容推荐逻辑，可修改 recommend_content_audio 和 recommend_content 函数。
