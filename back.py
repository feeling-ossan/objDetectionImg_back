from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
from ultralytics import YOLO

# 受信正常性確認用
# from PIL import Image
# from io import BytesIO

app = FastAPI()

def objDetectionExec(image):
    # YOLOの実行
    # load pretrained model.
    model = YOLO(model="yolov8n.pt")

    # inference
    resPred = model.predict(image, save=False, show=False)

    # plotメソッドでndarrayを取得
    resPlotted = resPred[0].plot()
    # yoloの推論結果はBGRで表現されていそう。

    return resPlotted


@app.post('/objDetection')
async def postExec(file: UploadFile = File(...)):
    # アップロードされた画像を読み込む
    readImg = await file.read()
    # 受信正常性確認用
    # image = Image.open(BytesIO(readImg))
    # image.show()
    np_image = np.frombuffer(readImg, dtype=np.uint8)
    uploadImg = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # 物体検知処理
    detectedImg = objDetectionExec(uploadImg)

    # 結果をバイナリデータに変換してストリーム化
    _, detectedImg_encoded = cv2.imencode('.jpg', detectedImg)
    detectedImg_byte_array = detectedImg_encoded.tobytes()

    # StreamingResponseを使用して結果をストリーミング
    return StreamingResponse(io.BytesIO(detectedImg_byte_array), media_type="image/jpeg")
