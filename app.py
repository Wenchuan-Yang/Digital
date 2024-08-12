from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import ffmpeg
import uvicorn

app = FastAPI()

def generate_video_stream(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 使用 ffmpeg 分离出音频流
    audio_stream = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='mp3')
        .run_async(pipe_stdout=True)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理视频帧，比如灰度处理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 编码为JPEG格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 从音频流中读取数据
        audio_data = audio_stream.stdout.read(1024)

        if not audio_data:
            break
        
        # 返回视频帧和音频数据
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
               b'--audio\r\n'
               b'Content-Type: audio/mpeg\r\n\r\n' + audio_data + b'\r\n')

    cap.release()
    audio_stream.stdout.close()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_video_stream('/home/ubuntu/digital/video/20231227-174452.mp4'),
                             media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
