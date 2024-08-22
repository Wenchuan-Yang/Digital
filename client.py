import asyncio
import websockets

async def send_audio_and_receive_video():
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # 读取音频文件
        with open("/home/ubuntu/digital/vocal/hello.mp3", "rb") as f:
            audio_data = f.read()

        # 将音频文件发送到服务端
        await websocket.send(audio_data)
        print("Sent audio file to server")

        # 接收服务端返回的视频文件
        video_data = await websocket.recv()
        print("Received video file from server")

        # 将视频文件保存为本地文件
        with open("output_video.mp4", "wb") as f:
            f.write(video_data)

if __name__ == "__main__":
    asyncio.run(send_audio_and_receive_video())
