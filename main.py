import asyncio
import websockets
import os

async def process_audio_and_generate_video(audio_data):
    # 将接收到的音频数据保存为临时文件
    with open("vocal/temp_audio.wav", "wb") as f:
        f.write(audio_data)

    # 处理音频并生成视频（这里假设已经有处理代码）
    # 在实际场景中，你可以使用ffmpeg或其他工具生成视频
    # 这里简单地模拟生成一个视频文件
    print("处理音频文件")

    # 读取生成的视频文件并返回
    with open("/home/ubuntu/digital/video/test-websocket.mp4", "rb") as f:
        video_data = f.read()

    return video_data

async def handler(websocket, path):
    try:
        async for message in websocket:
            print("Received audio file from client")

            # 处理音频文件并生成视频
            video_data = await process_audio_and_generate_video(message)

            # 将生成的视频文件发送回客户端
            await websocket.send(video_data)
            print("Sent video file to client")

    except websockets.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
