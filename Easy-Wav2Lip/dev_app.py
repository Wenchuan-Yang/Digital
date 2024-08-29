import asyncio
import websockets
import tempfile
import subprocess
from inference import parser, do_load, main as process

# 初始化模型或其他资源
args = parser.parse_args()
do_load(args.checkpoint_path)

async def process_audio_segment(audio_data, count):
    # 创建临时文件来保存音频数据
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file_path = temp_audio_file.name

    # 使用 ffmpeg 将 WebM 转换为 WAV
    output_wav_file = f"temp/input_{count}.wav"
    command = [
        "ffmpeg", "-y", "-i", temp_audio_file_path, output_wav_file
    ]
    process_result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 检查 ffmpeg 执行情况
    if process_result.returncode != 0:
        print(f"FFmpeg error: {process_result.stderr.decode()}")
        return None

    return output_wav_file

async def process_audio_and_return_video(websocket, path):
    print(f" path = {path}")
    count = 0
    try:
        while True:
            # 接收音频数据
            print(f" 开始接收数据, count = {count}")
            audio_data = await websocket.recv()

            print(f"接收到音频信息, type(audio_data) = {type(audio_data)}")
            
            # 处理音频数据并生成 wav 文件
            audio_path = await process_audio_segment(audio_data, count)
            if not audio_path:
                await websocket.send("Error processing audio data.".encode('utf-8'))
                continue

            # 处理音频并生成视频
            print("开始处理音频并生成视频")
            args.audio = audio_path
            try:
                temp_video_file = process(args, audio_path)
            except Exception as e:
                print(f"Video processing error: {e}")
                await websocket.send("Error generating video.".encode('utf-8'))
                continue

            # 逐块读取和发送视频数据
            print("开始返回视频流数据")
            with open(temp_video_file, 'rb') as video_file:
                chunk_size = 1024 * 1024  # 1MB per chunk
                while True:
                    chunk = video_file.read(chunk_size)
                    if not chunk:
                        break
                    await websocket.send(chunk)

            count += 1

    except Exception as e:
        print(f"Error: {e}")
        await websocket.send("Error processing audio and generating video.".encode('utf-8'))

async def main():
    async with websockets.serve(process_audio_and_return_video, "0.0.0.0", 8000):
        await asyncio.Future()  # 运行直到进程被终止

if __name__ == "__main__":
    asyncio.run(main())
