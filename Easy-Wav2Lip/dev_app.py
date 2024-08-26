import asyncio
import json
import tempfile
import subprocess
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
import websockets

async def handle_webrtc_signaling(websocket, path):
    pc = RTCPeerConnection()

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            await websocket.send(json.dumps({
                "type": "candidate",
                "candidate": {
                    'candidate': candidate.candidate,
                    'sdpMid': candidate.sdpMid,
                    'sdpMLineIndex': candidate.sdpMLineIndex,
                    'usernameFragment': candidate.usernameFragment
                }
            }))

    @pc.on("datachannel")
    async def on_datachannel(channel):
        @channel.on("message")
        async def on_message(message):
            # 假设接收到的消息是音频数据
            if isinstance(message, bytes):
                # 处理音频数据并生成视频片段
                video_data = await process_audio_to_video(message)
                # 将生成的视频片段发送回前端
                channel.send(video_data)

    while True:
        message = await websocket.recv()

        if isinstance(message, str):
            data = json.loads(message)
            if data["type"] == "offer":
                await pc.setRemoteDescription(RTCSessionDescription(data["sdp"], 'offer'))
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send(json.dumps({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                }))
            elif data["type"] == "candidate":
                candidate = data["candidate"]
                ice_candidate = RTCIceCandidate(
                    sdpMid=candidate.get("sdpMid"),
                    sdpMLineIndex=candidate.get("sdpMLineIndex"),
                    candidate=candidate.get("candidate")
                )
                await pc.addIceCandidate(ice_candidate)

async def process_audio_to_video(audio_data):
    # 创建一个临时文件来保存接收到的音频数据
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file_path = temp_audio_file.name

    # 使用 ffmpeg 将 WebM 转换为 WAV
    output_wav_file = temp_audio_file_path.replace('.webm', '.wav')
    command = [
        "ffmpeg", "-y", "-i", temp_audio_file_path, output_wav_file
    ]
    process_result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process_result.returncode != 0:
        print(f"FFmpeg error: {process_result.stderr.decode()}")
        return None

    # 处理音频生成视频
    video_data = generate_video_from_audio(output_wav_file)

    return video_data

def generate_video_from_audio(audio_path):
    # 此函数应根据实际需要生成视频
    # 假设我们使用 ffmpeg 创建一个简单的视频
    output_video_file = audio_path.replace('.wav', '.mp4')
    command = [
        "ffmpeg", "-y", "-loop", "1", "-i", "input_image.jpg", "-i", audio_path,
        "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-shortest", output_video_file
    ]
    process_result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process_result.returncode != 0:
        print(f"FFmpeg error: {process_result.stderr.decode()}")
        return None

    # 读取生成的视频文件并返回其内容
    with open(output_video_file, 'rb') as video_file:
        video_data = video_file.read()

    return video_data

async def main():
    async with websockets.serve(handle_webrtc_signaling, "0.0.0.0", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
