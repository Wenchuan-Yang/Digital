# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import time
import json
import gevent
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import os
import re
import numpy as np
from threading import Thread,Event
import multiprocessing

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from webrtc import HumanPlayer

import argparse

import shutil
import asyncio


app = Flask(__name__)
sockets = Sockets(app)
nerfreals = []
statreals = [] 

    
@sockets.route('/humanecho')
def echo_socket(ws):
    """
    Websocket回调函数，接收和发送WebSocket消息。
    
    Args:
        ws (WebSocket): WebSocket对象。
    
    Returns:
        str: 当WebSocket连接未建立或接收到空消息时，返回错误信息。
    
    """
    # 获取WebSocket对象
    #ws = request.environ.get('wsgi.websocket')
    # 如果没有获取到，返回错误信息
    print(f" ====连接建立！===")
    if not ws:
        print('未建立连接！')
        return 'Please use WebSocket'
    # 否则，循环接收和发送消息
    else:
        print('建立连接！')
        while True:
            message = ws.receive()           
            
            if not message or len(message)==0:
                return '输入信息为空'
            else:                                
                nerfreal.put_msg_txt(message)


def llm_response(message):
    from llm.LLM import LLM

    # 初始化模型为Gemini，模型路径为'gemini-pro'，API密钥为'Your API Key'，代理URL为None
    # 注意：该行已被注释掉，不会执行
    # llm = LLM().init_model('Gemini', model_path= 'gemini-pro',api_key='Your API Key', proxy_url=None)

    # 初始化模型为ChatGPT，模型路径为'gpt-3.5-turbo'，API密钥为'Your API Key'
    # 注意：该行已被注释掉，不会执行
    # llm = LLM().init_model('ChatGPT', model_path= 'gpt-3.5-turbo',api_key='Your API Key')

    # 初始化模型为VllmGPT，模型路径为'THUDM/chatglm3-6b'
    llm = LLM().init_model('VllmGPT', model_path= 'THUDM/chatglm3-6b')

    # 使用llm模型进行对话，输入message，获取response
    response = llm.chat(message)

    # 打印response
    print(response)

    # 返回response
    return response

@sockets.route('/humanchat')
def chat_socket(ws):
    # 获取WebSocket对象
    #ws = request.environ.get('wsgi.websocket')
    # 如果没有获取到，返回错误信息
    if not ws:
        print('未建立连接！')
        return 'Please use WebSocket'
    # 否则，循环接收和发送消息
    else:
        print('建立连接！')
        while True:
            message = ws.receive()           
            
            if len(message)==0:
                return '输入信息为空'
            else:
                res=llm_response(message)                           
                nerfreal.put_msg_txt(res)

#####webrtc###############################
pcs = set()

#@app.route('/offer', methods=['POST'])
async def offer(request):
    # 从请求中获取json参数
    print(f" 收到offer请求！")
    params = await request.json()
    print(f" 收到offer参数：{params}")
    # 创建RTCSessionDescription对象
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # 获取当前可用的sessionid
    sessionid = len(nerfreals)
    # 遍历statreals列表，找到第一个值为0的元素的索引
    for index,value in enumerate(statreals):
        if value == 0:
            sessionid = index
            break
    # 如果sessionid超出了nerfreals的长度，则打印提示信息并返回-1
    if sessionid>=len(nerfreals):
        print('reach max session')
        return -1
    # 将当前sessionid对应的statreals元素设置为1
    statreals[sessionid] = 1

    # 创建一个 RTCPeerConnection 对象
    pc = RTCPeerConnection()
    # 将pc对象添加到pcs集合中
    pcs.add(pc)

    # 当连接状态改变时执行以下回调函数
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        # 打印当前连接状态
        print(" offer - Connection state is %s" % pc.connectionState)
        # 如果连接状态为failed，则关闭连接并从pcs集合中移除pc对象，并将当前sessionid对应的statreals元素设置为0
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            statreals[sessionid] = 0
        # 如果连接状态为closed，则从pcs集合中移除pc对象，并将当前sessionid对应的statreals元素设置为0
        if pc.connectionState == "closed":
            pcs.discard(pc)
            statreals[sessionid] = 0

    # 创建一个HumanPlayer对象
    player = HumanPlayer(nerfreals[sessionid])
    # 将音频轨道添加到pc对象中
    audio_sender = pc.addTrack(player.audio)
    # 将视频轨道添加到pc对象中
    video_sender = pc.addTrack(player.video)

    # 设置pc对象的远程描述为offer
    await pc.setRemoteDescription(offer)

    # 创建answer
    answer = await pc.createAnswer()
    # 设置pc对象的本地描述为answer
    await pc.setLocalDescription(answer)

    # 返回一个包含sdp、type和sessionid的json响应
    # return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    print(f" 返回offer响应！")
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}
        ),
    )

async def human(request):
    # 打印日志，表示接收到一个 HTTP 请求，显示请求的方法和路径
    print(f" human() 接收到请求 {request.method} {request.path}")
    
    # 异步地获取请求体中的 JSON 数据，并将其解析为 Python 字典对象
    params = await request.json()
    # 打印解析得到的参数
    print(f"  params = {params}")

    # 从请求参数中获取 'sessionid' 值，如果未提供则默认为 0
    sessionid = params.get('sessionid', 0)

    # 如果请求参数中带有 'interrupt' 标志
    if params.get('interrupt'):
        # 暂停当前 session 对应的 nerfreals 对象的对话
        nerfreals[sessionid].pause_talk()

    # 检查 'type' 字段的值，如果是 'echo'
    if params['type'] == 'echo':
        # 将收到的文本消息放入当前 session 对应的 nerfreals 对象中
        nerfreals[sessionid].put_msg_txt(params['text'])
    elif params['type'] == 'chat':
        # 如果 'type' 字段的值是 'chat'
        # 使用 run_in_executor 在事件循环中运行阻塞的 LLM（大型语言模型）响应计算
        res = await asyncio.get_event_loop().run_in_executor(None, llm_response(params['text']))                         
        # 将生成的响应文本放入当前 session 对应的 nerfreals 对象中
        nerfreals[sessionid].put_msg_txt(res)

    # 返回一个 HTTP 响应，响应内容为 JSON 格式，表示操作成功
    return web.Response(
        content_type="application/json",  # 设置响应的内容类型为 JSON
        text=json.dumps({"code": 0, "data": "ok"})  # 将响应内容序列化为 JSON 字符串
    )


async def set_audiotype(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)    
    nerfreals[sessionid].set_curr_state(params['audiotype'],params['reinit'])

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data":"ok"}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')

async def run(push_url):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("run - Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[0])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")

    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")

    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')
   
    parser.add_argument('--num_rays', type=int, default=4096 * 16, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### loss set
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    
    parser.add_argument('--bg_img', type=str, default='white', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### else
    parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")

    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000, help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # asr
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    #parser.add_argument('--asr_model', type=str, default='deepspeech')
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto') #
    # parser.add_argument('--asr_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
    # parser.add_argument('--asr_model', type=str, default='facebook/hubert-large-ls960-ft')

    parser.add_argument('--asr_save_feats', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--fullbody', action='store_true', help="fullbody human")
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='wav2lip_avatar_1')
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)

    # parser.add_argument('--customvideo', action='store_true', help="custom video")
    # parser.add_argument('--customvideo_img', type=str, default='data/customvideo/img')
    # parser.add_argument('--customvideo_imgnum', type=int, default=1)

    parser.add_argument('--customvideo_config', type=str, default='')

    parser.add_argument('--tts', type=str, default='edgetts') #xtts gpt-sovits
    parser.add_argument('--REF_FILE', type=str, default=None)
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='wav2lip') #musetalk wav2lip

    parser.add_argument('--transport', type=str, default='webrtc') #rtmp webrtc rtcpush
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=2)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010)

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    if opt.model == 'ernerf':
        print(f"         nerf model: {opt.model}")
    elif opt.model == 'wav2lip':
        print(f" ====================== wav2lip model: {opt.model} ================================ ")
        from lipreal import LipReal
        print(opt)
        for _ in range(opt.max_session):
            nerfreal = LipReal(opt)
            nerfreals.append(nerfreal)
    
    for _ in range(opt.max_session):
        statreals.append(0)


    #############################################################################
    appasync = web.Application()  # 创建一个 aiohttp 异步 Web 应用实例。
    appasync.on_shutdown.append(on_shutdown)  # 在应用关闭时调用 on_shutdown 函数，用于清理资源或保存状态。
    appasync.router.add_post("/offer", offer)  # 添加一个 POST 路由，处理 /offer 路径上的请求，并调用 offer 函数。
    appasync.router.add_post("/human", human)  # 添加一个 POST 路由，处理 /human 路径上的请求，并调用 human 函数。
    appasync.router.add_post("/set_audiotype", set_audiotype)  # 添加一个 POST 路由，处理 /set_audiotype 路径上的请求，并调用 set_audiotype 函数。
    appasync.router.add_static('/', path='web')  # 设置静态文件服务，路径为 /，静态文件目录为 'web'。

    # 配置默认的 CORS（跨域资源共享）设置，允许所有域名的请求。
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,  # 允许发送身份验证信息（如 Cookie）。
                expose_headers="*",  # 允许所有响应头暴露给客户端。
                allow_headers="*",  # 允许所有请求头发送到服务器。
            )
        })

    # 为所有路由配置 CORS 设置，使每个路由都遵循上面定义的 CORS 规则。
    for route in list(appasync.router.routes()):
        cors.add(route)

    # 根据不同的传输方式设置默认的 HTML 页面名称。
    pagename = 'webrtcapi.html'  # 默认页面名称为 webrtcapi.html。

    # 输出启动服务器的消息，显示服务器的 IP 地址和端口，以及对应的页面路径。
    print('start http server; http://<serverip>:' + str(opt.listenport) + '/' + pagename)

    # 定义一个函数，用于运行 HTTP 服务器。
    def run_server(runner):
        loop = asyncio.new_event_loop()  # 创建一个新的事件循环。
        asyncio.set_event_loop(loop)  # 将这个事件循环设置为当前线程的默认事件循环。
        loop.run_until_complete(runner.setup())  # 运行 runner 的 setup 方法，完成服务器的初始化。
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)  # 创建一个 TCPSite 实例，绑定到指定的 IP 和端口。
        loop.run_until_complete(site.start())  # 启动 HTTP 服务器。
        print("============1=================")
        # if opt.transport == 'rtcpush':
        #     loop.run_until_complete(run(opt.push_url))  # 如果传输方式为 rtcpush，调用 run 函数启动推流服务。
        loop.run_forever()  # 运行事件循环，保持服务器持续运行。
        print("============1=================")

        

    # # 直接运行 run_server 函数，启动服务器并传入 appasync 的运行器。
    run_server(web.AppRunner(appasync))

    # 以下代码段是被注释掉的旧实现或备用代码。

    # app.on_shutdown.append(on_shutdown)  # 为了兼容可能的其他实现，这行代码是注释掉的旧实现。

    # app.router.add_post("/offer", offer)  # 这个路由添加是冗余的，因为之前已经添加过，因此被注释掉。

    # print('start websocket server')  # 这是一个启动 WebSocket 服务器的旧实现的开始部分。

    # print(f'start websocket server; ws://<serverip>:' + str(opt.listenport) + '/ws')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # # 创建一个 WSGI 服务器，绑定到 0.0.0.0:8000，并使用 WebSocketHandler 处理 WebSocket 请求。
    # server.serve_forever()  # 启动服务器，保持运行，监听来自客户端的 WebSocket 连接。
