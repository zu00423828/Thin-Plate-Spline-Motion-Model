from logging import root
import pickle
import cv2
import numpy as np
import os
import torch
from demo import create_image_animation
import face_alignment
import subprocess
from tqdm import trange
from utils.crop_video import process_video
from gpen.face_enhancement import FaceEnhancement


def concat_video(left, right, out_path):
    video1 = cv2.VideoCapture(left)
    video2 = cv2.VideoCapture(right)
    fps = video1.get(5)
    out = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (512, 256))
    while video1.isOpened():
        ret, frame1 = video1.read()
        if not ret:
            break
        ret, frame2 = video2.read()
        if not ret:
            break
        frame = np.concatenate([frame1, frame2], axis=1)
        out.write(frame)
    video1.release()
    video2.release()
    out.release()
    # command=f'ffmpeg -i /tmp/temp_concat.mp4 -i /tmp/temp.wav  {out_path}'
    # subprocess.call(command,shell=True)


def check(dir):
    source_video = cv2.VideoCapture(f'{dir}/source.mp4')
    result_video = cv2.VideoCapture(f'{dir}/driving.mp4')
    out = cv2.VideoWriter(
        f'{dir}/check.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (256, 256))
    while source_video.isOpened():
        ret, frame1 = source_video.read()
        if not ret:
            break
        ret, frame2 = result_video.read()
        if not ret:
            break
        frame = (frame1/2)+(frame2/2)
        frame = frame.astype(np.uint8)
        out.write(frame)
    out.release()
    source_video.release()
    result_video.release()


def video_gpen_process(origin_video_path, model_dir, out_video_path='/tmp/paste_temp.mp4'):
    processer = FaceEnhancement(base_dir=model_dir, in_size=512, model='GPEN-BFR-512', sr_scale=2,
                                use_sr=False, sr_model=None)
    full_video = cv2.VideoCapture(origin_video_path)
    h = int(full_video.get(4))
    w = int(full_video.get(3))
    fps = full_video.get(5)
    frame_count = int(full_video.get(7))
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
        *'XVID'), fps, (w, h))
    for _ in trange(frame_count):
        _, frame = full_video.read()
        img_out, orig_faces, enhanced_faces = processer.process(
            frame, aligned=False)
        img_out = cv2.resize(img_out, (256, 256))
        out_video.write(img_out)
    return out_video_path


def make_image_animation_dataflow(source_path, driving_origin_path, result_path, model_dir, use_crop=False, crf=0, use_gfp=True, use_best=False, face_data=None):
    config_path = f"{os.path.split(os.path.realpath(__file__))[0]}/config/end2end.yaml"
    if use_crop:
        print('crop driving video', flush=True)
        driving_video_path = process_video(
            driving_origin_path, '/tmp/driving.mp4', min_frames=15, face_data=face_data)
        torch.cuda.empty_cache()
    else:
        driving_video_path = driving_origin_path
    command = f"ffmpeg -y -i {driving_video_path} /tmp/temp.wav "
    subprocess.call(command, shell=True)
    # driving_video_path = video_gpen_process(
    #     driving_video_path, model_dir, out_video_path='/tmp/driving_enhace.mp4')
    print('create animation', flush=True)
    safa_model_path = f'{model_dir}/final_3DV.tar'
    safa_video = create_image_animation(source_path, driving_video_path, '/tmp/temp.mp4', config_path,
                                        safa_model_path, with_eye=True, relative=True, adapt_scale=True, use_best_frame=use_best)
    torch.cuda.empty_cache()
    # print('extract landmark', flush=True)
    # ldmk_path = extract_landmark(safa_video, '/tmp/ldmk.pkl')
    # print('blur mouth video', flush=True)
    # torch.cuda.empty_cache()
    # safa_video = blur_video_mouth(
    #     safa_video, ldmk_path, '/tmp/blur.avi', kernel=3)
    print('enhaance process', flush=True)
    # paste_video_path = video_gfpgan_process(
    #     safa_video, ldmk_path, use_gfp, model_dir=model_dir)
    paste_video_path = video_gpen_process(safa_video, model_dir)
    # -preset veryslow
    command = f"ffmpeg -y -i {paste_video_path} -i /tmp/temp.wav  -crf  {crf} -vcodec h264  {result_path} "
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    # inference_animation_dataflow('new_test/source_all.mp4','new_test/driving_all.mp4','temp','finish.mp4','ckpt/final_3DV.tar')
    # make_animation_dataflow('test1/1.mp4','test1/1.mp4','test1/temp','finish_t.mp4','ckpt/final_3DV.tar',add_audo=True)
    # make_animation_dataflow('finish.mp4','finish_2/driving_all.mp4','finish_2/temp','finish2.mp4','ckpt/final_3DV.tar',add_audo=True)
    # concat_video('/home/yuan/repo/my_safa/01_18/1.mp4','/home/yuan/repo/my_safa/01_18/out/1_1.mp4','concat.mp4')

    # root='/home/yuan/hdd/safa_test/01_18_2'
    # make_image_animation_dataflow(f'{root}/EP010-08.jpg',f'{root}/1.mp4',f'{root}/1_gfpgan.mp4','ckpt/final_3DV.tar',use_crop=False)
    # concat_video(f'{root}/1_gfpgan.mp4',f'{root}/out/1.mp4','concat2.mp4')
    root = '/home/yuan/hdd/05_19_1'
    driving_video_path = os.path.join(
        root, 'lip_man.mp4')
    from pathlib import Path
    from glob import glob

    for image_path in sorted(glob(f'{root}/img/*g')):
        # image_input = os.path.join(root, image_path)
        image_input = image_path
        save_dir = os.path.join(root, Path(
            image_path).parent.name+'_out')
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, 'result_' +
                                Path(image_input).stem+'.mp4')
        if os.path.exists(out_path):
            continue
        make_image_animation_dataflow(
            image_input, driving_video_path, out_path, 'ckpt/', use_crop=True, face_data=os.path.join("datadir/preprocess/driving_woman/face.pkl"))

    root = '/home/yuan/hdd/05_19'
    for audio_path in sorted(glob(f'{root}/audio/*wav')):
        image_input = f"{root}/img/412.png"
        lip_dir = f'{root}/lip'
        os.makedirs(lip_dir, exist_ok=True)
        lip_path = os.path.join(lip_dir,
                                Path(audio_path).stem+'.mp4')
        if os.path.exists(lip_path) == False:
            generate_lip_video(
                "datadir/preprocess/driving_woman/face.pkl", audio_path, lip_path)
        save_dir = os.path.join(root, Path(
            image_input).parent.name+'_out')
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'result_' +
                                Path(lip_path).stem+'.mp4')
        if os.path.exists(out_path):
            continue
        make_image_animation_dataflow(
            image_input, lip_path, out_path, 'ckpt/', use_crop=True, crf=10, face_data="datadir/preprocess/driving_woman/face.pkl", use_best=True)

    root = '/home/yuan/hdd/05_23_custom'
    face_data = "datadir/preprocess/driving_woman/face.pkl"
    # face_data = '/home/yuan/hdd/driving_video/model2/face.pkl'
    for audio_path in sorted(glob(f'{root}/audio/woman.wav')):
        image_input = f"{root}/img/切cut-青輔02.png"
        lip_dir = f'{root}/lip3'
        os.makedirs(lip_dir, exist_ok=True)
        lip_path = os.path.join(lip_dir, 'result_' +
                                Path(audio_path).stem+'.mp4')
        if os.path.exists(lip_path) == False:
            generate_lip_video(
                face_data, audio_path, lip_path, start_seconds=3)
        save_dir = os.path.join(root, Path(
            image_input).parent.name+'_out')
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, 'result_書慧' +
                                Path(image_input).stem+'.mp4')
        if os.path.exists(out_path):
            continue
        make_image_animation_dataflow(
            image_input, lip_path, out_path, 'ckpt/', use_crop=True, face_data=face_data)
