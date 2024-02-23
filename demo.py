from modules.avd_network import AVDNetwork
from modules.dense_motion import DenseMotionNetwork
from modules.keypoint_detector import KPDetector
from modules.inpainting_network import InpaintingNetwork
import torch
from skimage import img_as_ubyte
from skimage.transform import resize
import imageio
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
import sys
import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] < 3:
    raise Exception(
        "You must use Python 3 or higher. Recommended version is Python 3.9")


def relative_kp(kp_source, kp_driving, kp_driving_initial):

    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(
        kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new


def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)
        config = yaml.load(f, yaml.FullLoader)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                   **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])

    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()

    return inpainting, kp_detector, dense_motion_network, avd_network


def make_animation(source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network, device, mode='relative'):
    assert mode in ['standard', 'relative', 'avd']
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(
            np.float32)).permute(0, 4, 1, 2, 3).to(device)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode == 'relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                      kp_driving_initial=kp_driving_initial)
            elif mode == 'avd':
                kp_norm = avd_network(kp_source, kp_driving)
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                kp_source=kp_source, bg_param=None,
                                                dropout_flag=False)
            out = inpainting_network(source, dense_motion)

            predictions.append(np.transpose(
                out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def make_animation_new(source_image, driving_reader, inpainting_network, kp_detector, dense_motion_network, avd_network, result_video_path='/tmp/temp.mp4', fps=30, duration=100, mode='relative', device='cuda'):
    assert mode in ['standard', 'relative', 'avd']
    writer = imageio.get_writer(result_video_path, fps=fps)
    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        kp_source = kp_detector(source)
        kp_driving_initial = None
        for frame in tqdm(driving_reader, total=int(fps*duration)):
            frame = resize(frame, (256, 256))[..., :3]
            driving_frame = torch.tensor(frame[np.newaxis].astype(
                np.float32)).permute(0, 3, 1, 2)
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if kp_driving_initial is None:
                driving_init = driving_frame.clone()
                kp_driving_initial = kp_detector(driving_init)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode == 'relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                      kp_driving_initial=kp_driving_initial)
            elif mode == 'avd':
                kp_norm = avd_network(kp_source, kp_driving)
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                kp_source=kp_source, bg_param=None,
                                                dropout_flag=False)
            out = inpainting_network(source, dense_motion)
            prediction = np.transpose(
                out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            prediction = (prediction*255).astype(np.uint8)
            writer.append_data(prediction)
    driving_reader.close()
    writer.close()


def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        try:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except:
            pass
    return frame_num


def create_image_animation(source_image_pth, driving_video_pth, result_video_pth, config, checkpoint, use_best_frame=False, device='cuda'):
    source_image = imageio.imread(source_image_pth)
    reader = imageio.get_reader(driving_video_pth)
    fps = reader.get_meta_data()['fps']
    duration = reader.get_meta_data()['duration']
    source_image = resize(source_image, (256, 256))[..., :3]
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
        config_path=config, checkpoint_path=checkpoint, device=device)

    if use_best_frame:
        i = find_best_frame(source_image, reader, device)
        print("Best frame: " + str(i))
        driving_video = [resize(im, (256, 256))[..., :3] for im in reader]
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, inpainting,
                                             kp_detector, dense_motion_network, avd_network, device=device, mode='relative')
        predictions_backward = make_animation(source_image, driving_backward, inpainting,
                                              kp_detector, dense_motion_network, avd_network, device=device, mode='relative')
        predictions = predictions_backward[::-1] + predictions_forward[1:]
        imageio.mimsave(opt.result_video, [img_as_ubyte(
            frame) for frame in predictions], fps=fps)
    else:
        predictions = make_animation_new(source_image, reader, inpainting, kp_detector, dense_motion_network,
                                         avd_network, result_video_pth, fps, duration)


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints/vox.pth.tar',
                        help="path to checkpoint to restore")

    parser.add_argument(
        "--source_image", default='./assets/source.png', help="path to source image")
    parser.add_argument(
        "--driving_video", default='./assets/driving.mp4', help="path to driving video")
    parser.add_argument(
        "--result_video", default='./result.mp4', help="path to output")

    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')

    parser.add_argument("--mode", default='relative', choices=['standard', 'relative', 'avd'],
                        help="Animate mode: ['standard', 'relative', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--cpu", dest="cpu",
                        action="store_true", help="cpu mode.")

    opt = parser.parse_args()

    # source_image = imageio.imread(opt.source_image)
    # reader = imageio.get_reader(opt.driving_video)
    # fps = reader.get_meta_data()['fps']
    # driving_video = []
    # try:
    #     for im in reader:
    #         driving_video.append(im)
    # except RuntimeError:
    #     pass
    # reader.close()

    # if opt.cpu:
    #     device = torch.device('cpu')
    # else:
    #     device = torch.device('cuda')

    # source_image = resize(source_image, opt.img_shape)[..., :3]
    # driving_video = [resize(frame, opt.img_shape)[..., :3]
    #                  for frame in driving_video]
    # inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
    #     config_path=opt.config, checkpoint_path=opt.checkpoint, device=device)

    # if opt.find_best_frame:
    #     i = find_best_frame(source_image, driving_video, opt.cpu)
    #     print("Best frame: " + str(i))
    #     driving_forward = driving_video[i:]
    #     driving_backward = driving_video[:(i+1)][::-1]
    #     predictions_forward = make_animation(source_image, driving_forward, inpainting,
    #                                          kp_detector, dense_motion_network, avd_network, device=device, mode=opt.mode)
    #     predictions_backward = make_animation(source_image, driving_backward, inpainting,
    #                                           kp_detector, dense_motion_network, avd_network, device=device, mode=opt.mode)
    #     predictions = predictions_backward[::-1] + predictions_forward[1:]
    # else:
    #     predictions = make_animation(source_image, driving_video, inpainting, kp_detector,
    #                                  dense_motion_network, avd_network, device=device, mode=opt.mode)

    # imageio.mimsave(opt.result_video, [img_as_ubyte(
    #     frame) for frame in predictions], fps=fps)
create_image_animation('test/0429_1-ok.png', 'test/driving.mp4',
                       'test/result.mp4', 'config/vox-256.yaml', 'ckpt/vox.pth.tar')
