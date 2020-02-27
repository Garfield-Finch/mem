import numpy as np
from PIL import Image
# import matplotlib
# import cv2
import pickle
import os
from tqdm import tqdm


def vis_pose_from_npy():
    file_name = '/p300/dataset/iPER/kpt_imgs/001/1/1/000.npy'

    img_np = np.load(file_name)

    img_np = np.sum(img_np, axis=2)
    print(img_np.dtype)
    print(img_np.shape)

    # matplotlib.image.imsave('name.png', img_np0)

    img_np = img_np * 255
    im = Image.fromarray(img_np)
    if im.mode == "F":
        im = im.convert('RGB')
    im.save("your_file.png")


def vis_pose_from_tensor(tsr_in, nm_file_out='yourfile'):
    img_np = tsr_in.to('cpu').numpy()

    # img_np = np.sum(img_np, axis=1)
    # img_np = np.expand_dims(img_np[0][0], axis=0)
    img_np = img_np[0][0]
    print(img_np.shape)
    # img_np = img_np[0].newaxis(0)

    img_np = img_np * 255
    # img_np = img_np[1]
    im = Image.fromarray(img_np.astype('uint8')).convert("RGB")
    # if im.mode == "F":
    #     im = im.convert('RGB')
    im.save(nm_file_out + ".png")


def _translate_kpts(joints, img_size):
    return np.array([img_size * (i + 1) / 2 for i in joints])


def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """

    :param input_image: numpy.array
    :param joints:
    :param draw_edges:
    :param vis:
    :param radius:
    :return:
    """
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([120, 120, 120]),  # Changed color white to a light gray, since the background is white.
    }

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    image = input_image.copy()
    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white'
    ]

    if joints.shape[1] == 19:
        # parent indices -1 means no parents
        parents = np.array([
            1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, -1, 13, -1, -1, 15, 16
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            17: 'light_green',
            18: 'light_green',
            14: 'purple'
        }
    elif joints.shape[1] == 14:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    else:
        print('Unknown skeleton!!')

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            # cv2.circle(image, (point[0], point[1]), radius, colors['white'], -1)
            cv2.circle(image, (point[0], point[1]), radius, [int(i) for i in colors['white']], -1)
            # cv2.circle(image, (point[0], point[1]), radius - 1, colors[jcolors[child]], -1)
            cv2.circle(image, (point[0], point[1]), radius - 1, [int(i) for i in colors[jcolors[child]]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1, colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            # cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1, colors[jcolors[pa_id]], -1)
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1, [int(i) for i in colors[jcolors[pa_id]]], -1)
            if child not in ecolors.keys():
                print('bad')
                import ipdb
                ipdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     [int(i) for i in colors[ecolors[child]]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image


def _gen_skeleton():
    root_path = "/p300/dataset/iPER/images_HD/"
    lst_root_dir = os.listdir(root_path)
    lst_root_dir.sort()

    for root_dir in lst_root_dir:
        path_root_dir = os.path.join(root_path, root_dir)
        lst_sub_dir = os.listdir(path_root_dir)
        lst_sub_dir.sort()

        for sub_dir in lst_sub_dir:
            path_sub_dir = os.path.join(root_path, root_dir, sub_dir)
            lst_img_dir = os.listdir(path_sub_dir)
            lst_img_dir.sort()

            for img_dir in lst_img_dir:
                path_img_dir = os.path.join(root_path, root_dir, sub_dir, img_dir)
                lst_img = os.listdir(path_img_dir)
                lst_img.sort()

                file_nm = os.path.join(path_img_dir, 'kps.pkl').replace('images_HD', 'smpls')

                print(f'file_nm:{file_nm}')
                if file_nm == '/p300/dataset/iPER/smpls/001/1/1/kps.pkl':
                    continue

                # file_nm = "/p300/dataset/iPER/smpls/001/1/1/kps.pkl"
                with open(file_nm, 'rb') as fo:  # 读取pkl文件数据
                    dict_data = pickle.load(fo, encoding='bytes')
                data = dict_data["kps"]
                # save_dir_path = "/p300/dataset/iPER/skeletons/001/1/1/"
                save_dir_path = path_img_dir.replace('images_HD', 'skeletons')
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)

                for img_nm in tqdm(lst_img):
                    img_number = int(img_nm[:-4])
                    kps = data[img_number]
                    img_in = (np.ones([1024, 1024, 3]) * 255).astype(np.uint8)
                    kps = _translate_kpts(kps, img_in.shape[0])
                    img_out = draw_skeleton(input_image=img_in, joints=kps)

                    im = Image.fromarray(img_out)
                    img_save_path = os.path.join(save_dir_path, img_nm).replace('.jpg', '.png')
                    im.save(img_save_path)


def _gen_namelist():
    ans = []
    root_path = "/p300/dataset/iPER/images_HD/"
    lst_root_dir = os.listdir(root_path)
    lst_root_dir.sort()

    for root_dir in lst_root_dir:
        path_root_dir = os.path.join(root_path, root_dir)
        lst_sub_dir = os.listdir(path_root_dir)
        lst_sub_dir.sort()

        for sub_dir in lst_sub_dir:
            path_sub_dir = os.path.join(root_path, root_dir, sub_dir)
            lst_img_dir = os.listdir(path_sub_dir)
            lst_img_dir.sort()

            for img_dir in lst_img_dir:
                path_img_dir = os.path.join(root_path, root_dir, sub_dir, img_dir)
                lst_img = os.listdir(path_img_dir)
                lst_img.sort()

                item_nm_list = os.path.join(root_dir, sub_dir, img_dir)
                ans.append(item_nm_list)

    print(ans)
    print(len(ans))

    # # Write to file
    with open("/p300/dataset/iPER/train_reserve.txt", 'w') as file:
        for i in tqdm(range(124)):
            file.write(str(ans[i]) + '\n')

    with open("/p300/dataset/iPER/val_reserve.txt", 'w') as file:
        for i in tqdm(range(124, 124 + 41)):
            file.write(str(ans[i]) + '\n')

    with open("/p300/dataset/iPER/test_reserve.txt", 'w') as file:
        for i in tqdm(range(124 + 41, len(ans))):
            file.write(str(ans[i]) + '\n')


if __name__ == "__main__":
    pass
    # _gen_skeleton()
    # _gen_namelist()
