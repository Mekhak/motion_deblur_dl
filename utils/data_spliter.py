import os
import shutil
from tqdm import tqdm

def move_to_train_test_dirs(train_or_test = "train"):
    root_dir = "E:\\motion_deblur\\GOPRO_Large"
    data_dir = os.path.join(root_dir, train_or_test)

    new_data_dir = os.path.join(root_dir, train_or_test + "_new")
    new_data_blur_dir = os.path.join(new_data_dir, "blur")
    new_data_sharp_dir = os.path.join(new_data_dir, "sharp")

    os.mkdir(new_data_dir)
    os.mkdir(new_data_blur_dir)
    os.mkdir(new_data_sharp_dir)


    i = 0
    # iterate over all sub-folders, move all images in "train" or "test" dir to "new_data_dir"
    for sub_data_dir_name in tqdm(os.listdir(data_dir)):
        # print(sub_data_dir_name)
        for dir_name in os.listdir(os.path.join(data_dir, sub_data_dir_name)):
            # print(dir_name)
            if dir_name == "blur":
                for file_name in os.listdir(os.path.join(data_dir, sub_data_dir_name, dir_name)):
                    src_blur = os.path.join(data_dir, sub_data_dir_name, "blur", file_name)
                    dst_blur = os.path.join(new_data_blur_dir, str(i) + ".png")

                    src_sharp = os.path.join(data_dir, sub_data_dir_name, "sharp", file_name)
                    dst_sharp = os.path.join(new_data_sharp_dir, str(i) + ".png")

                    shutil.copyfile(src_blur, dst_blur)
                    # os.rename(dst_blur, os.path.join(dst_blur, str(i) + ".png"))

                    shutil.copyfile(src_sharp, dst_sharp)
                    # os.rename(dst_sharp, os.path.join(dst_sharp, str(i) + ".png"))
                    i += 1


# move_to_train_test_dirs("test")
# move_to_train_test_dirs("train")