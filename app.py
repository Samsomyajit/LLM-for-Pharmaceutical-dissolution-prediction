# pip install omnipose
from cellpose_omni import io, models
from omnipose.gpu import use_gpu
from pre import pre_crop
from matplotlib import pyplot as plt
import os


root_path = r"D:\Dataset\battery\ningde\trainingdata\3.5"
subfolders = [f.name for f in os.scandir(root_path) if f.is_dir()]

for subfolder in subfolders:
    img_path = os.path.join(root_path, subfolder)
    # img_path = r"D:\Dataset\battery\ningde\trainingdata\4\240922"
    imgs, files = pre_crop(img_path)

    plt.imshow(imgs[0], cmap='gray')
    plt.axis('off')
    plt.show()

    model = models.CellposeModel(gpu=True, model_type="cyto2")

    masks, flows, styles = model.eval(
        imgs,
        channels=[0, 0],
        omni=True,
        invert=False,
        diameter=15,
        # tile=True,
        # resample=True,
        # cellprob_threshold=0,
        mask_threshold=0.0,
        flow_threshold=0.0,
        # min_size=0,
    )

    print("masks.max() =", int(masks[0].max()))

    # check masks size
    print("masks shape =", masks[0].shape)
    print(f"mask type = {masks[0].dtype}")

    # 保存（建议 TIF + 分目录）
    io.save_masks(
        imgs,
        masks,
        flows,
        files,
        tif=True,
        png=False,
        in_folders=True,
        save_flows=True,
        save_outlines=True,
        save_ncolor=True,
    )

    print("Saved masks for", subfolder)

else:
    print("All done!")
