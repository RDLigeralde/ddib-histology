# %%
import os
from CLAM.dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from CLAM.wsi_core.WholeSlideImage import WholeSlideImage
from torchvision import transforms
import argparse

# %%
def get_tiles(wsi_dir: str, h5_dir: str, extensions = ['.qptiff', '.tif', '.svs']) -> None:
    """
    filler for now
    """
    out_dir = os.path.dirname(h5_dir)
    for wsi_file in os.listdir(wsi_dir):
        split = os.path.splitext(wsi_file)
        ext = split[1]
        if ext in extensions:
            wsi_path = os.path.join(wsi_dir, wsi_file)
            h5 = wsi_file.replace(ext, '.h5')
            h5_path = os.path.join(h5_dir, h5)

            whole = WholeSlideImage(wsi_path)
            wsi = whole.wsi
            img_transforms = transforms.Compose([])
            dataset = Whole_Slide_Bag_FP(file_path = h5_path, wsi = wsi, img_transforms = img_transforms)
            wsi_outdir = os.path.join(out_dir, split[0])
            if not os.path.exists(wsi_outdir):
                os.makedirs(wsi_outdir)

            for i in range(len(dataset)):
                current = dataset[i]
                img, coord = current['img'], current['coord']
                patch_fname = f"patch_{i}_{coord[0]}_{coord[1]}.png"
                patch_path = os.path.join(wsi_outdir, patch_fname)
                if not os.path.exists(patch_path):
                    img.save(patch_path)

# %%

#python images_from_h5s.py --wsi_dir "/project/kimlab_hshisto/WSI/bright" --h5_dir "/home/roblig22/ddib/datasets/histology/tilings/bright/patches"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', type=str)
    parser.add_argument('--h5_dir', type=str)
    parser.add_argument('--extensions', type=str, nargs='*', default=['.qptiff', '.tif', '.svs'])

    args = parser.parse_args()
    get_tiles(args.wsi_dir, args.h5_dir, args.extensions)

if __name__ == "__main__":
    main()

