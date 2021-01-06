from pynuscenes.nuscenes_dataset import NuscenesDataset
from tqdm import tqdm

def main():
    nusc = NuscenesDataset("/DATA/datasets/nuscenes", 'v1.0-mini', 'mini_train', 'config/nuscenes.yml')
    classes = {}
    for frame in tqdm(nusc):
        for ann in frame['anns']:
            if ann['category_id'] in classes:
                classes[ann['category_id']][0] += 1
                classes[ann['category_id']][1] += ann['box_3d'].center[2]
            else:
                classes[ann['category_id']] = [1,ann['box_3d'].center[2]]

    print("AVG_HEIGHT = { \n    0 : 1.0, ")
    for cat, arr in classes.items():
        print("    {:2} :{:.5f}, ".format(cat, arr[1]/arr[0]))
    print("}")

    for i in range(1, 10):
        assert(i in classes.keys()), "Height not calculated for all keys!"

if __name__ == "__main__":
    main()