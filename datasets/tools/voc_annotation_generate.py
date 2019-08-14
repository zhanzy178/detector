import argparse
import os
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

def parsing():
    parser = argparse.ArgumentParser(add_help='Preprocess program to process voc2007 annotation format to suitable format.')
    parser.add_argument('voc_root', type=str, help='VOC2007 dataset root')

    return parser.parse_args()

def process_set(imgset_file, anno_root, output_root):
    filename = os.path.splitext(os.path.basename(imgset_file))[0]
    output_filename = filename + '_anno.json'
    output_path = os.path.join(output_root, output_filename)

    annotations = []
    with open(imgset_file, 'r') as f:
        for imgid in tqdm(f.readlines()):
            xml_anno_path = os.path.join(anno_root, imgid.strip()+'.xml')
            if not os.path.exists(xml_anno_path):
                print('Warning: not exist path "%s"!!!'%xml_anno_path)
                continue
            # parsing xml
            tree = ET.parse(xml_anno_path)
            img_filename = tree.find('filename').text
            img_width = int(tree.find('size').find('width').text)
            img_height = int(tree.find('size').find('height').text)
            bboxes = []
            for obj in tree.findall('object'):
                bboxes.append(dict(
                        name=obj.find('name').text,
                        pose=obj.find('pose').text,
                        truncated=int(obj.find('truncated').text),
                        difficult=int(obj.find('difficult').text),
                        bbox=[
                            int(obj.find('bndbox').find('xmin').text),
                            int(obj.find('bndbox').find('ymin').text),
                            int(obj.find('bndbox').find('xmax').text),
                            int(obj.find('bndbox').find('ymax').text)
                        ]
                    ))
            annotations.append(dict(
                filename=img_filename,
                width=img_width,
                height=img_height,
                bboxes=bboxes
            ))

    print('annotations save to "%s"....'%output_path)
    with open(output_path, 'w') as f:
        json.dump(annotations, f)


def annotation_generate(voc_root):
    imgset_root = os.path.join(voc_root, 'ImageSets/Main')
    anno_root = os.path.join(voc_root, 'Annotations')

    process_set(os.path.join(imgset_root, 'trainval.txt'), anno_root, voc_root)
    process_set(os.path.join(imgset_root, 'train.txt'), anno_root, voc_root)
    process_set(os.path.join(imgset_root, 'val.txt'), anno_root, voc_root)
    process_set(os.path.join(imgset_root, 'test.txt'), anno_root, voc_root)


if __name__ == '__main__':
    args = parsing()
    annotation_generate(args.voc_root)