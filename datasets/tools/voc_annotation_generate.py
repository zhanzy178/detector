import argparse
import os
import xml.dom.minidom
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
            DOMTree = xml.dom.minidom.parse(xml_anno_path)
            annotation = DOMTree.documentElement
            img_filename = annotation.getElementsByTagName('filename')[0].childNodes[0].data
            img_width = int(annotation.getElementsByTagName('size')[0].getElementsByTagName('width')[0].childNodes[0].data)
            img_height = int(annotation.getElementsByTagName('size')[0].getElementsByTagName('height')[0].childNodes[0].data)
            bboxes = []
            for obj in annotation.getElementsByTagName('object'):
                bboxes.append(dict(
                    name=obj.getElementsByTagName('name')[0].childNodes[0].data,
                    pose=obj.getElementsByTagName('pose')[0].childNodes[0].data,
                    truncated=int(obj.getElementsByTagName('truncated')[0].childNodes[0].data),
                    difficult=int(obj.getElementsByTagName('difficult')[0].childNodes[0].data),
                    bbox=[
                        int(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].childNodes[0].data),
                        int(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].childNodes[0].data),
                        int(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].childNodes[0].data),
                        int(obj.getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].childNodes[0].data)
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