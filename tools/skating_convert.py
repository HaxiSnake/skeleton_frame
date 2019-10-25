import os
import argparse
import json
import shutil
import sys
import pickle

import numpy as np
import torch
import skvideo.io

from numpy.lib.format import open_memmap

# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
stgcn_path="/home/jiangdong/workspace/st-gcn/"
sys.path.append(stgcn_path)
import tools
import tools.utils as utils
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skating Data Converter.')
    # region arguments yapf: disable
    parser.add_argument('--openpose',
        default='/home/jiangdong/opt/openpose/build',
        help='Path to openpose')
    parser.add_argument(
        '--data_path', default='/home/jiangdong/workspace/st-gcn/data/Skating',help="Path to dataset")
    parser.add_argument(
        '--out_folder', default='/home/jiangdong/workspace/st-gcn/output/Skating',help="Path to save files")
    parser.add_argument(
        '--model_folder', default='/home/jiangdong/workspace/st-gcn/models',help="Path to model folder")
    arg = parser.parse_args()
    arg.trainfile=os.path.join(arg.data_path,"train.csv")
    arg.testfile=os.path.join(arg.data_path,"val.csv")
    arg.labelfile=os.path.join(arg.data_path,"classInd.csv")
    openpose='{}/examples/openpose/openpose.bin'.format(arg.openpose)
    def _count_lines(filename):
        with open(filename) as f:
            count=-1
            for count,_ in enumerate(f):
                pass
            count+=1
        return count

    def _video_loader(filename):
        with open(filename) as f:
            for line in f.readlines():
                info=line.strip()
                video_name , _,label=info.split(" ")
                yield video_name,str(int(label)+1)

    def pose_estimation(openpose,out_folder,video_path,model_folder,info,p):
        video_name=video_path.split('/')[-1].split('.')[0]
        output_snippets_dir=os.path.join(out_folder,'openpose_estimation/snippets/{}'.format(video_name))
        output_sequence_dir = os.path.join(out_folder,'{}_data/'.format(p))
        output_sequence_path = '{}/{}.json'.format(output_sequence_dir, video_name)
        # pose estimation
        openpose_args = dict(
            video=video_path,
            write_json=output_snippets_dir,
            display=0,
            render_pose=0, 
            model_pose='COCO',
            model_folder=model_folder)
        command_line = openpose + ' '
        command_line += ' '.join(['--{} {}'.format(k, v) for k, v in openpose_args.items()])
        shutil.rmtree(output_snippets_dir, ignore_errors=True)
        os.makedirs(output_snippets_dir)
        print(command_line)
        os.system(command_line)
        # pack openpose ouputs
        video = utils.video.get_video_frames(video_path)
        height, width, _ = video[0].shape
        video_info = utils.openpose.json_pack(
            output_snippets_dir, video_name, width, height, label_index=info["label_index"],label=info["label"])
        if not os.path.exists(output_sequence_dir):
            os.makedirs(output_sequence_dir)
        with open(output_sequence_path, 'w') as outfile:
            json.dump(video_info, outfile)
        if len(video_info['data']) == 0:
            print('Can not find pose estimation results of %s'%(video_name))
            return
        else:
            print('%s Pose estimation complete.'%(video_name))    
    print(os.getcwd())
    label_names={}
    with open(arg.labelfile) as lf:
        for line in lf.readlines():
            index,label_name=line.strip().split(" ")
            label_names[index]=label_name
    print(label_names)
    part = ['train', 'val']
    for p in part:
        csvfile=os.path.join(arg.data_path,"{}.csv".format(p))
        label_file={}
        total_count = _count_lines(csvfile)
        count=0
        for nameinfo,label in _video_loader(csvfile):
            try:
                filename=nameinfo.split('/')[3]+".mp4"
#                 category=filename.split("_")[0]
                category=label_names[label]
                info={}
                info['label_index']=int(label)
                info['has_skeleton']=True
                info['label']=label_names[label]
                name_for_labelfile=filename.split('.')[0]
                label_file[name_for_labelfile]=info
                video_path = os.path.join(arg.data_path,category,filename)
                pose_estimation(openpose,arg.out_folder,video_path,arg.model_folder,info,p)
                count+=1
                print("%4.2f %% of %s has been processed"%(count*100/total_count,p))
            except Exception as e:
                print(e)
        label_save_path=os.path.join(arg.out_folder,"{}_label.json".format(p))
        with open(label_save_path,"w") as f:
            json.dump(label_file,f)
