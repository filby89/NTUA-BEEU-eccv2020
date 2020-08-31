import torch.utils.data as data
import cv2
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import torch
import torchvision.transforms.functional as tF

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])
    
    @property
    def min_frame(self):
        return int(self._data[2])

    @property
    def max_frame(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, mode,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, context=False):

        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self.bold_path = "/gpu-data/filby/BoLD/BOLD_public"

        self.context = context

        self.categorical_emotions = ["Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence", "Happiness",
                               "Pleasure", "Excitement", "Surprise", "Sympathy", "Doubt/Confusion", "Disconnect",
                               "Fatigue", "Embarrassment", "Yearning", "Disapproval", "Aversion", "Annoyance", "Anger",
                               "Sensitivity", "Sadness", "Disquietment", "Fear", "Pain", "Suffering"]

        self.continuous_emotions = ["Valence", "Arousal", "Dominance"]

        self.attributes = ["Gender", "Age", "Ethnicity"]

        header = ["video", "person_id", "min_frame", "max_frame"] + self.categorical_emotions + self.continuous_emotions + self.attributes + ["annotation_confidence"]
        
        # self.df = pd.read_csv(os.path.join(self.bold_path, "annotations/{}_extra.csv".format(mode)))
        self.df = pd.read_csv(os.path.join(self.bold_path, "annotations/{}.csv".format(mode)), names=header)
        self.df["joints_path"] = self.df["video"].apply(rreplace,args=[".mp4",".npy",1])

        self.video_list = self.df["video"]
        self.mode = mode

        self.embeddings = np.load("glove_840B_embeddings.npy")

    def get_context(self, image, joints, format="cv2"):
        joints = joints.reshape((18,3))
        joints[joints[:,2]<0.1] = np.nan
        joints[np.isnan(joints[:,2])] = np.nan

        joint_min_x = int(round(np.nanmin(joints[:,0])))
        joint_min_y = int(round(np.nanmin(joints[:,1])))

        joint_max_x = int(round(np.nanmax(joints[:,0])))
        joint_max_y = int(round(np.nanmax(joints[:,1])))

        expand_x = int(round(10/100 * (joint_max_x-joint_min_x)))
        expand_y = int(round(10/100 * (joint_max_y-joint_min_y)))

        if format == "cv2":
            image[max(0, joint_min_x - expand_x):min(joint_max_x + expand_x, image.shape[1])] = [0,0,0]
        elif format == "PIL":
            bottom = min(joint_max_y+expand_y, image.height)
            right = min(joint_max_x+expand_x,image.width)
            top = max(0,joint_min_y-expand_y)
            left = max(0,joint_min_x-expand_x)
            image = np.array(image)
            if len(image.shape) == 3:
                image[top:bottom,left:right] = [0,0,0]
            else:
                image[top:bottom,left:right] = np.min(image)
            return Image.fromarray(image)


    def get_bounding_box(self, image, joints, format="cv2"):
        joints = joints.reshape((18,3))
        joints[joints[:,2]<0.1] = np.nan
        joints[np.isnan(joints[:,2])] = np.nan

        joint_min_x = int(round(np.nanmin(joints[:,0])))
        joint_min_y = int(round(np.nanmin(joints[:,1])))

        joint_max_x = int(round(np.nanmax(joints[:,0])))
        joint_max_y = int(round(np.nanmax(joints[:,1])))

        expand_x = int(round(100/100 * (joint_max_x-joint_min_x)))
        expand_y = int(round(100/100 * (joint_max_y-joint_min_y)))

        if format == "cv2":
            return image[max(0,joint_min_y-expand_y):min(joint_max_y+expand_y, image.shape[0]), max(0,joint_min_x-expand_x):min(joint_max_x+expand_x,image.shape[1])]
        elif format == "PIL":
            bottom = min(joint_max_y+expand_y, image.height)
            right = min(joint_max_x+expand_x,image.width)
            top = max(0,joint_min_y-expand_y)
            left = max(0,joint_min_x-expand_x)
            return tF.crop(image, top, left, bottom-top ,right-left)


    def joints(self, index):
        sample = self.df.iloc[index]

        joints_path = os.path.join(self.bold_path, "joints", sample["joints_path"])

        joints18 = np.load(joints_path)
        joints18[:,0] -= joints18[0,0]

        return joints18

    def _load_image(self, directory, idx, index, mode="body"):
        joints = self.joints(index)

        poi_joints = joints[joints[:, 0] + 1 == idx]
        sample = self.df.iloc[index]
        poi_joints = poi_joints[(poi_joints[:, 1] == sample["person_id"]), 2:]
        if self.modality == 'RGB' or self.modality == 'RGBDiff':

            frame = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert("RGB")
          
            if mode == "context":
                if poi_joints.size == 0:
                    return [frame]
                context = self.get_context(frame, poi_joints, format="PIL")
                return [context]

            if poi_joints.size == 0:
                body = frame
                pass #just do the whole frame
            else:
                body = self.get_bounding_box(frame, poi_joints, format="PIL")

                if body.size == 0:
                    print(poi_joints)
                    body = frame

            return [body]

            # return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            frame_x = Image.open(os.path.join(directory, self.image_tmpl.format('flow_x', idx))).convert('L')
            frame_y = Image.open(os.path.join(directory, self.image_tmpl.format('flow_y', idx))).convert('L')
            # frame = cv2.imread(os.path.join(directory, 'img_{:05d}.jpg'.format(idx)))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if mode == "context":
                if poi_joints.size == 0:
                    return [frame_x, frame_y]
                context_x = self.get_context(frame_x, poi_joints, format="PIL")
                context_y = self.get_context(frame_y, poi_joints, format="PIL")
                return [context_x, context_y]

            if poi_joints.size == 0:
                body_x = frame_x
                body_y = frame_y
                pass #just do the whole frame
            else:
                body_x = self.get_bounding_box(frame_x, poi_joints, format="PIL")
                body_y = self.get_bounding_box(frame_y, poi_joints, format="PIL")

                if body_x.size == 0:
                    body_x = frame_x
                    body_y = frame_y


            return [body_x, body_y]


    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) # + (record.min_frame+1)
            # print(record.num_frames, record.min_frame, record.max_frame)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        sample = self.df.iloc[index]

        fname = os.path.join(self.bold_path,"videos",self.df.iloc[index]["video"])

        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))-1

        capture.release()

        record_path = os.path.join(self.bold_path,"test_raw",sample["video"][4:-4])

        record = VideoRecord([record_path, frame_count, sample["min_frame"], sample["max_frame"]])

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices, index)

    def get(self, record, indices, index):

        images = list()
        # print(indices)
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
              
                seg_imgs = self._load_image(record.path, p, index, mode="body")

                images.extend(seg_imgs)

                if self.context:
                    seg_imgs = self._load_image(record.path, p, index, mode="context")
                    images.extend(seg_imgs)


                if p < record.num_frames:
                    p += 1


        if self.mode != "test":
            categorical = self.df.iloc[index][self.categorical_emotions]

            continuous = self.df.iloc[index][self.continuous_emotions]
            continuous = continuous/10.0 # normalize to 0 - 1

            if self.transform is None:
                process_data = images
            else:
                process_data = self.transform(images)

            return process_data, torch.tensor(self.embeddings).float(), torch.tensor(categorical).float(), torch.tensor(continuous).float(), self.df.iloc[index]["video"]
        else:
            process_data = self.transform(images)
            return process_data, torch.tensor(self.embeddings).float()

    def __len__(self):
        return len(self.df)
