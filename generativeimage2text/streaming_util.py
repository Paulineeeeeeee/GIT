import jsonlines
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import BertTokenizer
from torchvision.transforms import functional as F
import numpy as np
import pandas as pd

import jax.numpy as jnp


def write_log(args,im):
    with open(f"./log/{args['exp_name']}_bs{args['bs']}_lr{args['lr']}_im{im}_log.txt","a") as f:
        f.write(f"bs = {args['bs']}, num_epoch = {args['epoch']},\n")
        f.write(f"lr = {args['lr']}, wd = {args['wd']},\n")

class ImageProcessor:
    def __init__(self, pixel_mean, pixel_std):
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def preprocess(self, inputs):
        """Preprocess images. Normalize pixels for non-padded pixels."""
        mean = jnp.asarray(self.pixel_mean, dtype=jnp.float32).reshape(1, 1, 1, 3)
        std = jnp.asarray(self.pixel_std, dtype=jnp.float32).reshape(1, 1, 1, 3)
        inputs = (inputs - mean) / std
        return inputs


def get_images(id, data_path, length, num_images,trsfm):
    # num_images default is 6
    # f"{des}{id}-{i}-{len(episode)}.jpg"
    path = data_path+id
    images = []
    if length >=num_images:
        for i in range(num_images):
            # get last six images
            image_path = f"{path}-{length-num_images+i}-{length}.jpg"
            images.append(trsfm(Image.open(image_path).convert('RGB')))
          
    else:
        for i in range(length):
            # get last six images
            image_path = f"{path}-{i}-{length}.jpg"            
            images.append(trsfm(Image.open(image_path).convert('RGB')))
        shape = images[0].shape
        for i in range(num_images-length):
            images.append(torch.zeros(shape))
        # for i in range(num_images-length):
        #     images.append()
    # Image.open(image_path).convert('RGB')
    images_stacked = torch.stack(images, dim=0)
    return images_stacked


def get_llama_images(id, data_path, length, num_images,trsfm):
    # num_images default is 6
    # f"{des}{id}-{i}-{len(episode)}.jpg"
    path = data_path
    images = []
    for i in range(length):
            # get last six images
            image_path = f"{path}/{i}.png"       
            image = Image.open(image_path).convert('RGB')     
            image_np = np.array(image) / 255.0 
            images.append(trsfm.preprocess(image_np))
    
    images_stacked = jnp.stack(images, axis=0)

    return images_stacked


def trsfm(image_size = (224,224),split = 'VALID'):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    if split == 'TRAIN':
        ret = v2.Compose(
                [
                    # v2.RandomResizedCrop(size=image_size, scale=(0.5,1.0), interpolation=InterpolationMode.BICUBIC),
                    v2.RandomResizedCrop(size=image_size, interpolation=InterpolationMode.BICUBIC),
                    v2.RandomAffine(degrees=0,translate=(0.2, 0.2), scale=(0.75, 1)),
                    v2.ColorJitter(brightness=0.5, contrast=0.5),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean, std)
                ]
            )
    else:
        ret = v2.Compose(
                [
                    v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean, std)
                ]
            )
    return ret


def kmeans(init_centers, data, weights, num_iters=1):
  """Run kmeans on weighted data.


  Args:
    init_centers: array in shape (k, d);
    data: array in shape (n, d); All data points.
    weights: array in shape (n,); Weights of the data points.
    num_iters: int;
  Returns:
    new_centers: (k, d)
    counts: (k,), num_data assigned to each center
  """
  k = init_centers.shape[0]
  def step_fn(_, centers_counts):
    centers, _ = centers_counts
    # TODO(zhouxy): We might want to try other distance functions.
    distances = jnp.linalg.norm(
        data[:, None] - centers[None, :], axis=2)  # (n, k)
    assignments = jnp.argmin(distances, axis=1)  # (n,)
    weighted_data = data * weights[:, None]  # (n, d)
    one_hot_assignments = jax.nn.one_hot(assignments, k)  # (n, k)
    # NOTE: The following stop_gradient is optional, since both one_hot and
    # argmin are not differentiable anyway.
    # one_hot_assignments = jax.lax.stop_gradient(one_hot_assignments)
    weighted_sums = jnp.dot(
        one_hot_assignments.T, weighted_data)  # (k, d)
    counts = jnp.dot(one_hot_assignments.astype(jnp.int32).T, weights)  # (k,)
    # If the cluster is empty (which can happen from the second iteration),
    # we just retain the original cluster center.
    new_centers = weighted_sums / jnp.maximum(counts[:, None], 1)
    new_centers = jnp.where(
        jnp.broadcast_to(counts[:, None], new_centers.shape) == 0,
        centers, new_centers)

    return new_centers, counts
  new_centers, counts = jax.lax.fori_loop(
      0, num_iters, step_fn,
      (init_centers, jnp.zeros((k,), dtype=jnp.int32)))
  # new_centers = jax.lax.stop_gradient(new_centers)
  # counts = jax.lax.stop_gradient(counts)
  return new_centers, counts


class AITW_Dataset(Dataset):
    def __init__(self, jsonl_file, data_path,split,tokenizer,transform, num_images=6 ):
        self.data = []
        self.num_images = num_images
        self.data_path = data_path
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Read data from JSONL file
        with jsonlines.open(jsonl_file) as reader:
            size = sum(1 for i in reader)
        
        with jsonlines.open(jsonl_file) as reader:
            if split == 'TRAIN':
                limit = int(size*0.8)
                for i,lines in enumerate(reader):
                    if i < limit:
                        self.data.append(lines)
                    else:
                        break
            if split == 'VALID':
                limit = int(size*0.8)
                for i,lines in enumerate(reader):
                    if i >= limit:
                        self.data.append(lines)

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, samples):
        max_length = 0
        for x in samples:
            max_length = max(max_length,x['caption_tokens'].shape[0])
        for x in samples:
            t = x['caption_tokens']
            n = x['need_predict']
            b1 = torch.zeros(max_length, dtype=t.dtype, device=t.device)
            b2 = torch.zeros(max_length, dtype=n.dtype, device=n.device)
            b1[:t.shape[0]] = t
            b2[:n.shape[0]] = n
            x['caption_tokens'] = b1
            x['need_predict'] = b2
        image_batch = torch.stack([x['image'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
        }
        return data

    def __getitem__(self, idx):
        # {"id": "18375519518960921438", "goal_info": "Is it going to rain this weekend?", "episode_length": 8}
        # 10002872452831025023-0-14.jpg
        item = self.data[idx]
        images = get_images(item['id'],self.data_path,item['episode_length'],self.num_images,self.transform)

        max_text_len = 40 #from train.py
        target = item['goal_info']
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
        # print('get_data')
        # print(prefix)
        # print(target)
        # print(need_predict)
        # print(payload)
        if len(payload) > max_text_len:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        # Convert other fields to tensors as needed
        data = {
            'caption_tokens': torch.tensor(input_ids),
            'need_predict': torch.tensor(need_predict),
            'image': images,
            # 'rect' field can be fed in 'caption', which tells the bounding box
            # region of the image that is described by the caption. In this case,
            # we can optionally crop the region.
            # 'caption': {},
            # this iteration can be used for crop-size selection so that all GPUs
            # can process the image with the same input size
            # 'iteration': 0
        }
        
        return data
class AITW_Dataset_V2(Dataset):
    def __init__(self, data_path,split,tokenizer,transform, num_images=6 ):
        self.data = []
        self.num_images = num_images
        self.data_path = data_path
        self.transform = transform
        self.tokenizer = tokenizer
        # category = ['google_apps','general','install','web_shopping']
        category = ['web_shopping']
        # Read data from JSONL file
        for cat in category:
            jsonl_file = f'/local/pauline/GIT/test/no_miss_{cat}_train.jsonl'
            with jsonlines.open(jsonl_file) as reader:
                size = sum(1 for i in reader)
            
            with jsonlines.open(jsonl_file) as reader:
                if split == 'TRAIN':
                    limit = int(size*0.8)
                    for i,lines in enumerate(reader):
                        if i < limit:
                            self.data.append(lines)
                        else:
                            break
                if split == 'VALID':
                    limit = int(size*0.8)
                    for i,lines in enumerate(reader):
                        if i >= limit:
                            self.data.append(lines)

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, samples):
        max_length = 0
        for x in samples:
            max_length = max(max_length,x['caption_tokens'].shape[0])
        for x in samples:
            t = x['caption_tokens']
            n = x['need_predict']
            b1 = torch.zeros(max_length, dtype=t.dtype, device=t.device)
            b2 = torch.zeros(max_length, dtype=n.dtype, device=n.device)
            b1[:t.shape[0]] = t
            b2[:n.shape[0]] = n
            x['caption_tokens'] = b1
            x['need_predict'] = b2
        image_batch = torch.stack([x['image'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
        }
        return data

    def __getitem__(self, idx):
        # {"id": "18375519518960921438", "goal_info": "Is it going to rain this weekend?", "episode_length": 8}
        # 10002872452831025023-0-14.jpg
        item = self.data[idx]
        path = self.data_path + item['category']+'/'
        images = get_images(item['id'],path,item['episode_length'],self.num_images,self.transform)

        max_text_len = 40 #from train.py
        target = item['goal_info']
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
        # print('get_data')
        # print(prefix)
        # print(target)
        # print(need_predict)
        # print(payload)
        if len(payload) > max_text_len:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        # Convert other fields to tensors as needed
        data = {
            'caption_tokens': torch.tensor(input_ids),
            'need_predict': torch.tensor(need_predict),
            'image': images,
            # 'rect' field can be fed in 'caption', which tells the bounding box
            # region of the image that is described by the caption. In this case,
            # we can optionally crop the region.
            # 'caption': {},
            # this iteration can be used for crop-size selection so that all GPUs
            # can process the image with the same input size
            # 'iteration': 0
        }
        
        return data

class LlamaTouch(Dataset):
    def __init__(self, split, tokenizer, transform, image_encoder, num_images=6, streaming_buffer_size=16, kmeans_num_iters=10):
        self.data = []
        self.num_images = num_images
        self.data_path = '/home/pauline/llamatouch_dataset/'
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_encoder = image_encoder
        self.streaming_buffer_size = streaming_buffer_size
        self.kmeans_num_iters = kmeans_num_iters

        df = pd.read_csv('/home/pauline/llamatouch_dataset/llamatouch_task_metadata.tsv', sep='\t')

        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)

        # Split into train and validation
        train_df = df[:int(0.8 * len(df))]
        valid_df = df[int(0.8 * len(df)):]

        # Save train and validation data to self.data
        self.data = train_df if split == 'TRAIN' else valid_df


    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, samples):
        max_length = 0
        for x in samples:
            max_length = max(max_length,x['caption_tokens'].shape[0])
        for x in samples:
            t = x['caption_tokens']
            n = x['need_predict']
            b1 = torch.zeros(max_length, dtype=t.dtype, device=t.device)
            b2 = torch.zeros(max_length, dtype=n.dtype, device=n.device)
            b1[:t.shape[0]] = t
            b2[:n.shape[0]] = n
            x['caption_tokens'] = b1
            x['need_predict'] = b2
        image_batch = torch.stack([x['image'] for x in samples])
        caption_batch = torch.stack([x['caption_tokens'] for x in samples])
        predict_batch = torch.stack([x['need_predict'] for x in samples])
        data = {
            'caption_tokens': caption_batch,
            'need_predict': predict_batch,
            'image': image_batch,
        }
        return data

    def get_visual_features(self, images, train=False):
        visual_features = self.image_encoder(images, train=train)  # (B, hw, D)
        if self.num_images > 0:  # video model
            num_tokens = visual_features.shape[1]
            visual_features = visual_features.reshape(
                (-1, self.num_images) + visual_features.shape[1:]
            )  # (B // t, t, hw, D)
            visual_feat_dim = visual_features.shape[-1]
            temp_emb = self.param(
                    'temperal_embedding',
                    nn.initializers.zeros,
                    (self.num_frames, 1, 1, visual_feat_dim),
                )
            visual_features = visual_features + temp_emb[
                    None, :, 0]  # (B // t, t, hw, D)
            
            visual_features = visual_features.reshape(
                    visual_features.shape[0], self.num_frames * num_tokens,
                    visual_features.shape[-1],
                )  # (B // t, t * hw, D)

        else:  # image model
            visual_features = visual_features.reshape(
                visual_features.shape[0], -1, visual_features.shape[-1],
            )  # (B, hw, D)
        return visual_features

    def get_streaming_features(self, features, train = False):
        """Get streaming features.

        Args:
        features: (video_batch_size, num_tot_tokens, dim)
        train: bool
        Returns:
        streaming_features: (video_batch_size, num_streaming_tokens, dim)
        """
        # NOTE: We can also implement this under
        # CaptioningFlaxModel.pool_video_feature. Put them here in a separate
        # StreamingCaptioningFlaxModel to make it less entangled with existing code.
        # The behaviours of temporal_mean_pool/ spatial_mean_pool are the same as
        # setting "frame_fuse_fn" in CaptioningFlaxModel.
        unused_video_batch_size, _, dim = features.shape

        del train
        def streaming_feature_extractor(feature):

            kmeans_num_iters = 1
            centers = feature[:self.streaming_buffer_size]
            weights = jnp.ones((feature.shape[0],), dtype=jnp.int32)
            streaming_feature = kmeans(
                centers, feature, weights=weights,
                num_iters=kmeans_num_iters)[0]

            return streaming_feature

        streaming_features = jax.vmap(
            streaming_feature_extractor,
            in_axes=0, out_axes=0, axis_name='batch')(features)
        return streaming_features
    
    def __getitem__(self, idx):
        # episode	        category path	            description	        nsteps	app
        # 84143002711104077	general	 general/trace_11	Open the settings	4	    Settings
        item = self.data.iloc[idx]
        path = self.data_path + item['path']+'/'
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        processor = ImageProcessor(pixel_mean=mean, pixel_std=std)
        images = get_llama_images(item['episode'],path,item['nsteps'],self.num_images,processor)
        visual_features = self.get_visual_features(images)  # (batch_size, num_tokens, dim)
        visual_features = self.get_streaming_features(visual_features)  # (video_batch_size, new_num_vis_tokens, proj_dim)
        
        max_text_len = 40 #from train.py
        target = item['description']
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
  
        if len(payload) > max_text_len:
            payload = payload[-(max_text_len - 2):]
            need_predict = need_predict[-(max_text_len - 2):]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        # Convert other fields to tensors as needed
        data = {
            'caption_tokens': torch.tensor(input_ids),
            'need_predict': torch.tensor(need_predict),
            'image': torch.tensor(visual_features),
            # 'rect' field can be fed in 'caption', which tells the bounding box
            # region of the image that is described by the caption. In this case,
            # we can optionally crop the region.
            # 'caption': {},
            # this iteration can be used for crop-size selection so that all GPUs
            # can process the image with the same input size
            # 'iteration': 0
        }
        
        return data



def main():
    annotations_file = '/local/pauline/GIT/preprocessing/no_miss_general_train.jsonl'
    data_path = '/home/pauline/no-miss-AITW/general/'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = AITW_Dataset(annotations_file,data_path,'TRAIN',tokenizer,transform = trsfm())
    for i in range(5):
        print(dataset[i])
    # trainloader = DataLoader(dataset, batch_size=32, num_workers=2, collate_fn = dataset.collate_fn)
    # print(next(iter(trainloader))['caption_tokens'])
    # print(next(iter(trainloader))['need_predict'])
    # for i,data in enumerate(trainloader):
    #     if i == 1:
    #         break
    #     print(data)


if __name__ == '__main__':
    main()
    # print_demo_images()



