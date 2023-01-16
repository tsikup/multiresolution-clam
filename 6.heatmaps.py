import os
import torch
import natsort
import argparse
import openslide
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from torchvision import transforms
from wholeslidedata.annotation.parser import AnnotationParser, QuPathAnnotationParser
from wholeslidedata.image.wholeslideimage import WholeSlideImage

from definitions import ROOT_DIR
from utils.config import get_config
from he_preprocessing.utils.timer import Timer
from he_preprocessing.utils.image import is_blurry, keep_tile, pad_image
from pytorch.data_helpers.wsi.utils import (
    create_batch_sampler,
    visHeatmap,
    whole_slide_files_from_folder_factory,
)
from pytorch.data_helpers.wsi import MaskedTiledAnnotationHook
from pytorch.models.ssl_features.vit import ViT
from pytorch.models.classification.clam import CLAM_Features_PL


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        "--slides-dir",
        dest="slides_dir",
        help="Directory containing the pathology slides.",
        required=True,
    )

    argparser.add_argument(
        "--annotations-dir",
        dest="annotations_dir",
        help="Directory containing the annotation files.",
        required=True,
    )

    argparser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory to save hdf5 files for each slide.",
        required=True,
    )

    argparser.add_argument(
        "--image-extension",
        dest="image_extension",
        default=".ndpi",
        help="Extension of digital pathology slides.",
        required=False,
    )

    argparser.add_argument(
        "--ann-extension",
        dest="ann_extension",
        default=".geojson",
        help="Extension of digital pathology annotations.",
        required=False,
    )
    
    argparser.add_argument(
        "--model-ckpt",
        dest="model_ckpt",
        default=None,
        help="CLAM model checkpoint.",
        required=True,
    )
    
    argparser.add_argument(
        "--labels-csv",
        dest="labels_csv",
        default=None,
        help="Labels for each slide.",
        required=True,
    )
    
    argparser.add_argument(
        "--label",
        dest="label",
        default=None,
        help="Which label to use.",
        required=True,
    )

    argparser.add_argument(
        "--tile-size",
        dest="tile_size",
        type=int,
        default=512,
        help="Tile size.",
        required=False,
    )

    argparser.add_argument(
        "--blurriness-threshold",
        dest="blurriness_threshold",
        type=int,
        default=None,
        help="Blurriness threshold for target.",
        required=False,
    )

    argparser.add_argument(
        "--intersection-percentage",
        dest="intersection_percentage",
        type=float,
        default=1.0,
        help="Intersection threshold between tiled patch and annotation to include patch.",
        required=False,
    )

    argparser.add_argument(
        "--stride-overlap-percentage",
        dest="stride_overlap_percentage",
        type=float,
        default=0.0,
        help="Stride overlap percentage between sampled tile windows.",
        required=False,
    )

    argparser.add_argument(
        "--tissue-percentage",
        dest="tissue_percentage",
        type=float,
        default=0.5,
        help="Tissue percentage to keep tile.",
        required=False,
    )

    argparser.add_argument(
        "--config",
        dest="config",
        default=os.path.join(ROOT_DIR, "assets/configs/seg_config.yml"),
        help="Config file to use.",
        required=False,
    )

    args = argparser.parse_args()
    return args

TISSUE_LABEL = 1
TUMOR_LABEL = 2
TARGET_SPACING = 0.5
CONTEXT_SPACING = 2.0

def get_models(clam_ckpt, ckpt_dir_feature_extractor='/mimer/NOBACKUP/groups/foukakis_ai/niktsi/models'):
    vit = ViT(arch="small", ckpt=os.path.join(ckpt_dir_feature_extractor, "vits_tcga_brca_dino.pt"))
    vit = vit.cuda()
    vit.eval()
    
    clam = CLAM_Features_PL.load_from_checkpoint(clam_ckpt, strict=False)
    clam = clam.cuda()
    clam.eval()
    
    return vit, clam

def compute_embeddings(model, x, transform):
    with torch.no_grad():
        x = transform(x).unsqueeze(0).cuda()
        o = np.squeeze(model.forward(x).detach().cpu().numpy())
    return o

def qc_tile(x_target, x_context, blurriness=500, tile_size=512, tissue_threshold=0.5):
    if is_blurry(
        x_target,
        threshold=blurriness,
        normalize=True,
        masked=False,
    ):
        print('Tile is blurry...')
        return None, None
    
    x_target = pad_image(
        x_target,
        tile_size,
        value=230,
    )
    
    x_context = pad_image(
        x_context,
        tile_size,
        value=230,
    )
    
    if not keep_tile(
        x_target,
        tile_size=tile_size,
        tissue_threshold=tissue_threshold,
        pad=True,
    ):
        print('Tile doesn\'t have enough tissue.')
        return None, None
    
    return x_target, x_context

def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    trnsfrms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return trnsfrms_val

def extract_features(model, x_target, x_context, transform):
    features_target = compute_embeddings(model, x_target, transform)
    features_context = compute_embeddings(model, x_context, transform)
    return {'target': features_target, 'context': features_context}

def transform_features(features):
    keys = list(features.keys())
    keys.sort()
    if keys == []:
        return None, None
    
    features_target = []
    features_context = []
    for key in keys:
        features_target.append(torch.from_numpy(features[key]['target']))
        features_context.append(torch.from_numpy(features[key]['context']))
        
    features_target = torch.vstack(features_target).to('cuda')
    features_context = torch.vstack(features_context).to('cuda')
    return features_target, features_context

def infer_single_slide(model, features, features_context, label, reverse_label_dict, k=1):
    with torch.no_grad():
        logits, Y_prob, Y_hat, A, model_results_dict = model.forward(
            h=features,
            h_context=features_context,
            label=label,
            instance_eval=False,
            return_features=False,
            attention_only=False,
        )
        
        Y_hat = Y_hat.item()

        A = A.view(-1, 1).cpu().numpy()

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	

        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A


def get_files(
    slides_dir,
    annotations_dir,
    tile_size,
    labels,
    stride_overlap_percentage,
    file_type="mrwsi",
):

    if ann_extension == ".geojson":
        parser = QuPathAnnotationParser
    else:
        parser = AnnotationParser
    parser = parser(
        labels=labels,
        hooks=(
            MaskedTiledAnnotationHook(
                tile_size=tile_size,
                ratio=1,
                overlap=int(tile_size * stride_overlap_percentage),
                label_names=list(labels.keys()),
                full_coverage=True,
            ),
        ),
    )

    image_files = whole_slide_files_from_folder_factory(
        slides_dir,
        file_type,
        excludes=[
            "mask",
        ],
        filters=[
            slide_extension,
        ],
        image_backend="openslide",
    )

    annotation_files = whole_slide_files_from_folder_factory(
        annotations_dir,
        "wsa",
        excludes=["tif"],
        filters=[ann_extension],
        annotation_parser=parser,
    )
    return image_files, annotation_files


def extract_data(
    model,
    feature_extractor,
    transform,
    labels_df: pd.DataFrame,
    label: str,
    n_classes: int,
    output_root_dir: Path,
    image_files,
    annotation_files,
    slide_extension,
    ann_extension,
    file_type,
    tile_size,
    batch_size,
    tissue_percentage,
    intersection_percentage,
    stride_overlap_percentage,
    labels,
    spacing,
    blurriness_threshold: Dict[str, int],
    label_rename_dict: Dict[str, int] = None,
    base_label=0,
    seed=123,
):
    
    for image_idx, image_file in enumerate(image_files):
        print("##############")
        print(image_file)
        print("##############")

        try:
            batch_sampler, batch_ref_sampler, batch_shape = create_batch_sampler(
                image_files=[image_file],
                annotation_files=annotation_files,
                slide_extension=slide_extension,
                ann_extension=ann_extension,
                file_type=file_type,
                tile_size=tile_size,
                batch_size=batch_size,
                tissue_percentage=tissue_percentage,
                stride_overlap_percentage=stride_overlap_percentage,
                intersection_percentage=intersection_percentage,
                blurriness_threshold=blurriness_threshold,
                labels=labels,
                spacing=spacing,
                extract_graph=False,
                seed=seed,
            )
        except ValueError:
            print(f"No annotations found in {image_file}")
            continue

        image_name = image_file.path.stem
        
        y = labels_df.loc[labels_df['image_name'] == image_name, label].to_numpy()[0]
        raw_y = y
            
        if y == 'Unknown':
            continue
        else:
            if label_rename_dict is not None:
                if y in label_rename_dict.keys():
                    y = label_rename_dict[y]
                else:
                    continue
        
        y = torch.from_numpy(np.array([y])).cuda()
        
        output_dir = Path(output_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        target_key = ("target", spacing["target"])
        context_key = ("context", spacing["context"])
        shape = (tile_size, tile_size, 3)
        
        features = {}
        qc_failed_features = {}
        
        idx = 0
        idx_failed = 0
        while True:
            ref_batch = batch_ref_sampler.batch()
            if ref_batch == []:
                break
            try:
                total_labels = len(batch_ref_sampler)
            except TypeError:
                total_labels = -1
            
            wsi = WholeSlideImage(os.path.join(
                args.slides_dir, image_name + args.image_extension
            ))
            
            wsi_openslide = openslide.OpenSlide(os.path.join(
                args.slides_dir, image_name + args.image_extension
            ))
            
            level = wsi.get_level_from_spacing(
                wsi.get_real_spacing(
                    spacing['target']
                )
            )
        
            downsample = wsi.downsamplings[level]
            
            center_point = ref_batch[0]['point']
                
            ann_idx = f'{ref_batch[0]["reference"].wsa_index}_{ref_batch[0]["reference"].annotation_index}'
            print(
                f"************ Processing {ann_idx} out of {total_labels} total labels. {image_file}, {image_idx} out of {len(image_files)} ************"
            )

            x_batch, y_batch = batch_sampler.batch(ref_batch)
            
            if x_batch is None:
                break
            
            x_target = x_batch[0][target_key][shape]
            x_context = x_batch[0][context_key][shape]
            
            x_target, x_context = qc_tile(
                x_target,
                x_context,
                blurriness=blurriness_threshold['target'],
                tile_size=tile_size,
                tissue_threshold=tissue_percentage
            )

            if x_target is None or x_context is None:
                qc_failed_features[idx_failed] = {
                    'center_point': center_point,
                    'downsample': downsample,
                }
                
                idx_failed += 1
                continue
            
            _features = extract_features(feature_extractor, x_target, x_context, transform)
            
            features[idx] = {
                'center_point': center_point,
                'downsample': downsample,
                'target': _features['target'],
                'context': _features['context']
            }
            
            idx += 1
            
        features_keys = list(features.keys())
        features_keys.sort()
        
        features_target, features_context = transform_features(features)
        if features_target is None:
            continue
        
        ids, preds_str, probs, A = infer_single_slide(model, features_target, features_context, y, {v-base_label: k for k, v in label_rename_dict.items()}, k=n_classes)
        
        center_points = np.array(
            [
                [features[idx]['center_point'].x, features[idx]['center_point'].y] for idx in features_keys
            ]
        )
        
        features_keys = list(qc_failed_features.keys())
        features_keys.sort()
        qc_failed_center_points = np.array(
            [
                [qc_failed_features[idx]['center_point'].x, qc_failed_features[idx]['center_point'].y] for idx in features_keys
            ]
        )
        
        patch_downsample = features[0]['downsample']
        
        del features
        del qc_failed_features
        
        heatmap = visHeatmap(wsi_openslide, A, center_points, patch_downsample,
                   vis_downsample=8,
                   coords_are_center=True,
                   top_left=None, bot_right=None,
                   patch_size=(512, 512),
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4,
                   blur=False, overlap=0.0,
                   segment=False, use_holes=True,
                   convert_to_percentiles=True,
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample = 1,
                   cmap='coolwarm')
        
        heatmap.save(os.path.join(output_dir, 'heatmap_' + f'y_{raw_y}_pred_{preds_str[0]}' + '_' + image_name + '.png'))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    timer = Timer()

    args = get_args()

    config, _ = get_config(args.config)

    slide_extension = args.image_extension
    if not slide_extension.startswith("."):
        slide_extension = "." + slide_extension

    ann_extension = args.ann_extension
    if not ann_extension.startswith("."):
        ann_extension = "." + ann_extension

    file_type = "mrwsi"

    labels = {"tumor": TUMOR_LABEL}

    spacing = {
        "target": TARGET_SPACING,
        "context": CONTEXT_SPACING,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files, annotation_files = get_files(
        slides_dir=args.slides_dir,
        annotations_dir=args.annotations_dir,
        file_type=file_type,
        tile_size=args.tile_size,
        labels=labels,
        stride_overlap_percentage=args.stride_overlap_percentage,
    )

    image_files = natsort.natsorted(image_files, key=str)

    blurriness_threshold = dict(
        target=args.blurriness_threshold, context=None
    )
    
    labels_df = pd.read_csv(args.labels_csv)
    labels_df['image_name'] = labels_df['image_name'].apply(lambda x: Path(x).stem)
    label = args.label
    
    vit, model = get_models(clam_ckpt=args.model_ckpt)
    
    transform = eval_transforms(pretrained=False)

    extract_data(
        model,
        vit,
        transform,
        labels_df,
        label,
        2,
        output_dir,
        image_files,
        annotation_files,
        slide_extension,
        ann_extension,
        file_type,
        args.tile_size,
        1,
        args.tissue_percentage,
        args.intersection_percentage,
        args.stride_overlap_percentage,
        labels,
        spacing,
        blurriness_threshold,
        label_rename_dict={'II':2, 'III':3},
        base_label=2,
        seed=np.random.randint(2**32 - 1),
    )
