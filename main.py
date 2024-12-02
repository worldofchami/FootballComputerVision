from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import torch
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked
import numpy as np
import umap
from sklearn.cluster import KMeans
from sports.common.team import TeamClassifier

# REDUCER = umap.UMAP(n_components=3)
# CLUSTERING_MODEL = KMeans(n_clusters=2)

# SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
# EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

PITCH_DETECTION_MODEL = YOLO("./models/pitch-detection.pt")
PLAYER_DETECTION_MODEL = YOLO("./models/player-detection.pt")

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

SOURCE_VIDEO_PATH = "./content/121364_0.mp4"
TARGET_VIDEO_PATH = "./output/clip_1.mp4"

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.Color.from_hex("#FFD700"),
    thickness=2
)

triangle_annotator = sv.TriangleAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
    base=20,
    height=17
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
    text_color=sv.Color.from_hex("#000000"),
    text_position=sv.Position.BOTTOM_CENTER
)

tracker = sv.ByteTrack()
tracker.reset()

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)

STRIDE = 30

def extract_crops():
    global SOURCE_VIDEO_PATH

    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=STRIDE)

    crops = []

    for frame in tqdm(frame_generator, desc="Collecting crops"):
        result = PLAYER_DETECTION_MODEL(frame, conf=0.3)[0]

        detections = sv.Detections.from_ultralytics(result)

        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]

        crops += [
            sv.crop_image(frame, xyxy)
            for xyxy
            in detections.xyxy
        ]
    
    return crops

def resolve_gks_team_id(players_detections: sv.Detections, gk_detections: sv.Detections):
    gks_xy = gk_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_centroid = players_xy[players_detections.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_detections.class_id == 1].mean(axis=0)

    gk_team_ids = []
    for gk_xy in gks_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        
        gk_team_ids.append(0 if dist_0 < dist_1 else 1)

    return np.array(gk_team_ids)

"""Manually grouping teams"""

# BATCH_SIZE = 32

# crops = extract_crops()
# crops = [sv.cv2_to_pillow(crop) for crop in crops]

# batches = chunked(crops, BATCH_SIZE)

# data = []

# with torch.no_grad():
#     for batch in tqdm(batches, desc="Embeddings extraction"):
#         inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
#         outputs = EMBEDDINGS_MODEL(**inputs)
#         embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

#         data.append(embeddings)

# data = np.concatenate(data)

# projections = REDUCER.fit_transform(data)

# clusters = CLUSTERING_MODEL.fit_predict(projections)

# team_0 = [
#     crop
#     for crop, cluster
#     in zip(crops, clusters)
#     if clusters == 0
# ]

"""Auto Grouping Teams"""
crops = extract_crops()
team_classifier = TeamClassifier(device=DEVICE)
team_classifier.fit(crops)

# with video_sink:
# for frame in tqdm(frame_generator, total=video_info.total_frames):

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=STRIDE)

frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL(frame, conf=0.3)[0]
detections = sv.Detections.from_ultralytics(result)

ball_detections = detections[detections.class_id == BALL_ID]
# ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections, px=10)

all_detections = detections[detections.class_id != BALL_ID]
all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
all_detections = tracker.update_with_detections(detections=all_detections)

player_detections = all_detections[all_detections.class_id == PLAYER_ID]
players_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
player_detections.class_id = team_classifier.predict(players_crops)

goalkeeper_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
goalkeeper_detections.class_id = resolve_gks_team_id(player_detections, goalkeeper_detections)

referee_detections = all_detections[all_detections.class_id == REFEREE_ID]
referee_detections.class_id -= 1

all_detections = sv.Detections.merge([player_detections, goalkeeper_detections, referee_detections])

labels = [
    f"#{tracker_id}"
    for tracker_id
    in all_detections.tracker_id
]

annotated_frame = frame.copy()

annotated_frame = triangle_annotator.annotate(
    scene=annotated_frame,
    detections=all_detections
)

annotated_frame = ellipse_annotator.annotate(
    scene=annotated_frame,
    detections=ball_detections
)

annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=all_detections,
    labels=labels
)

sv.plot_image(annotated_frame)

# video_sink.write_frame(annotated_frame)