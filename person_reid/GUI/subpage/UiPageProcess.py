import os
import sys
import json
import csv
import cv2
import datetime
from GUI.libs import qt_sql
from GUI.libs import img_show_and_encoder
from GUI.libs.draw_box_api import draw_chinese_box
import Algorithm.libs.config.model_cfgs as cfgs
import traceback
from collections import defaultdict
import numpy as np
from functools import partial
from Algorithm.reid_outer_api import ReidPipeline
from PySide6.QtCore import Qt, QModelIndex
from PySide6.QtWidgets import (
                                QFileDialog,
                                QHeaderView,
                                QDataWidgetMapper,
                                QTableView,
                                QMessageBox,
                                QHBoxLayout,
                                QPushButton,
                                QVBoxLayout,
                                QWidget,
                                QLineEdit,
                            )
from PySide6.QtGui import QImage, QPixmap, QColor, QStandardItemModel, QStandardItem
from PySide6.QtSql import QSqlRelation, QSqlRelationalTableModel, QSqlTableModel
from PySide6.QtCore import Qt, Signal, QObject, QThread, QTimer, QMetaObject, Q_ARG
from ultralytics.utils.plotting import Annotator, colors

GUI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(GUI_DIR, "config")
OUTPUTS_DIR = os.path.join(GUI_DIR, "outputs")

class ProcessThread(QObject):
    debug_msg = Signal(str)
    show_match_status = Signal(str)
    show_match_id = Signal(str)
    show_match_dist = Signal(str)
    progress_bar = Signal(int)
    show_img = Signal(np.ndarray)
    show_target_img = Signal(np.ndarray)
    table_info_list = Signal(list)
    
    def __init__(
        self, base_feat_lists, base_idx_lists, dims=None,
        target_class="person", device_info="cpu", modalities=None,
    ):
        QObject.__init__(self)
        if dims is None:
            dims = cfgs.DIMS
        self.reid_pipeline = ReidPipeline(
            base_feat_lists, base_idx_lists, dims=dims,
            target_class=target_class, device_info=device_info, modalities=modalities,
        )
        self.proc_source_url = ''
        self.proc_ir_source_url = ''
        self.proc_source_type = None
        self._ir_capture = None
        self._ir_static_frame = None
        self.skip_frames = 6
        self.stop_dtc = False
        self.continue_dtc = True
        self.match_thresh = cfgs.MATCH_DIST_THRESH
        self.match_min_margin = cfgs.MATCH_MIN_MARGIN
        self.match_min_cosine_sim = cfgs.MATCH_MIN_COSINE_SIM
        self.track_lock_dist_thresh = cfgs.TRACK_LOCK_DIST_THRESH
        self.is_track = False
        self.is_show_no_match_item = False
        self.had_track_id_dict = dict()

        self.save_matches = True  # 是否保存匹配结果
        self.save_dir = "outputs/matches"  # 保存目录
        self.save_threshold = 0.3  # 保存阈值，只保存相似度高于此值的结果
        self.max_matches = 5  # 每个目标保存的最大匹配数量
        
        # 帧计数用于控制UI更新频率
        self.ui_update_counter = 0
        self.ui_update_interval = 3  # 每处理3帧更新一次UI，降低更新频率避免递归重绘

    def save_match_result(self, proc_img, target_box, search_label, similarity, frame_count):
        """保存匹配结果"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
             
        # 裁剪目标图像
        crop_img = proc_img[int(target_box[1]):int(target_box[3]), 
                          int(target_box[0]):int(target_box[2])]
        
        # 生成文件名：ID_相似度_帧号.jpg
        filename = f"{search_label}_{similarity:.3f}_{frame_count}.jpg"
        save_path = os.path.join(self.save_dir, filename)
        
        # 在图像上添加相似度信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Similarity: {similarity:.3f}"
        cv2.putText(crop_img, text, (10, 30), font, 0.7, (0, 255, 0), 2)
        
        # 保存图像
        cv2.imwrite(save_path, crop_img)

    def reload_faiss(self, dims=None):
        if dims is None:
            dims = cfgs.DIMS
        base_feat_lists, base_idx_lists, modality_list = qt_sql.load_sql_feat_info(
            cfgs.DB_PATH, cfgs.DB_NAME,
        )
        self.reid_pipeline.reload_search_engine(
            base_feat_lists, base_idx_lists, dims, modalities=modality_list,
        )

    def _read_ir_frame(self, frame_count=0):
        if not self.proc_ir_source_url:
            return None
        if self.proc_ir_source_url.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff')):
            if not hasattr(self, '_ir_static_frame') or self._ir_static_frame is None:
                self._ir_static_frame = cv2.imread(self.proc_ir_source_url)
            return self._ir_static_frame
        if self._ir_capture is None:
            self._ir_capture = cv2.VideoCapture(self.proc_ir_source_url)
        ok, ir_frame = self._ir_capture.read()
        if not ok:
            self._ir_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, ir_frame = self._ir_capture.read()
        return ir_frame if ok else None

    def _reset_ir_reader(self):
        self._ir_static_frame = None
        if self._ir_capture is not None:
            self._ir_capture.release()
            self._ir_capture = None

    def proc_start_run_dir_type(self):
        proc_dir_index = 0
        start_img_list = os.listdir(self.proc_source_url)
        start_img_path_list = []
        for e_img in start_img_list:
            if e_img.lower().endswith(".jpg") or e_img.lower().endswith(".png") or e_img.lower().endswith(".jpeg") or \
                    e_img.lower().endswith(".mp4") or e_img.lower().endswith(".mkv") or e_img.lower().endswith(".avi") or e_img.lower().endswith(".flv"):
                start_img_path_list.append(os.path.join(self.proc_source_url, e_img))
        start_img_path_list = sorted(start_img_path_list)
        all_count = len(start_img_path_list)
        is_first_frame = True
        while True:
            if self.stop_dtc:
                break
            if self.continue_dtc:
                if len(start_img_path_list) < proc_dir_index + 1:
                    break
                target_file_path = start_img_path_list[proc_dir_index]
                if target_file_path.endswith(".jpg") or target_file_path.endswith(".png") or target_file_path.endswith(".jpeg"):
                    proc_img = cv2.imread(target_file_path)
                    proc_dir_index += 1
                    frame_count = 1
                    _inter_type = "image"
                elif target_file_path.endswith(".mp4") or target_file_path.endswith(".mkv") or target_file_path.endswith(".avi") or target_file_path.endswith(".flv"):
                    _inter_type = "video"
                    if is_first_frame:
                        self.reid_pipeline.reset_track()
                        track_history = defaultdict(list)
                        start_img = cv2.VideoCapture(target_file_path)
                        _fps = start_img.get(cv2.CAP_PROP_FPS)
                        flag_read, proc_img = start_img.read()
                        is_first_frame = False
                        self.had_track_id_dict = dict()
                        frame_count = 0
                    else:
                        flag_read, proc_img = start_img.read()
                        if not flag_read:
                            is_first_frame = True
                            proc_dir_index += 1
                            frame_count = 0
                            continue
                        else:
                            frame_count += 1
                else:
                    proc_dir_index += 1
                    continue
                _image = proc_img.copy()
                if _inter_type == "image":
                    boxes, track_ids, labels = self.reid_pipeline.detect(proc_img, class_idx_list=self.reid_pipeline._target_class_idx_list, format=_inter_type)
                    self.draw_box(_image, boxes, labels)
                else:
                    # sample frame
                    if frame_count % self.skip_frames != 0:
                        continue
                    boxes, track_ids, labels = self.reid_pipeline.detect(proc_img, class_idx_list=self.reid_pipeline._target_class_idx_list, format=_inter_type, is_track=self.is_track)
                    filter_bbox_list = []
                    filter_trackid_list = []
                    had_search_trackid_list = []
                    if self.is_track and track_ids is not None:
                        self.draw_track(_image,boxes,track_ids,labels,track_history)
                        for bbox, track_id in zip(boxes, track_ids):
                            if track_id not in self.had_track_id_dict:
                                self.had_track_id_dict[track_id] = [bbox, None]
                                filter_bbox_list.append(bbox)
                                filter_trackid_list.append(track_id)
                            else:
                                if self.had_track_id_dict[track_id][1] is not None:
                                    had_search_label = self.had_track_id_dict[track_id][1]
                                    had_search_trackid_list.append([bbox, had_search_label])
                        _image = self.draw_match(_image, [row[0] for row in had_search_trackid_list], [row[1] for row in had_search_trackid_list])
                        boxes = filter_bbox_list
                    else:
                        self.draw_box(_image, boxes, labels)
                ir_frame = self._read_ir_frame(frame_count)
                search_labels_list, search_dist_list, target_box_list, before_sort_list, before_dist_list = self.reid_pipeline.search(
                    proc_img, boxes, self.match_thresh, ir_img=ir_frame,
                )
                if _inter_type == 'video':
                    happend_time = round((frame_count + 1)/_fps, 3)
                    if self.is_track and track_ids is not None:
                        for _idx, _e_sort_box in enumerate(boxes):
                            if before_sort_list[_idx] != "unknown":
                                dist_val = before_dist_list[_idx]
                                if dist_val is not None and dist_val <= self.track_lock_dist_thresh:
                                    self.had_track_id_dict[filter_trackid_list[_idx]][1] = before_sort_list[_idx]
                        search_labels_list.extend([row[1] for row in had_search_trackid_list])
                        target_box_list.extend([row[0] for row in had_search_trackid_list])
                        search_dist_list.extend(["None"]*len(had_search_trackid_list))
                else:
                    happend_time = 0.0
                if len(target_box_list) > 0:
                    _image = self.draw_match(_image, target_box_list, search_labels_list)
                    for _idx, _e_dist in enumerate(search_dist_list):

                        similarity = 1 - search_dist_list[_idx]  # 转换距离为相似度
                         # 只保存相似度高于阈值的结果
                        if self.save_matches and similarity > self.save_threshold:
                            self.save_match_result(
                                proc_img,
                                target_box_list[_idx],
                                search_labels_list[_idx],
                                similarity,
                                frame_count
                            )
                        rows = [target_file_path, self.reid_pipeline._target_class, frame_count, happend_time, search_dist_list[_idx], "{}:{}".format("命中", search_labels_list[_idx]),"[{}]".format(','.join(map(str,map(int, target_box_list[_idx][:4]))))]
                        self.table_info_list.emit(rows)
                        bb = target_box_list[_idx]
                        crop_img =  proc_img[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2]),:]
                        self.show_target_img.emit(crop_img)
                        self.show_match_dist.emit(str(search_dist_list[_idx]))
                        self.show_match_id.emit(search_labels_list[_idx])
                        self.show_match_status.emit("命中")
                else:
                    if self.is_show_no_match_item:
                        rows = [target_file_path, self.reid_pipeline._target_class, frame_count, happend_time, "None", "未命中", "None"]
                        self.table_info_list.emit(rows)
                    crop_img = np.full((128,128,3),255,dtype=np.uint8)
                    self.show_target_img.emit(crop_img)
                    self.show_match_dist.emit("None")
                    self.show_match_id.emit("None")
                    self.show_match_status.emit("未命中")
                
                # 控制UI更新频率，避免递归重绘
                self.ui_update_counter += 1
                if self.ui_update_counter >= self.ui_update_interval:
                    self.ui_update_counter = 0
                    self.show_img.emit(_image)
                
                process_value = int((proc_dir_index)/all_count*1000)
                self.progress_bar.emit(process_value)
                if process_value == 1000:
                    break

    def draw_track(self,frame, boxes, track_ids, clss, track_history):
        for box, track_id, cls in zip(boxes, track_ids, clss):
            annotator = Annotator(frame, example=str(cfgs.YOLO_LABELS))
            annotator.box_label(box, "track_id:{}".format(track_id), color=colors(track_id, True))
            bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center   
            track = track_history[track_id]  # Tracking Lines plot
            track.append((float(bbox_center[0]), float(bbox_center[1])))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=colors(track_id, True), thickness=4)

    def draw_box(self, _image, boxes, labels):
        for _idx, (box, cls) in enumerate(zip(boxes, labels)):
            annotator = Annotator(_image,example=str(cfgs.YOLO_LABELS))
            annotator.box_label(box, self.reid_pipeline._target_class, color=colors(int(cls), True))

    def draw_match(self, _image, boxes, match_ids):
        for _idx, (box, cls) in enumerate(zip(boxes, match_ids)):
            _image = draw_chinese_box(
                _image,
                font=os.path.join(cfgs.MODELS_DIR, "SimHei.ttf"),
                box=box,
                label=cls,
                color=(0,255,0),
            )
        return _image
    
    def proc_start_run_media_type(self):
        _fps = 0
        all_count = 0
        if self.proc_source_type == 'image':
            start_img = cv2.imread(self.proc_source_url)
            all_count = 1
        elif self.proc_source_type == 'video':
            start_img = cv2.VideoCapture(self.proc_source_url)
            _fps = start_img.get(cv2.CAP_PROP_FPS)
            all_count = start_img.get(cv2.CAP_PROP_FRAME_COUNT)
            track_history = defaultdict(list)
        frame_count = 1
        self.had_track_id_dict = dict()
        while True:
            if self.stop_dtc:
                break
            _inter_type = "image"
            if self.continue_dtc:
                if self.proc_source_type == 'video':
                    _inter_type = "video"
                    flag_read, proc_img = start_img.read()
                    frame_count += 1
                    if not flag_read:
                        self.progress_bar.emit(1000)
                        break
                else:
                    proc_img = start_img
                _image = proc_img.copy()
                if _inter_type == "image":
                    boxes, track_ids, labels = self.reid_pipeline.detect(proc_img, class_idx_list=self.reid_pipeline._target_class_idx_list, format=_inter_type)
                    self.draw_box(_image, boxes, labels)
                else:
                    # sample frame
                    if frame_count % self.skip_frames != 0:
                        continue
                    boxes, track_ids, labels = self.reid_pipeline.detect(proc_img, class_idx_list=self.reid_pipeline._target_class_idx_list, format=_inter_type, is_track=self.is_track)
                    filter_bbox_list = []
                    filter_trackid_list = []
                    had_search_trackid_list = []
                    if self.is_track and track_ids is not None:
                        self.draw_track(_image,boxes,track_ids,labels,track_history)
                        for bbox, track_id in zip(boxes, track_ids):
                            if track_id not in self.had_track_id_dict:
                                self.had_track_id_dict[track_id] = [bbox, None]
                                filter_bbox_list.append(bbox)
                                filter_trackid_list.append(track_id)
                            else:
                                if self.had_track_id_dict[track_id][1] is not None:
                                    had_search_label = self.had_track_id_dict[track_id][1]
                                    had_search_trackid_list.append([bbox, had_search_label])
                        _image = self.draw_match(_image, [row[0] for row in had_search_trackid_list], [row[1] for row in had_search_trackid_list])
                        boxes = filter_bbox_list
                    else:
                        self.draw_box(_image, boxes, labels)
                ir_frame = self._read_ir_frame(frame_count)
                search_labels_list, search_dist_list, target_box_list, before_sort_list, before_dist_list = self.reid_pipeline.search(
                    proc_img, boxes, self.match_thresh, ir_img=ir_frame,
                )
                if _inter_type == 'video':
                    happend_time = round((frame_count + 1)/_fps, 3)
                    if self.is_track and track_ids is not None:
                        for _idx, _e_sort_box in enumerate(boxes):
                            if before_sort_list[_idx] != "unknown":
                                dist_val = before_dist_list[_idx]
                                if dist_val is not None and dist_val <= self.track_lock_dist_thresh:
                                    self.had_track_id_dict[filter_trackid_list[_idx]][1] = before_sort_list[_idx]
                        ## had_search_trackid
                        search_labels_list.extend([row[1] for row in had_search_trackid_list])
                        target_box_list.extend([row[0] for row in had_search_trackid_list])
                        search_dist_list.extend(["None"]*len(had_search_trackid_list))
                else:
                    happend_time = 0.0
                if len(target_box_list) > 0:
                    _image = self.draw_match(_image, target_box_list, search_labels_list)
                    for _idx, _e_dist in enumerate(search_dist_list):
                        rows = [self.proc_source_url, self.reid_pipeline._target_class, frame_count, happend_time, search_dist_list[_idx], "{}:{}".format("命中", search_labels_list[_idx]),"[{}]".format(','.join(map(str,map(int, target_box_list[_idx][:4]))))]
                        self.table_info_list.emit(rows)
                        bb = target_box_list[_idx]
                        crop_img =  proc_img[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2]),:]
                        self.show_target_img.emit(crop_img)
                        self.show_match_dist.emit(str(search_dist_list[_idx]))
                        self.show_match_id.emit(search_labels_list[_idx])
                        self.show_match_status.emit("命中")
                else:
                    if self.is_show_no_match_item:
                        rows = [self.proc_source_url, self.reid_pipeline._target_class, frame_count, happend_time, "None", "未命中", "None"]
                        self.table_info_list.emit(rows)
                    crop_img = np.full((128,128,3),255,dtype=np.uint8)
                    self.show_target_img.emit(crop_img)
                    self.show_match_dist.emit("None")
                    self.show_match_id.emit("None")
                    self.show_match_status.emit("未命中")
                
                # 控制UI更新频率，避免递归重绘
                self.ui_update_counter += 1
                if self.ui_update_counter >= self.ui_update_interval:
                    self.ui_update_counter = 0
                    self.show_img.emit(_image)
                
                process_value = int(frame_count/all_count*1000)
                self.progress_bar.emit(process_value)
                if process_value == 1000:
                    break
                

    def proc_start_run_func(self):
        try:
            if self.proc_source_type == 'image' or self.proc_source_type == 'video':
                self.proc_start_run_media_type()
            elif self.proc_source_type == 'dir':
                self.proc_start_run_dir_type()
        except Exception as e:
            print(traceback.print_exc())
            self.debug_msg.emit("%s"%e)
        finally:
            self.progress_bar.emit(1000)
        

class PageProcess:
    main_process_thread = Signal()
    is_save_video = False
    is_save_csv = False
    video_writer = None
    frame_images = {}  # 存储帧数对应的图片信息
    current_match_image = None  # Store current matched image
    save_matches = True  # Enable saving matches
    save_threshold = 0.7  # Only save matches above this similarity threshold

    def set_proc_page(self):
        base_feat_lists, base_idx_lists, modality_list = qt_sql.load_sql_feat_info(
            cfgs.DB_PATH, cfgs.DB_NAME,
        )
        self.proc_class = ProcessThread(
            base_feat_lists, base_idx_lists, dims=cfgs.DIMS,
            target_class="person", device_info="cpu", modalities=modality_list,
        )
        self.reid_pipeline = self.proc_class.reid_pipeline
        self._setup_proc_ir_widgets()
        self.process_file_button.clicked.connect(self.proc_open_file_func)
        self.process_ir_button.clicked.connect(self.proc_open_ir_file_func)
        self.process_dir_button.clicked.connect(self.proc_open_dir_func)
        self._process_info_model = QStandardItemModel(self)
        self._process_info_model.setHorizontalHeaderLabels(['来源', '类型', "帧数", "时间",'距离', '是否命中', "位置[x1,y1,x2,y2]"])
        self.process_table_show.setModel(self._process_info_model)
        
        # 添加表格点击事件
        self.process_table_show.clicked.connect(self.on_table_clicked)
        
        # 添加图片点击事件
        self.match_show_img.mousePressEvent = self.save_match_image
        
        self.proc_class.debug_msg.connect(lambda x: self.show_status(x))
        self.proc_class.show_img.connect(lambda x: self.show_image(x, self.det_pano_img))
        self.proc_class.show_target_img.connect(lambda x: self.show_image(x, self.match_show_img))
        self.proc_class.show_match_id.connect(lambda x: self.match_show_id.setText(x))
        self.proc_class.show_match_status.connect(lambda x:self.match_show_status.setText(x))
        self.proc_class.show_match_dist.connect(lambda x: self.match_show_dist.setText(x))
        self.proc_class.progress_bar.connect(lambda x: self.show_progress_bar(x))
        self.proc_class.table_info_list.connect(lambda x: self.show_table_proc_stage(x))

        self.process_thread = QThread()
        self.main_process_thread.connect(self.proc_class.proc_start_run_func)
        self.proc_class.moveToThread(self.process_thread)

        self.proc_run_button.clicked.connect(self.proc_run_or_continue)
        self.proc_stop_button.clicked.connect(self.proc_stop)

        ## save checkout choice
        self.istrack_checkbox.toggled.connect(self.istrack_checkbox_setting)
        self.save_media_checkbox.toggled.connect(self.save_media_checkbox_setting)
        self.save_csv_checkbox.toggled.connect(self.save_csv_checkbox_setting)
    

    def register_video_writer(self):
        date_now = datetime.datetime.now()
        year = date_now.year
        month = date_now.month
        day = date_now.day
        hour = date_now.hour
        second = date_now.second
        result_dir = f"{year}_{month}_{hour}_{day}_{second}.mp4"
        file_path = os.path.join(OUTPUTS_DIR, "video", result_dir)
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        video_writer = cv2.VideoWriter(file_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                max(int(25//self.proc_class.skip_frames),25),
                                (1280,720))
        return video_writer
    
    def istrack_checkbox_setting(self):
        self.proc_class.reid_pipeline.reset_track()
        if self.istrack_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: {} state had been changed to False.'.format('istrack_checkbox'))
            self.proc_class.is_track = False
        elif self.istrack_checkbox.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: {} state had been changed to True.'.format('istrack_checkbox'))
            self.proc_class.is_track = True
            self.proc_class.had_track_id_dict = dict()

    def save_media_checkbox_setting(self):
        if self.save_media_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: {} state had been changed to False.'.format('save_media_checkbox'))
            self.is_save_video = False
        elif self.save_media_checkbox.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: {} state had been changed to True.'.format('save_media_checkbox'))
            self.is_save_video = True

    def save_csv_checkbox_setting(self):
        if self.save_csv_checkbox.checkState() == Qt.CheckState.Unchecked:
            self.show_status('NOTE: {} state had been changed to False.'.format('save_csv_checkbox'))
            self.is_save_csv = False
        elif self.save_csv_checkbox.checkState() == Qt.CheckState.Checked:
            self.show_status('NOTE: {} state had been changed to True.'.format('save_csv_checkbox'))
            self.is_save_csv = True

    def proc_run_or_continue(self):
        if self.proc_class.proc_source_url == '':
            self.show_status('Please select the media video/image or dir source before starting detection...')
            self.proc_run_button.setChecked(False)
        else:
            self.proc_class.stop_dtc = False
            if self.proc_run_button.isChecked():
                self.proc_run_button.setChecked(True)
                self.show_status("Reid process......")
                self.proc_class.continue_dtc = True
                self.process_thread.start()
                self.main_process_thread.emit()
            else:
                self.proc_class.continue_dtc = False
                self.show_status("Pause...")
                self.proc_run_button.setChecked(False)
    
    def _setup_proc_ir_widgets(self):
        self.process_ir_button = QPushButton("打开红外源", self.proc)
        self.process_ir_button.setObjectName("process_ir_button")
        self.process_ir_button.setMinimumHeight(28)
        self.process_ir_edit = QLineEdit(self.proc)
        self.process_ir_edit.setObjectName("process_ir_edit")
        self.process_ir_edit.setReadOnly(True)
        self.process_ir_edit.setPlaceholderText("  可选：红外视频/图像（与可见光帧对齐）")
        if hasattr(self, "horizontalLayout_proc_top"):
            self.horizontalLayout_proc_top.addWidget(self.process_ir_button)
            self.horizontalLayout_proc_top.addWidget(self.process_ir_edit)
        elif hasattr(self, "process_file_edit") and self.process_file_edit.parentWidget():
            layout = self.process_file_edit.parentWidget().layout()
            if layout is not None:
                layout.addWidget(self.process_ir_button)
                layout.addWidget(self.process_ir_edit)

    def proc_open_ir_file_func(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        config_file = os.path.join(CONFIG_DIR, 'proc_ir_fold.json')
        if os.path.exists(config_file):
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            open_fold = config.get('open_fold', os.getcwd())
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()
        else:
            config = {}
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(
            self, 'Infrared video/image', open_fold,
            "Media(*.mp4 *.mkv *.avi *.flv *.jpg *.png *.jpeg *.tif *.tiff)",
        )
        if name:
            self.proc_class.proc_ir_source_url = name
            self.proc_class._reset_ir_reader()
            config['open_fold'] = os.path.dirname(name)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            self.process_ir_edit.setText(name)

    def proc_stop(self):
        if self.process_thread.isRunning():
            self.process_thread.quit()
        self.proc_class.stop_dtc = True
        self.proc_class._reset_ir_reader()
        self.proc_class.reid_pipeline.reset_track()
        self.proc_run_button.setChecked(False)
        self.progress_bar.setValue(0)
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
    
    def show_status(self, msg):
        print(msg)
    
    def show_progress_bar(self, x):
        if x == 1000:
            self.proc_stop()
            if self.is_save_csv:
                self.save_as_csv()
        else:
            self.progress_bar.setValue(x)

    def set_sample_frame(self, num):
        self.proc_class.skip_frames = num

    def show_table_proc_stage(self, rows):
        items = [QStandardItem(str(item)) for item in rows]
        self._process_info_model.appendRow(items)
        self.process_table_show.scrollToBottom()

    def set_dist_thresh(self, dist):
        self.dist_rank_thresh = dist

    def proc_open_dir_func(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        config_file = os.path.join(CONFIG_DIR, 'proc_fold_dir.json')
        if os.path.exists(config_file):  
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            open_fold = config['open_fold']     
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()
        else:
            config = dict()
            open_fold = config['open_fold'] = os.getcwd()
        name = QFileDialog.getExistingDirectory(None, '选择文件夹', open_fold, QFileDialog.ShowDirsOnly)
        if name:
            self.proc_class.proc_source_url = name
            config['open_fold'] = name
            self.proc_class.proc_source_type = 'dir'
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.process_dir_edit.setText(name)
            self.process_file_edit.clear()
            self.proc_stop()
    
    def proc_open_file_func(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        config_file = os.path.join(CONFIG_DIR, 'proc_fold.json')
        if os.path.exists(config_file):  
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            open_fold = config['open_fold']     
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()
        else:
            config = dict()
            open_fold = config['open_fold'] = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.jpg *.png *.jpeg)")
        if name:
            self.proc_class.proc_source_url = name
            suffix_name = os.path.basename(name)
            config['open_fold'] = os.path.dirname(name)
            if suffix_name.endswith(".jpg") or suffix_name.endswith(".png") or suffix_name.endswith(".jpeg"):
                self.proc_class.proc_source_type = 'image'
            if suffix_name.endswith(".mp4") or suffix_name.endswith(".mkv") or suffix_name.endswith(".avi") or suffix_name.endswith(".flv"):
                self.proc_class.proc_source_type = 'video'
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.process_file_edit.setText(name)
            self.process_dir_edit.clear()
            self.proc_stop()
    
    def show_image(self, img_src, label):
        try:
            if img_src is None or img_src.size == 0:
                return
            
            ih, iw, _ = img_src.shape
            if ih == 0 or iw == 0:
                return
                
            w = label.geometry().width()
            h = label.geometry().height()
            
            if w == 0 or h == 0:
                return
                
            # keep the original data ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            
            # 直接设置图片（信号已经在主线程处理）
            if label.isVisible():
                label.setPixmap(QPixmap.fromImage(img))

            # 如果是匹配图片，保存图片信息
            if label.objectName() == "match_show_img":
                try:
                    current_row = self._process_info_model.rowCount() - 1
                    if current_row >= 0:
                        frame_num = self._process_info_model.item(current_row, 2).text()
                        match_id = self.match_show_id.text()
                        match_status = self.match_show_status.text()
                        match_dist = self.match_show_dist.text()
                        
                        if "命中" in match_status:
                            self.frame_images[frame_num] = {
                                'crop_img': img_src.copy(),
                                'id': match_id,
                                'status': match_status,
                                'dist': match_dist
                            }
                            self.current_match_image = img_src.copy()
                except Exception as inner_e:
                    print("Error saving match info:", str(inner_e))

        except Exception as e:
            print("Error in show_image:", repr(e))
            print(traceback.print_exc())
        finally:
            try:
                if label.objectName() == "det_pano_img" and self.is_save_video:
                    if self.video_writer is None:
                        self.video_writer = self.register_video_writer()
                    img_src_ = cv2.resize(img_src, (1280, 720))
                    self.video_writer.write(img_src_)
            except Exception as e:
                print("Error saving video frame:", str(e))

    def save_as_csv(self):
        date_now = datetime.datetime.now()
        year = date_now.year
        month = date_now.month
        day = date_now.day
        hour = date_now.hour
        second = date_now.second
        result_dir = f"{year}_{month}_{hour}_{day}_{second}.csv"
        file_path = os.path.join(OUTPUTS_DIR, "csv", result_dir)
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in range(self._process_info_model.rowCount()):
                row_data = []
                for column in range(self._process_info_model.columnCount()):
                    item = self._process_info_model.item(row, column)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append("")  # 如果项目为空，则写入空字符串
                writer.writerow(row_data)

    def on_table_clicked(self, index):
        # 获取点击的行
        row = index.row()
        # 获取帧数列的值（第3列）
        frame_num = self._process_info_model.item(row, 2).text()
        match_info = self._process_info_model.item(row, 5).text()  # 获取匹配信息
        
        # 如果该帧的图片信息存在且是命中的结果
        if frame_num in self.frame_images and "命中" in match_info:
            frame_info = self.frame_images[frame_num]
            # 显示图片
            self.show_image(frame_info['crop_img'], self.match_show_img)
            # 更新匹配信息
            self.match_show_id.setText(frame_info['id'])
            self.match_show_status.setText(frame_info['status'])
            self.match_show_dist.setText(frame_info['dist'])
            
            # 保存当前图片用于后续保存操作
            self.current_match_image = frame_info['crop_img'].copy()

    def save_match_image(self, event):
        if self.current_match_image is not None:
            try:
                # Get current match info
                match_id = self.match_show_id.text()
                match_dist = self.match_show_dist.text()
                
                # Generate default filename with timestamp
                date_now = datetime.datetime.now()
                timestamp = f"{date_now.year}_{date_now.month}_{date_now.day}_{date_now.hour}_{date_now.minute}_{date_now.second}"
                default_filename = f"match_{match_id}_{match_dist}_{timestamp}.jpg"
                
                # Get default save directory from config or create new one
                os.makedirs(CONFIG_DIR, exist_ok=True)
                config_file = os.path.join(CONFIG_DIR, 'save_matches.json')
                save_fold = os.getcwd()
                
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            if 'save_fold' in config and os.path.exists(config['save_fold']):
                                save_fold = config['save_fold']
                    except:
                        pass
                
                # Open file dialog for user to choose save location
                save_path, _ = QFileDialog.getSaveFileName(
                    None,  # 使用None作为父窗口，确保对话框总是显示在最前面
                    '保存匹配图片',
                    os.path.join(save_fold, default_filename),
                    'Images (*.jpg *.png)'
                )
                
                if save_path:
                    # Update config with new save directory
                    config = {'save_fold': os.path.dirname(save_path)}
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)
                    
                    # Save the image
                    cv2.imwrite(save_path, self.current_match_image)
                    
                    # Show success message
                    QMessageBox.information(
                        None,
                        "保存成功",
                        f"图片已保存到:\n{save_path}",
                        QMessageBox.StandardButton.Ok
                    )
                    self.show_status(f"已保存匹配图片到 {save_path}")
            except Exception as e:
                QMessageBox.warning(
                    None,
                    "保存失败",
                    f"保存图片时发生错误:\n{str(e)}",
                    QMessageBox.StandardButton.Ok
                )
                self.show_status(f"保存图片失败: {str(e)}")
