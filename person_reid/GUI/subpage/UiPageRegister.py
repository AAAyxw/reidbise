import os
import sys
import json
import cv2
from GUI.libs import img_show_and_encoder
import Algorithm.libs.config.model_cfgs as cfgs
from GUI.libs import qt_sql
from ultralytics.utils.plotting import Annotator, colors
from PySide6.QtCore import Qt, QModelIndex
from PySide6.QtWidgets import (
                                QFileDialog,
                                QDataWidgetMapper,
                                QTableView,
                                QMessageBox,
                                QHBoxLayout, 
                                QPushButton, 
                                QVBoxLayout, 
                                QWidget, 
                            )
from PySide6.QtGui import QKeySequence
from PySide6.QtSql import QSqlRelation, QSqlRelationalTableModel, QSqlTableModel

class PageRegister:
    
    def set_reg_page(self):
        self.reg_source_img_url = None
        self.reg_crop_img  = None
        self.reg_id = None
        self.reg_file_open.clicked.connect(self.reg_open_file_func)
        self.reg_file_process.clicked.connect(self.reg_process_file_func)
        self.reg_processd_box.currentTextChanged.connect(self.reg_choose_target_func)
        self.reg_to_makesure_reg.clicked.connect(self.register_to_sql_func)
    
    def register_to_sql_func(self):
        if self.reg_crop_img is not None and self.reg_id_lineEdit.text() != "":
            name = self.reg_id_lineEdit.text()
            category = self.reid_pipeline._target_class
            reply = QMessageBox.question(self, '确认', '请再次确认将类型:{},ID名为【{}】注册到数据库中吗?'.format(category, name),
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                box = ','.join(map(str, self.reg_box))
                img_feature = self.reid_pipeline.extract(self.reg_crop_img)
                feature_string = ','.join(map(str, img_feature))
                img_base64_encoder = img_show_and_encoder.base64_encoder(cv2.resize(self.reg_crop_img, (128, 256)))
                # print(feature_string)
                # print(img_base64_encoder)
                # name, category, box, path, image
                qt_sql._add_register(cfgs.DB_PATH, cfgs.DB_NAME, name, category, box, feature_string, img_base64_encoder)
                self.sql_model.submitAll()
                self.mag_total_reg_num.display(self.all_db_showTb.model().rowCount())
                self.proc_class.reload_faiss()


    def reg_choose_target_func(self):
        if self.reg_source_img_url is not None:
            _reg_choose_label = self.reg_processd_box.currentText()
            bb = self.reg_box_dict[_reg_choose_label]
            crop_img =  self._reg_img[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2]),:]
            img_show_and_encoder.show_image(crop_img, self.reg_show_crop_img)
            self.reg_crop_img = crop_img.copy()
            self.reg_box = [int(bb[0]),int(bb[1]),int(bb[2]),int(bb[3])]

    def reg_process_file_func(self):
        if self.reg_source_img_url is not None:
            _image = cv2.imread(self.reg_source_img_url)
            self._reg_img = _image.copy()
            print (self.reid_pipeline._target_class_idx_list)
            bboxs, track_ids, labels = self.reid_pipeline.detect(_image, self.reid_pipeline._target_class_idx_list, format='image')
            self.reg_label_list = []
            self.reg_box_dict = {}
            for _idx, (box, cls) in enumerate(zip(bboxs, labels)):
                annotator = Annotator(_image, example=str(cfgs.YOLO_LABELS))
                annotator.box_label(box, self.reid_pipeline._target_class, color=colors(int(0), True))
                _reg_label = "{}_{}".format(self.reid_pipeline._target_class, _idx)
                self.reg_label_list.append(_reg_label)
                self.reg_box_dict[_reg_label] = box
            img_show_and_encoder.show_image(_image, self.reg_show_main_img)
            self.reg_processd_box.clear()
            self.reg_processd_box.addItems(self.reg_label_list)

    def reg_open_file_func(self):
        config_file = 'config/reg_fold.json'
        if os.path.exists(config_file):  
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            open_fold = config['open_fold']     
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()
        else:
            config = dict()
            open_fold = config['open_fold'] = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'image', open_fold, "Pic File(*.JPG *.PNG *.JPEG *.jpg *.png *.jpeg)")
        if name:
            self.reg_source_img_url = name
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)  
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # TODO 
            img_show_and_encoder.show_image(cv2.imread(name), self.reg_show_main_img)
            self.reg_file_lineEdit.setText(name)

