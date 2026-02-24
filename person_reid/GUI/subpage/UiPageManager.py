import os
import sys
from PySide6.QtCore import Qt, QModelIndex
from PySide6.QtWidgets import (
                                QApplication,
                                QAbstractItemView,
                                QDataWidgetMapper,
                                QTableView,
                                QMessageBox,
                                QHeaderView,
                                QHBoxLayout, 
                                QPushButton, 
                                QVBoxLayout, 
                                QWidget, 
                            )
from PySide6.QtGui import QKeySequence
from PySide6.QtSql import QSqlRelation, QSqlRelationalTableModel, QSqlTableModel
from PySide6.QtCore import Qt, Slot
from GUI.libs import img_show_and_encoder
from GUI.libs import qt_sql
import Algorithm.libs.config.model_cfgs as cfgs


class PageManager:
    def set_mag_page(self):
        qt_sql.init_db(cfgs.DB_PATH, cfgs.DB_NAME)
        self.sql_model = QSqlRelationalTableModel(self.all_db_showTb)
        self.sql_model.setEditStrategy(QSqlTableModel.OnManualSubmit)
        self.sql_model.setTable(cfgs.DB_NAME)
        self.sql_model.setHeaderData(self.sql_model.fieldIndex("id"), Qt.Horizontal, self.tr("ID"))
        self.sql_model.setHeaderData(self.sql_model.fieldIndex("name"), Qt.Horizontal, self.tr("Name"))
        self.sql_model.setHeaderData(self.sql_model.fieldIndex("category"), Qt.Horizontal, self.tr("Category"))
        self.sql_model.setHeaderData(self.sql_model.fieldIndex("box"), Qt.Horizontal, self.tr("Boxs [x1,y1,x2,y2]"))
        self.sql_model.setHeaderData(self.sql_model.fieldIndex("feat"), Qt.Horizontal, self.tr("Feat"))
        self.sql_model.setHeaderData(self.sql_model.fieldIndex("image"), Qt.Horizontal, self.tr("Image"))
        
        if not self.sql_model.select():
            print(self.sql_model.lastError())
        self.all_db_showTb.setModel(self.sql_model)
        self.all_db_showTb.setColumnHidden(self.sql_model.fieldIndex("image"), True)
        self.all_db_showTb.setColumnHidden(self.sql_model.fieldIndex("feat"), True)
        self.all_db_showTb.setSelectionMode(QAbstractItemView.SingleSelection)
        self.all_db_showTb.setSelectionBehavior(QTableView.SelectRows)
        self.all_db_showTb.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.all_db_showTb.resizeColumnsToContents()
        self.all_db_showTb.clicked.connect(self.show_img_details)
        self.all_db_showTb.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mag_delete_button.clicked.connect(self.deleteButtonClicked)
        
        self.sql_mapper = QDataWidgetMapper(self)
        self.sql_mapper.setModel(self.sql_model)
        self.sql_mapper.addMapping(self.mag_id_nameEdit, self.sql_model.fieldIndex("name"))
        self.sql_mapper.addMapping(self.mag_del_edit, self.sql_model.fieldIndex("name"))

        selection_model = self.all_db_showTb.selectionModel()
        selection_model.currentRowChanged.connect(self.sql_mapper.setCurrentModelIndex)

        #self.all_db_showTb.setCurrentIndex(self.sql_model.index(0, 0))
        self.mag_nameedit_button.clicked.connect(self.name_edit_changed)
        self.mag_total_reg_num.display(self.all_db_showTb.model().rowCount())

    def show_img_details(self, index: QModelIndex):
        row = index.row()
        img64_data = self.sql_model.record(row).value("image")
        img_show_and_encoder.show_image(img_show_and_encoder.base64_decoder(img64_data), self.mag_grly_img_show)

    def name_edit_changed(self):
        current_index = self.sql_mapper.currentIndex()
        self.sql_model.setData(self.sql_model.index(current_index, self.sql_model.fieldIndex("name")), self.mag_id_nameEdit.text())
        self.sql_model.submitAll()
        self.proc_class.reload_faiss()

    def deleteButtonClicked(self):
        selected_row = self.all_db_showTb.currentIndex().row()
        if selected_row >= 0:
            # double check QMessageBox
            reply = QMessageBox.question(self, 'Confirmation', 'Are you sure you want to delete this row?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                if not self.sql_model.removeRow(selected_row):
                    print("Error removing row:", self.sql_model.lastError())
                else:
                    if not self.sql_model.submitAll():
                        print("Error submitting changes:", self.sql_model.lastError())
                    else:
                        self.mag_total_reg_num.display(self.all_db_showTb.model().rowCount())
                        self.proc_class.reload_faiss()
                        print("Row deleted successfully")
        else:
            print("No row selected")