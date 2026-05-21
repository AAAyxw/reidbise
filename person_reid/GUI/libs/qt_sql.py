from PySide6.QtSql import QSqlDatabase, QSqlQuery
from Algorithm.libs.logger.log import get_logger

log_info = get_logger(__name__)

_REID_COLUMNS = (
    "id integer primary key, "
    "name text, category varchar, box text, feat text, image varchar, "
    "modality varchar, image_ir varchar"
)


def init_db(db_path, db_name):
    db = QSqlDatabase.addDatabase("QSQLITE")
    db.setDatabaseName(db_path)
    if not db.open():
        print("Error: Unable to open database")
        log_info.error("{}_{} Unable to open database.".format(db_path, db_name))
        return False
    query = QSqlQuery()
    query.exec_("CREATE TABLE IF NOT EXISTS {} ({})".format(db_name, _REID_COLUMNS))
    _migrate_reid_table(query, db_name)
    return True


def _table_has_column(query, db_name, column):
    query.exec_("PRAGMA table_info({})".format(db_name))
    while query.next():
        if query.value(1) == column:
            return True
    return False


def _migrate_reid_table(query, db_name):
    for col_name, col_type in (("modality", "varchar"), ("image_ir", "varchar")):
        if not _table_has_column(query, db_name, col_name):
            query.exec_("ALTER TABLE {} ADD COLUMN {} {}".format(db_name, col_name, col_type))


def check(func, *args):
    if not func(*args):
        raise ValueError(func.__self__.lastError())


def load_sql_feat_info(db_path, db_name):
    db = QSqlDatabase.addDatabase("QSQLITE")
    db.setDatabaseName(db_path)
    if not db.open():
        print("Error: Unable to open database")
        log_info.error("{}_{} Unable to open database.".format(db_path, db_name))
        return [], [], []
    query = QSqlQuery()
    query.exec("SELECT name, feat, modality FROM {}".format(db_name))
    feat_list = []
    label_list = []
    modality_list = []
    if not query.isActive():
        log_info.error("{}_{} query feature error.".format(db_path, db_name))
        print("Error:", query.lastError().text())
    else:
        while query.next():
            label_list.append(query.value(0))
            feat_list.append(list(map(float, query.value(1).split(','))))
            modality_list.append(query.value(2) or "visible")
    return feat_list, label_list, modality_list


def _add_register(db_path, db_name, name, category, box, feat, image, modality="visible", image_ir=""):
    db = QSqlDatabase.addDatabase("QSQLITE")
    db.setDatabaseName(db_path)
    if not db.open():
        print("Error: Unable to open database")
        log_info.error("{}_{} Unable to open database.".format(db_path, db_name))
        return False
    q = QSqlQuery()
    insert_sql = (
        "insert into {}(name, category, box, feat, image, modality, image_ir) "
        "values(?, ?, ?, ?, ?, ?, ?)"
    ).format(db_name)
    check(q.prepare, insert_sql)
    q.addBindValue(name)
    q.addBindValue(category)
    q.addBindValue(box)
    q.addBindValue(feat)
    q.addBindValue(image)
    q.addBindValue(modality)
    q.addBindValue(image_ir or "")
    q.exec()
    if q.lastError().isValid():
        print("Error:", q.lastError().text())
    else:
        print("Query executed successfully")
