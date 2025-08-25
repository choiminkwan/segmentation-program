# segmentprog.py
# 폴더 기반 로딩(원본/마스크), 인덱스 페어링(O↔M), 방향키로 세트 이동, Save All(일괄 저장)
# Pen 미리보기 매끈화, 한/영 단축키, 줌/투명도(%), 밝기/명암 조절
# Opacity / Brightness / Contrast 헤더 1:1:1 배치, 슬라이더-수치 간격 축소
# 빈 캔버스에서 중앙의 흰 점 방지(라벨 숨김)

import sys, os, glob
import numpy as np
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QSpinBox,
    QFrame, QSlider, QScrollArea, QListWidget, QListWidgetItem, QTabWidget,
    QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox
)
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import Qt

# ---------- 유틸 ----------
def np_to_qimage_rgb(img_rgb: np.ndarray) -> QImage:
    h, w, _ = img_rgb.shape
    img_rgb = np.ascontiguousarray(img_rgb)
    q = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return q.copy()

def np_to_qimage_rgba(img_rgba: np.ndarray) -> QImage:
    h, w, _ = img_rgba.shape
    img_rgba = np.ascontiguousarray(img_rgba)
    q = QImage(img_rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
    return q.copy()

def apply_brightness_contrast(rgb: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    if rgb is None:
        return None
    out = rgb.astype(np.int16)

    # contrast: -100..100
    if contrast != 0:
        c = float(contrast)
        factor = (259 * (c + 255)) / (255 * (259 - c))
        out = factor * (out - 128) + 128

    # brightness: -100..100
    if brightness != 0:
        out = out + int(brightness)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# ---------- 1:1 라벨 ----------
class ImageLabel(QLabel):
    sig_click_drag = QtCore.pyqtSignal(int, int, bool)  # x,y,is_drag
    sig_release    = QtCore.pyqtSignal()                # 펜 스트로크 종료
    sig_dbl        = QtCore.pyqtSignal()                # 더블클릭(폴리곤 완료)

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setMouseTracking(True)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.sig_click_drag.emit(e.x(), e.y(), False)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.sig_click_drag.emit(e.x(), e.y(), True)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.sig_release.emit()
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.sig_dbl.emit()
        super().mouseDoubleClickEvent(e)

# ---------- 페어 구조 ----------
class PairItem:
    def __init__(self, orig_path: str | None, mask_path: str | None):
        self.orig_path = orig_path
        self.mask_path = mask_path
        self.mask_arr  = None
        self.modified  = False

# ---------- 메인 ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dental Mask Editor — Folders + Arrow-Key Sets + Batch Save")
        self.resize(1320, 860)

        # 상태
        self.img_rgb = None
        self.gray = None
        self.mask = None
        self.overlay_alpha = 184   # 0..255
        self.opacity_step = 16
        self.mode = "pen"          # pen / polygon
        self.draw_op = "add"       # add / sub
        self.brush_size = 14       # 1~30

        # 밝기/명암
        self.brightness = 0        # -100..100
        self.contrast   = 0        # -100..100
        self.bc_step    = 5

        # 폴리곤 상태
        self.poly_points = []
        self.poly_active = False
        self.poly_click_block = False

        # Undo
        self.undo_stack = []
        self.max_undo = 30

        # 폴더/페어
        self.dir_originals = None
        self.dir_masks     = None
        self.pairs: list[PairItem] = []
        self.current_pair_idx = None

        # Pen 프리뷰(매끈화)
        self.pen_base_mask = None
        self.pen_stroke_pts = []
        self.pen_start_xy = None
        self.pen_shift_snap = False

        # 줌
        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 6.0
        self.zoom_step = 1.25

        # ===== Root =====
        root = QWidget(); self.setCentralWidget(root)
        root_layout = QVBoxLayout(root); root_layout.setContentsMargins(8,8,8,8); root_layout.setSpacing(8)

        # ===== 헤더 =====
        header = QFrame(); header.setObjectName("Header")
        hl = QHBoxLayout(header); hl.setContentsMargins(12,8,12,8); hl.setSpacing(12)

        btn_orig_dir = QPushButton("Open Originals Folder"); btn_orig_dir.clicked.connect(self.on_choose_orig_dir)
        btn_mask_dir = QPushButton("Open Masks Folder");     btn_mask_dir.clicked.connect(self.on_choose_mask_dir)
        hl.addWidget(btn_orig_dir); hl.addWidget(btn_mask_dir)
        hl.addStretch(1)

        # 공통으로 쓸 사이즈 설정
        def make_slider():
            s = QSlider(Qt.Horizontal)
            s.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            s.setMinimumWidth(220)
            return s
        def make_value_label():
            lab = QLabel("0")
            lab.setFixedWidth(44)
            lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return lab

        # Opacity (stretch=1)
        opbox = QFrame(); opl = QHBoxLayout(opbox); opl.setContentsMargins(0,0,0,0); opl.setSpacing(4)
        opl.addWidget(QLabel("Opacity"))
        self.opacity_slider = make_slider()
        self.opacity_slider.setRange(0,255); self.opacity_slider.setValue(self.overlay_alpha)
        self.opacity_slider.valueChanged.connect(self.set_overlay_alpha)
        self.lbl_opacity = make_value_label(); self.lbl_opacity.setText(self.opacity_text())
        opl.addWidget(self.opacity_slider, 1); opl.addWidget(self.lbl_opacity)
        hl.addWidget(opbox, 1)

        # Brightness (stretch=1)
        bri_box = QFrame(); brl = QHBoxLayout(bri_box); brl.setContentsMargins(0,0,0,0); brl.setSpacing(4)
        brl.addWidget(QLabel("Brightness"))
        self.bri_slider = make_slider()
        self.bri_slider.setRange(-100, 100); self.bri_slider.setValue(self.brightness)
        self.bri_slider.valueChanged.connect(self.set_brightness)
        self.lbl_bri = make_value_label(); self.lbl_bri.setText(str(self.brightness))
        brl.addWidget(self.bri_slider, 1); brl.addWidget(self.lbl_bri)
        hl.addWidget(bri_box, 1)

        # Contrast (stretch=1)
        con_box = QFrame(); col = QHBoxLayout(con_box); col.setContentsMargins(0,0,0,0); col.setSpacing(4)
        col.addWidget(QLabel("Contrast"))
        self.con_slider = make_slider()
        self.con_slider.setRange(-100, 100); self.con_slider.setValue(self.contrast)
        self.con_slider.valueChanged.connect(self.set_contrast)
        self.lbl_con = make_value_label(); self.lbl_con.setText(str(self.contrast))
        col.addWidget(self.con_slider, 1); col.addWidget(self.lbl_con)
        hl.addWidget(con_box, 1)

        btn_save_all = QPushButton("Save All"); btn_save_all.setObjectName("Primary"); btn_save_all.clicked.connect(self.on_save_all)
        hl.addWidget(btn_save_all)

        root_layout.addWidget(header)

        # ===== 본문 =====
        content = QFrame(); content.setObjectName("Content")
        cl = QHBoxLayout(content); cl.setContentsMargins(12,12,12,12); cl.setSpacing(12)

        # --- 좌측: Tabs + Pairs 리스트 ---
        left_panel = QFrame(); left_panel.setObjectName("Panel")
        ll = QVBoxLayout(left_panel); ll.setContentsMargins(12,12,12,12); ll.setSpacing(12)

        ll.addWidget(QLabel("Tools", objectName="SectionTitle"))

        self.tools_tabs = QTabWidget(); self.tools_tabs.setTabPosition(QTabWidget.North)
        self.tools_tabs.currentChanged.connect(self.on_tools_tab_changed)

        # Pen 탭
        pen_tab = QWidget(); pen_l = QVBoxLayout(pen_tab); pen_l.setContentsMargins(6,6,6,6); pen_l.setSpacing(8)
        self.btn_fill_toggle_pen = QPushButton("Fill / Erase"); self.btn_fill_toggle_pen.setCheckable(True)
        self.btn_fill_toggle_pen.toggled.connect(lambda ch: self.set_fill_mode(ch))
        pen_l.addWidget(self.btn_fill_toggle_pen)
        row = QHBoxLayout(); row.setSpacing(8); row.addWidget(QLabel("Brush"))
        self.brush_slider = QSlider(Qt.Horizontal); self.brush_slider.setRange(1,30); self.brush_slider.setValue(self.brush_size)
        self.brush_spin = QSpinBox(); self.brush_spin.setRange(1,30); self.brush_spin.setValue(self.brush_size); self.brush_spin.setFixedWidth(64)
        self.brush_slider.valueChanged.connect(self.set_brush_size); self.brush_spin.valueChanged.connect(self.set_brush_size)
        row.addWidget(self.brush_slider); row.addWidget(self.brush_spin); pen_l.addLayout(row)
        btn_undo_pen = QPushButton("Undo"); btn_undo_pen.clicked.connect(self.on_ctrl_z); pen_l.addWidget(btn_undo_pen)
        pen_l.addStretch(1)
        self.tools_tabs.addTab(pen_tab, "Pen")

        # Polygon 탭
        poly_tab = QWidget(); poly_l = QVBoxLayout(poly_tab); poly_l.setContentsMargins(6,6,6,6); poly_l.setSpacing(8)
        self.btn_fill_toggle_poly = QPushButton("Fill / Erase"); self.btn_fill_toggle_poly.setCheckable(True)
        self.btn_fill_toggle_poly.toggled.connect(lambda ch: self.set_fill_mode(ch))
        poly_l.addWidget(self.btn_fill_toggle_poly)
        r2 = QHBoxLayout(); r2.setSpacing(8)
        self.btn_poly_done   = QPushButton("Complete");  self.btn_poly_done.clicked.connect(lambda: self.finish_polygon(True, False))
        self.btn_poly_undo   = QPushButton("Undo");      self.btn_poly_undo.clicked.connect(self.on_ctrl_z)
        self.btn_poly_cancel = QPushButton("Cancel");    self.btn_poly_cancel.clicked.connect(lambda: self.finish_polygon(False, False))
        r2.addWidget(self.btn_poly_done); r2.addWidget(self.btn_poly_undo); r2.addWidget(self.btn_poly_cancel)
        poly_l.addLayout(r2); poly_l.addStretch(1)
        self.tools_tabs.addTab(poly_tab, "Polygon")

        ll.addWidget(self.tools_tabs)
        ll.addWidget(QLabel("Dataset (Original ↔ Mask)", objectName="SectionTitle"))
        self.list_pairs = QListWidget()
        self.list_pairs.itemDoubleClicked.connect(self.on_open_selected_pair)
        ll.addWidget(self.list_pairs, 1)

        # --- 우측: Canvas ---
        right_panel = QFrame(); right_panel.setObjectName("Panel")
        rl = QVBoxLayout(right_panel); rl.setContentsMargins(12,12,12,12); rl.setSpacing(8)

        title_bar = QFrame(); tbar = QHBoxLayout(title_bar); tbar.setContentsMargins(0,0,0,0); tbar.setSpacing(8)
        right_title = QLabel("Canvas (Original + Mask Overlay)", objectName="SectionTitle")
        tbar.addWidget(right_title); tbar.addStretch(1)
        zoom_box = QFrame(); zl = QHBoxLayout(zoom_box); zl.setContentsMargins(0,0,0,0); zl.setSpacing(6)
        self.btn_zoom_out = QPushButton("–"); self.btn_zoom_out.setFixedWidth(36); self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.lbl_zoom = QLabel(self.zoom_text()); self.lbl_zoom.setMinimumWidth(56); self.lbl_zoom.setAlignment(Qt.AlignCenter)
        self.btn_zoom_in  = QPushButton("+"); self.btn_zoom_in.setFixedWidth(36); self.btn_zoom_in.clicked.connect(self.zoom_in)
        zl.addWidget(self.btn_zoom_out); zl.addWidget(self.lbl_zoom); zl.addWidget(self.btn_zoom_in)
        tbar.addWidget(zoom_box)
        rl.addWidget(title_bar)

        self.view_label = ImageLabel()
        self.view_label.sig_click_drag.connect(self.on_paint_or_click)
        self.view_label.sig_release.connect(self.on_stroke_end)
        self.view_label.sig_dbl.connect(lambda: self.finish_polygon(True, True))  # 더블클릭만 1클릭 무시
        # 빈 이미지일 때 1px 흰 점 방지
        self.view_label.setAutoFillBackground(False)
        self.view_label.setStyleSheet("background: transparent;")
        self.view_label.setVisible(False)

        self.scroll = QScrollArea(); self.scroll.setWidget(self.view_label); self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)

        canvas_frame = QFrame(); canvas_frame.setObjectName("CanvasFrame")
        cfl = QVBoxLayout(canvas_frame); cfl.setContentsMargins(8,8,8,8); cfl.setSpacing(4)
        cfl.addWidget(self.scroll)
        rl.addWidget(canvas_frame, 1)

        cl.addWidget(left_panel, 0)
        cl.addWidget(right_panel, 1)
        root_layout.addWidget(content, 1)

        # ===== 스타일 =====
        self.setStyleSheet("""
        QMainWindow { background: #1c1c1e; }
        #Header { background: #26262a; border: 1px solid #4a4a50; border-radius: 8px; }
        #Content { background: #141416; border: 1px solid #2a2a2e; border-radius: 8px; }
        #Panel { background: #26262a; border: 1px solid #4a4a50; border-radius: 12px; }
        #CanvasFrame { background: #303036; border: 1px solid #808088; border-radius: 8px; }
        QLabel { color: #e6e6ec; }
        QLabel#SectionTitle { color: #aaaaaf; font-weight: 600; }
        QPushButton {
            background: #303036; color: #e6e6ec; border: 1px solid #4a4a50; border-radius: 8px; padding: 6px 12px;
        }
        QPushButton:hover { background: #3a3a40; }
        QPushButton:checked { background: #3c4248; }
        QPushButton#Primary { background: #007aff; color: white; border: 1px solid #007aff; }
        QPushButton#Primary:hover { background: #1b8dff; }
        QSlider::groove:horizontal { height: 4px; background: #4a4a50; border-radius: 2px; }
        QSlider::handle:horizontal { width: 14px; height: 14px; margin: -6px 0; background: #007aff; border-radius: 7px; }
        QListWidget { background: #303036; color: #e6e6ec; border: 1px solid #4a4a50; border-radius: 8px; }
        QListWidget::item { padding: 6px 8px; }
        QListWidget::item:selected { background: #3c4248; }
        QScrollArea { background: #303036; border: none; }
        """)

        # ===== 단축키 =====
        QtWidgets.QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.on_ctrl_z)
        QtWidgets.QShortcut(QKeySequence("P"), self, activated=lambda: self.select_tab("polygon"))
        QtWidgets.QShortcut(QKeySequence("B"), self, activated=lambda: self.select_tab("pen"))
        QtWidgets.QShortcut(QKeySequence("E"), self, activated=self.toggle_fill_mode)
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_BracketLeft),  self, activated=lambda: self.nudge_brush(-1))
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_BracketRight), self, activated=lambda: self.nudge_brush(+1))
        QtWidgets.QShortcut(QKeySequence.ZoomIn,  self, activated=self.zoom_in)
        QtWidgets.QShortcut(QKeySequence.ZoomOut, self, activated=self.zoom_out)
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Return), self, activated=lambda: self.finish_polygon(True, False))
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Enter),  self, activated=lambda: self.finish_polygon(True, False))
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Escape), self, activated=lambda: self.finish_polygon(False, False))
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Backspace), self, activated=self.on_ctrl_z)
        # 방향키 세트 이동
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Left),  self, activated=self.prev_pair)
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Up),    self, activated=self.prev_pair)
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Right), self, activated=self.next_pair)
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Down),  self, activated=self.next_pair)

        self.installEventFilter(self)
        self.set_fill_mode(False)

    # ===== 이벤트 필터(한/영 공통 + 글로벌 키) =====
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            fw = QApplication.focusWidget()
            if isinstance(fw, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
                return super().eventFilter(obj, event)

            key = event.key()
            mods = event.modifiers()
            ch = event.text()

            if mods & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
                return super().eventFilter(obj, event)

            # 방향키: 세트 이동
            if key in (Qt.Key_Left, Qt.Key_Up):
                self.prev_pair(); return True
            if key in (Qt.Key_Right, Qt.Key_Down):
                self.next_pair(); return True

            # 브러시 대괄호
            if key == Qt.Key_BracketLeft:  self.nudge_brush(-1); return True
            if key == Qt.Key_BracketRight: self.nudge_brush(+1); return True

            # 모드 토글 (한/영)
            if ch in ("p","P","ㅔ"): self.select_tab("polygon"); return True
            if ch in ("b","B","ㅠ"): self.select_tab("pen"); return True
            if ch in ("e","E","ㄷ"): self.toggle_fill_mode(); return True

            # 줌: +/= , -/_
            if ch in ("+","="): self.zoom_in(); return True
            if ch in ("-","_"): self.zoom_out(); return True

            # 투명도: <, >
            if ch in ("<", ","): self.change_opacity(-self.opacity_step); return True
            if ch in (">", "."): self.change_opacity(+self.opacity_step); return True

            # 밝기: k/l (한글 ㅏ/ㅣ 포함)
            if ch in ("k","K","ㅏ"): self.set_brightness(self.brightness - self.bc_step); self.bri_slider.setValue(self.brightness); return True
            if ch in ("l","L","ㅣ"): self.set_brightness(self.brightness + self.bc_step); self.bri_slider.setValue(self.brightness); return True
            # 명암: ; / '
            if ch in (";",):        self.set_contrast(self.contrast - self.bc_step); self.con_slider.setValue(self.contrast); return True
            if ch in ("'", '"'):    self.set_contrast(self.contrast + self.bc_step); self.con_slider.setValue(self.contrast); return True

        return super().eventFilter(obj, event)

    # ===== 폴더 선택 =====
    def on_choose_orig_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Originals Folder", self.dir_originals or "")
        if not path: return
        self.dir_originals = path
        self.rebuild_pairs()

    def on_choose_mask_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Masks Folder", self.dir_masks or "")
        if not path: return
        self.dir_masks = path
        self.rebuild_pairs()

    # ===== 페어 빌드 (파일명 무시, 인덱스 기준) =====
    def rebuild_pairs(self):
        self.commit_current_mask_to_pair()

        self.pairs.clear()
        self.list_pairs.clear()
        self.current_pair_idx = None
        self.img_rgb = self.gray = self.mask = None
        self.refresh()

        if not self.dir_originals:
            return

        def collect_images(folder):
            if not folder:
                return []
            exts = ["jpg","jpeg","png","bmp","tif","tiff"]
            paths = []
            for ext in exts:
                paths += glob.glob(os.path.join(folder, f"*.{ext}"))
            return sorted(paths)

        orig_paths = collect_images(self.dir_originals)
        mask_paths = collect_images(self.dir_masks) if self.dir_masks else []

        N = max(len(orig_paths), len(mask_paths))
        for i in range(N):
            op = orig_paths[i] if i < len(orig_paths) else None
            mp = mask_paths[i] if i < len(mask_paths) else None
            self.pairs.append(PairItem(op, mp))

        for i, pair in enumerate(self.pairs):
            o_item = QListWidgetItem(f"O: {os.path.basename(pair.orig_path)}" if pair.orig_path else "O: (none)")
            o_item.setData(Qt.UserRole, i); self.list_pairs.addItem(o_item)
            m_item = QListWidgetItem(f"M: {os.path.basename(pair.mask_path)}" if pair.mask_path else "M: (none)")
            m_item.setData(Qt.UserRole, i); self.list_pairs.addItem(m_item)

        if self.pairs:
            first_idx = 0
            for idx, p in enumerate(self.pairs):
                if p.orig_path: first_idx = idx; break
            self.load_pair_into_canvas(first_idx)
            self.update_pair_selection()

    def on_open_selected_pair(self, item: QListWidgetItem):
        idx = item.data(Qt.UserRole)
        if idx is None: return
        self.load_pair_into_canvas(int(idx))
        self.update_pair_selection()

    def update_pair_selection(self):
        if self.current_pair_idx is None: return
        row = self.current_pair_idx * 2
        if 0 <= row < self.list_pairs.count():
            self.list_pairs.setCurrentRow(row)

    def load_pair_into_canvas(self, idx: int):
        if idx < 0 or idx >= len(self.pairs): return
        self.commit_current_mask_to_pair()

        pair = self.pairs[idx]
        if not pair.orig_path:
            QMessageBox.information(self, "Info", "This pair has no original image."); return

        # 원본
        bgr = cv2.imread(pair.orig_path, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.critical(self, "Error", f"Failed to load image: {pair.orig_path}"); return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.img_rgb = rgb
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 마스크
        if pair.mask_arr is not None and pair.mask_arr.shape == self.gray.shape:
            self.mask = pair.mask_arr.copy()
        else:
            m = None
            if pair.mask_path and os.path.exists(pair.mask_path):
                m = cv2.imread(pair.mask_path, cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape != self.gray.shape:
                    m = cv2.resize(m, (self.gray.shape[1], self.gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            if m is None:
                m = np.zeros(self.gray.shape, np.uint8)
            _, m_bin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
            self.mask = m_bin.astype(np.uint8)

        self.undo_stack.clear()
        self.current_pair_idx = idx
        self.refresh()
        self.statusBar().showMessage(f"Loaded set {idx+1}/{len(self.pairs)}", 1500)

    def commit_current_mask_to_pair(self):
        if self.current_pair_idx is None or self.mask is None: return
        if not (0 <= self.current_pair_idx < len(self.pairs)): return
        pair = self.pairs[self.current_pair_idx]
        if pair.mask_arr is None or not np.array_equal(pair.mask_arr, self.mask):
            pair.mask_arr = self.mask.copy()
            pair.modified = True

    # ===== 방향키 이동 =====
    def next_pair(self):
        if not self.pairs: return
        start = self.current_pair_idx if self.current_pair_idx is not None else -1
        n = len(self.pairs)
        for step in range(1, n+1):
            idx = (start + step) % n
            if self.pairs[idx].orig_path:
                self.load_pair_into_canvas(idx)
                self.update_pair_selection()
                break

    def prev_pair(self):
        if not self.pairs: return
        start = self.current_pair_idx if self.current_pair_idx is not None else 0
        n = len(self.pairs)
        for step in range(1, n+1):
            idx = (start - step) % n
            if self.pairs[idx].orig_path:
                self.load_pair_into_canvas(idx)
                self.update_pair_selection()
                break

    # ===== 저장(일괄) =====
    def on_save_all(self):
        if not self.pairs:
            QMessageBox.information(self, "Info", "No pairs to save."); return
        self.commit_current_mask_to_pair()

        modified = [p for p in self.pairs if p.modified and p.mask_arr is not None]
        if not modified:
            QMessageBox.information(self, "Info", "No modified masks to save."); return

        parent = QFileDialog.getExistingDirectory(self, "Choose parent folder to create 'result'", "")
        if not parent: return
        out_dir = os.path.join(parent, "result")
        os.makedirs(out_dir, exist_ok=True)

        saved = 0
        for p in modified:
            if p.mask_path:
                out_name = os.path.basename(p.mask_path)
            else:
                stem = os.path.splitext(os.path.basename(p.orig_path))[0]
                out_name = f"{stem}_mask.png"
            out_path = os.path.join(out_dir, out_name)

            m = (p.mask_arr > 0).astype(np.uint8) * 255
            ext = os.path.splitext(out_path)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                cv2.imwrite(out_path, m, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(out_path, m)
            saved += 1
            p.modified = False

        QMessageBox.information(self, "Saved", f"Saved {saved} mask(s) to:\n{out_dir}")

    # ===== 투명도/밝기/명암/줌 =====
    def opacity_text(self) -> str:
        return f"{int(round(self.overlay_alpha * 100 / 255))}%"

    def change_opacity(self, delta: int):
        new_v = int(max(0, min(255, self.overlay_alpha + delta)))
        self.opacity_slider.setValue(new_v)

    def set_overlay_alpha(self, v: int):
        self.overlay_alpha = int(max(0, min(255, v)))
        if hasattr(self, "lbl_opacity"):
            self.lbl_opacity.setText(self.opacity_text())
        if self.img_rgb is not None:
            self.refresh()

    def set_brightness(self, v: int):
        self.brightness = int(max(-100, min(100, v)))
        if hasattr(self, "lbl_bri"):
            self.lbl_bri.setText(str(self.brightness))
        if self.img_rgb is not None:
            self.refresh()

    def set_contrast(self, v: int):
        self.contrast = int(max(-100, min(100, v)))
        if hasattr(self, "lbl_con"):
            self.lbl_con.setText(str(self.contrast))
        if self.img_rgb is not None:
            self.refresh()

    def zoom_text(self) -> str:
        return f"{int(round(self.zoom * 100))}%"

    def set_zoom(self, z: float):
        z = max(self.min_zoom, min(self.max_zoom, z))
        if abs(z - self.zoom) < 1e-6: return
        self.zoom = z
        if hasattr(self, "lbl_zoom"): self.lbl_zoom.setText(self.zoom_text())
        self.refresh()

    def zoom_in(self):  self.set_zoom(self.zoom * self.zoom_step)
    def zoom_out(self): self.set_zoom(self.zoom / self.zoom_step)

    # ===== 모드/Undo =====
    def on_ctrl_z(self):
        if self.mode == "polygon" and self.poly_points:
            self.poly_points.pop()
            self.refresh()
        else:
            self.on_undo()

    def on_tools_tab_changed(self, idx: int):
        self.mode = "pen" if idx == 0 else "polygon"
        if self.mode != "polygon":
            self.poly_points.clear(); self.poly_active = False; self.poly_click_block = False
        self.statusBar().showMessage(f"Mode: {self.mode}", 800)

    def select_tab(self, name: str):
        self.tools_tabs.setCurrentIndex(0 if name=="pen" else 1)

    def set_fill_mode(self, erase_checked: bool):
        self.draw_op = "sub" if erase_checked else "add"
        for btn in (self.btn_fill_toggle_pen, self.btn_fill_toggle_poly):
            if btn.isChecked() != erase_checked:
                btn.blockSignals(True); btn.setChecked(erase_checked); btn.blockSignals(False)
        self.statusBar().showMessage("Erase" if erase_checked else "Fill", 700)

    def toggle_fill_mode(self):
        self.set_fill_mode(not (self.draw_op == "sub"))

    def set_brush_size(self, v: int):
        v = int(max(1, min(30, v))); self.brush_size = v
        if self.brush_slider.value() != v:
            self.brush_slider.blockSignals(True); self.brush_slider.setValue(v); self.brush_slider.blockSignals(False)
        if self.brush_spin.value() != v:
            self.brush_spin.blockSignals(True); self.brush_spin.setValue(v); self.brush_spin.blockSignals(False)
        self.statusBar().showMessage(f"Brush size: {v}", 700)

    def nudge_brush(self, d: int): self.set_brush_size(self.brush_size + d)

    # ===== 편집 =====
    def push_undo(self):
        if self.mask is None: return
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > self.max_undo: self.undo_stack.pop(0)

    def on_undo(self):
        if not self.undo_stack: return
        self.mask = self.undo_stack.pop(-1); self.mark_current_modified(); self.refresh()

    def mark_current_modified(self):
        if self.current_pair_idx is None: return
        if 0 <= self.current_pair_idx < len(self.pairs):
            self.pairs[self.current_pair_idx].modified = True
            self.pairs[self.current_pair_idx].mask_arr = self.mask.copy()

    # Pen 미리보기 갱신
    def update_pen_preview(self, ix: int, iy: int, final: bool):
        if self.pen_base_mask is None: return
        temp = self.pen_base_mask.copy()
        thickness = max(1, int(self.brush_size)); line_type = cv2.LINE_AA
        color = 255 if self.draw_op == "add" else 0

        if self.pen_shift_snap and self.pen_start_xy is not None:
            pts = np.array([self.pen_start_xy, (ix, iy)], np.int32)
            cv2.polylines(temp, [pts], False, color, thickness, line_type)
        else:
            if len(self.pen_stroke_pts) >= 2:
                pts = np.array(self.pen_stroke_pts, np.int32)
                cv2.polylines(temp, [pts], False, color, thickness, line_type)
            else:
                cv2.circle(temp, (ix, iy), max(1, thickness//2), color, -1, line_type)

        self.mask = temp
        if final:
            self.pen_base_mask = None
            self.pen_stroke_pts = []
            self.pen_start_xy = None
            self.pen_shift_snap = False
            self.mark_current_modified()

        self.refresh()

    def on_paint_or_click(self, x: int, y: int, is_drag: bool):
        if self.gray is None: return
        H, W = self.gray.shape

        # 캔버스 → 이미지 좌표(줌 보정)
        z = max(1e-6, float(self.zoom))
        ix = int(x / z); iy = int(y / z)
        if ix < 0 or iy < 0 or ix >= W or iy >= H: return

        if self.mode == "pen":
            self.pen_shift_snap = bool(QtGui.QGuiApplication.keyboardModifiers() & Qt.ShiftModifier)
            if not is_drag:
                self.push_undo()
                self.pen_base_mask = self.mask.copy()
                self.pen_stroke_pts = [(ix, iy)]
                self.pen_start_xy = (ix, iy)
                self.update_pen_preview(ix, iy, final=False)
            else:
                if self.pen_base_mask is None:
                    self.pen_base_mask = self.mask.copy()
                    self.pen_stroke_pts = [(ix, iy)]
                    self.pen_start_xy = (ix, iy)
                else:
                    self.pen_stroke_pts.append((ix, iy))
                self.update_pen_preview(ix, iy, final=False)

        elif self.mode == "polygon":
            if not is_drag:
                if self.poly_click_block:
                    self.poly_click_block = False
                    return
                if not self.poly_active:
                    self.poly_points = []
                    self.poly_active = True
                self.poly_points.append((ix, iy))
                self.refresh()

    def on_stroke_end(self):
        if self.mode == "pen" and self.pen_base_mask is not None:
            if self.pen_stroke_pts:
                ix, iy = self.pen_stroke_pts[-1]
            else:
                ix = iy = 0
            self.update_pen_preview(ix, iy, final=True)

    def finish_polygon(self, apply=True, block_next_click=False):
        if not self.poly_points:
            self.poly_active = False
            self.poly_click_block = bool(block_next_click)
            return

        if apply and len(self.poly_points) >= 3:
            self.push_undo()
            pts = np.array(self.poly_points, np.int32).reshape((-1, 1, 2))
            canvas = np.zeros_like(self.mask)
            cv2.fillPoly(canvas, [pts], 255)
            if self.draw_op == "add":
                self.mask = cv2.bitwise_or(self.mask, canvas)
            else:
                self.mask[canvas > 0] = 0
            self.mark_current_modified()

        self.poly_points.clear()
        self.poly_active = False
        self.poly_click_block = bool(block_next_click)
        self.refresh()

    # ===== 렌더 =====
    def refresh(self):
        # 이미지 없으면 라벨을 숨겨 1px 흰 점 방지
        if self.img_rgb is None:
            self.view_label.clear()
            self.view_label.setVisible(False)
            return

        self.view_label.setVisible(True)

        # 밝기/명암 반영된 베이스
        base_rgb = apply_brightness_contrast(self.img_rgb, self.brightness, self.contrast)

        H, W = self.gray.shape
        pix = QPixmap.fromImage(np_to_qimage_rgb(base_rgb))
        painter = QtGui.QPainter(pix)
        try:
            if self.mode == "polygon" and len(self.poly_points) >= 1:
                qpts = [QtCore.QPointF(x, y) for (x, y) in self.poly_points]
                pen = QtGui.QPen(QtGui.QColor("#00ff88"), 2)
                painter.setPen(pen)
                for i in range(1, len(qpts)):
                    painter.drawLine(qpts[i-1], qpts[i])
                if len(qpts) >= 3:
                    path = QtGui.QPainterPath()
                    path.moveTo(qpts[0])
                    for p in qpts[1:]:
                        path.lineTo(p)
                    path.closeSubpath()
                    painter.fillPath(path, QtGui.QColor(0, 255, 136, 64))

            if self.mask is not None and self.overlay_alpha > 0:
                rgba = np.zeros((H, W, 4), dtype=np.uint8)
                rgba[..., 0] = 255
                rgba[..., 3] = (self.mask > 0).astype(np.uint8) * self.overlay_alpha
                ov = QPixmap.fromImage(np_to_qimage_rgba(rgba))
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
                painter.drawPixmap(0, 0, ov)
        finally:
            painter.end()

        # 줌
        if abs(self.zoom - 1.0) > 1e-6:
            sw = max(1, int(W * self.zoom))
            sh = max(1, int(H * self.zoom))
            pix = pix.scaled(sw, sh, Qt.KeepAspectRatio, Qt.FastTransformation)

        self.view_label.setPixmap(pix)
        self.view_label.setFixedSize(pix.size())

# ---------- 엔트리 ----------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()