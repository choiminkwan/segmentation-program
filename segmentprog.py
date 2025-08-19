# dental_mask_editor_figma.py
# 사이드 탭형 툴바(Pen/Polygon) + 디렉토리, 우측 캔버스 / 다크테마
# 한/영 단축키, 줌 UI 캔버스 타이틀 오른쪽, < / > 로 투명도 조절 + %표시
# Enter 완료 시 다음 클릭 씹힘 방지(더블클릭만 1클릭 무시), Pen 스트로크 매끈화(프리뷰-확정 방식, Shift 직선 스냅)
# 의존성: PyQt5, opencv-python, numpy

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

# ---------- 메인 ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dental Mask Editor — Side Tabs + Zoom (KR/EN Hotkeys)")
        self.resize(1280, 820)

        # 상태
        self.img_rgb = None
        self.gray = None
        self.mask = None
        self.overlay_alpha = 184   # 0..255
        self.opacity_step = 16     # <, > 조절 스텝
        self.mode = "pen"          # pen / polygon
        self.draw_op = "add"       # add / sub
        self.brush_size = 14       # 1~30

        # 폴리곤 상태
        self.poly_points = []
        self.poly_active = False          # 현재 폴리곤 세션 진행 중인지
        self.poly_click_block = False     # 완료/취소 직후 클릭 1회 무시

        self.undo_stack = []; self.max_undo = 30
        self.current_dir = None

        # Pen 스트로크 미리보기 상태
        self.pen_base_mask = None         # 스트로크 시작 시점의 마스크 복사본
        self.pen_stroke_pts = []          # 드래그 포인트 누적
        self.pen_start_xy = None          # 스트로크 시작점
        self.pen_shift_snap = False       # Shift 직선 스냅 여부

        # 줌 상태
        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 6.0
        self.zoom_step = 1.25

        # ===== Root =====
        root = QWidget(); self.setCentralWidget(root)
        root_layout = QVBoxLayout(root); root_layout.setContentsMargins(8,8,8,8); root_layout.setSpacing(8)

        # ===== 헤더 (파일/투명도/저장) =====
        header = QFrame(); header.setObjectName("Header")
        hl = QHBoxLayout(header); hl.setContentsMargins(12,8,12,8); hl.setSpacing(12)

        btn_open_img = QPushButton("Open Original"); btn_open_img.clicked.connect(self.on_open_image)
        btn_open_msk = QPushButton("Open Mask");     btn_open_msk.clicked.connect(self.on_open_mask)
        btn_load_dir = QPushButton("Load Directory"); btn_load_dir.clicked.connect(self.on_load_directory)
        hl.addWidget(btn_open_img); hl.addWidget(btn_open_msk); hl.addWidget(btn_load_dir)
        hl.addStretch(1)

        # Opacity 슬라이더 + %표시 라벨
        opbox = QFrame(); opl = QHBoxLayout(opbox); opl.setContentsMargins(0,0,0,0); opl.setSpacing(8)
        opl.addWidget(QLabel("Opacity"))
        self.opacity_slider = QSlider(Qt.Horizontal); self.opacity_slider.setRange(0,255); self.opacity_slider.setValue(self.overlay_alpha)
        self.opacity_slider.valueChanged.connect(lambda v: self.set_overlay_alpha(v))
        self.lbl_opacity = QLabel(self.opacity_text())
        self.lbl_opacity.setMinimumWidth(40)
        self.lbl_opacity.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        opl.addWidget(self.opacity_slider); opl.addWidget(self.lbl_opacity)
        hl.addWidget(opbox, 1)

        btn_save = QPushButton("Save"); btn_save.setObjectName("Primary"); btn_save.clicked.connect(self.on_save_mask)
        hl.addWidget(btn_save)

        root_layout.addWidget(header)

        # ===== 본문 =====
        content = QFrame(); content.setObjectName("Content")
        cl = QHBoxLayout(content); cl.setContentsMargins(12,12,12,12); cl.setSpacing(12)

        # --- 좌측: Tabs + Directory ---
        left_panel = QFrame(); left_panel.setObjectName("Panel")
        ll = QVBoxLayout(left_panel); ll.setContentsMargins(12,12,12,12); ll.setSpacing(12)

        tools_title = QLabel("Tools"); tools_title.setObjectName("SectionTitle")
        ll.addWidget(tools_title)

        self.tools_tabs = QTabWidget(); self.tools_tabs.setTabPosition(QTabWidget.North)
        self.tools_tabs.currentChanged.connect(self.on_tools_tab_changed)

        # Pen 탭
        pen_tab = QWidget(); pen_l = QVBoxLayout(pen_tab); pen_l.setContentsMargins(6,6,6,6); pen_l.setSpacing(8)
        self.btn_fill_toggle_pen = QPushButton("Fill / Erase")
        self.btn_fill_toggle_pen.setCheckable(True)
        self.btn_fill_toggle_pen.toggled.connect(lambda ch: self.set_fill_mode(ch))
        pen_l.addWidget(self.btn_fill_toggle_pen)
        pen_brush_row = QHBoxLayout(); pen_brush_row.setSpacing(8)
        pen_brush_row.addWidget(QLabel("Brush"))
        self.brush_slider = QSlider(Qt.Horizontal); self.brush_slider.setRange(1,30); self.brush_slider.setValue(self.brush_size)
        self.brush_spin = QSpinBox(); self.brush_spin.setRange(1,30); self.brush_spin.setValue(self.brush_size); self.brush_spin.setFixedWidth(64)
        self.brush_slider.valueChanged.connect(self.set_brush_size)
        self.brush_spin.valueChanged.connect(self.set_brush_size)
        pen_brush_row.addWidget(self.brush_slider); pen_brush_row.addWidget(self.brush_spin)
        pen_l.addLayout(pen_brush_row)
        btn_undo_pen = QPushButton("Undo"); btn_undo_pen.clicked.connect(self.on_ctrl_z)
        pen_l.addWidget(btn_undo_pen)
        pen_l.addStretch(1)
        self.tools_tabs.addTab(pen_tab, "Pen")

        # Polygon 탭
        poly_tab = QWidget(); poly_l = QVBoxLayout(poly_tab); poly_l.setContentsMargins(6,6,6,6); poly_l.setSpacing(8)
        self.btn_fill_toggle_poly = QPushButton("Fill / Erase")
        self.btn_fill_toggle_poly.setCheckable(True)
        self.btn_fill_toggle_poly.toggled.connect(lambda ch: self.set_fill_mode(ch))
        poly_l.addWidget(self.btn_fill_toggle_poly)
        row_poly_btns = QHBoxLayout(); row_poly_btns.setSpacing(8)
        self.btn_poly_done   = QPushButton("Complete");  self.btn_poly_done.clicked.connect(lambda: self.finish_polygon(True, False))
        self.btn_poly_undo   = QPushButton("Undo");      self.btn_poly_undo.clicked.connect(self.on_ctrl_z)
        self.btn_poly_cancel = QPushButton("Cancel");    self.btn_poly_cancel.clicked.connect(lambda: self.finish_polygon(False, False))
        row_poly_btns.addWidget(self.btn_poly_done); row_poly_btns.addWidget(self.btn_poly_undo); row_poly_btns.addWidget(self.btn_poly_cancel)
        poly_l.addLayout(row_poly_btns)
        poly_l.addStretch(1)
        self.tools_tabs.addTab(poly_tab, "Polygon")

        ll.addWidget(self.tools_tabs)

        dir_title = QLabel("Directory"); dir_title.setObjectName("SectionTitle")
        self.list_files = QListWidget(); self.list_files.itemDoubleClicked.connect(self.on_open_selected_from_list)
        ll.addWidget(dir_title); ll.addWidget(self.list_files, 1)

        # --- 우측: Canvas ---
        right_panel = QFrame(); right_panel.setObjectName("Panel")
        rl = QVBoxLayout(right_panel); rl.setContentsMargins(12,12,12,12); rl.setSpacing(8)

        # 타이틀 바: 좌측 제목 / 우측 Zoom UI
        title_bar = QFrame(); tbar = QHBoxLayout(title_bar); tbar.setContentsMargins(0,0,0,0); tbar.setSpacing(8)
        right_title = QLabel("Canvas (Original + Mask Overlay)"); right_title.setObjectName("SectionTitle")
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
        # 더블클릭 완료는 잔여클릭 방지를 위해 block_next_click=True
        self.view_label.sig_dbl.connect(lambda: self.finish_polygon(True, True))

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

        # ===== 단축키 (QShortcut + 이벤트필터) =====
        QtWidgets.QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.on_ctrl_z)
        QtWidgets.QShortcut(QKeySequence("P"), self, activated=lambda: self.select_tab("polygon"))
        QtWidgets.QShortcut(QKeySequence("B"), self, activated=lambda: self.select_tab("pen"))
        QtWidgets.QShortcut(QKeySequence("E"), self, activated=self.toggle_fill_mode)
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_BracketLeft),  self, activated=lambda: self.nudge_brush(-1))
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_BracketRight), self, activated=lambda: self.nudge_brush(+1))
        QtWidgets.QShortcut(QKeySequence.ZoomIn,  self, activated=self.zoom_in)
        QtWidgets.QShortcut(QKeySequence.ZoomOut, self, activated=self.zoom_out)
        # Enter/Return 완료는 block_next_click=False (클릭 씹힘 방지)
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Return), self, activated=lambda: self.finish_polygon(True, False))
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Enter),  self, activated=lambda: self.finish_polygon(True, False))
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Escape), self, activated=lambda: self.finish_polygon(False, False))
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Backspace), self, activated=self.on_ctrl_z)

        # 한/영 단축키 이벤트 필터
        self.installEventFilter(self)

        self.set_fill_mode(False)  # Fill 기본

    # ===== 이벤트 필터: 한글/영문 단축키 공통 처리 =====
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

            # 브러시 대괄호
            if key == Qt.Key_BracketLeft:
                self.nudge_brush(-1); return True
            if key == Qt.Key_BracketRight:
                self.nudge_brush(+1); return True

            # 모드 토글 (한/영)
            if ch in ("p", "P", "ㅔ"):
                self.select_tab("polygon"); return True
            if ch in ("b", "B", "ㅠ"):
                self.select_tab("pen"); return True
            if ch in ("e", "E", "ㄷ"):
                self.toggle_fill_mode(); return True

            # 줌: +/= , -/_
            if ch in ("+", "="):
                self.zoom_in(); return True
            if ch in ("-", "_"):
                self.zoom_out(); return True

            # 투명도: <, >  (보조로 , . 도 허용)
            if ch in ("<", ","):
                self.change_opacity(-self.opacity_step); return True
            if ch in (">", "."):
                self.change_opacity(+self.opacity_step); return True

        return super().eventFilter(obj, event)

    # ----- 투명도 -----
    def opacity_text(self) -> str:
        return f"{int(round(self.overlay_alpha * 100 / 255))}%"

    def change_opacity(self, delta: int):
        new_v = int(max(0, min(255, self.overlay_alpha + delta)))
        self.opacity_slider.setValue(new_v)  # 슬라이더와 동기화 → set_overlay_alpha 호출

    def set_overlay_alpha(self, v: int):
        self.overlay_alpha = int(max(0, min(255, v)))
        if hasattr(self, "lbl_opacity"):
            self.lbl_opacity.setText(self.opacity_text())
        if self.img_rgb is not None:
            self.refresh()

    # ----- 줌 -----
    def zoom_text(self) -> str:
        return f"{int(round(self.zoom * 100))}%"

    def set_zoom(self, z: float):
        z = max(self.min_zoom, min(self.max_zoom, z))
        if abs(z - self.zoom) < 1e-6:
            return
        self.zoom = z
        if hasattr(self, "lbl_zoom"):
            self.lbl_zoom.setText(self.zoom_text())
        self.refresh()

    def zoom_in(self):
        self.set_zoom(self.zoom * self.zoom_step)

    def zoom_out(self):
        self.set_zoom(self.zoom / self.zoom_step)

    # ----- 컨텍스트 Undo -----
    def on_ctrl_z(self):
        if self.mode == "polygon" and self.poly_points:
            self.poly_points.pop()
            self.refresh()
        else:
            self.on_undo()

    # ----- 탭/모드 -----
    def on_tools_tab_changed(self, idx: int):
        self.mode = "pen" if idx == 0 else "polygon"
        # 폴리곤에서 다른 모드로 전환될 때 세션 정리
        if self.mode != "polygon":
            self.poly_points.clear()
            self.poly_active = False
            self.poly_click_block = False
        self.statusBar().showMessage(f"Mode: {self.mode}", 800)

    def select_tab(self, name: str):
        if name == "pen":
            self.tools_tabs.setCurrentIndex(0)
        elif name == "polygon":
            self.tools_tabs.setCurrentIndex(1)

    # ----- Fill/Erase -----
    def set_fill_mode(self, erase_checked: bool):
        self.draw_op = "sub" if erase_checked else "add"
        for btn in (self.btn_fill_toggle_pen, self.btn_fill_toggle_poly):
            if btn.isChecked() != erase_checked:
                btn.blockSignals(True); btn.setChecked(erase_checked); btn.blockSignals(False)
        self.statusBar().showMessage("Erase" if erase_checked else "Fill", 700)

    def toggle_fill_mode(self):
        self.set_fill_mode(not (self.draw_op == "sub"))

    # ----- 브러시 크기 -----
    def set_brush_size(self, v: int):
        v = int(max(1, min(30, v)))
        self.brush_size = v
        if self.brush_slider.value() != v:
            self.brush_slider.blockSignals(True); self.brush_slider.setValue(v); self.brush_slider.blockSignals(False)
        if self.brush_spin.value() != v:
            self.brush_spin.blockSignals(True); self.brush_spin.setValue(v); self.brush_spin.blockSignals(False)
        self.statusBar().showMessage(f"Brush size: {v}", 700)

    def nudge_brush(self, delta: int):
        self.set_brush_size(self.brush_size + delta)

    # ----- 파일/디렉토리 -----
    def on_load_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Choose Directory", self.current_dir or "")
        if not path: return
        self.current_dir = path
        self.list_files.clear()
        exts = ["jpg","jpeg","png","bmp","tif","tiff"]
        files = sorted(sum([glob.glob(os.path.join(path, f"*.{ext}")) for ext in exts], []))
        for fp in files:
            it = QListWidgetItem(os.path.basename(fp)); it.setData(Qt.UserRole, fp)
            self.list_files.addItem(it)
        if files: self.list_files.setCurrentRow(0)

    def on_open_selected_from_list(self, item: QListWidgetItem):
        fp = item.data(Qt.UserRole)
        if not fp: return
        self.load_image(fp)
        base, _ = os.path.splitext(fp)
        for mfp in [base + "_mask.png", base + "_mask.jpg", base + ".png"]:
            if os.path.exists(mfp):
                self.load_mask(mfp, resize_to_image=True); break

    def on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Original", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not path: return
        self.load_image(path)

    def on_open_mask(self):
        if self.gray is None:
            QMessageBox.information(self, "Info", "Open the original image first."); return
        path, _ = QFileDialog.getOpenFileName(self, "Open Mask", "", "Images (*.jpg *.jpeg *.png)")
        if not path: return
        self.load_mask(path, resize_to_image=True)

    def on_save_mask(self):
        if self.mask is None:
            QMessageBox.information(self, "Info", "No mask to save."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Mask", "teeth_mask.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path: return
        m = (self.mask > 0).astype(np.uint8) * 255
        ext = os.path.splitext(path)[1].lower()
        if ext in [".jpg",".jpeg"]:
            cv2.imwrite(path, m, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(path, m)
        self.statusBar().showMessage(f"Saved: {path}", 3000)

    # ----- 로드 -----
    def load_image(self, path: str):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            QMessageBox.critical(self, "Error", f"Failed to load image: {path}"); return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.img_rgb = rgb
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.mask is None or self.mask.shape != self.gray.shape:
            self.mask = np.zeros(self.gray.shape, np.uint8)
        self.undo_stack.clear()
        self.refresh()

    def load_mask(self, path: str, resize_to_image=False):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            QMessageBox.critical(self, "Error", f"Failed to load mask: {path}"); return
        if resize_to_image and m.shape != self.gray.shape:
            m = cv2.resize(m, (self.gray.shape[1], self.gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, m_bin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        self.mask = m_bin.astype(np.uint8)
        self.undo_stack.clear()
        self.refresh()

    # ----- 편집 -----
    def push_undo(self):
        if self.mask is None: return
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > self.max_undo: self.undo_stack.pop(0)

    def on_undo(self):
        if not self.undo_stack: return
        self.mask = self.undo_stack.pop(-1); self.refresh()

    # Pen 프리뷰 갱신(드래그 중/릴리즈 시 공통)
    def update_pen_preview(self, ix: int, iy: int, final: bool):
        if self.pen_base_mask is None:
            return
        temp = self.pen_base_mask.copy()
        thickness = max(1, int(self.brush_size))
        line_type = cv2.LINE_AA

        if self.pen_shift_snap and self.pen_start_xy is not None:
            pts = np.array([self.pen_start_xy, (ix, iy)], np.int32)
            if self.draw_op == "add":
                cv2.polylines(temp, [pts], False, 255, thickness, line_type)
            else:
                cv2.polylines(temp, [pts], False, 0,   thickness, line_type)
        else:
            if len(self.pen_stroke_pts) >= 2:
                pts = np.array(self.pen_stroke_pts, np.int32)
                if self.draw_op == "add":
                    cv2.polylines(temp, [pts], False, 255, thickness, line_type)
                else:
                    cv2.polylines(temp, [pts], False, 0,   thickness, line_type)
            else:
                # 점만 있을 때는 점으로 표시
                if self.draw_op == "add":
                    cv2.circle(temp, (ix, iy), max(1, thickness // 2), 255, -1, line_type)
                else:
                    cv2.circle(temp, (ix, iy), max(1, thickness // 2), 0,   -1, line_type)

        self.mask = temp
        if final:
            # 스트로크 종료 → 프리뷰 상태 해제
            self.pen_base_mask = None
            self.pen_stroke_pts = []
            self.pen_start_xy = None
            self.pen_shift_snap = False

        self.refresh()

    def on_paint_or_click(self, x: int, y: int, is_drag: bool):
        if self.gray is None: return
        H, W = self.gray.shape

        # 줌 좌표 → 이미지 좌표
        z = max(1e-6, float(self.zoom))
        ix = int(x / z); iy = int(y / z)
        if ix < 0 or iy < 0 or ix >= W or iy >= H: return

        if self.mode == "pen":
            # Shift 스냅 상태 읽기
            self.pen_shift_snap = bool(QtGui.QGuiApplication.keyboardModifiers() & Qt.ShiftModifier)

            if not is_drag:
                # 스트로크 시작: Undo 저장 + 베이스 확보
                self.push_undo()
                self.pen_base_mask = self.mask.copy()
                self.pen_stroke_pts = [(ix, iy)]
                self.pen_start_xy = (ix, iy)
                self.update_pen_preview(ix, iy, final=False)
            else:
                # 드래그 중: 포인트 누적 후 프리뷰 갱신(매끄럽게)
                if self.pen_base_mask is None:
                    self.pen_base_mask = self.mask.copy()
                    self.pen_stroke_pts = [(ix, iy)]
                    self.pen_start_xy = (ix, iy)
                else:
                    self.pen_stroke_pts.append((ix, iy))
                self.update_pen_preview(ix, iy, final=False)

        elif self.mode == "polygon":
            if not is_drag:
                # 폴리곤 완료/취소 직후 들어오는 클릭 1회 무시(더블클릭 후 버블 방지)
                if self.poly_click_block:
                    self.poly_click_block = False
                    return
                # 세션 시작
                if not self.poly_active:
                    self.poly_points = []
                    self.poly_active = True
                self.poly_points.append((ix, iy))
                self.refresh()

    def on_stroke_end(self):
        # 펜 스트로크 확정(릴리즈 시)
        if self.mode == "pen" and self.pen_base_mask is not None:
            # 마지막 좌표는 이미 on_paint_or_click에서 전달되었을 수 있으므로
            if self.pen_stroke_pts:
                ix, iy = self.pen_stroke_pts[-1]
            else:
                ix = iy = 0
            self.update_pen_preview(ix, iy, final=True)

    def finish_polygon(self, apply=True, block_next_click=False):
        # 점이 없더라도 완료/취소시 세션 종료
        if not self.poly_points:
            self.poly_active = False
            # 더블클릭일 때만 다음 클릭 무시
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

        # 세션 종료
        self.poly_points.clear()
        self.poly_active = False
        # 더블클릭일 때만 클릭 차단
        self.poly_click_block = bool(block_next_click)
        self.refresh()

    # ----- 렌더 -----
    def refresh(self):
        if self.img_rgb is None:
            self.view_label.clear(); return

        H, W = self.gray.shape
        pix = QPixmap.fromImage(np_to_qimage_rgb(self.img_rgb))
        painter = QtGui.QPainter(pix)
        try:
            # 폴리곤 프리뷰
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

            # 마스크 오버레이
            if self.mask is not None and self.overlay_alpha > 0:
                rgba = np.zeros((H, W, 4), dtype=np.uint8)
                rgba[...,0] = 255
                rgba[...,3] = (self.mask > 0).astype(np.uint8) * self.overlay_alpha
                ov = QPixmap.fromImage(np_to_qimage_rgba(rgba))
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
                painter.drawPixmap(0, 0, ov)
        finally:
            painter.end()

        # 줌 스케일 적용
        if abs(self.zoom - 1.0) > 1e-6:
            sw = max(1, int(W * self.zoom))
            sh = max(1, int(H * self.zoom))
            pix = pix.scaled(sw, sh, Qt.KeepAspectRatio, Qt.FastTransformation)

        self.view_label.setPixmap(pix)
        self.view_label.setFixedSize(pix.size())

    # ----- 줌 텍스트 -----
    def zoom_text(self) -> str:
        return f"{int(round(self.zoom * 100))}%"

# ---------- 엔트리 ----------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

#todo 
# 원본에 대한 명암 조절 기능
# 원본 로드, 마스크 로드 제작
 