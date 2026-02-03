import cv2
import numpy as np
from ultralytics import YOLO
import json
import time
from typing import Dict, List, Optional, Tuple, Any

class TaskStartPointDetector:
    """
    Task Start Point Detection (TSPD) Algoritması - Gruplu Görev Yönetimi
    
    Video akışında nesne değişimlerini ve hareket modellerini analiz ederek
    gerçek görev geçişlerini tespit eder. Görevleri gruplara ayırarak,
    grup içi sıra değişikliklerini tolere eder.
    """
    
    def __init__(self, yolo_model_path: str, confidence_threshold: float = 0.7, task_definition_mode: str = "manual"):
        """
        TSPD sınıfını başlatır
        
        Args:
            yolo_model_path: YOLO model dosyasının yolu
            confidence_threshold: YOLO tahmin güven eşiği
            task_definition_mode: Görev tanımlama modu ("manual", "semantic", "contextual")
        """
        self.yolo_model = YOLO(yolo_model_path)
        #self.yolo_model.to('cpu')  # CPU'yu zorla
        self.confidence_threshold = confidence_threshold
        self.task_definition_mode = task_definition_mode
        
        # Algoritma değişkenleri
        self.yolo_previous = None
        self.yolo_current = None
        self.n = 0  # Görev sayacı
        self.s = 0  # Başlangıç frame numarası
        
        # Veri saklama
        self.task_history = []
        self.frame_buffer = []
        self.detection_log = []
        
        # Görev tanımlama sistemi
        self.task_definitions = self._initialize_task_definitions()
        self.object_stability_buffer = []
        self.min_stability_frames = 2
        
        # GRUPLU GÖREV YÖNETİMİ
        self.task_groups = [
            {
                'name': 'Grup 1: Fikstur',
                'tasks': ['fixture'], #buradan fixture kaldır ilk başta hataya düşüyor
                'strict_order': False,  # Grup içinde sıra önemli
                'completion_required': True  # Tüm görevlerin yapılması gerekli
            },
            {
                'name': 'Grup 2: Reflektor',
                'tasks': ['reflector'], #buradan fixture kaldır ilk başta hataya düşüyor
                'strict_order': False,  # Grup içinde sıra önemli
                'completion_required': True  # Tüm görevlerin yapılması gerekli
            },
            {
                'name': 'Grup 3: PCB',
                'tasks': ['pcb'],
                'strict_order': True,
                'completion_required': True
            },
            {
                'name': 'Grup 4: Buton',
                'tasks': ['button_hand'],
                'strict_order': True,
                'completion_required': True
            },
            {
                'name': 'Grup 5: Vidalama',
                'tasks': ['screwing'],
                'strict_order': True,
                'completion_required': True
            },
            {
                'name': 'Grup 6: Konnektorler',
                'tasks': ['test_connector', 'black_connector'], #, 'power_connector'
                'strict_order': False,
                'completion_required': True
            },
            {
                'name': 'Grup 7: Buton',
                'tasks': ['button_hand'],
                'strict_order': True,
                'completion_required': True
            },
            {
                'name': 'Grup 8: Power konnektor',
                'tasks': [ "power_connector"],
                'strict_order': True,
                'completion_required': True
            },
            {
                'name': 'Grup 9: Fikstur',
                'tasks': ['fixture'],
                'strict_order': True,
                'completion_required': True
            }
        ]
        
        # Tüm görevlerin düz listesi (referans için)
        self.task_order = []
        for group in self.task_groups:
            self.task_order.extend(group['tasks'])
        
        # Grup takip değişkenleri
        self.current_group_idx = 0
        self.current_group_completed_tasks = set()
        self.group_violations = []
        
        self.expected_task_idx = 0
        self.current_task_name = None
        self.last_detections = []
        
        # Sınıf bazlı renkler (BGR)
        self.class_colors = {
            'reflector': (0, 0, 255),
            'pcb': (0, 255, 0),
            'screwing': (255, 0, 0),
            'fixture': (0, 165, 255),
            'button_hand': (255, 0, 255),
            'black_connector': (0, 128, 0),
            'test_connector': (128, 0, 128),
            'power_connector': (0, 255, 255),
        }
        
        # Uyarı yönetimi
        self.order_violations = []
        self.order_violation_frames_remaining = 0
        self.order_violation_message = None
        self.violation_cooldown_frames = 0
        self.stable_mismatch_label = None
        self.stable_mismatch_count = 0
        self.min_mismatch_stability_frames = 3
        
        # Son aktif görev takibi
        self.last_active_task_name = None
        
    def _initialize_task_definitions(self) -> Dict[str, Any]:
        """Görev tanımlama yapılarını başlatır"""
        return {}
    
    def get_current_group(self) -> Optional[Dict]:
        """Mevcut grup bilgisini döndürür"""
        if self.current_group_idx < len(self.task_groups):
            return self.task_groups[self.current_group_idx]
        return None
    
    def is_task_in_current_group(self, task_name: str) -> bool:
        """Görevin mevcut grupta olup olmadığını kontrol eder"""
        current_group = self.get_current_group()
        if current_group is None:
            return False
        return task_name in current_group['tasks']
    
    def is_current_group_completed(self) -> bool:
        """Mevcut grubun tamamlanıp tamamlanmadiğını kontrol eder"""
        current_group = self.get_current_group()
        if current_group is None:
            return True
        
        required_tasks = set(current_group['tasks'])
        return required_tasks.issubset(self.current_group_completed_tasks)
    
    def check_group_violation(self, detected_task: str) -> Tuple[bool, str]:
        """
        Grup ihlalini kontrol eder
        
        Returns:
            (ihlal_var_mı, hata_mesajı)
        """
        current_group = self.get_current_group()
        if current_group is None:
            return False, ""
        
        # Tespit edilen görevin hangi grupta olduğunu bul
        detected_group_idx = None
        for idx, group in enumerate(self.task_groups):
            if detected_task in group['tasks']:
                detected_group_idx = idx
                break
        
        if detected_group_idx is None:
            return True, f"Tanımsız görev: {detected_task}"
        
        # Önceki gruplara ait görev tespiti (geçmiş grup)
        if detected_group_idx < self.current_group_idx:
            return False, ""  # Sessizce yoksay
        
        # Gelecek gruplara ait görev tespiti (atlama)
        if detected_group_idx > self.current_group_idx:
            # Mevcut grup tamamlanmış mı kontrol et
            if not self.is_current_group_completed():
                incomplete = set(current_group['tasks']) - self.current_group_completed_tasks
                return True, f"Grup atlandı! {current_group['name']} tamamlanmadi. Eksik: {', '.join(incomplete)}"
        
        # Aynı gruptaki görev
        if detected_group_idx == self.current_group_idx:
            # Grup içinde sıra kontrolü
            if current_group.get('strict_order', False):
                # Sıralı grup: beklenen görevi kontrol et
                expected_idx = len(self.current_group_completed_tasks)
                if expected_idx < len(current_group['tasks']):
                    expected_task = current_group['tasks'][expected_idx]
                    if detected_task != expected_task:
                        return True, f"Grup içi sıra hatası: {expected_task} bekleniyor, {detected_task} tespit edildi"
            # Sırasız grup: tamamlanmış olup olmadığını kontrol et
            if detected_task in self.current_group_completed_tasks:
                return False, ""  # Tekrar yapılan görev, sorun değil
        
        return False, ""
        
    def yolo_detect(self, frame, expected_object_name: Optional[str] = None) -> Optional[str]:
        """
        YOLO ile nesne tespiti yapar.
        Mümkünse beklenen nesneyi, yoksa en yüksek confidence'a sahip nesneyi döndürür.
        """
        results = self.yolo_model(frame, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            self.last_detections = []
            return None, None

        all_boxes = results[0].boxes
        confidences_all = all_boxes.conf.cpu().numpy()
        classes_all = all_boxes.cls.cpu().numpy().astype(int)
        boxes_all = all_boxes.xyxy.cpu().numpy()

        all_detections = []
        best_expected_detection = None
        best_overall_detection = None
        max_overall_conf = 0.0

        for idx in range(len(confidences_all)):
            conf_val = float(confidences_all[idx])
            if conf_val < self.confidence_threshold:
                continue

            class_id_val = int(classes_all[idx])
            class_name_val = self.yolo_model.names[class_id_val]
            bbox_val = boxes_all[idx].tolist()

            current_detection_info = {
                'class': class_name_val,
                'confidence': conf_val,
                'bbox': bbox_val
            }
            all_detections.append(current_detection_info)

            if expected_object_name and class_name_val == expected_object_name:
                if best_expected_detection is None or conf_val > best_expected_detection['confidence']:
                    best_expected_detection = current_detection_info

            if conf_val > max_overall_conf:
                max_overall_conf = conf_val
                best_overall_detection = current_detection_info

        self.last_detections = all_detections

        if best_expected_detection:
            return best_expected_detection['class'], best_expected_detection

        if best_overall_detection:
            return best_overall_detection['class'], best_overall_detection

        return None, None
    
    def run_TETE(self, previous_object: str, frame_range: List[np.ndarray], duration: int):
        """TETE (Temporal Event Tracking Engine) simülasyonu"""
        tete_result = {
            'previous_object': previous_object,
            'duration_frames': duration,
            'frame_count': len(frame_range),
            'analysis_timestamp': time.time()
        }
        
        print(f"🔍 TETE Analysis - Object: {previous_object}, Duration: {duration} frames")
        return tete_result
    
    def detect_task_transition(self, frame_i: int, frame_fi: np.ndarray) -> Optional[Dict]:
        """
        Ana TSPD algoritması - Gruplu görev yönetimi ile görev geçişlerini tespit eder
        """
        self.frame_buffer.append(frame_fi.copy())
        
        # Mevcut grup bilgisi
        current_group = self.get_current_group()
        if current_group is None:
            return None
        
        # Beklenen görev (grup içinden henüz yapılmamış ilk görev)
        expected_name = None
        if current_group.get('strict_order', False):
            # Sıralı grup: sıradaki görevi bekle
            completed_count = len(self.current_group_completed_tasks)
            if completed_count < len(current_group['tasks']):
                expected_name = current_group['tasks'][completed_count]
        else:
            # Sırasız grup: herhangi bir tamamlanmamış görevi bekle
            remaining = set(current_group['tasks']) - self.current_group_completed_tasks
            expected_name = list(remaining)[0] if remaining else None
        
        # YOLO tespiti
        detected_name, current_detection_info = self.yolo_detect(frame_fi, expected_name)
        self.yolo_current = detected_name
        
        # Detection log
        log_entry = {
            'frame': frame_i,
            'detection': detected_name,
            'info': current_detection_info,
            'event': 'detection',
            'expected': expected_name,
            'current_task': self.current_task_name,
            'current_group': current_group['name'],
            'group_progress': f"{len(self.current_group_completed_tasks)}/{len(current_group['tasks'])}"
        }
        self.detection_log.append(log_entry)
        
        # Tespit yoksa - görev devam ediyor veya başlangıç durumu
        if detected_name is None:
            log_entry['event'] = 'no_detection'
            return None
        
        detected_norm = str(detected_name).strip().lower()
        
        # Grup ihlali kontrolü
        is_violation, violation_msg = self.check_group_violation(detected_norm)
        
        if is_violation:
            log_entry['event'] = 'group_violation'
            
            # Cooldown kontrolü
            if self.violation_cooldown_frames > 0:
                return None
            
            # Kararlılık kontrolü
            if self.stable_mismatch_label == detected_norm:
                self.stable_mismatch_count += 1
            else:
                self.stable_mismatch_label = detected_norm
                self.stable_mismatch_count = 1
            
            if self.stable_mismatch_count < self.min_mismatch_stability_frames:
                return None
            
            print(f"⚠️ GRUP İHLALİ! {violation_msg} - Frame {frame_i}")
            
            violation_record = {
                'frame': frame_i,
                'expected_group': current_group['name'],
                'detected': detected_norm,
                'violation_message': violation_msg,
                'timestamp': time.time()
            }
            self.group_violations.append(violation_record)
            log_entry['details'] = violation_record
            
            self.order_violation_message = violation_msg
            fps_val = getattr(self, 'video_fps', 30) or 30
            self.order_violation_frames_remaining = max(self.order_violation_frames_remaining, int(2 * fps_val))
            self.violation_cooldown_frames = int(2 * fps_val)
            return None
        
        # Geçerli tespit - grup içinde
        if self.is_task_in_current_group(detected_norm):
            # Mismatch kararlılık sıfırlama
            self.stable_mismatch_label = None
            self.stable_mismatch_count = 0
            
            # Yeni görev başlangıcı
            if self.current_task_name is None:
                log_entry['event'] = 'task_start'
                
                self.current_task_name = detected_norm
                self.last_active_task_name = detected_norm
                self.s = frame_i
                if self.n > 0:
                    self.n += 1
                else:
                    self.n = 1
                
                # Grup takibine ekle
                self.current_group_completed_tasks.add(detected_norm)
                
                # İlk görevi task_history'e ekle
                task_info = {
                    'task_number': self.n,
                    'start_frame': frame_i,
                    'end_frame': None,  # Henüz bitmedi
                    'duration': None,
                    'previous_object': None,
                    'current_object': detected_norm,
                    'tete_analysis': None,
                    'timestamp': time.time(),
                    'status': 'started',
                    'group': current_group['name']
                }
                self.task_history.append(task_info)
                
                print(f"🎯 Görev başladı - Frame {frame_i}: {detected_name} [{current_group['name']}]")
                print(f"   Grup ilerleme: {len(self.current_group_completed_tasks)}/{len(current_group['tasks'])}")
                
                # Grup tamamlandi mı kontrol et
                if self.is_current_group_completed():
                    print(f"✅ {current_group['name']} TAMAMLANDI!")
                    self.current_group_idx += 1
                    self.current_group_completed_tasks.clear()
                
                return None
            
            # Aynı görev devam ediyor
            elif self.current_task_name == detected_norm:
                log_entry['event'] = 'task_ongoing'
                return None
            
            # Görev geçişi (aynı grup içinde veya gruplar arası)
            else:
                # Sırasız gruplarda zaten yapılmış görevi sessizce yoksay
                if not current_group.get('strict_order', False) and detected_norm in self.current_group_completed_tasks:
                    return None
                
                log_entry['event'] = 'task_transition'
                
                Dn = frame_i - self.s
                previous_object = self.current_task_name
                current_object = detected_norm
                
                # Önceki görevin (task_history'deki son görev) end_frame ve duration değerlerini güncelle
                if self.task_history:
                    last_task = self.task_history[-1]
                    if last_task.get('end_frame') is None:
                        last_task['end_frame'] = frame_i
                        last_task['duration'] = Dn
                
                self.n += 1
                print(f"🎯 GÖREV DEĞİŞİMİ TESPİT EDİLDİ!")
                print(f"   Görev #{self.n}")
                print(f"   {previous_object} → {current_object}")
                print(f"   Süre: {Dn} frame")
                print(f"   Grup: {current_group['name']}")
                
                task_frames = self.frame_buffer[-(frame_i - self.s + 1):]
                tete_result = self.run_TETE(previous_object, task_frames, Dn)
                
                task_info = {
                    'task_number': self.n,
                    'start_frame': frame_i,
                    'end_frame': None,  # Sonraki geçişte güncellenecek
                    'duration': None,
                    'previous_object': previous_object,
                    'current_object': current_object,
                    'tete_analysis': tete_result,
                    'timestamp': time.time(),
                    'status': 'started',
                    'group': current_group['name']
                }
                self.task_history.append(task_info)
                
                self.s = frame_i
                self.current_task_name = current_object
                self.last_active_task_name = current_object
                
                # Grup takibine ekle
                self.current_group_completed_tasks.add(detected_norm)
                
                # Grup tamamlandi mı kontrol et
                if self.is_current_group_completed():
                    print(f"✅ {current_group['name']} TAMAMLANDI!")
                    self.current_group_idx += 1
                    self.current_group_completed_tasks.clear()
                
                if len(self.frame_buffer) > 100:
                    self.frame_buffer = self.frame_buffer[-50:]
                
                return task_info
        
        return None
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True, mask_path: Optional[str] = None) -> List[Dict]:
        """Video dosyasını işler ve görev geçişlerini tespit eder"""
        cap = cv2.VideoCapture(video_path) #video_path
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            raise ValueError(f"Video dosyası açılamadı: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video bilgileri:")
        print(f"   FPS: {fps}")
        print(f"   Toplam frame: {total_frames}")
        print(f"   Çözünürlük: {width}x{height}")
        print(f"   Süre: {total_frames/fps:.2f} saniye")
        print()
        
        # Maske yükleme
        mask_binary = None
        if mask_path:
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    if mask_img.shape != (height, width):
                        mask_img = cv2.resize(mask_img, (width, height))
                    _, mask_binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                    print(f"🗺️ Maske yüklendi: {mask_path}")
            except Exception as e:
                print(f"⚠️ Maske yüklenemedi: {e}")
        
        out_writer = None
        self.video_fps = fps
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        prev_time = time.time()
        fps_smooth = float(fps)
        
        FRAME_SKIP_RATE = 1
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                task_transition = None
                
                if frame_count % FRAME_SKIP_RATE == 0:
                    frame_for_detection = frame
                    if mask_binary is not None:
                        frame_for_detection = cv2.bitwise_and(frame, frame, mask=mask_binary)
                    
                    task_transition = self.detect_task_transition(frame_count, frame_for_detection)
                
                # Görselleştirme
                display_frame = frame.copy()
                if mask_binary is not None:
                    outside = cv2.bitwise_not(mask_binary)
                    dark = np.zeros_like(display_frame)
                    display_frame = cv2.add(
                        cv2.bitwise_and(display_frame, display_frame, mask=mask_binary),
                        cv2.bitwise_and(dark, dark, mask=outside)
                    )
                
                # FPS hesaplama
                now_time = time.time()
                dt = now_time - prev_time
                if dt > 0:
                    fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
                prev_time = now_time
                
                # Tespitleri çiz
                if self.last_detections:
                    for det in self.last_detections:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        label = f"{det['class']} {det['confidence']:.2f}"
                        color = self.class_colors.get(det['class'].lower(), (0, 255, 255))
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, label, (x1, max(0, y1 - 10)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # FPS ve Frame - Sağ tarafta
                fps_text = f"FPS: {fps_smooth:.1f}"
                frame_text = f"Frame: {frame_count}"
                
                # Metin boyutlarını hesapla ve sağa hizala
                fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                frame_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                
                cv2.putText(display_frame, fps_text, (width - fps_size[0] - 20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(display_frame, frame_text, (width - frame_size[0] - 20, 80),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Sol tarafta tüm görevleri listele
                y_position = 30
                line_height = 35
                
                # Tamamlanmış görevleri ve sürelerini takip et
                task_durations = {}
                for t in self.task_history:
                    task_name = t.get('current_object')
                    if task_name:
                        duration = t.get('duration')
                        if duration is not None and fps > 0:
                            duration_sec = duration / fps
                            task_durations[task_name] = duration_sec
                
                # İhlal yapılan görevleri takip et (grup bazlı)
                violated_groups = set()
                for v in self.group_violations:
                    violated_groups.add(v.get('expected_group'))
                
                # Tüm gruplar ve görevleri listele
                global_task_idx = 0
                for group_idx, group in enumerate(self.task_groups):
                    for task_name in group['tasks']:
                        global_task_idx += 1
                        
                        # Görevin durumunu kontrol et
                        is_completed = False
                        has_violation = False
                        
                        # Geçmiş gruplar tamamen tamamlanmış sayılır
                        if group_idx < self.current_group_idx:
                            is_completed = True
                            # Bu grupta ihlal var mı kontrol et
                            if group['name'] in violated_groups:
                                has_violation = True
                        # Mevcut grupta tamamlananları kontrol et
                        elif group_idx == self.current_group_idx:
                            is_completed = task_name in self.current_group_completed_tasks
                            if group['name'] in violated_groups:
                                has_violation = True
                        
                        # Süre bilgisi
                        duration_str = ""
                        if task_name in task_durations:
                            duration_str = f" ({task_durations[task_name]:.1f}s)"
                        
                        # Renk: Beyaz (başlamadı), Yeşil (doğru tamamlandı), Kırmızı (ihlal)
                        if is_completed:
                            if has_violation:
                                color = (0, 0, 255)  # Kırmızı (BGR) - ihlal
                            else:
                                color = (0, 255, 0)  # Yeşil (BGR) - doğru
                        else:
                            color = (255, 255, 255)  # Beyaz (BGR) - başlamadı
                        
                        # Görev metni
                        task_display = f"{global_task_idx}. {task_name}{duration_str}"
                        
                        cv2.putText(display_frame, task_display, (10, y_position),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_position += line_height
                
                # Geçiş vurgusu
                if task_transition:
                    cv2.rectangle(display_frame, (0, 0), (width, height), (0, 255, 255), 5)
                    cv2.putText(display_frame, "TASK TRANSITION!", (width//4, height//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                # İhlal uyarısı
                if self.order_violation_frames_remaining > 0:
                    cv2.rectangle(display_frame, (0, 0), (width, height), (0, 0, 255), 6)
                    msg = self.order_violation_message or "GRUP İHLALİ"
                    cv2.putText(display_frame, msg, (20, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                    self.order_violation_frames_remaining -= 1
                
                if self.violation_cooldown_frames > 0:
                    self.violation_cooldown_frames -= 1
                
                if display:
                    cv2.imshow('TSPD - Grouped Task Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if out_writer:
                    out_writer.write(display_frame)
                
                frame_count += 1
                
        finally:
            # Video bittiğinde son görevin süresini güncelle
            if self.task_history:
                last_task = self.task_history[-1]
                if last_task.get('end_frame') is None:
                    last_task['end_frame'] = frame_count - 1
                    last_task['duration'] = (frame_count - 1) - last_task['start_frame']
            
            cap.release()
            if out_writer:
                out_writer.release()
            if display:
                cv2.destroyAllWindows()
        
        return self.task_history
    
    def save_results(self, output_file: str):
        """
        Sonuçları JSON dosyasına kaydeder
        
        Args:
            output_file: Çıkış dosyasının yolu
        """
        fps_value = getattr(self, 'video_fps', None)
        tasks_with_durations = []
        total_duration_seconds = 0.0
        for t in self.task_history:
            duration_val = t.get('duration')
            duration_frames = int(duration_val) if duration_val is not None else 0
            duration_seconds = float(duration_frames) / float(fps_value) if fps_value and fps_value > 0 and duration_val is not None else None
            if duration_seconds is not None:
                total_duration_seconds += duration_seconds
            t_out = dict(t)
            t_out['duration_seconds'] = duration_seconds
            tasks_with_durations.append(t_out)
        
        # Toplam süre hesaplarken None değerleri filtrele
        total_duration_frames = sum(int(t.get('duration', 0)) for t in self.task_history if t.get('duration') is not None)
        
        results = {
            'total_tasks': self.n,
            'total_frames_processed': len(self.detection_log),
            'task_history': tasks_with_durations,
            'detection_log': self.detection_log,
            'order_violations': self.order_violations,
            'totals': {
                'total_duration_frames': total_duration_frames,
                'total_duration_seconds': total_duration_seconds if fps_value and fps_value > 0 else None,
                'video_fps': fps_value
            },
            'algorithm_settings': {
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Sonuçlar kaydedildi: {output_file}")
    
    def print_summary(self):
        """
        Analiz özetini yazdırır
        """
        print("\n" + "="*50)
        print("📊 TSPD ANALİZ ÖZETİ")
        print("="*50)
        print(f"🎯 Toplam tespit edilen görev: {self.n}")
        print(f"📝 İşlenen toplam frame: {len(self.detection_log)}")
        print()
        
        if self.task_history:
            print("📋 GÖREV DETAYLARI:")
            for task in self.task_history:
                print(f"   Görev #{task['task_number']}:")
                print(f"      Nesne değişimi: {task['previous_object']} → {task['current_object']}")
                print(f"      Frame aralığı: {task['start_frame']}-{task['end_frame']}")
                print(f"      Süre: {task['duration']} frame")
                print()


def main():
    """
    Ana fonksiyon - TSPD algoritmasını çalıştırır
    """
    # Kullanım örneği
    print("🚀 TSPD (Task Start Point Detection) Algoritması")
    print("="*60)
    
    # Model ve video yolları (bu kısımları kendi dosya yollarınızla değiştirin)
    YOLO_MODEL_PATH = "C:/Users/ali.donbaloglu/Desktop/Montaj_proces/Modeller/K_Fold/yolov8n_fold2/weights/best.pt"  # YOLO model dosyasının yolu
    VIDEO_PATH = "C:/Users/ali.donbaloglu/Desktop/Montaj_proces/input_video/part18_yeni.mp4"   # Video dosyasının yolu
    OUTPUT_VIDEO_PATH = "C:/Users/ali.donbaloglu/Desktop/Montaj_proces/output_video/part18_yeni_output_tspd.mp4"  # Çıkış video dosyası (opsiyonel)
    RESULTS_JSON_PATH = "C:/Users/ali.donbaloglu/Desktop/Montaj_proces/output_video/part18_yeni_tspd_results.json"  # Sonuçlar JSON dosyası
    MASK_PATH = "C:/Users/ali.donbaloglu/Desktop/Montaj_proces/input_video/mask_part1.png"  # Maske dosyasının yolu (opsiyonel - None olabilir)
    
    try:
        # TSPD detector'ı başlat
        detector = TaskStartPointDetector(
            yolo_model_path=YOLO_MODEL_PATH,
            confidence_threshold=0.6
        )
        
        print(f"✅ YOLO model yüklendi: {YOLO_MODEL_PATH}")
        print(f"📹 Video işlenecek: {VIDEO_PATH}")
        print()
        
        # Video işleme
        task_transitions = detector.process_video(
            video_path=VIDEO_PATH,
            output_path=OUTPUT_VIDEO_PATH,
            display=True,
            mask_path=MASK_PATH  # Maske dosyasını kullan
        )
        
        # Sonuçları göster
        detector.print_summary()
        
        # Sonuçları kaydet
        detector.save_results(RESULTS_JSON_PATH)
        
        print(f"\n✅ İşlem tamamlandi!")
        print(f"📊 {len(task_transitions)} görev geçişi tespit edildi.")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        print("Lütfen model ve video dosya yollarını kontrol edin.")


if __name__ == "__main__":
    main()