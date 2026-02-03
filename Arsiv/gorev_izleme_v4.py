import cv2
import numpy as np
from ultralytics import YOLO
import json
import time
from typing import Dict, List, Optional, Tuple, Any

class TaskStartPointDetector:
    """
    Task Start Point Detection (TSPD) Algoritması
    
    Video akışında nesne değişimlerini ve hareket modellerini analiz ederek
    gerçek görev geçişlerini tespit eder.
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
        self.object_stability_buffer = []  # Nesne kararlılığı için
        self.min_stability_frames = 3  # Minimum kararlı frame sayısı
        
        # Sıralı görev takibi: yalnızca ilk tespitler önemlidir
        self.task_order = ["fixture", # bazı görevelri gruplamak lazım grup içinde sıra değişiklikleri küçük hata olarak algılansın ama hepsinin yapılması kontrol edilsin
                           "reflector",
                            "pcb",
                            "button_hand",
                            "screwing",
                            "test_connector",
                            "black_connector",
                            "button_hand",
                            "power_connector",
                            "fixture"]
        self.expected_task_idx = 0  # Beklenen bir sonraki görevin indeksini izler
        self.current_task_name = None  # Başlamış olan mevcut görev
        self.last_detections = []  # Son frame'deki tüm tespitler (görselleştirme için)
        # Sınıf bazlı renkler (BGR)
        self.class_colors = {
            'reflector': (0, 0, 255),    # Kırmızı
            'pcb': (0, 255, 0),        # Yeşil
            'screwing': (255, 0, 0),   # Mavi
            'fixture': (0, 165, 255),  # Turuncu
            'button_hand': (255, 0, 255), # Magenta
            'black_connector': (0, 128, 0), # Koyu yeşil
            'test_connector': (128, 0, 128), # Mor
            "power_connector" : (0,255,255),
        }
        # Sıra ihlali uyarıları
        self.order_violations = []
        self.order_violation_frames_remaining = 0
        self.order_violation_message = None
        # Uyarı debounce ve kararlılık takibi
        self.violation_cooldown_frames = 0
        self.stable_mismatch_label = None
        self.stable_mismatch_count = 0
        self.min_mismatch_stability_frames = 3 
        
        # Görev bitişi ve bekleme takibi
        self.waiting = False
        self.wait_start_frame = None
        self.last_task_index = None
        self.wait_periods = []
        self.wait_min_seconds = 2.0  # Bekleme kabul eşiği (saniye)
        self.last_active_task_name = None  # Alt eşik beklemede ekranda göstermek için
        
    def _initialize_task_definitions(self) -> Dict[str, Any]:
        """
        Görev tanımlama yapılarını başlatır
        """
        # Şimdilik basit bir başlangıç yapısı döndürüyoruz.
        # İleride "manual", "semantic" veya "contextual" modlarına göre
        # daha gelişmiş tanımlar eklenebilir.
        return {}
        
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

            # 1. Beklenen nesneyi ara
            if expected_object_name and class_name_val == expected_object_name:
                if best_expected_detection is None or conf_val > best_expected_detection['confidence']:
                    best_expected_detection = current_detection_info

            # 2. Genel olarak en iyi nesneyi de takip et
            if conf_val > max_overall_conf:
                max_overall_conf = conf_val
                best_overall_detection = current_detection_info

        # Son tespitleri çizim için sakla
        self.last_detections = all_detections

        # Karar: Beklenen nesne bulunduysa onu döndür
        if best_expected_detection:
            return best_expected_detection['class'], best_expected_detection

        # Beklenen nesne bulunamadıysa, genel olarak en iyiyi döndür
        # (Bu, sıra hatası tespiti için gereklidir)
        if best_overall_detection:
            return best_overall_detection['class'], best_overall_detection

        return None, None
    
    def run_TETE(self, previous_object: str, frame_range: List[np.ndarray], duration: int):
        """
        TETE (Temporal Event Tracking Engine) simülasyonu
        
        Args:
            previous_object: Önceki görevdeki nesne
            frame_range: Görev süresince olan frame'ler
            duration: Görev süresi (frame sayısı)
        """
        # TETE analizi (burada basit bir implementasyon)
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
        Ana TSPD algoritması - Görev geçişlerini tespit eder
        (Senaryo 1'i çözmek için mantığı yeniden yapılandırılmış versiyon)
        
        Args:
            frame_i: Frame numarası
            frame_fi: Frame görüntüsü
            
        Returns:
            Görev geçiş bilgileri veya None
        """
        # Frame'i buffer'a ekle
        self.frame_buffer.append(frame_fi.copy())
        
        # 1. Beklenen nesneyi ÖNCE belirle
        expected_name = None
        if self.expected_task_idx < len(self.task_order):
            expected_name = self.task_order[self.expected_task_idx]

        # 2. YOLO'yu beklenen nesneye öncelik vererek çağır
        detected_name, current_detection_info = self.yolo_detect(frame_fi, expected_name)
        self.yolo_current = detected_name
        
        # Detection log'a ekle (her frame için)
        log_entry = {
            'frame': frame_i,
            'detection': detected_name,
            'info': current_detection_info,
            'event': 'detection',
            'expected': expected_name,
            'current_task': self.current_task_name
        }
        self.detection_log.append(log_entry)
        
        # 3. YENİ MANTIK AKIŞI
        
        # DURUM A: Hiçbir nesne tespit edilmedi (veya eşiğin altında)
        if detected_name is None:
            log_entry['event'] = 'no_detection'
            # Eğer aktif bir görev varsa, onu bitir ve beklemeye geç
            if self.current_task_name is not None and not self.waiting:
                Dn = frame_i - self.s
                previous_object = self.current_task_name
                task_frames = self.frame_buffer[-(frame_i - self.s + 1):]
                tete_result = self.run_TETE(previous_object, task_frames, Dn)
                task_info = {
                    'task_number': self.n if self.n > 0 else 1,
                    'start_frame': self.s,
                    'end_frame': frame_i,
                    'duration': Dn,
                    'previous_object': previous_object,
                    'current_object': None,
                    'tete_analysis': tete_result,
                    'timestamp': time.time(),
                    'status': 'completed_no_next'
                }
                self.task_history.append(task_info)
                self.last_task_index = len(self.task_history) - 1
                # Bekleme başlasın
                self.waiting = True
                self.wait_start_frame = frame_i
                self.last_active_task_name = previous_object
                self.current_task_name = None
                self.s = frame_i
                
                log_entry['event'] = 'task_end_object_lost'
                log_entry['details'] = {'ended_object': previous_object}
            return None

        # Buraya geldiysek, bir nesne tespit edildi
        detected_norm = str(detected_name).strip().lower()
        if self.expected_task_idx >= len(self.task_order):
            # Tüm görevler zaten başlatıldı/tamamlandı sayılır
            return None
        
        # DURUM B: Tespit edilen nesne, BEKLENEN nesne (Doğru sıra)
        if detected_norm == expected_name:
            # Mismatch kararlılık takibini sıfırla (doğru sıraya dönüldü)
            self.stable_mismatch_label = None
            self.stable_mismatch_count = 0
            
            if self.current_task_name is None:
                # B.1: Bu, 'bekleme' (waiting) sonrası YENİ BİR GÖREVİN BAŞLANGICI
                log_entry['event'] = 'task_start'
                
                if self.waiting and self.last_task_index is not None:
                    wait_frames = frame_i - (self.wait_start_frame or frame_i)
                    try:
                        fps_val = getattr(self, 'video_fps', 30) or 30
                        wait_seconds = float(wait_frames) / float(fps_val) if fps_val > 0 else 0.0
                        wait_min = float(getattr(self, 'wait_min_seconds', 2.0))
                        if wait_seconds >= wait_min:
                            adj_seconds = max(0.0, wait_seconds - wait_min)
                            adj_frames = int(round(adj_seconds * float(fps_val))) if fps_val > 0 else 0
                            self.task_history[self.last_task_index]['waiting_after_frames'] = adj_frames
                            self.task_history[self.last_task_index]['waiting_after_seconds'] = adj_seconds
                            self.task_history[self.last_task_index]['waiting_after_frames_raw'] = int(wait_frames)
                            self.task_history[self.last_task_index]['waiting_after_seconds_raw'] = wait_seconds
                            self.wait_periods.append({
                                'task_index': self.last_task_index,
                                'task_number': self.task_history[self.last_task_index].get('task_number'),
                                'start_frame': int(self.wait_start_frame or frame_i),
                                'end_frame': int(frame_i),
                                'duration_frames_raw': int(wait_frames),
                                'duration_seconds_raw': wait_seconds,
                                'duration_frames': adj_frames,
                                'duration_seconds': adj_seconds
                            })
                        if self.task_history[self.last_task_index].get('current_object') is None:
                            self.task_history[self.last_task_index]['current_object'] = detected_norm
                    except Exception:
                        pass
                    self.waiting = False
                    self.wait_start_frame = None
                
                # Görev başlangıcını işaretle
                self.current_task_name = detected_norm
                self.last_active_task_name = detected_norm
                self.s = frame_i
                if self.n > 0: self.n += 1
                else: self.n = 1
                
                if self.n == 1: print(f"📍 İlk görev başladı - Frame {frame_i}: {detected_name}")
                else: print(f"📍 Yeni görev başladı - Frame {frame_i}: {detected_name}")
                
                self.expected_task_idx += 1
                return None # Bu bir 'başlangıç', 'geçiş' değil
            
            elif self.current_task_name == detected_norm:
                # B.2: Halen aynı görevin içindeyiz, bir şey yapma
                log_entry['event'] = 'task_ongoing'
                return None
                
            else:
                # B.3: Bu, 'reflector'dan 'pcb'ye GÖREV GEÇİŞİ (SENARYO 1'İN ÇÖZÜMÜ)
                log_entry['event'] = 'task_transition'
                
                Dn = frame_i - self.s
                previous_object = self.current_task_name
                current_object = detected_norm
                
                self.n += 1
                print(f"🎯 GÖREV DEĞİŞİMİ TESPİT EDİLDİ!")
                print(f"   Görev #{self.n}")
                print(f"   {previous_object} → {current_object}")
                print(f"   Süre: {Dn} frame")
                print(f"   Frame aralığı: {self.s} - {frame_i}")
                
                task_frames = self.frame_buffer[-(frame_i - self.s + 1):]
                tete_result = self.run_TETE(previous_object, task_frames, Dn)
                
                task_info = {
                    'task_number': self.n,
                    'start_frame': self.s,
                    'end_frame': frame_i,
                    'duration': Dn,
                    'previous_object': previous_object,
                    'current_object': current_object,
                    'tete_analysis': tete_result,
                    'timestamp': time.time(),
                    'status': 'completed_transition' # Yeni status eklendi
                }
                self.task_history.append(task_info)
                self.last_task_index = len(self.task_history) - 1 # Olası bir 'wait' için indeksi ayarla
                
                # Güncellemeler
                self.s = frame_i
                self.current_task_name = current_object
                self.last_active_task_name = current_object
                self.expected_task_idx += 1
                
                if len(self.frame_buffer) > 100:
                    self.frame_buffer = self.frame_buffer[-50:]
                
                return task_info

        # DURUM C: Tespit edilen nesne, BEKLENEN nesne DEĞİL (Sıra Hatası)
        else:
            log_entry['event'] = 'order_violation'
            
            # Tamamlanmış aşamalara ait tekrar tespitleri sessizce yoksay
            violation_type = "bilinmeyen"
            try:
                detected_idx = self.task_order.index(detected_norm)
                if detected_idx < self.expected_task_idx:
                    # Önceki aşamaya ait tekrar tespiti: uyarı verme, sadece yoksay
                    log_entry['event'] = 'past_task_ignored'
                    return None
                elif detected_idx > self.expected_task_idx:
                    violation_type = "sirayi atlama"
            except ValueError:
                detected_idx = None
                violation_type = "tanimsiz sinif"

            # Uyarı tekrarını önlemek için cooldown uygula
            if self.violation_cooldown_frames and self.violation_cooldown_frames > 0:
                return None

            # Kararlılık kontrolü
            if self.stable_mismatch_label == detected_norm:
                self.stable_mismatch_count += 1
            else:
                self.stable_mismatch_label = detected_norm
                self.stable_mismatch_count = 1

            if self.stable_mismatch_count < self.min_mismatch_stability_frames:
                return None # Henüz kararlı değil

            warn_msg = f"SIRA HATASI! Beklenen: {expected_name}, Tespit: {detected_norm} ({violation_type})"
            print(f"⚠️ {warn_msg} - Frame {frame_i}")

            violation_record = {
                'frame': frame_i,
                'expected': expected_name,
                'detected': detected_norm,
                'violation_type': violation_type,
                'timestamp': time.time()
            }
            self.order_violations.append(violation_record)
            log_entry['details'] = violation_record
            
            # Görsel uyarı
            self.order_violation_message = warn_msg
            fps_val = getattr(self, 'video_fps', 30) or 30
            self.order_violation_frames_remaining = max(self.order_violation_frames_remaining, int(2 * fps_val))
            self.violation_cooldown_frames = int(2 * fps_val)
            return None
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True, mask_path: Optional[str] = None) -> List[Dict]:
        """
        Video dosyasını işler ve görev geçişlerini tespit eder
        
        Args:
            video_path: Video dosyasının yolu
            output_path: Çıkış video dosyasının yolu (opsiyonel)
            display: Video gösterimini açık/kapalı
            mask_path: İsteğe bağlı maske yolu (beyaz alanlar işlenecek)
            
        Returns:
            Tespit edilen görevlerin listesi video_path
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video dosyası açılamadı: {video_path}")
        
        # Video bilgileri
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
        
        # Maske yükle (opsiyonel). Beyaz alanlar işlenecek, siyah alanlar yoksayılacak.
        mask_binary = None
        try:
            if mask_path is None:
                mask_path = "mask2.png"
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if mask_img.shape[1] != width or mask_img.shape[0] != height:
                    mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
                _, mask_binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                print(f"🗺️ Maske yüklendi: {mask_path}")
            else:
                print(f"⚠️ Maske okunamadı: {mask_path}. Maske olmadan devam ediliyor.")
        except Exception:
            print("⚠️ Maske yükleme sırasında hata. Maske olmadan devam ediliyor.")
            mask_binary = None
        
        # Çıkış video yazıcısı (opsiyonel)
        out_writer = None
        # FPS bilgisini sınıf seviyesinde sakla (JSON için saniye hesaplama)
        self.video_fps = fps
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        prev_time = time.time()
        fps_smooth = float(fps) if fps > 0 else 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Maske içini işle (tespit için)
                frame_for_detection = frame
                if mask_binary is not None:
                    frame_for_detection = cv2.bitwise_and(frame, frame, mask=mask_binary)
                
                # TSPD algoritmasını çalıştır
                task_transition = self.detect_task_transition(frame_count, frame_for_detection)
                
                # Görselleştirme
                display_frame = frame.copy()
                if mask_binary is not None:
                    outside = cv2.bitwise_not(mask_binary)
                    dark = np.zeros_like(display_frame)
                    # Dış bölgeyi karart
                    display_frame = cv2.add(
                        cv2.bitwise_and(display_frame, display_frame, mask=mask_binary),
                        cv2.bitwise_and(dark, dark, mask=outside)
                    )
                
                # Anlık FPS hesapla (işleme FPS'i) ve yumuşat
                now_time = time.time()
                dt = now_time - prev_time
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps
                prev_time = now_time
                
                # Tüm tespitleri çiz (bbox + etiket)
                if hasattr(self, 'last_detections') and self.last_detections:
                    for det in self.last_detections:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        label = f"{det['class']} {det['confidence']:.2f}"
                        color = self.class_colors.get(str(det['class']).lower(), (0, 255, 255))
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame,
                                    label,
                                    (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Görev numarası ve adı / bekleme göstergesi
                task_text = "Task: 0"
                if self.current_task_name:
                    task_text = f"Task: {self.n} - {self.current_task_name}"
                elif getattr(self, 'waiting', False) and getattr(self, 'last_task_index', None) is not None:
                    wait_frames_raw = frame_count - (self.wait_start_frame or frame_count)
                    fps_val = getattr(self, 'video_fps', 30) or 30
                    wait_seconds_raw = (float(wait_frames_raw) / float(fps_val)) if fps_val > 0 else 0.0
                    # Ekranda yalnızca eşik ve üstünü göster, ilk 2 saniyeyi düş
                    if wait_seconds_raw >= float(getattr(self, 'wait_min_seconds', 2.0)):
                        adj_seconds = max(0.0, wait_seconds_raw - float(getattr(self, 'wait_min_seconds', 2.0)))
                        adj_frames = int(round(adj_seconds * float(fps_val))) if fps_val > 0 else 0
                        last_task_num = self.task_history[self.last_task_index].get('task_number', '?') if self.task_history else '?'
                        task_text = f"Task: {last_task_num} - delay {adj_frames}f ({adj_seconds:.1f}s)"
                    else:
                        # Eşik altı ise mevcut son görevi göstermeye devam edelim
                        last_task_num = self.task_history[self.last_task_index].get('task_number', '?') if self.task_history else (self.n or 0)
                        last_task_name = self.last_active_task_name or self.current_task_name or ""
                        task_text = f"Task: {last_task_num} - {last_task_name}" if last_task_name else ""
                cv2.putText(display_frame,
                          task_text if task_text else (f"Task: {self.n} - {self.current_task_name}" if self.current_task_name else ""),
                          (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # FPS bilgisini göster (işleme FPS'i)
                cv2.putText(display_frame,
                          f"FPS: {fps_smooth:.1f}",
                          (10, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Frame numarasını göster
                cv2.putText(display_frame, 
                          f"Frame: {frame_count}", 
                          (10, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Görev geçişi varsa vurgula
                if task_transition:
                    cv2.rectangle(display_frame, (0, 0), (width, height), (0, 255, 255), 5)
                    cv2.putText(display_frame, 
                              "TASK TRANSITION!", 
                              (width//4, height//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

                # Sıra ihlali görsel uyarısı
                if self.order_violation_frames_remaining and self.order_violation_frames_remaining > 0:
                    cv2.rectangle(display_frame, (0, 0), (width, height), (0, 0, 255), 6)
                    msg = self.order_violation_message or "SIRA HATASI"
                    cv2.putText(display_frame,
                                msg,
                                (max(10, width//20), max(50, height//10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                    self.order_violation_frames_remaining -= 1
                # Violation cooldown sayacı
                if self.violation_cooldown_frames and self.violation_cooldown_frames > 0:
                    self.violation_cooldown_frames -= 1
                
                # Video gösterimi
                if display:
                    cv2.imshow('TSPD - Task Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Kullanıcı tarafından durduruldu.")
                        break
                
                # Çıkış videosuna kaydet
                if out_writer:
                    out_writer.write(display_frame)
                
                frame_count += 1
                
                # Progress bar
                if frame_count % (total_frames // 20) == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"⏳ İlerleme: {progress:.1f}%")
                    
        except KeyboardInterrupt:
            print("\n⚠️ İşlem kullanıcı tarafından durduruldu.")
        
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            if display:
                cv2.destroyAllWindows()
            # Video sonunda açık bekleme varsa kapat ve kaydet
            if getattr(self, 'waiting', False) and self.last_task_index is not None:
                end_frame_final = max(0, frame_count - 1)
                wait_frames_final = end_frame_final - (self.wait_start_frame or end_frame_final)
                if wait_frames_final < 0:
                    wait_frames_final = 0
                fps_val = getattr(self, 'video_fps', 30) or 30
                try:
                    wait_seconds_final = float(wait_frames_final) / float(fps_val) if fps_val > 0 else 0.0
                    wait_min = float(getattr(self, 'wait_min_seconds', 2.0))
                    if wait_seconds_final >= wait_min:
                        adj_seconds_f = max(0.0, wait_seconds_final - wait_min)
                        adj_frames_f = int(round(adj_seconds_f * float(fps_val))) if fps_val > 0 else 0
                        self.task_history[self.last_task_index]['waiting_after_frames'] = adj_frames_f
                        self.task_history[self.last_task_index]['waiting_after_seconds'] = adj_seconds_f
                        # Ham değerleri de ekle
                        self.task_history[self.last_task_index]['waiting_after_frames_raw'] = int(wait_frames_final)
                        self.task_history[self.last_task_index]['waiting_after_seconds_raw'] = wait_seconds_final
                        self.wait_periods.append({
                            'task_index': self.last_task_index,
                            'task_number': self.task_history[self.last_task_index].get('task_number'),
                            'start_frame': int(self.wait_start_frame or end_frame_final),
                            'end_frame': int(end_frame_final),
                            'duration_frames_raw': int(wait_frames_final),
                            'duration_seconds_raw': wait_seconds_final,
                            'duration_frames': adj_frames_f,
                            'duration_seconds': adj_seconds_f,
                            'closed_on_video_end': True
                        })
                except Exception:
                    pass
                self.waiting = False
                self.wait_start_frame = None
        
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
            duration_frames = int(t.get('duration', 0))
            duration_seconds = float(duration_frames) / float(fps_value) if fps_value and fps_value > 0 else None
            if duration_seconds is not None:
                total_duration_seconds += duration_seconds
            t_out = dict(t)
            t_out['duration_seconds'] = duration_seconds
            tasks_with_durations.append(t_out)
        
        results = {
            'total_tasks': self.n,
            'total_frames_processed': len(self.detection_log),
            'task_history': tasks_with_durations,
            'detection_log': self.detection_log,
            'order_violations': self.order_violations,
            'wait_periods': self.wait_periods,
            'totals': {
                'total_duration_frames': sum(int(t.get('duration', 0)) for t in self.task_history),
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
                if 'waiting_after_frames' in task:
                    print(f"      Bekleme: {task['waiting_after_frames']} frame")
                print()
        if getattr(self, 'wait_periods', None):
            print("⏱️ BEKLEME PERIYOTLARI:")
            for wp in self.wait_periods:
                print(f"   Görev #{wp.get('task_number')}: {wp.get('start_frame')} - {wp.get('end_frame')} ({wp.get('duration_frames')} frame)")
                print()


def main():
    """
    Ana fonksiyon - TSPD algoritmasını çalıştırır
    """
    # Kullanım örneği
    print("🚀 TSPD (Task Start Point Detection) Algoritması")
    print("="*60)
    
    # Model ve video yolları (bu kısımları kendi dosya yollarınızla değiştirin)
    YOLO_MODEL_PATH = "Modeller/Makale/Yolov11/M_model/runs/detect/train/weights/best.pt"  # YOLO model dosyasının yolu
    VIDEO_PATH = "C:/Users/ali.donbaloglu/Desktop/Montaj_proces/input_video/eski/part10.mp4"   # Video dosyasının yolu
    OUTPUT_VIDEO_PATH = "output_tspd.mp4"  # Çıkış video dosyası (opsiyonel)
    RESULTS_JSON_PATH = "tspd_results.json"  # Sonuçlar JSON dosyası
    
    try:
        # TSPD detector'ı başlat
        detector = TaskStartPointDetector(
            yolo_model_path=YOLO_MODEL_PATH,
            confidence_threshold=0.7
        )
        
        print(f"✅ YOLO model yüklendi: {YOLO_MODEL_PATH}")
        print(f"📹 Video işlenecek: {VIDEO_PATH}")
        print()
        
        # Video işleme
        task_transitions = detector.process_video(
            video_path=VIDEO_PATH,
            output_path=OUTPUT_VIDEO_PATH,
            display=True
        )
        
        # Sonuçları göster
        detector.print_summary()
        
        # Sonuçları kaydet
        detector.save_results(RESULTS_JSON_PATH)
        
        print(f"\n✅ İşlem tamamlandı!")
        print(f"📊 {len(task_transitions)} görev geçişi tespit edildi.")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        print("Lütfen model ve video dosya yollarını kontrol edin.")


if __name__ == "__main__":
    main()