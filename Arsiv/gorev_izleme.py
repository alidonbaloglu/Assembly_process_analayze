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
        self.task_order = ["fikstur",
                           "hausing",
                            "pcb",
                            "buton_el",
                            "vidalama",
                            "test_konnektor",
                            "siyah_konnektor",
                            "buton_el",
                            "fikstur"]
        self.expected_task_idx = 0  # Beklenen bir sonraki görevin indeksini izler
        self.current_task_name = None  # Başlamış olan mevcut görev
        self.last_detections = []  # Son frame'deki tüm tespitler (görselleştirme için)
        # Sınıf bazlı renkler (BGR)
        self.class_colors = {
            'hausing': (0, 0, 255),    # Kırmızı
            'pcb': (0, 255, 0),        # Yeşil
            'vidalama': (255, 0, 0),   # Mavi
            'fikstur': (0, 165, 255),  # Turuncu
            'buton_el': (255, 0, 255), # Magenta
            'siyah_konnektor': (0, 128, 0), # Koyu yeşil
            'test_konnektor': (128, 0, 128) # Mor
        }
        
    def _initialize_task_definitions(self) -> Dict[str, Any]:
        """
        Görev tanımlama yapılarını başlatır
        """
        # Şimdilik basit bir başlangıç yapısı döndürüyoruz.
        # İleride "manual", "semantic" veya "contextual" modlarına göre
        # daha gelişmiş tanımlar eklenebilir.
        return {}
        
    def yolo_detect(self, frame) -> Optional[str]:
        """
        YOLO ile nesne tespiti yapar ve en yüksek confidence'a sahip nesneyi döndürür
        
        Args:
            frame: Video frame'i
            
        Returns:
            Tespit edilen nesne sınıfı (string) veya None
        """
        results = self.yolo_model(frame, verbose=False)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Tüm tespitleri sakla (görselleştirme için)
            all_boxes = results[0].boxes
            confidences_all = all_boxes.conf.cpu().numpy()
            classes_all = all_boxes.cls.cpu().numpy().astype(int)
            boxes_all = all_boxes.xyxy.cpu().numpy()
            
            all_detections = []
            for idx in range(len(confidences_all)):
                conf_val = float(confidences_all[idx])
                if conf_val < self.confidence_threshold:
                    continue
                class_id_val = int(classes_all[idx])
                class_name_val = self.yolo_model.names[class_id_val]
                bbox_val = boxes_all[idx].tolist()
                all_detections.append({
                    'class': class_name_val,
                    'confidence': conf_val,
                    'bbox': bbox_val
                })
            # Son frame için tespitleri sakla
            self.last_detections = all_detections
            
            if len(confidences_all) > 0:
                # En yüksek confidence'a sahip tespiti seç
                max_conf_idx = int(np.argmax(confidences_all))
                max_confidence = float(confidences_all[max_conf_idx])
                if max_confidence >= self.confidence_threshold:
                    class_id = int(classes_all[max_conf_idx])
                    class_name = self.yolo_model.names[class_id]
                    detection_info = {
                        'class': class_name,
                        'confidence': max_confidence,
                        'bbox': boxes_all[max_conf_idx].tolist()
                    }
                    return class_name, detection_info
        
        # Tespit yoksa boş liste tut
        self.last_detections = []
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
        
        Args:
            frame_i: Frame numarası
            frame_fi: Frame görüntüsü
            
        Returns:
            Görev geçiş bilgileri veya None
        """
        # Frame'i buffer'a ekle
        self.frame_buffer.append(frame_fi.copy())
        
        # Şu anki frame'de YOLO tespiti
        detected_name, current_detection_info = self.yolo_detect(frame_fi)
        self.yolo_current = detected_name
        
        # Detection log'a ekle (her frame için)
        self.detection_log.append({
            'frame': frame_i,
            'detection': detected_name,
            'info': current_detection_info,
            'event': 'detection'
        })
        
        # Yalnızca tanımlı görev sınıfları ve belirlenen sıraya göre ilerle
        if detected_name is None:
            return None
        
        detected_norm = str(detected_name).strip().lower()
        if self.expected_task_idx >= len(self.task_order):
            # Tüm görevler zaten başlatıldı/tamamlandı sayılır
            return None
        
        expected_name = self.task_order[self.expected_task_idx]
        if detected_norm != expected_name:
            # Sırada olmayan veya tekrar eden tespitleri yoksay
            return None
        
        # Beklenen görev tespit edildi
        if self.current_task_name is None:
            # İlk görev başlangıcı (örn. hausing)
            self.current_task_name = detected_norm
            self.s = frame_i
            self.n = 1
            self.detection_log.append({
                'frame': frame_i,
                'detection': detected_name,
                'info': current_detection_info,
                'event': 'task_start'
            })
            print(f"📍 İlk görev başladı - Frame {frame_i}: {detected_name}")
            
            # Sadece başlangıcı işaretle, geçiş yaratma
            self.expected_task_idx += 1
            return None
        
        # Buraya gelindiyse, bir sonraki beklenen görev tespit edilmiştir -> geçiş
        Dn = frame_i - self.s
        previous_object = self.current_task_name
        current_object = detected_norm
        
        self.n += 1
        print(f"🎯 GÖREV DEĞİŞİMİ TESPİT EDİLDİ!")
        print(f"   Görev #{self.n}")
        print(f"   {previous_object} → {current_object}")
        print(f"   Süre: {Dn} frame")
        print(f"   Frame aralığı: {self.s} - {frame_i}")
        
        # TETE analizi
        task_frames = self.frame_buffer[-(frame_i - self.s + 1):]
        tete_result = self.run_TETE(previous_object, task_frames, Dn)
        
        # Görev geçiş bilgisini kaydet
        task_info = {
            'task_number': self.n,
            'start_frame': self.s,
            'end_frame': frame_i,
            'duration': Dn,
            'previous_object': previous_object,
            'current_object': current_object,
            'tete_analysis': tete_result,
            'timestamp': time.time()
        }
        self.task_history.append(task_info)
        
        # Güncellemeler
        self.s = frame_i
        self.current_task_name = current_object
        self.expected_task_idx += 1
        
        # Buffer'ı temizle (bellek tasarrufu için)
        if len(self.frame_buffer) > 100:
            self.frame_buffer = self.frame_buffer[-50:]
        
        return task_info
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True, mask_path: Optional[str] = None) -> List[Dict]:
        """
        Video dosyasını işler ve görev geçişlerini tespit eder
        
        Args:
            video_path: Video dosyasının yolu
            output_path: Çıkış video dosyasının yolu (opsiyonel)
            display: Video gösterimini açık/kapalı
            mask_path: İsteğe bağlı maske yolu (beyaz alanlar işlenecek)
            
        Returns:
            Tespit edilen görevlerin listesi
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
                
                # Görev numarası ve adı (sol üstte object yazısı kaldırıldı)
                task_text = "Task: 0"
                if self.current_task_name:
                    task_text = f"Task: {self.n} - {self.current_task_name}"
                cv2.putText(display_frame,
                          task_text,
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
                print()


def main():
    """
    Ana fonksiyon - TSPD algoritmasını çalıştırır
    """
    # Kullanım örneği
    print("🚀 TSPD (Task Start Point Detection) Algoritması")
    print("="*60)
    
    # Model ve video yolları (bu kısımları kendi dosya yollarınızla değiştirin)
    YOLO_MODEL_PATH = "C:/Users/ali.donbaloglu/Desktop/Montaj_proces/Modeller/montaj_full_m_buton.pt"  # YOLO model dosyasının yolu
    VIDEO_PATH = "C:/Users/ali.donbaloglu/Desktop/Montaj_proces/input_video/part10.mp4"   # Video dosyasının yolu
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