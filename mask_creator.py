"""
Maske Oluşturma Uygulaması
==========================
Video üzerinde maske alanı çizerek maske dosyası oluşturmanızı sağlar.

Kullanım:
---------
1. Video dosyasını seçin
2. Sol fare tuşu ile çizim yapın (basılı tutarak)
3. Sağ fare tuşu ile son çizimi geri alın
4. Klavye kısayolları:
   - 'c' : Tüm çizimleri temizle
   - 's' : Maskeyi kaydet
   - 'f' : Çizim alanını doldur (polygon modunda)
   - 'm' : Mod değiştir (serbest çizim / polygon)
   - '+' : Fırça boyutunu artır
   - '-' : Fırça boyutunu azalt
   - 'q' : Çıkış
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from datetime import datetime

class MaskCreator:
    def __init__(self):
        self.frame = None
        self.original_frame = None
        self.mask = None
        self.drawing = False
        self.points = []  # Polygon noktaları
        self.all_polygons = []  # Tüm polygon'lar
        self.freehand_points = []  # Serbest çizim noktaları
        self.all_freehand = []  # Tüm serbest çizimler
        self.brush_size = 15
        self.mode = "freehand"  # "freehand" veya "polygon"
        self.video_path = None
        self.output_dir = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Fare olaylarını işler"""
        if self.mode == "freehand":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.freehand_points = [(x, y)]
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.freehand_points.append((x, y))
                    # Anlık çizim
                    if len(self.freehand_points) >= 2:
                        cv2.line(self.frame, self.freehand_points[-2], self.freehand_points[-1], 
                                (0, 255, 0), self.brush_size)
                        cv2.line(self.mask, self.freehand_points[-2], self.freehand_points[-1], 
                                255, self.brush_size)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                if len(self.freehand_points) > 1:
                    self.all_freehand.append(self.freehand_points.copy())
                self.freehand_points = []
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Son serbest çizimi geri al
                if self.all_freehand:
                    self.all_freehand.pop()
                    self.redraw()
                    
        elif self.mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
                if len(self.points) > 1:
                    cv2.line(self.frame, self.points[-2], self.points[-1], (0, 255, 0), 2)
                    
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Son polygon noktasını geri al
                if self.points:
                    self.points.pop()
                    self.redraw()
                elif self.all_polygons:
                    self.all_polygons.pop()
                    self.redraw()
    
    def redraw(self):
        """Tüm çizimleri yeniden çizer"""
        self.frame = self.original_frame.copy()
        self.mask = np.zeros((self.frame.shape[0], self.frame.shape[1]), dtype=np.uint8)
        
        # Doldurulmuş polygon'ları çiz
        for polygon in self.all_polygons:
            pts = np.array(polygon, np.int32)
            cv2.fillPoly(self.mask, [pts], 255)
            cv2.fillPoly(self.frame, [pts], (0, 255, 0))
            # Yarı saydam overlay
            alpha = 0.4
            overlay = self.original_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            self.frame = cv2.addWeighted(overlay, alpha, self.original_frame, 1 - alpha, 0)
        
        # Serbest çizimleri çiz
        for freehand in self.all_freehand:
            for i in range(len(freehand) - 1):
                cv2.line(self.frame, freehand[i], freehand[i+1], (0, 255, 0), self.brush_size)
                cv2.line(self.mask, freehand[i], freehand[i+1], 255, self.brush_size)
        
        # Mevcut polygon noktalarını çiz
        for i, pt in enumerate(self.points):
            cv2.circle(self.frame, pt, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(self.frame, self.points[i-1], pt, (0, 255, 0), 2)
    
    def fill_polygon(self):
        """Mevcut polygon'u doldurur"""
        if len(self.points) >= 3:
            self.all_polygons.append(self.points.copy())
            self.points = []
            self.redraw()
            print("✅ Polygon dolduruldu!")
        else:
            print("⚠️ En az 3 nokta gerekli!")
    
    def clear_all(self):
        """Tüm çizimleri temizler"""
        self.points = []
        self.all_polygons = []
        self.all_freehand = []
        self.freehand_points = []
        self.frame = self.original_frame.copy()
        self.mask = np.zeros((self.frame.shape[0], self.frame.shape[1]), dtype=np.uint8)
        print("🗑️ Tüm çizimler temizlendi!")
    
    def save_mask(self):
        """Maskeyi kaydeder"""
        if self.mask is None:
            print("⚠️ Kaydedilecek maske yok!")
            return
        
        # Maske içinde beyaz alan var mı kontrol et
        if np.sum(self.mask) == 0:
            print("⚠️ Maske boş! Önce bir alan çizin.")
            return
        
        # Çıkış dizini
        if self.output_dir is None:
            self.output_dir = os.path.dirname(self.video_path)
        
        # Dosya adı oluştur
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_filename = f"mask_{video_name}_{timestamp}.png"
        mask_path = os.path.join(self.output_dir, mask_filename)
        
        # Kaydet
        cv2.imwrite(mask_path, self.mask)
        print(f"✅ Maske kaydedildi: {mask_path}")
        
        # Basit bir maske de kaydet (timestamp'siz)
        simple_mask_path = os.path.join(self.output_dir, f"mask_{video_name}.png")
        cv2.imwrite(simple_mask_path, self.mask)
        print(f"✅ Basit maske kaydedildi: {simple_mask_path}")
        
        return mask_path
    
    def select_video(self):
        """Video dosyası seçim penceresi açar"""
        root = tk.Tk()
        root.withdraw()  # Ana pencereyi gizle
        
        file_path = filedialog.askopenfilename(
            title="Video Dosyası Seçin",
            filetypes=[
                ("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("MP4 Dosyaları", "*.mp4"),
                ("AVI Dosyaları", "*.avi"),
                ("Tüm Dosyalar", "*.*")
            ],
            initialdir="C:/Users/ali.donbaloglu/Desktop/Montaj_proces/input_video"
        )
        
        root.destroy()
        return file_path
    
    def run(self, video_path=None):
        """Ana uygulama döngüsü"""
        # Video seçimi
        if video_path is None:
            video_path = self.select_video()
            
        if not video_path:
            print("❌ Video seçilmedi!")
            return
        
        self.video_path = video_path
        print(f"📹 Video yükleniyor: {video_path}")
        
        # Video aç
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Video açılamadı: {video_path}")
            return
        
        # İlk frame'i al
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Video frame'i okunamadı!")
            return
        
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        self.mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        height, width = frame.shape[:2]
        print(f"📐 Video boyutu: {width}x{height}")
        print()
        print("=" * 50)
        print("🎨 MASKE OLUŞTURMA ARACI")
        print("=" * 50)
        print(f"📌 Mevcut Mod: {self.mode.upper()}")
        print(f"🖌️ Fırça Boyutu: {self.brush_size}")
        print()
        print("Klavye Kısayolları:")
        print("  [Sol Tık]  : Çiz / Nokta ekle")
        print("  [Sağ Tık]  : Son çizimi geri al")
        print("  [M]        : Mod değiştir (serbest/polygon)")
        print("  [F]        : Polygon'u doldur")
        print("  [C]        : Tüm çizimleri temizle")
        print("  [S]        : Maskeyi kaydet")
        print("  [+]        : Fırça boyutunu artır")
        print("  [-]        : Fırça boyutunu azalt")
        print("  [Q/ESC]    : Çıkış")
        print("=" * 50)
        
        # Pencere oluştur
        window_name = "Maske Olusturma - Cizim yapin, S ile kaydedin"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            # Bilgi metni ekle
            display = self.frame.copy()
            
            # Yarı saydam panel oluştur (sağ üst köşede)
            panel_width = 280
            panel_height = 280
            panel_x = width - panel_width - 10
            panel_y = 10
            
            # Yarı saydam siyah arka plan
            overlay = display.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
            
            # Başlık
            cv2.putText(display, "KLAVYE KISAYOLLARI", (panel_x + 10, panel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Kısayol listesi
            shortcuts = [
                f"Mod: {self.mode.upper()}",
                f"Firca: {self.brush_size}",
                "-------------------",
                "[Sol Tik] Ciz/Nokta ekle",
                "[Sag Tik] Geri al",
                "[M] Mod degistir",
                "[F] Polygon doldur",
                "[C] Temizle",
                "[S] Kaydet",
                "[+/-] Firca boyutu",
                "[Q/ESC] Cikis"
            ]
            
            y_offset = panel_y + 55
            for shortcut in shortcuts:
                color = (255, 255, 255)
                if shortcut.startswith("Mod:"):
                    color = (0, 255, 0)  # Yeşil
                elif shortcut.startswith("Firca:"):
                    color = (255, 165, 0)  # Turuncu
                elif shortcut.startswith("---"):
                    color = (100, 100, 100)  # Gri
                    
                cv2.putText(display, shortcut, (panel_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 22
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q veya ESC
                break
                
            elif key == ord('s'):  # Kaydet
                self.save_mask()
                
            elif key == ord('c'):  # Temizle
                self.clear_all()
                
            elif key == ord('f'):  # Doldur
                self.fill_polygon()
                
            elif key == ord('m'):  # Mod değiştir
                if self.mode == "freehand":
                    self.mode = "polygon"
                else:
                    self.mode = "freehand"
                print(f"🔄 Mod değiştirildi: {self.mode.upper()}")
                
            elif key == ord('+') or key == ord('='):  # Fırça büyüt
                self.brush_size = min(50, self.brush_size + 2)
                print(f"🖌️ Fırça boyutu: {self.brush_size}")
                
            elif key == ord('-') or key == ord('_'):  # Fırça küçült
                self.brush_size = max(2, self.brush_size - 2)
                print(f"🖌️ Fırça boyutu: {self.brush_size}")
        
        cv2.destroyAllWindows()
        print("\n👋 Uygulama kapatıldı.")


def main():
    print("🚀 Maske Oluşturma Uygulaması Başlatılıyor...")
    print()
    
    creator = MaskCreator()
    creator.run()


if __name__ == "__main__":
    main()
