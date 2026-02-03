import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import json
import os
from datetime import datetime
from pathlib import Path


class VideoTaskLabeler:
    """Video üzerinde görev etiketleme arayüzü"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("TSPD Görev Etiketleme Arayüzü")
        self.root.geometry("1400x900")
        
        # Video değişkenleri
        self.video_path = None
        self.video_capture = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_image = None
        self.updating_slider = False  # Sonsuz döngü önleme bayrağı
        
        # Etiketleme değişkenleri
        self.tasks = []
        self.current_task = None
        self.task_counter = 1
        
        # Nesne tipleri
        self.object_types = [
            "fixture",
            "reflector",
            "pcb",
            "button_hand",
            "screwing",
            "test_connector",
            "black_connector",
            "power_connector"
        ]
        
        # Görev tanımları (task_groups'taki sıraya göre)
        # Görev sırasına göre atamak için sıralı liste
        self.task_groups_order = [
            {'name': 'Fikstur', 'tasks': ['fixture']},
            {'name': 'Reflektor', 'tasks': ['reflector']},
            {'name': 'PCB', 'tasks': ['pcb']},
            {'name': 'Buton', 'tasks': ['button_hand']},
            {'name': 'Vidalama', 'tasks': ['screwing']},
            {'name': 'Konnektorler', 'tasks': ['test_connector', 'black_connector']},
            {'name': 'Buton', 'tasks': ['button_hand']},
            {'name': 'Power konnektor', 'tasks': ['power_connector']},
            {'name': 'Fikstur', 'tasks': ['fixture']}
        ]
        
        # Varsayılan görev adı eşleştirmesi (basit durumlar için)
        self.group_names = {
            "fixture": "Fikstur",
            "reflector": "Reflektor",
            "pcb": "PCB",
            "button_hand": "Buton",
            "screwing": "Vidalama",
            "test_connector": "Konnektorler",
            "black_connector": "Konnektorler",
            "power_connector": "Power konnektor"
        }
        
        self.create_widgets()
    
    def _get_next_group_for_object(self, object_name: str) -> str:
        """
        Nesne için sıradaki uygun grubu belirler.
        Aynı nesne birden fazla grupta varsa (örn: button_hand Grup 4 ve 7'de),
        daha önce kullanılmamış sıradaki grubu seçer.
        """
        # Bu nesneyi içeren grupları bul
        matching_groups = []
        for idx, group in enumerate(self.task_groups_order):
            if object_name in group['tasks']:
                matching_groups.append((idx, group['name']))
        
        if not matching_groups:
            return f"Bilinmeyen Grup"
        
        # Daha önce bu nesne için hangi gruplar kullanılmış?
        used_groups_for_object = set()
        for task in self.tasks:
            if task['current_object'] == object_name:
                used_groups_for_object.add(task['group'])
        
        # Kullanılmamış ilk grubu bul
        for idx, group_name in matching_groups:
            if group_name not in used_groups_for_object:
                return group_name
        
        # Tüm gruplar kullanılmışsa, varsayılan olarak ilk grubu döndür
        return matching_groups[0][1]
        
    def create_widgets(self):
        """Arayüz bileşenlerini oluştur"""
        
        # Ana frame'ler
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        video_frame = ttk.Frame(self.root, padding="10")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Grid ağırlıkları
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)
        
        # ============ ÜST PANEL (Video Yükleme) ============
        ttk.Button(top_frame, text="📁 Video Yükle", 
                  command=self.load_video, width=20).pack(side=tk.LEFT, padx=5)
        
        self.video_info_label = ttk.Label(top_frame, 
                                          text="Video yüklenmedi", 
                                          foreground="gray")
        self.video_info_label.pack(side=tk.LEFT, padx=20)
        
        # ============ VIDEO GÖRÜNTÜLEME PANELİ ============
        self.video_label = ttk.Label(video_frame, text="Video yüklendiğinde burada görünecek",
                                     background="black", foreground="white",
                                     anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # ============ KONTROL PANELİ ============
        
        # Frame kontrolü
        frame_control = ttk.LabelFrame(control_frame, text="Frame Kontrolü", padding="10")
        frame_control.pack(fill=tk.X, pady=5)
        
        # Frame slider
        slider_frame = ttk.Frame(frame_control)
        slider_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, 
                                      orient=tk.HORIZONTAL,
                                      command=self.on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.frame_number_label = ttk.Label(slider_frame, text="0 / 0", width=15)
        self.frame_number_label.pack(side=tk.LEFT)
        
        # Frame butonları
        button_frame = ttk.Frame(frame_control)
        button_frame.pack(pady=5)
        
        ttk.Button(button_frame, text="⏮ İlk Frame", 
                  command=self.goto_first_frame, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="◀ -10 Frame", 
                  command=lambda: self.skip_frames(-10), width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="◀ Geri", 
                  command=lambda: self.skip_frames(-1), width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="İleri ▶", 
                  command=lambda: self.skip_frames(1), width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="+10 Frame ▶", 
                  command=lambda: self.skip_frames(10), width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Son Frame ⏭", 
                  command=self.goto_last_frame, width=12).pack(side=tk.LEFT, padx=2)
        
        # Frame atlama
        jump_frame = ttk.Frame(frame_control)
        jump_frame.pack(pady=5)
        
        ttk.Label(jump_frame, text="Frame'e Git:").pack(side=tk.LEFT)
        self.jump_entry = ttk.Entry(jump_frame, width=10)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(jump_frame, text="Git", 
                  command=self.jump_to_frame).pack(side=tk.LEFT)
        
        # ============ SAĞ PANEL (Etiketleme) ============
        
        # Görev etiketleme
        labeling_frame = ttk.LabelFrame(right_frame, text="Görev Etiketleme", padding="10")
        labeling_frame.pack(fill=tk.X, pady=5)
        
        # Nesne seçimi
        ttk.Label(labeling_frame, text="Nesne Tipi:").pack(anchor=tk.W, pady=5)
        self.object_var = tk.StringVar()
        self.object_combo = ttk.Combobox(labeling_frame, 
                                         textvariable=self.object_var,
                                         values=self.object_types,
                                         state="readonly",
                                         width=25)
        self.object_combo.pack(fill=tk.X, pady=5)
        self.object_combo.current(0)
        
        # Önceki nesne
        ttk.Label(labeling_frame, text="Önceki Nesne:").pack(anchor=tk.W, pady=5)
        self.prev_object_var = tk.StringVar()
        self.prev_object_combo = ttk.Combobox(labeling_frame, 
                                              textvariable=self.prev_object_var,
                                              values=self.object_types,
                                              state="readonly",
                                              width=25)
        self.prev_object_combo.pack(fill=tk.X, pady=5)
        self.prev_object_combo.current(0)
        
        # Mevcut görev bilgisi
        self.current_task_label = ttk.Label(labeling_frame, 
                                            text="Başlangıç frame'i işaretlenmedi",
                                            foreground="orange",
                                            wraplength=300)
        self.current_task_label.pack(fill=tk.X, pady=10)
        
        # Etiketleme butonları
        ttk.Button(labeling_frame, text="🏁 Görev Başlangıcı İşaretle",
                  command=self.mark_task_start,
                  width=30).pack(fill=tk.X, pady=3)
        
        ttk.Button(labeling_frame, text="🏁 Görev Bitişi İşaretle",
                  command=self.mark_task_end,
                  width=30).pack(fill=tk.X, pady=3)
        
        ttk.Button(labeling_frame, text="❌ Mevcut Görevi İptal",
                  command=self.cancel_current_task,
                  width=30).pack(fill=tk.X, pady=3)
        
        # Kayıtlı görevler
        tasks_frame = ttk.LabelFrame(right_frame, text="Kayıtlı Görevler", padding="10")
        tasks_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Görev listesi
        list_frame = ttk.Frame(tasks_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tasks_listbox = tk.Listbox(list_frame, 
                                        yscrollcommand=scrollbar.set,
                                        font=("Courier", 9))
        self.tasks_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.tasks_listbox.yview)
        
        # Görev listesi butonları
        list_buttons = ttk.Frame(tasks_frame)
        list_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(list_buttons, text="Seçili Göreve Git",
                  command=self.goto_selected_task).pack(side=tk.LEFT, padx=2)
        ttk.Button(list_buttons, text="Sil",
                  command=self.delete_selected_task).pack(side=tk.LEFT, padx=2)
        
        # Kaydetme
        save_frame = ttk.LabelFrame(right_frame, text="Kaydet", padding="10")
        save_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(save_frame, text="💾 JSON Olarak Kaydet",
                  command=self.save_to_json,
                  width=30).pack(fill=tk.X, pady=3)
        
        self.save_info_label = ttk.Label(save_frame, text="", foreground="green")
        self.save_info_label.pack()
        
    def load_video(self):
        """Video dosyası yükle"""
        file_path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[
                ("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        # Önceki videoyu kapat
        if self.video_capture is not None:
            self.video_capture.release()
        
        # Yeni videoyu aç
        self.video_path = file_path
        self.video_capture = cv2.VideoCapture(file_path)
        
        if not self.video_capture.isOpened():
            messagebox.showerror("Hata", "Video açılamadı!")
            return
        
        # Video bilgilerini al
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        
        # Slider'ı ayarla
        self.frame_slider.config(to=self.total_frames - 1)
        
        # İlk frame'i göster
        self.current_frame_number = 0
        self.show_frame(0)
        
        # Bilgi etiketini güncelle
        filename = Path(file_path).name
        self.video_info_label.config(
            text=f"Video: {filename} | Toplam Frame: {self.total_frames} | FPS: {self.fps:.2f}",
            foreground="green"
        )
        
        # Görevleri sıfırla
        self.tasks = []
        self.current_task = None
        self.task_counter = 1
        self.update_tasks_list()
        
    def show_frame(self, frame_number):
        """Belirli bir frame'i göster"""
        if self.video_capture is None:
            return
        
        # Frame'i oku
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_capture.read()
        
        if not ret:
            return
        
        self.current_frame_number = frame_number
        self.current_frame_image = frame.copy()
        
        # Frame'i gösterebilmek için boyutlandır
        display_frame = frame.copy()
        
        # Görev işaretlerini çiz
        for task in self.tasks:
            if task['start_frame'] == frame_number:
                cv2.putText(display_frame, f"START T{task['task_number']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if task['end_frame'] == frame_number:
                cv2.putText(display_frame, f"END T{task['task_number']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Mevcut görev varsa göster
        if self.current_task is not None:
            cv2.putText(display_frame, f"CURRENT TASK (Start: {self.current_task['start_frame']})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # BGR'den RGB'ye çevir
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # PIL Image'e çevir
        image = Image.fromarray(frame_rgb)
        
        # Arayüze sığdır
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        if label_width > 1 and label_height > 1:
            # Aspect ratio'yu koru
            img_aspect = image.width / image.height
            label_aspect = label_width / label_height
            
            if img_aspect > label_aspect:
                new_width = label_width
                new_height = int(label_width / img_aspect)
            else:
                new_height = label_height
                new_width = int(label_height * img_aspect)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # PhotoImage'e çevir ve göster
        photo = ImageTk.PhotoImage(image)
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo  # Referansı tut
        
        # Frame numarasını güncelle
        self.frame_number_label.config(text=f"{frame_number} / {self.total_frames}")
        
        # Slider'ı güncelle (callback'i tetiklemeyi engelle)
        self.updating_slider = True
        self.frame_slider.set(frame_number)
        self.updating_slider = False
        
    def on_slider_change(self, value):
        """Slider değiştiğinde"""
        # Programatik güncellemelerde callback'i atla
        if self.updating_slider:
            return
        
        frame_number = int(float(value))
        self.show_frame(frame_number)
        
    def skip_frames(self, count):
        """Frame'leri atla"""
        if self.video_capture is None:
            return
        
        new_frame = max(0, min(self.total_frames - 1, 
                              self.current_frame_number + count))
        self.show_frame(new_frame)
        
    def goto_first_frame(self):
        """İlk frame'e git"""
        if self.video_capture is None:
            return
        self.show_frame(0)
        
    def goto_last_frame(self):
        """Son frame'e git"""
        if self.video_capture is None:
            return
        self.show_frame(self.total_frames - 1)
        
    def jump_to_frame(self):
        """Belirli bir frame'e atla"""
        if self.video_capture is None:
            return
        
        try:
            frame_number = int(self.jump_entry.get())
            if 0 <= frame_number < self.total_frames:
                self.show_frame(frame_number)
            else:
                messagebox.showwarning("Uyarı", 
                                      f"Frame numarası 0-{self.total_frames-1} arasında olmalı!")
        except ValueError:
            messagebox.showerror("Hata", "Geçerli bir sayı girin!")
            
    def mark_task_start(self):
        """Görev başlangıcını işaretle"""
        if self.video_capture is None:
            messagebox.showwarning("Uyarı", "Önce bir video yükleyin!")
            return
        
        if self.current_task is not None:
            messagebox.showwarning("Uyarı", 
                                  "Zaten bir görev başlatıldı! Önce bitişini işaretleyin veya iptal edin.")
            return
        
        current_object = self.object_var.get()
        previous_object = self.prev_object_var.get()
        
        self.current_task = {
            'task_number': self.task_counter,
            'start_frame': self.current_frame_number,
            'current_object': current_object,
            'previous_object': previous_object
        }
        
        self.current_task_label.config(
            text=f"Görev {self.task_counter} başlatıldı\n"
                 f"Frame: {self.current_frame_number}\n"
                 f"Nesne: {current_object}\n"
                 f"Önceki: {previous_object}\n"
                 f"Bitiş frame'i işaretleyin!",
            foreground="orange"
        )
        
        self.show_frame(self.current_frame_number)
        
    def mark_task_end(self):
        """Görev bitişini işaretle"""
        if self.video_capture is None:
            messagebox.showwarning("Uyarı", "Önce bir video yükleyin!")
            return
        
        if self.current_task is None:
            messagebox.showwarning("Uyarı", "Önce bir görev başlatın!")
            return
        
        if self.current_frame_number <= self.current_task['start_frame']:
            messagebox.showwarning("Uyarı", 
                                  "Bitiş frame'i başlangıç frame'inden büyük olmalı!")
            return
        
        # Görevi tamamla
        self.current_task['end_frame'] = self.current_frame_number
        self.current_task['duration'] = (self.current_frame_number - 
                                         self.current_task['start_frame'])
        self.current_task['duration_seconds'] = self.current_task['duration'] / self.fps
        
        # Akıllı grup atama - önceki görevlere bakarak doğru grubu seç
        current_object = self.current_task['current_object']
        assigned_group = self._get_next_group_for_object(current_object)
        self.current_task['group'] = assigned_group
        
        # Listeye ekle
        self.tasks.append(self.current_task.copy())
        self.task_counter += 1
        
        # Sıfırla
        self.current_task = None
        self.current_task_label.config(
            text="Görev tamamlandı ve kaydedildi!\n"
                 "Yeni görev başlatabilirsiniz.",
            foreground="green"
        )
        
        # Listeyi güncelle
        self.update_tasks_list()
        
        # Bir sonraki nesne için önceki nesneyi güncelle
        if len(self.tasks) > 0:
            last_object = self.tasks[-1]['current_object']
            try:
                idx = self.object_types.index(last_object)
                self.prev_object_combo.current(idx)
            except ValueError:
                pass
        
        self.show_frame(self.current_frame_number)
        
    def cancel_current_task(self):
        """Mevcut görevi iptal et"""
        if self.current_task is None:
            messagebox.showinfo("Bilgi", "İptal edilecek görev yok.")
            return
        
        self.current_task = None
        self.current_task_label.config(
            text="Görev iptal edildi.",
            foreground="gray"
        )
        self.show_frame(self.current_frame_number)
        
    def update_tasks_list(self):
        """Görev listesini güncelle"""
        self.tasks_listbox.delete(0, tk.END)
        
        for task in self.tasks:
            task_str = (f"T{task['task_number']}: {task['current_object']} "
                       f"[{task['start_frame']}-{task['end_frame']}] "
                       f"({task['duration_seconds']:.2f}s)")
            self.tasks_listbox.insert(tk.END, task_str)
            
    def goto_selected_task(self):
        """Seçili görevin başlangıcına git"""
        selection = self.tasks_listbox.curselection()
        if not selection:
            messagebox.showinfo("Bilgi", "Önce bir görev seçin!")
            return
        
        task_idx = selection[0]
        task = self.tasks[task_idx]
        self.show_frame(task['start_frame'])
        
    def delete_selected_task(self):
        """Seçili görevi sil"""
        selection = self.tasks_listbox.curselection()
        if not selection:
            messagebox.showinfo("Bilgi", "Önce bir görev seçin!")
            return
        
        task_idx = selection[0]
        task = self.tasks[task_idx]
        
        result = messagebox.askyesno("Onay", 
                                     f"Görev {task['task_number']}'i silmek istediğinizden emin misiniz?")
        if result:
            del self.tasks[task_idx]
            self.update_tasks_list()
            
    def save_to_json(self):
        """Görevleri JSON formatında kaydet"""
        if not self.tasks:
            messagebox.showwarning("Uyarı", "Kaydedilecek görev yok!")
            return
        
        if self.video_path is None:
            messagebox.showwarning("Uyarı", "Video yüklü değil!")
            return
        
        # Varsayılan dosya adı
        video_name = Path(self.video_path).stem
        default_filename = f"{video_name}_gercek_sonuclar.json"
        default_path = os.path.join(
            os.path.dirname(self.video_path),
            default_filename
        )
        
        # Kayıt yeri sor
        file_path = filedialog.asksaveasfilename(
            title="JSON Dosyasını Kaydet",
            initialfile=default_filename,
            initialdir=os.path.dirname(self.video_path),
            defaultextension=".json",
            filetypes=[("JSON Dosyaları", "*.json"), ("Tüm Dosyalar", "*.*")]
        )
        
        if not file_path:
            return
        
        # JSON formatında hazırla
        output_data = {
            "total_tasks": len(self.tasks),
            "total_frames_processed": self.total_frames,
            "video_file": Path(self.video_path).name,
            "fps": self.fps,
            "labeled_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_history": self.tasks
        }
        
        # Kaydet
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.save_info_label.config(
                text=f"✓ Kaydedildi: {Path(file_path).name}",
                foreground="green"
            )
            messagebox.showinfo("Başarılı", 
                               f"Görevler başarıyla kaydedildi!\n{file_path}")
        except Exception as e:
            messagebox.showerror("Hata", f"Kaydetme hatası: {str(e)}")
            
    def on_closing(self):
        """Pencere kapatılırken"""
        if self.video_capture is not None:
            self.video_capture.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VideoTaskLabeler(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
