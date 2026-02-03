import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

class TSPDComparator:
    """TSPD sistem çıktılarını gerçek sonuçlarla karşılaştırır"""
    
    def __init__(self, predicted_file: str, ground_truth_file: str):
        """
        Args:
            predicted_file: Programın ürettiği sonuç dosyası
            ground_truth_file: Gerçek sonuçların bulunduğu dosya
        """
        self.predicted_file = predicted_file
        self.ground_truth_file = ground_truth_file
        self.predicted_data = None
        self.ground_truth_data = None
        self.comparison_results = {}
        
    def load_data(self):
        """JSON dosyalarını yükle"""
        try:
            with open(self.predicted_file, 'r', encoding='utf-8') as f:
                self.predicted_data = json.load(f)
            print(f"✓ Tahmin dosyası yüklendi: {self.predicted_file}")
            
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                self.ground_truth_data = json.load(f)
            print(f"✓ Gerçek sonuç dosyası yüklendi: {self.ground_truth_file}")
            
            return True
        except FileNotFoundError as e:
            print(f"✗ Dosya bulunamadı: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ JSON parse hatası: {e}")
            return False
    
    def compare_task_detection(self) -> Dict:
        """Görev tespiti doğruluğunu karşılaştır"""
        pred_tasks = self.predicted_data['task_history']
        true_tasks = self.ground_truth_data['task_history']
        
        comparison = {
            'total_predicted_tasks': len(pred_tasks),
            'total_true_tasks': len(true_tasks),
            'task_matches': [],
            'correct_detections': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 0.0
        }
        
        # task_number'a göre görevleri bir dict'te topla
        true_tasks_dict = {task.get('task_number', i): task for i, task in enumerate(true_tasks)}
        
        # Her tahmin görevini gerçek görevle karşılaştır
        for pred in pred_tasks:
            pred_task_num = pred.get('task_number', None)
            true = true_tasks_dict.get(pred_task_num)
            
            # Eğer gerçek görev bulunamazsa, indeks ile eşleştir (fallback)
            if true is None and pred_task_num is not None:
                # İlk N görev adedinde fallback
                idx = min(pred_task_num - 2, len(true_tasks) - 1)  # task_number 2'den başlıyor
                true = true_tasks[idx] if idx >= 0 and idx < len(true_tasks) else None
            
            if true is None:
                continue  # Eşleşen görev yoksa atla
            
            i = pred_task_num
            task_comparison = {
                'task_number': pred_task_num,
                'matches': {},
                'mismatches': {}
            }
            
            # Nesne tespiti
            pred_obj = pred.get('current_object', '')
            true_obj = true.get('current_object', '')
            if pred_obj == true_obj:
                task_comparison['matches']['current_object'] = pred_obj
                comparison['correct_detections'] += 1
            else:
                task_comparison['mismatches']['current_object'] = {
                    'predicted': pred_obj,
                    'actual': true_obj
                }
            
            # Frame karşılaştırması
            pred_start = pred.get('start_frame', 0)
            true_start = true.get('start_frame', 0)
            pred_end = pred.get('end_frame', 0)
            true_end = true.get('end_frame', 0)
            
            frame_diff_start = abs(pred_start - true_start)
            frame_diff_end = abs(pred_end - true_end)
            
            task_comparison['frame_analysis'] = {
                'start_frame_diff': frame_diff_start,
                'end_frame_diff': frame_diff_end,
                'predicted_range': [pred_start, pred_end],
                'actual_range': [true_start, true_end]
            }
            
            # Süre karşılaştırması
            pred_duration = pred.get('duration_seconds', 0)
            true_duration = true.get('duration_seconds', 0)
            duration_diff = abs(pred_duration - true_duration)
            duration_error_percent = (duration_diff / true_duration * 100) if true_duration > 0 else 0
            
            task_comparison['duration_analysis'] = {
                'predicted_seconds': round(pred_duration, 2),
                'actual_seconds': round(true_duration, 2),
                'difference_seconds': round(duration_diff, 2),
                'error_percent': round(duration_error_percent, 2)
            }
            
            # Grup karşılaştırması - sadece adları karşılaştır (Grup X: kısmını çıkar)
            pred_group = pred.get('group', '')
            true_group = true.get('group', '')
            
            # "Grup X: " kısmını çıkar
            pred_group_name = pred_group.split(': ')[-1] if ': ' in pred_group else pred_group
            true_group_name = true_group.split(': ')[-1] if ': ' in true_group else true_group
            
            if pred_group_name == true_group_name:
                task_comparison['matches']['group'] = pred_group_name
            else:
                task_comparison['mismatches']['group'] = {
                    'predicted': pred_group_name,
                    'actual': true_group_name
                }
            
            comparison['task_matches'].append(task_comparison)
        
        # Accuracy hesapla
        total_comparisons = len(pred_tasks)
        if total_comparisons > 0:
            comparison['accuracy'] = round((comparison['correct_detections'] / total_comparisons) * 100, 2)
        
        return comparison
    
    def calculate_frame_metrics(self) -> Dict:
        """Frame bazlı metrikleri hesapla (IoU, Overlap vb.)"""
        pred_tasks = self.predicted_data['task_history']
        true_tasks = self.ground_truth_data['task_history']
        
        metrics = {
            'task_overlaps': [],
            'avg_iou': 0.0,
            'avg_frame_error': 0.0,
            'total_frame_diff': 0
        }
        
        total_iou = 0
        total_frame_error = 0
        
        for pred, true in zip(pred_tasks, true_tasks):
            pred_start = pred.get('start_frame', 0)
            pred_end = pred.get('end_frame', 0)
            true_start = true.get('start_frame', 0)
            true_end = true.get('end_frame', 0)
            
            # IoU (Intersection over Union) hesapla
            intersection_start = max(pred_start, true_start)
            intersection_end = min(pred_end, true_end)
            intersection = max(0, intersection_end - intersection_start)
            
            union_start = min(pred_start, true_start)
            union_end = max(pred_end, true_end)
            union = union_end - union_start
            
            iou = (intersection / union) if union > 0 else 0
            total_iou += iou
            
            # Frame hatası
            start_error = abs(pred_start - true_start)
            end_error = abs(pred_end - true_end)
            avg_error = (start_error + end_error) / 2
            total_frame_error += avg_error
            
            overlap_info = {
                'task_number': pred.get('task_number', 0),
                'iou': round(iou, 3),
                'start_frame_error': start_error,
                'end_frame_error': end_error,
                'avg_frame_error': round(avg_error, 2)
            }
            
            metrics['task_overlaps'].append(overlap_info)
        
        num_tasks = len(pred_tasks)
        if num_tasks > 0:
            metrics['avg_iou'] = round(total_iou / num_tasks, 3)
            metrics['avg_frame_error'] = round(total_frame_error / num_tasks, 2)
        
        return metrics
    
    def calculate_duration_metrics(self) -> Dict:
        """Süre tahminlerinin metriklerini hesapla"""
        pred_tasks = self.predicted_data['task_history']
        true_tasks = self.ground_truth_data['task_history']
        
        metrics = {
            'task_durations': [],
            'avg_duration_error_seconds': 0.0,
            'avg_duration_error_percent': 0.0,
            'max_duration_error': 0.0,
            'min_duration_error': float('inf')
        }
        
        total_error_seconds = 0
        total_error_percent = 0
        
        for pred, true in zip(pred_tasks, true_tasks):
            pred_dur = pred.get('duration_seconds', 0)
            true_dur = true.get('duration_seconds', 0)
            
            error_seconds = abs(pred_dur - true_dur)
            error_percent = (error_seconds / true_dur * 100) if true_dur > 0 else 0
            
            total_error_seconds += error_seconds
            total_error_percent += error_percent
            
            metrics['max_duration_error'] = max(metrics['max_duration_error'], error_seconds)
            metrics['min_duration_error'] = min(metrics['min_duration_error'], error_seconds)
            
            duration_info = {
                'task_number': pred.get('task_number', 0),
                'predicted': round(pred_dur, 2),
                'actual': round(true_dur, 2),
                'error_seconds': round(error_seconds, 2),
                'error_percent': round(error_percent, 2)
            }
            
            metrics['task_durations'].append(duration_info)
        
        num_tasks = len(pred_tasks)
        if num_tasks > 0:
            metrics['avg_duration_error_seconds'] = round(total_error_seconds / num_tasks, 2)
            metrics['avg_duration_error_percent'] = round(total_error_percent / num_tasks, 2)
        
        if metrics['min_duration_error'] == float('inf'):
            metrics['min_duration_error'] = 0.0
        
        metrics['max_duration_error'] = round(metrics['max_duration_error'], 2)
        metrics['min_duration_error'] = round(metrics['min_duration_error'], 2)
        
        return metrics
    
    def generate_report(self) -> str:
        """Detaylı karşılaştırma raporu oluştur"""
        report = []
        report.append("=" * 80)
        report.append("TSPD SİSTEM PERFORMANS KARŞILAŞTIRMA RAPORU")
        report.append("=" * 80)
        report.append("")
        
        # Dosya bilgileri
        report.append("DOSYA BİLGİLERİ:")
        report.append(f"Tahmin Dosyası: {Path(self.predicted_file).name}")
        report.append(f"Gerçek Sonuç Dosyası: {Path(self.ground_truth_file).name}")
        report.append("")
        
        # Genel bilgiler
        report.append("GENEL BİLGİLER:")
        report.append(f"Toplam İşlenen Frame: {self.predicted_data.get('total_frames_processed', 0)}")
        report.append(f"Tahmin Edilen Görev Sayısı: {self.predicted_data.get('total_tasks', 0)}")
        report.append(f"Gerçek Görev Sayısı: {self.ground_truth_data.get('total_tasks', 0)}")
        report.append("")
        
        # Görev tespiti sonuçları
        task_comp = self.comparison_results['task_detection']
        report.append("GÖREV TESPİTİ SONUÇLARI:")
        report.append(f"Doğru Nesne Tespitleri: {task_comp['correct_detections']}/{task_comp['total_predicted_tasks']}")
        report.append(f"Nesne Tespiti Doğruluğu: %{task_comp['accuracy']}")
        report.append("")
        
        # Frame metrikleri
        frame_metrics = self.comparison_results['frame_metrics']
        report.append("FRAME ANALİZİ:")
        report.append(f"Ortalama IoU (Intersection over Union): {frame_metrics['avg_iou']}")
        report.append(f"Ortalama Frame Hatası: {frame_metrics['avg_frame_error']} frame")
        report.append("")
        
        # Süre metrikleri
        duration_metrics = self.comparison_results['duration_metrics']
        report.append("SÜRE ANALİZİ:")
        report.append(f"Ortalama Süre Hatası: {duration_metrics['avg_duration_error_seconds']} saniye")
        report.append(f"Ortalama Süre Hata Yüzdesi: %{duration_metrics['avg_duration_error_percent']}")
        report.append(f"Maksimum Süre Hatası: {duration_metrics['max_duration_error']} saniye")
        report.append(f"Minimum Süre Hatası: {duration_metrics['min_duration_error']} saniye")
        report.append("")
        
        # Görev detayları
        report.append("GÖREV BAZLI DETAYLAR:")
        report.append("-" * 80)
        
        for task in task_comp['task_matches']:
            task_num = task['task_number']
            report.append(f"\nGÖREV {task_num}:")
            
            # Nesne tespiti
            if 'current_object' in task['matches']:
                report.append(f"  ✓ Nesne Tespiti: {task['matches']['current_object']} (DOĞRU)")
            elif 'current_object' in task['mismatches']:
                mismatch = task['mismatches']['current_object']
                report.append(f"  ✗ Nesne Tespiti: Tahmin={mismatch['predicted']}, Gerçek={mismatch['actual']} (YANLIŞ)")
            
            # Frame analizi
            frame_ana = task['frame_analysis']
            report.append(f"  Frame Aralığı: Tahmin={frame_ana['predicted_range']}, Gerçek={frame_ana['actual_range']}")
            report.append(f"  Frame Farkı: Başlangıç={frame_ana['start_frame_diff']}, Bitiş={frame_ana['end_frame_diff']}")
            
            # Süre analizi
            dur_ana = task['duration_analysis']
            report.append(f"  Süre: Tahmin={dur_ana['predicted_seconds']}s, Gerçek={dur_ana['actual_seconds']}s")
            report.append(f"  Süre Farkı: {dur_ana['difference_seconds']}s (%{dur_ana['error_percent']} hata)")
            
            # Grup
            if 'group' in task['matches']:
                report.append(f"  ✓ Grup: {task['matches']['group']} (DOĞRU)")
            elif 'group' in task['mismatches']:
                mismatch = task['mismatches']['group']
                report.append(f"  ✗ Grup: Tahmin={mismatch['predicted']}, Gerçek={mismatch['actual']} (YANLIŞ)")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, output_dir: str = None):
        """Raporu dosyaya kaydet"""
        if output_dir is None:
            output_dir = os.path.dirname(self.predicted_file)
        
        # Dosya adından part numarasını al
        filename = Path(self.predicted_file).stem
        report_filename = f"{filename}_comparison_report.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        report_text = self.generate_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n✓ Rapor kaydedildi: {report_path}")
        return report_path
    
    def export_to_excel(self, output_dir: str = None):
        """Sonuçları Excel dosyasına aktar"""
        if output_dir is None:
            output_dir = os.path.dirname(self.predicted_file)
        
        filename = Path(self.predicted_file).stem
        excel_filename = f"{filename}_comparison.xlsx"
        excel_path = os.path.join(output_dir, excel_filename)
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Görev karşılaştırması
            task_comp = self.comparison_results['task_detection']
            task_data = []
            
            for task in task_comp['task_matches']:
                row = {
                    'Görev No': task['task_number'],
                    'Tahmin Edilen Nesne': task['mismatches'].get('current_object', {}).get('predicted', 
                                           task['matches'].get('current_object', '')),
                    'Gerçek Nesne': task['mismatches'].get('current_object', {}).get('actual', 
                                    task['matches'].get('current_object', '')),
                    'Nesne Doğru mu?': 'current_object' in task['matches'],
                    'Tahmin Frame Başlangıç': task['frame_analysis']['predicted_range'][0],
                    'Gerçek Frame Başlangıç': task['frame_analysis']['actual_range'][0],
                    'Tahmin Frame Bitiş': task['frame_analysis']['predicted_range'][1],
                    'Gerçek Frame Bitiş': task['frame_analysis']['actual_range'][1],
                    'Başlangıç Frame Farkı': task['frame_analysis']['start_frame_diff'],
                    'Bitiş Frame Farkı': task['frame_analysis']['end_frame_diff'],
                    'Tahmin Süresi (s)': task['duration_analysis']['predicted_seconds'],
                    'Gerçek Süre (s)': task['duration_analysis']['actual_seconds'],
                    'Süre Farkı (s)': task['duration_analysis']['difference_seconds'],
                    'Süre Hata (%)': task['duration_analysis']['error_percent'],
                    'Tahmin Grup': task['mismatches'].get('group', {}).get('predicted', 
                                   task['matches'].get('group', '')),
                    'Gerçek Grup': task['mismatches'].get('group', {}).get('actual', 
                                   task['matches'].get('group', '')),
                    'Grup Doğru mu?': 'group' in task['matches']
                }
                task_data.append(row)
            
            df_tasks = pd.DataFrame(task_data)
            df_tasks.to_excel(writer, sheet_name='Görev Karşılaştırması', index=False)
            
            # Frame metrikleri
            frame_metrics = self.comparison_results['frame_metrics']
            df_frame = pd.DataFrame(frame_metrics['task_overlaps'])
            df_frame.to_excel(writer, sheet_name='Frame Metrikleri', index=False)
            
            # Süre metrikleri
            duration_metrics = self.comparison_results['duration_metrics']
            df_duration = pd.DataFrame(duration_metrics['task_durations'])
            df_duration.to_excel(writer, sheet_name='Süre Metrikleri', index=False)
            
            # Özet metrikler
            summary_data = {
                'Metrik': [
                    'Toplam Görev Sayısı',
                    'Doğru Nesne Tespiti',
                    'Nesne Tespiti Doğruluğu (%)',
                    'Ortalama IoU',
                    'Ortalama Frame Hatası',
                    'Ortalama Süre Hatası (s)',
                    'Ortalama Süre Hata (%)',
                    'Max Süre Hatası (s)',
                    'Min Süre Hatası (s)'
                ],
                'Değer': [
                    task_comp['total_predicted_tasks'],
                    task_comp['correct_detections'],
                    task_comp['accuracy'],
                    frame_metrics['avg_iou'],
                    frame_metrics['avg_frame_error'],
                    duration_metrics['avg_duration_error_seconds'],
                    duration_metrics['avg_duration_error_percent'],
                    duration_metrics['max_duration_error'],
                    duration_metrics['min_duration_error']
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Özet', index=False)
        
        print(f"✓ Excel raporu kaydedildi: {excel_path}")
        return excel_path
    
    def run_comparison(self, save_reports: bool = True):
        """Tüm karşılaştırmayı çalıştır"""
        print("\nKarşılaştırma başlatılıyor...")
        print("-" * 80)
        
        # Verileri yükle
        if not self.load_data():
            return False
        
        # Karşılaştırmaları yap
        print("\nGörev tespiti karşılaştırılıyor...")
        self.comparison_results['task_detection'] = self.compare_task_detection()
        
        print("Frame metrikleri hesaplanıyor...")
        self.comparison_results['frame_metrics'] = self.calculate_frame_metrics()
        
        print("Süre metrikleri hesaplanıyor...")
        self.comparison_results['duration_metrics'] = self.calculate_duration_metrics()
        
        # Raporu göster
        print("\n" + self.generate_report())
        
        # Raporları kaydet
        if save_reports:
            self.save_report()
            self.export_to_excel()
        
        return True


def main():
    """Ana fonksiyon - Karşılaştırmayı çalıştır"""
    
    # Dosya yollarını belirle
    base_dir = r"C:/Users/ali.donbaloglu/Desktop/Montaj_proces/Sistem_test"
    
    # Part18 için karşılaştırma
    predicted_file = os.path.join(base_dir, "Part5_tspd_results.json")
    ground_truth_file = os.path.join(base_dir, "part5_gercek_sonuclar.json")
    
    # Karşılaştırıcıyı oluştur ve çalıştır
    comparator = TSPDComparator(predicted_file, ground_truth_file)
    comparator.run_comparison(save_reports=True)
    
    print("\n✓ Karşılaştırma tamamlandı!")


if __name__ == "__main__":
    main()
