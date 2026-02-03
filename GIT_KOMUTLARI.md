# Git Komutları Rehberi

## 🔧 Temel Kurulum

```bash
# Git reposu başlatma
git init

# Uzak repo ekleme
git remote add origin https://github.com/KULLANICI/REPO.git
```

---

## 📤 Değişiklikleri Yükleme

```bash
# Tüm değişiklikleri hazırla
git add .

# Tek dosya hazırla
git add dosya_adi.dart

# Commit oluştur
git commit -m "Açıklama mesajı"

# GitHub'a gönder
git push origin main
```

---

## 📥 Değişiklikleri Çekme

```bash
# Uzak repodan çek
git pull origin main
```

---

## 📋 Durum Kontrolü

```bash
# Değişiklikleri gör
git status

# Commit geçmişi
git log --oneline -10
```

---

## 🌿 Branch (Dal) İşlemleri

```bash
# Yeni branch oluştur
git checkout -b yeni-ozellik

# Branch değiştir
git checkout main

# Branch listele
git branch

# Branch'ı main'e birleştir
git checkout main
git merge yeni-ozellik
```

---

## ↩️ Geri Alma İşlemleri

```bash
# Son commit'i geri al (değişiklikler kalır)
git reset --soft HEAD~1

# Tek dosyayı eski haline getir
git checkout -- dosya_adi.dart

# Tüm değişiklikleri geri al
git checkout -- .
```

---

## 📝 Hızlı Kullanım

Değişiklik yaptıktan sonra:
```bash
git add .
git commit -m "Değişiklik açıklaması"
git push origin main
```
