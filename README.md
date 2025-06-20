
````markdown
# 🖐 Virtual Hand Mouse (کنترل موس با دست)

پروژه‌ای ساده و کاربردی برای کنترل موس سیستم با استفاده از دوربین و تشخیص حرکت دست. این پروژه از بینایی ماشین و شناسایی حرکات دست برای اجرای فرمان‌های موس (حرکت، کلیک چپ، کلیک راست) استفاده می‌کند.

---

## 📌 معرفی پروژه

با استفاده از تکنولوژی‌هایی مانند OpenCV و MediaPipe، این پروژه توانایی تشخیص انگشتان دست را دارد و آن را به عملکردهای موس در ویندوز تبدیل می‌کند.

- ✋ 1 انگشت بالا = حرکت موس  
- ✌ 2 انگشت بالا = کلیک چپ  
- 🤟 3 انگشت بالا = کلیک راست

---

## 🛠️ تکنولوژی‌ها و کتابخانه‌ها

| ابزار/کتابخانه | توضیح |
|----------------|--------|
| `Python`       | زبان برنامه‌نویسی اصلی پروژه |
| `OpenCV`       | دریافت تصویر زنده از وب‌کم |
| `MediaPipe`    | تشخیص موقعیت دست و انگشت‌ها |
| `PyAutoGUI`    | کنترل موس روی سیستم عامل |

---

## 🔧 نحوه نصب و اجرای پروژه (برای برنامه‌نویسان)

### 1. نصب پایتون  
- دانلود از [python.org](https://www.python.org)  
- حین نصب، گزینه `Add Python to PATH` را فعال کنید

### 2. نصب پیش‌نیازها

```bash
pip install opencv-python mediapipe pyautogui
````

### 3. اجرای پروژه

```bash
python mouse_curser.py
```

---

## 📦 ساخت فایل اجرایی (exe)

برای اجرای پروژه روی سیستم‌هایی که پایتون یا کتابخانه‌ها را ندارند:

```bash
pyinstaller --onefile --noconsole --add-data "C:/Path/To/Python/Lib/site-packages/mediapipe/modules;mediapipe/modules" mouse_curser.py
```

> ⚠ مسیر بالا را با مسیر واقعی پایتون در سیستم خود جایگزین کنید.

📁 اگر فایل `.exe` را به سیستم دیگری منتقل کردید، پوشه `mediapipe/modules` را کنار فایل قرار دهید تا بدون خطا اجرا شود.

---

## 👨‍💻 طریقه استفاده برای کاربران نهایی

1. فایل اجرایی `.exe` را اجرا کنید
2. دست خود را جلوی دوربین نگه دارید
3. انگشت‌ها را به یکی از این صورت‌ها بالا ببرید:

   * یک انگشت: حرکت موس
   * دو انگشت: کلیک چپ
   * سه انگشت: کلیک راست
4. برای خروج از برنامه کلید `Q` را فشار دهید یا پنجره را ببندید

---

## 🖼 پیش‌نمایش عملکرد

📷 وب‌کم روشن می‌شود
📌 دست شما تشخیص داده می‌شود
🖱 حرکت موس طبق موقعیت دست انجام می‌شود
✅ فرمان‌های کلیک با انگشت انجام می‌شود

---

## ❗ مشکلات رایج و راه‌حل

| مشکل                    | راه‌حل                                             |
| ----------------------- | -------------------------------------------------- |
| برنامه سریع بسته می‌شود | مطمئن شوید وب‌کم فعال است و فایل درست اجرا شده     |
| خطای Mediapipe modules  | پوشه mediapipe/modules را کنار فایل .exe قرار دهید |
| موس کار نمی‌کند         | نور محیط مناسب باشد و دست در مرکز تصویر باشد       |

---

## 👤 نویسنده

پروژه توسط \zahir.fakhar@gmail.com ساخته شده است — قابل استفاده برای پروژه‌های دانشگاهی، تمرینی و یادگیری بینایی ماشین.

---

## ⚖️ مجوز

استفاده از کد آزاد است برای اهداف آموزشی و غیرتجاری.

```

