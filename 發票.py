import numpy as np
import cv2
from keras.models import load_model
import pytesseract
import re
import requests
#爬蟲
url = 'https://invoice.etax.nat.gov.tw/index.html'
web = requests.get(url)    # 取得網頁內容
web.encoding='utf-8'       # 因為該網頁編碼為 utf-8，加上 .encoding 避免亂碼

from bs4 import BeautifulSoup
soup = BeautifulSoup(web.text, "html.parser")                    # 轉換成標籤樹
td = soup.select('.container-fluid')[0].select('.etw-tbiggest')  # 取出中獎號碼的位置
ns = td[0].getText()  # 特別獎
n1 = td[1].getText()  # 特獎
# 頭獎，因為存入串列會出現 /n 換行符，使用 [-8:] 取出最後八碼
n2 = [td[2].getText()[-8:], td[3].getText()[-8:], td[4].getText()[-8:]] 

# 配置 Tesseract OCR 路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 載入模型
model = load_model('my_model.h5')
model.load_weights('my_model.weights.h5')
# 讀取圖片
image_path = r'numberpredictforpython2\ch10.png'
print(f'Loading {image_path}...')
img = cv2.imread(image_path)

# 檢查圖片大小
sp = img.shape
print(f'Width: {sp[1]} \nHeight: {sp[0]} \nChannels: {sp[2]}')

# 假設發票號碼位於特定區域 (例如: 位於圖片上方或中間)
# 根據發票設計，設定裁剪區域的範圍
# 這裡用範例坐標 (上邊界, 下邊界, 左邊界, 右邊界)
crop_top = 150   # 上邊界
crop_bottom = 250   # 下邊界
crop_left = 150   # 左邊界
crop_right = 350  # 右邊界

# 裁剪特定區域
cropped_img = img[crop_top:crop_bottom, crop_left:crop_right]

# 預處理圖片
img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
_, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

# 計算灰階平均值，確保黑底白字
avg_pixel_value = img_binary.mean()
print("裁剪區域的灰度平均值 =", avg_pixel_value)
if avg_pixel_value < 127:
    print("裁剪區域為黑底白字")
else:
    print("裁剪區域為白底黑字")
    img_binary = 255 - img_binary

# 字元分割
contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# 初始化結果
predicted_classes = []
output_img = cropped_img.copy()

# 預測每個字元並拼接成完整數字
recognized_text = ""
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    
    # 篩選輪廓：排除過小或寬高比例異常的輪廓
    aspect_ratio = w / float(h)
    if not (0.3 < aspect_ratio < 1.5 and w > 5 and h > 5):
        print(f"排除的輪廓: 寬高比例={aspect_ratio}, 寬={w}, 高={h}")
        continue

    # 設置內縮邊界的大小
    margin = 8
    max_side = max(w, h) + 2 * margin
    padded_img = np.ones((max_side, max_side), dtype=np.uint8) * 0
    dx = (max_side - w) // 2
    dy = (max_side - h) // 2
    padded_img[dy:dy + h, dx:dx + w] = img_binary[y:y + h, x:x + w]

    # 調整大小並標準化
    char_img_resized = cv2.resize(padded_img, (28, 28))
    char_img_resized = char_img_resized.astype('float32') / 255.0
    char_img_reshaped = char_img_resized.reshape(1, 28, 28, 1)

    # 預測
    predictions = model.predict(char_img_reshaped)
    predicted_class = np.argmax(predictions, axis=-1)[0]

    # 篩選非數字結果
    if str(predicted_class).isdigit():
        predicted_classes.append(predicted_class)
        recognized_text += str(predicted_class)

        # 畫出輪廓
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (128, 128, 128), 2)
        cv2.putText(output_img, str(predicted_class), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    else:
        print(f"非數字字符被忽略: {predicted_class}")

# 顯示拼接的數字
#print("OCR 偵測到的數字：", recognized_text)

# 使用正則表示式檢測可能的發票號碼
potential_numbers = re.findall(r'\b\d{8}\b', recognized_text)
print(f"發票號碼：{potential_numbers}")
result=int(potential_numbers[0])
if result == ns: print('對中 1000 萬元！')
if result == n1: print('對中 200 萬元！')
for i in n2:
    r_str = str(result)  # 將整數轉換為字串
    i_str = str(i)
    if result == i:
        print('對中 20 萬元！')
        break
    elif r_str[-7:] == i[-7:]:
        print('對中 4 萬元！')
        break
    elif r_str[-6:] == i[-6:]:
        print('對中 1 萬元！')
        break
    elif r_str[-5:] == i[-5:]:
        print('對中 4000 元！')
        break
    elif r_str[-4:] == i[-4:]:
        print('對中 1000 元！')
        break
    elif r_str[-3:] == i[-3:]:
        print('對中 200 元！')
        break
    else:
        print("沒中")
        break

# 儲存結果圖片
output_path = r"C:\Users\Lucas\Downloads\testnumber1_cropped.jpg"
cv2.imwrite(output_path, output_img)

# 顯示輸出影像
cv2.imshow('Detected Characters', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 結尾
#print('預測類別：', predicted_classes)
print('done')