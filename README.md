# Roi-Tracer
## 操作步驟:
1. 執行roiTrace_1.py
```linux
$ python3 roiTrace_1.py
```
2. 校正：
   a. 當顯示'Please stare at the up left edge...'，注視螢幕左上角邊緣數秒。
   b. 用餘光判斷，當顯示下一行'please stare at the down right edge...'，改成注視螢幕右下角邊緣數秒。
3. 理想上視線將會顯示在出現的黑屏上。黑屏左上角有眼睛圖片的二極化過程，左而右分別為:原圖=>equalizeHist=>GaussainBlur=>equalizeHist=>threshold後並且dilate。以照片為範例：
![再次](https://user-images.githubusercontent.com/50452986/172172606-e5ec963d-6b01-47d1-af03-95923a736868.PNG)

5. 將黑屏放大到全螢幕，即可看到黑屏中白點隨視線移動。
![成功用眼睛高度判斷視線_應該還要調整](https://user-images.githubusercontent.com/50452986/172172240-c42314e4-b24f-4966-9e56-100a06e39ff1.PNG)
**註：螢幕預設為約1920*1080解析度，若不同需在90行以及138行調整**
## 實現原理:
- 運用瞳孔位在眼睛中的寬度位置與眼睛的高度來計算視線的方向，之所以沒有用瞳孔之餘眼睛的高度來測量視線高度是因受限於frame的眼睛高度小，解析度十分低，且眼睛開大開小會影響眼睛高度，解析度很差。後來發現眼睛本身的高度更容易表達視線高度，因為往下看時眼睛較小，往上時較大。
- 開始前校正時取樣64次，取中間32個值平均做為參考點。
- 運行時用八個frame中的視線位置平均(低通濾波器)，解決視線跳動問題。

### 處理後之瞳孔影像：
![用現成程式碼調整原始照片參數成功](https://user-images.githubusercontent.com/50452986/172172037-728b4ccb-e03f-49f1-af82-b462361d472d.PNG)
![用現成程式碼調整原始照片參數成功1](https://user-images.githubusercontent.com/50452986/172172073-f5ad87ec-5e3b-4560-830d-f11eb34074ef.PNG)
![用現成程式碼調整原始照片參數成功2](https://user-images.githubusercontent.com/50452986/172172104-e3a94249-9478-4239-a28d-a482e730aa64.PNG)
![有等化的影片同樣是二值化](https://user-images.githubusercontent.com/50452986/172172341-77849dc4-b782-4652-990f-b4a197a7a15b.PNG)
### 使用模組：
- OpenCV: 影像處理。https://pypi.org/project/opencv-python/
- Numpy: 矩陣運算。https://numpy.org/
- dlib: 人臉辨識。https://pypi.org/project/dlib/
- math: 求歐幾里得範數hypot()。https://docs.python.org/3/library/math.html
- time: 校正時計時。https://www.google.com/search?q=python+time+documentation&rlz=1C1PRFI_enTW786TW786&oq=python+time+docu&aqs=chrome.1.69i57j0i512j0i8i30l3j0i8i10i30j0i8i30.4410j0j7&sourceid=chrome&ie=UTF-8
