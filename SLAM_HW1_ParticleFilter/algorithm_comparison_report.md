# 算法實現差異比較報告

## 比較對象
- **用戶代碼**: 當前實現
- **參考答案**: https://github.com/aaronzguan/Robot-Localization-using-Particle-Filters

## 重要說明
本報告只關注**邏輯差異**，不比較：
- 變數名稱差異
- 參數值差異
- 性能優化（如 vectorization）

---

## 1. Motion Model (`motion_model.py`)

### ✅ 無邏輯差異

**檢查點：**
- ✅ 噪聲應用方式：兩者都使用 `delta - noise`（減法）
- ✅ 方差計算公式：完全一致
- ✅ 角度包裝：實現相同
- ✅ 運動分量計算：邏輯一致

**細節：**
- 參考答案：`delta_trans = math.sqrt((u_t0[0] - u_t1[0]) ** 2 + (u_t0[1] - u_t1[1]) ** 2)`
- 用戶代碼：`delta_trans = math.sqrt((x_bar_prime - x_bar)**2 + (y_bar_prime - y_bar)**2)`
- **結論**: 數學上等價，只是變數名稱不同

---

## 2. Sensor Model (`sensor_model.py`)

### 🔴 差異 1: p_short 條件判斷（**重要**）

**位置：**
- 參考答案：`/tmp/ref_sensor_model.py:137`
- 用戶代碼：`code/sensor_model.py:540`

**差異：**
```python
# 參考答案（錯誤）
p_short[np.where((z_t < 0) & (z_t > zstar_t))] = 0

# 用戶代碼（正確）
p_short[np.where((z_t < 0) | (z_t > zstar_t))] = 0
```

**分析：**
- 參考答案使用 `&` (AND)，這在邏輯上是錯誤的
- 條件 `(z_t < 0) & (z_t > zstar_t)` 永遠不可能同時為真（一個數不能既小於0又大於zstar_t）
- 用戶代碼使用 `|` (OR) 是正確的：當 z_t < 0 **或** z_t > zstar_t 時，p_short 應該為 0

**影響：**
- 參考答案的實現會導致 p_short 在某些情況下不正確地保持非零值
- 用戶代碼的實現是正確的

---

### 🟡 差異 2: p_hit 除零保護

**位置：**
- 參考答案：`/tmp/ref_sensor_model.py:126-131`
- 用戶代碼：`code/sensor_model.py:513-528`

**差異：**
```python
# 參考答案（無除零保護）
eta = norm.cdf(self._max_range, loc=zstar_t, scale=self._sigma_hit) - norm.cdf(0, loc=zstar_t, scale=self._sigma_hit)
p_hit = norm.pdf(z_t, loc=zstar_t, scale=self._sigma_hit) / eta
p_hit[z_t > self._max_range] = 0
p_hit[z_t < 0] = 0

# 用戶代碼（有除零保護）
eta = norm.cdf(self._max_range, loc=zstar_t, scale=self._sigma_hit) - norm.cdf(0, loc=zstar_t, scale=self._sigma_hit)
p_hit = norm.pdf(z_t, loc=zstar_t, scale=self._sigma_hit)
valid_eta = eta > 1e-10
p_hit[valid_eta] = p_hit[valid_eta] / eta[valid_eta]
p_hit[~valid_eta] = 0.0
p_hit[z_t > self._max_range] = 0
p_hit[z_t < 0] = 0
```

**分析：**
- 用戶代碼增加了除零保護，當 `eta` 接近 0 時避免除以零
- 這是一個**改進**，提高了數值穩定性

**影響：**
- 在極端情況下（eta 接近 0），參考答案可能產生 NaN 或 Inf
- 用戶代碼更穩定

---

### 🟡 差異 3: p_max 返回類型

**位置：**
- 參考答案：`/tmp/ref_sensor_model.py:140-141`
- 用戶代碼：`code/sensor_model.py:543-549`

**差異：**
```python
# 參考答案
return z_t == self._max_range  # 返回 boolean array

# 用戶代碼
return (z_t == self._max_range).astype(float)  # 返回 float array
```

**分析：**
- 參考答案返回 boolean，用戶代碼返回 float
- 在後續計算中，boolean 會被自動轉換為 0/1，所以功能上等價
- 但用戶代碼更明確，避免了隱式類型轉換

**影響：**
- 功能上等價，但用戶代碼更明確

---

### ✅ 無差異: p_rand 實現

兩者實現完全一致。

---

## 3. Resampling (`resampling.py`)

### ✅ 無邏輯差異

**檢查點：**
- ✅ 低方差採樣算法：邏輯完全一致
- ✅ 權重正規化：時機和方式相同
- ✅ 累積權重計算：邏輯一致

**細節：**
- 變數名稱不同（`r` vs `random_offset`, `U` vs `threshold`, `i` vs `source_idx`, `j` vs `target_idx`）
- 但算法邏輯完全相同

---

## 4. Ray Casting

### 🟡 差異: 邊界檢查

**位置：**
- 參考答案：`/tmp/ref_sensor_model.py:70-78` (ray_casting)
- 用戶代碼：`code/sensor_model.py:423-446` (_ray_casting_lookup)

**差異：**
```python
# 參考答案（無邊界檢查）
origin_laser_x = int((x_t1[0] + self._offset * math.cos(theta_robot))//self._resolution)
origin_laser_y = int((x_t1[1] + self._offset * math.sin(theta_robot))//self._resolution)
zstar_t = raycast_map[origin_laser_y, origin_laser_x, theta_laser]

# 用戶代碼（有邊界檢查）
origin_laser_x = int((x_t1[0] + self._laser_offset * math.cos(theta_robot)) // self._map_resolution)
origin_laser_y = int((x_t1[1] + self._laser_offset * math.sin(theta_robot)) // self._map_resolution)
# Check bounds
height, width = raycast_map.shape[:2]
if origin_laser_y < 0 or origin_laser_y >= height or origin_laser_x < 0 or origin_laser_x >= width:
    return np.ones(len(theta_laser)) * self._max_range
zstar_t = raycast_map[origin_laser_y, origin_laser_x, theta_laser]
```

**分析：**
- 用戶代碼增加了邊界檢查，當粒子位置超出地圖範圍時返回 max_range
- 這是一個**改進**，避免了數組越界錯誤

**影響：**
- 參考答案在粒子超出地圖時可能崩潰
- 用戶代碼更健壯

---

## 5. Main Loop (`main.py`)

### ✅ 無邏輯差異

主要差異在於：
- 用戶代碼有更多功能（可視化、調試選項等）
- 核心算法邏輯一致

---

## 總結

### 關鍵發現

1. **🔴 參考答案的錯誤**：
   - `p_short` 條件判斷使用了錯誤的邏輯運算符 `&` 而不是 `|`
   - **用戶代碼是正確的**

2. **🟡 用戶代碼的改進**：
   - `p_hit` 增加了除零保護
   - `p_max` 明確返回 float 類型
   - Ray casting 增加了邊界檢查

3. **✅ 核心算法一致**：
   - Motion Model: 完全一致
   - Resampling: 完全一致
   - 主要邏輯流程: 完全一致

### 結論

用戶代碼在以下方面**優於**參考答案：
1. ✅ 修復了 `p_short` 的邏輯錯誤
2. ✅ 增加了數值穩定性（除零保護）
3. ✅ 增加了健壯性（邊界檢查）

**建議：** 用戶代碼的實現是正確且改進的，無需修改。

