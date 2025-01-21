import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import os
import warnings
warnings.filterwarnings('ignore')
# 字体文件夹路径
font_dir = r'D:\anaconda\envs\pytorch\lib\site-packages\matplotlib\mpl-data\fonts\ttf'
# 获取字体文件夹中的所有字体文件
font_files = fm.findSystemFonts(fontpaths=[font_dir])
# 将字体文件添加到字体管理器中
for font_file in font_files:
    fm.fontManager.addfont(font_file)
# 手动创建 FontProperties 实例来获取字体名称
fonts = [fm.FontProperties(fname=f).get_name() for f in font_files]
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  
# 加载样式
plt.style.use(['science', 'bright','no-latex','cjk-sc-font'])
from matplotlib.colors import ListedColormap

def interpolate_color(color1, color2, steps):
    # 计算每一步的颜色变化量
    delta_r = (color2[0] - color1[0]) / steps
    delta_g = (color2[1] - color1[1]) / steps
    delta_b = (color2[2] - color1[2]) / steps
    
    colors = []
    for i in range(steps + 1):
        r = int(color1[0] + delta_r * i)
        g = int(color1[1] + delta_g * i)
        b = int(color1[2] + delta_b * i)
        colors.append((r / 255.0, g / 255.0, b / 255.0))  # 归一化到 [0, 1]
    return colors

# 将十六进制颜色转换为RGB元组
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# 定义两种颜色（十六进制）
color_a_hex = "#C6DFDF"
color_b_hex = "#6a73cf"

# 转换为RGB
color_a = hex_to_rgb(color_a_hex)
color_b = hex_to_rgb(color_b_hex)

# 插值步数
steps_ab = 100  # A到B的步数

# 在A和B之间进行插值
colors_ab = interpolate_color(color_a, color_b, steps_ab)

# 创建颜色映射
cmap = ListedColormap(colors_ab)



class LevelSet:
    def __init__(self,image):
        self.image = image
        self.phi = self._create_initial_smooth_phi()

    def _create_easy_initial_phi(self):
        rows, cols = self.image.shape
        # 首先创建二值mask
        mask = np.ones((rows, cols))
        center_r, center_c = rows//2, cols//2
        r_radius, c_radius = rows//4, cols//4
        mask[center_r-r_radius:center_r+r_radius, 
            center_c-c_radius:center_c+c_radius] = 0
        
        # 计算到边界的距离
        distance = ndimage.distance_transform_edt(mask)  # 外部距离
        distance2 = ndimage.distance_transform_edt(1-mask)  # 内部距离
        
        # 组合成符号距离函数
        phi = distance - distance2
        return phi

    def _create_initial_smooth_phi(self):
        rows, cols = self.image.shape
        mask = np.ones((rows, cols))
        
        # 调整初始矩形的位置和大小
        center_r, center_c = rows//2, cols//2
        r_radius, c_radius = rows//3, cols//3  # 改为1/3以覆盖更多区域
        mask[center_r-r_radius:center_r+r_radius, 
            center_c-c_radius:center_c+c_radius] = 0
        
        distance = ndimage.distance_transform_edt(mask)
        distance2 = ndimage.distance_transform_edt(1-mask)
        return distance - distance2
    
    def _constant_evolution(self,F=1.0,dt=0.1,n_steps=100):
        evolution_history=[]
        for step in range (n_steps):
            gradient_y, gradient_x = np.gradient(self.phi)
            gradient_magnitude = np.sqrt(gradient_x**2+gradient_y**2)

            self.phi = self.phi - dt * F * gradient_magnitude
            if step % 10 == 0:
                
                evolution_history.append(self.phi.copy())
            
        return evolution_history

    def _curvature_evolution(self,F=1.0,alpha=1.0,dt=0.1,n_steps=100):
        evolution_history = []
        for step in range (n_steps):
            gradient_y, gradient_x = np.gradient(self.phi)
            # 计算二阶导数
            gradient_xx = np.gradient(gradient_x, axis=1)  # φxx
            gradient_yy = np.gradient(gradient_y, axis=0)  # φyy
            gradient_xy = np.gradient(gradient_x, axis=0)  # φxy

            # 计算梯度幅值
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # 计算曲率
            denominator = (gradient_magnitude**3 + 1e-10)  # 避免除零
            curvature = (
                gradient_xx * gradient_y**2 + 
                gradient_yy * gradient_x**2 - 
                2 * gradient_xy * gradient_x * gradient_y
            ) / denominator

            # 更新phi（常速度项 + 曲率项）
            self.phi = self.phi - dt * (F * gradient_magnitude + alpha * curvature)
            
            if step % 10 == 0:
                evolution_history.append(self.phi.copy())
        
        return evolution_history
        
    def _calculate_energy(self):
        """优化的能量计算"""
        # 计算区域项（Chan-Vese模型）
        H = 0.5 * (1 + 2/np.pi * np.arctan(self.phi/1.0))
        c1 = np.sum(self.image * H) / (np.sum(H) + 1e-10)
        c2 = np.sum(self.image * (1-H)) / (np.sum(1-H) + 1e-10)
        
        # 增加前景区域的权重
        region = np.sum(2.0 * (self.image - c1)**2 * H + 
                        (self.image - c2)**2 * (1-H))
        
        return region

    def _edge_stopping(self):
        """改进的边缘停止函数"""
        # 计算图像梯度
        gradient_y, gradient_x = np.gradient(self.image)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 使用更敏感的边缘停止函数
        return 1.0 / (1.0 + 3.0 * gradient_magnitude**2)  # 增加系数使其对边缘更敏感
    
    def _calculate_curvature(self):
        """计算曲率"""
        gradient_y, gradient_x = np.gradient(self.phi)
        gradient_xx = np.gradient(gradient_x, axis=1)
        gradient_yy = np.gradient(gradient_y, axis=0)
        gradient_xy = np.gradient(gradient_x, axis=0)
        
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        denominator = (gradient_magnitude**3 + 1e-10)
        
        curvature = (
            gradient_xx * gradient_y**2 + 
            gradient_yy * gradient_x**2 - 
            2 * gradient_xy * gradient_x * gradient_y
        ) / denominator
        
        return curvature
    

    def _check_convergence(self, old_phi, new_phi, threshold=1e-3):
        """检查是否收敛"""
        difference = np.sum(np.abs(new_phi - old_phi))
        return difference < threshold
    

    def _chan_vese_evolution(self, mu=0.1, nu=0.0, lambda1=1.0, lambda2=1.0, 
                            initial_dt=0.5, n_steps=2000, threshold=1e-3):
        """改进的Chan-Vese模型演化，使用自适应时间步长"""
        evolution_history = []
        g = self._edge_stopping()
        
        prev_energy = float('inf')
        no_improvement_count = 0
        min_dt = 0.01  # 最小时间步长
        dt = initial_dt  # 初始时间步长
        
        # 记录初始能量
        initial_energy = self._calculate_energy()
        best_energy = initial_energy
        best_phi = self.phi.copy()
        
        for step in range(n_steps):
            old_phi = self.phi.copy()
            
            # 常规的Chan-Vese更新
            H = 0.5 * (1 + 2/np.pi * np.arctan(self.phi/1.0))
            delta = 1.0 / (np.pi * (1.0 + (self.phi/1.0)**2))
            
            c1 = np.sum(self.image * H) / (np.sum(H) + 1e-10)
            c2 = np.sum(self.image * (1-H)) / (np.sum(1-H) + 1e-10)
            
            curvature = self._calculate_curvature()
            
            region_term = (
                lambda1 * (self.image - c1)**2 - 
                lambda2 * (self.image - c2)**2
            )
            
            update = g * delta * (mu * curvature - nu - region_term)
            self.phi = self.phi + dt * update
            
            # 计算当前能量
            current_energy = self._calculate_energy()
            
            # 自适应时间步长策略
            if current_energy < prev_energy:
                # 能量下降，可以适当增加步长
                dt = min(dt * 1.1, initial_dt)
                no_improvement_count = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_phi = self.phi.copy()
            else:
                # 能量上升，减小步长并恢复上一状态
                dt = max(dt * 0.5, min_dt)
                self.phi = old_phi
                no_improvement_count += 1
            
            # 提前停止条件
            if no_improvement_count >= 500:
                print(f"Stopping early at step {step}, energy reduction: {initial_energy - best_energy}")
                self.phi = best_phi  # 恢复到最佳状态
                break
            
            # 收敛条件：能量变化很小
            energy_change = abs(current_energy - prev_energy) / abs(initial_energy)
            if energy_change < threshold and step > 1000:  # 确保至少运行100步
                print(f"Converged at step {step}, energy reduction: {initial_energy - best_energy}")
                break
            
            prev_energy = current_energy
            
            if step % 10 == 0:
                evolution_history.append(self.phi.copy())
                print(f"Step {step}, Energy: {current_energy}, dt: {dt}")
        
        return evolution_history

  

    def visualize_evolution(self, evolution_history):
        """可视化演化过程"""
        plt.figure(figsize=(15, 5))
        n_frames = len(evolution_history)
        
        # 选择3个关键帧显示
        frames_to_show = [0, n_frames//2, -1]
        titles = ['初始状态', '中间状态', '最终状态']
        
        for i, (frame, title) in enumerate(zip(frames_to_show, titles)):
            plt.subplot(1, 3, i+1)
            plt.imshow(self.image, cmap='gray')
            plt.contour(evolution_history[frame], levels=[0], colors='r')
            plt.title(title)
        
        plt.tight_layout()
        plt.show()

    def visualize_distance_functions(self):
        """可视化二值化和欧式距离函数"""
        rows, cols = self.image.shape
        
        # 创建二值mask
        mask = np.ones((rows, cols))
        center_r, center_c = rows//2, cols//2
        r_radius, c_radius = rows//3, cols//3
        mask[center_r-r_radius:center_r+r_radius, 
             center_c-c_radius:center_c+c_radius] = 0
        
        # 计算欧式距离
        distance_outside = ndimage.distance_transform_edt(mask)
        distance_inside = ndimage.distance_transform_edt(1-mask)
        signed_distance = distance_outside - distance_inside
        
        # 创建图像
        plt.figure(figsize=(15, 5))
        
        # 显示二值mask
        plt.subplot(131)
        plt.imshow(mask, cmap='gray')
        plt.colorbar()
        plt.title('二值化Mask')
        
        # 显示欧式距离场
        plt.subplot(132)
        plt.imshow(signed_distance, cmap=cmap)
        plt.colorbar()
        plt.title('符号距离函数')
        
        # 显示3D表面
        ax = plt.subplot(133, projection='3d')
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        surf = ax.plot_surface(x, y, signed_distance, cmap=cmap)
        plt.colorbar(surf)
        plt.title('距离函数3D表面')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 读取图像
    import cv2
    image_path = r'C:\Users\30766\Desktop\5.jpg'
    image = cv2.imread(image_path, 0)
    
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 改进的预处理步骤
    image = cv2.resize(image, (200, 200))
    
    # 1. 先用高斯滤波减少噪声
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. 使用Canny边缘检测
    edges = cv2.Canny(image, 50, 150)
    
    # 3. 使用CLAHE增强对比度，但减小clip limit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # 4. 温和的锐化
    kernel = np.array([[-0.5,-0.5,-0.5],
                      [-0.5, 5,-0.5],
                      [-0.5,-0.5,-0.5]])
    image = cv2.filter2D(image, -1, kernel)
    
    # 5. 添加边缘信息，但使用更小的权重
    edge_weight = 0.1
    image = cv2.addWeighted(image, 1, edges.astype(np.uint8), edge_weight, 0)
    
    # 6. 最后用双边滤波平滑但保持边缘
    image = cv2.bilateralFilter(image, 9, 25, 25)
    
    # 归一化
    image = image.astype(np.float64) / 255.0
    
    # 创建level set对象
    ls = LevelSet(image)
    
    # 可视化距离函数
    ls.visualize_distance_functions()
    
    # 使用自适应时间步长的Chan-Vese
    history = ls._chan_vese_evolution(
        mu=0.8,           # 曲率权重
        nu=0.2,           # 区域权重
        lambda1=15.0,      # 前景权重
        lambda2=1.0,      # 背景权重
        initial_dt=20,   # 初始时间步长
        n_steps=5000,     # 最大迭代次数
        threshold=1e-5    # 收敛阈值
    )
    
    # 显示演化过程
    ls.visualize_evolution(history)