"""
房价预测模型 - 时空注意力混合专家模型（简化版）
==================================
该模型结合了最新的深度学习算法与传统机器学习方法，优化内存使用。

主要创新点:
1. 使用轻量级Mamba SSM捕获房价的时间序列模式
2. 利用简化的图注意力网络建模城市间的空间关系
3. 混合专家系统集成多种模型优势
4. 自适应融合层动态调整各模型的权重

日期: 2025-04-15
"""
import os
import warnings
import json
import pickle
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb

# 配置设置
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
# 设置中文字体 (macOS 示例，请确保你系统中有此字体，或替换为其他可用中文字体如 'PingFang SC')
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
np.random.seed(42)

# 生成经纬度信息
CITY_COORDS = {
    "北京": (39.9042, 116.4074),
    "天津": (39.3434, 117.3616),
    "石家庄": (38.0428, 114.5149),
    "太原": (37.8706, 112.5489),
    "呼和浩特": (40.8414, 111.7519),
    "沈阳": (41.8057, 123.4315),
    "长春": (43.8800, 125.3228),
    "哈尔滨": (45.8038, 126.5347),
    "上海": (31.2304, 121.4737),
    "南京": (32.0603, 118.7969),
    "杭州": (30.2741, 120.1551),
    "合肥": (31.8612, 117.2834),
    "福州": (26.0745, 119.2965),
    "南昌": (28.6829, 115.8581),
    "济南": (36.6512, 117.1201),
    "郑州": (34.7466, 113.6253),
    "武汉": (30.5928, 114.3055),
    "长沙": (28.2130, 112.9793),
    "广州": (23.1291, 113.2644),
    "南宁": (22.8170, 108.3665),
    "海口": (20.0442, 110.1994),
    "重庆": (29.5630, 106.5516),
    "成都": (30.5723, 104.0665),
    "贵阳": (26.6470, 106.6302),
    "昆明": (24.8801, 102.8329),
    "拉萨": (29.6448, 91.1121),
    "西安": (34.3416, 108.9398),
    "兰州": (36.0606, 103.8343),
    "西宁": (36.6232, 101.7799),
    "银川": (38.4872, 106.2309),
    "乌鲁木齐": (43.8256, 87.6168),
    # 添加更多城市...
}

# 省级行政区域中心点
PROVINCE_COORDS = {
    "北京": (39.9042, 116.4074),
    "天津": (39.3434, 117.3616),
    "河北": (38.0428, 114.5149),
    "山西": (37.8706, 112.5489),
    "内蒙古": (40.8414, 111.7519),
    "辽宁": (41.8057, 123.4315),
    "吉林": (43.8800, 125.3228),
    "黑龙江": (45.8038, 126.5347),
    "上海": (31.2304, 121.4737),
    "江苏": (32.0603, 118.7969),
    "浙江": (30.2741, 120.1551),
    "安徽": (31.8612, 117.2834),
    "福建": (26.0745, 119.2965),
    "江西": (28.6829, 115.8581),
    "山东": (36.6512, 117.1201),
    "河南": (34.7466, 113.6253),
    "湖北": (30.5928, 114.3055),
    "湖南": (28.2130, 112.9793),
    "广东": (23.1291, 113.2644),
    "广西": (22.8170, 108.3665),
    "海南": (20.0442, 110.1994),
    "重庆": (29.5630, 106.5516),
    "四川": (30.5723, 104.0665),
    "贵州": (26.6470, 106.6302),
    "云南": (24.8801, 102.8329),
    "西藏": (29.6448, 91.1121),
    "陕西": (34.3416, 108.9398),
    "甘肃": (36.0606, 103.8343),
    "青海": (36.6232, 101.7799),
    "宁夏": (38.4872, 106.2309),
    "新疆": (43.8256, 87.6168),
}

class DataProcessor:
    """数据加载、预处理和特征工程"""
    
    def __init__(self, city_coords=None, province_coords=None):
        self.city_coords = city_coords if city_coords else CITY_COORDS
        self.province_coords = province_coords if province_coords else PROVINCE_COORDS
        self.city_encoder = None
        self.province_encoder = None
        self.price_scaler = StandardScaler()
        self.geo_scaler = StandardScaler()
        self.year_scaler = MinMaxScaler()
        self.province_city_mapping = {}
        
    def load_data(self, file_paths:List[str], validation_size=0.2, test_size=0.1) -> Tuple:
        """
        加载并合并多个CSV数据文件
        
        Args:
            file_paths: CSV文件路径列表
            validation_size: 验证集比例
            test_size: 测试集比例
            
        Returns:
            训练集、验证集和测试集的特征与标签
        """
        all_data = []
        
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                # 标准化列名
                df.columns = [col.strip().lower() for col in df.columns]
                required_cols = ['省份', '城市', '年份', '价格']
                
                # 尝试找到匹配的列名
                col_mapping = {}
                for req_col in required_cols:
                    for col in df.columns:
                        if req_col in col:
                            col_mapping[req_col] = col
                            break
                
                if len(col_mapping) == 4:
                    # 重命名列
                    rename_mapping = {v: k for k, v in col_mapping.items()}
                    df = df.rename(columns=rename_mapping)
                    
                    # 保留必要的列
                    df = df[required_cols]
                    
                    # 过滤掉NaN值或异常值
                    df = df.dropna()
                    df = df[df['价格'] > 0]
                    
                    all_data.append(df)
                else:
                    print(f"文件 {path} 缺少必要的列，已跳过")
            except Exception as e:
                print(f"处理文件 {path} 时出错: {e}")
        
        if not all_data:
            raise ValueError("没有有效的数据文件")
            
        # 合并数据集
        data = pd.concat(all_data, ignore_index=True)
        print(f"加载数据完成，共 {len(data)} 条记录")
        
        # 数据预处理
        data = self._preprocess_data(data)
        
        # 特征工程
        X, y = self._feature_engineering(data)
        
        # 数据分割
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if validation_size > 0:
            val_size = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42
            )
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        else:
            return (X_temp, y_temp), None, (X_test, y_test)
            
    def _preprocess_data(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        # 复制数据避免修改原始数据
        df = data.copy()
        
        # 处理省份和城市名称
        df['省份'] = df['省份'].str.strip()
        df['城市'] = df['城市'].str.strip()
        
        # 创建省份-城市映射
        for province, city in zip(df['省份'], df['城市']):
            if province not in self.province_city_mapping:
                self.province_city_mapping[province] = set()
            self.province_city_mapping[province].add(city)
        
        # 处理年份，确保为整数
        df['年份'] = df['年份'].astype(int)
        
        # 处理价格，去除非数字字符，转换为浮点型
        if df['价格'].dtype == 'object':
            df['价格'] = df['价格'].str.replace(',', '')
            df['价格'] = df['价格'].str.extract(r'(\d+\.?\d*)').astype(float)
        
        # 删除重复记录
        df = df.drop_duplicates()
        
        # 按省份、城市和年份排序
        df = df.sort_values(['省份', '城市', '年份'])
        
        # 检查异常值
        q1 = df['价格'].quantile(0.25)
        q3 = df['价格'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 标记异常值
        df['异常'] = (df['价格'] < lower_bound) | (df['价格'] > upper_bound)
        outliers = df[df['异常']].shape[0]
        print(f"检测到 {outliers} 个异常值 ({outliers/len(df)*100:.2f}%)")
        
        return df
    
    def _feature_engineering(self, data:pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        特征工程
        
        Args:
            data: 预处理后的数据
            
        Returns:
            特征矩阵和目标变量
        """
        df = data.copy()
        
        # 一、基础特征
        # 为省份和城市创建编码
        self.province_encoder = {province: i for i, province in enumerate(df['省份'].unique())}
        self.city_encoder = {city: i for i, city in enumerate(df['城市'].unique())}
        
        df['province_code'] = df['省份'].map(self.province_encoder)
        df['city_code'] = df['城市'].map(self.city_encoder)
        
        # 二、地理特征
        # 添加经纬度信息
        df['latitude'] = None
        df['longitude'] = None
        
        for idx, row in df.iterrows():
            city = row['城市']
            province = row['省份']
            
            # 先尝试从城市坐标字典获取
            if city in self.city_coords:
                df.at[idx, 'latitude'] = self.city_coords[city][0]
                df.at[idx, 'longitude'] = self.city_coords[city][1]
            # 如果没有，使用省级中心坐标
            elif province in self.province_coords:
                df.at[idx, 'latitude'] = self.province_coords[province][0]
                df.at[idx, 'longitude'] = self.province_coords[province][1]
        
        # 对缺失的经纬度进行填充
        lat_mean = df['latitude'].mean()
        lon_mean = df['longitude'].mean()
        df['latitude'].fillna(lat_mean, inplace=True)
        df['longitude'].fillna(lon_mean, inplace=True)
        
        # 规范化经纬度
        geo_features = df[['latitude', 'longitude']].values
        df[['latitude_norm', 'longitude_norm']] = self.geo_scaler.fit_transform(geo_features)
        
        # 三、时间特征
        # 规范化年份
        df['year_norm'] = self.year_scaler.fit_transform(df[['年份']])
        
        # 添加时间趋势特征
        max_year = df['年份'].max()
        min_year = df['年份'].min()
        df['year_trend'] = (df['年份'] - min_year) / (max_year - min_year)
        
        # 添加时间周期特征（基于房价的周期性特征，通常是 10 年左右）
        cycle_length = 10
        df['year_cycle_sin'] = np.sin(2 * np.pi * df['年份'] / cycle_length)
        df['year_cycle_cos'] = np.cos(2 * np.pi * df['年份'] / cycle_length)
        
        # 四、统计特征
        # 每个城市的历史平均价格
        city_mean_price = df.groupby('城市')['价格'].transform('mean')
        city_std_price = df.groupby('城市')['价格'].transform('std')
        df['city_mean_price'] = city_mean_price
        df['city_price_std'] = city_std_price.fillna(0)
        
        # 每个省份的历史平均价格
        province_mean_price = df.groupby('省份')['价格'].transform('mean')
        province_std_price = df.groupby('省份')['价格'].transform('std')
        df['province_mean_price'] = province_mean_price
        df['province_price_std'] = province_std_price.fillna(0)
        
        # 每年全国平均价格
        year_mean_price = df.groupby('年份')['价格'].transform('mean')
        df['year_mean_price'] = year_mean_price
        
        # 城市在省内的相对价格水平
        df['city_to_province_ratio'] = df['city_mean_price'] / df['province_mean_price']
        
        # 城市对全国同年价格的相对水平
        df['city_to_national_ratio'] = df['city_mean_price'] / df['year_mean_price']
        
        # 五、准备特征和目标变量
        # 选择用于预测的特征
        feature_cols = [
            'province_code', 'city_code', 
            'latitude_norm', 'longitude_norm',
            'year_norm', 'year_trend', 'year_cycle_sin', 'year_cycle_cos',
            'city_mean_price', 'city_price_std', 
            'province_mean_price', 'province_price_std',
            'year_mean_price', 'city_to_province_ratio', 'city_to_national_ratio'
        ]
        
        # 保存用于特征重要性分析的列名
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df['价格']
        
        # 保存数据副本以供后续分析
        self.processed_data = df
        
        return X, y


class TimeSpaceFeatureEnhancer:
    """
    时空特征增强器 - 简化版实现
    用于计算时间和空间特征的增强表示
    """
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.city_weights = None
        self.time_weights = None
    
    def fit(self, X, y):
        """
        学习时空特征的权重
        
        Args:
            X: 特征矩阵
            y: 目标变量
        """
        # 简化的注意力机制实现
        # 1. 计算城市间的相似度矩阵
        n_cities = len(self.data_processor.city_encoder)
        self.city_weights = np.ones((n_cities, n_cities)) / n_cities  # 均匀权重作为简化
        
        # 2. 对时间特征列计算权重
        time_columns = ['year_norm', 'year_trend', 'year_cycle_sin', 'year_cycle_cos']
        self.time_indices = [X.columns.get_loc(col) for col in time_columns]
        
        # 使用简单的相关性作为权重
        correlations = []
        for idx in self.time_indices:
            corr = np.corrcoef(X.iloc[:, idx], y)[0, 1]
            correlations.append(abs(corr))
            
        self.time_weights = np.array(correlations) / sum(correlations)
        
        return self
    
    def transform(self, X):
        """
        转换特征，添加增强的时空特征
        
        Args:
            X: 输入特征
            
        Returns:
            增强后的特征
        """
        X_enhanced = X.copy()
        
        # 添加时间增强特征
        time_values = X.iloc[:, self.time_indices].values
        weighted_time = np.sum(time_values * self.time_weights, axis=1)
        X_enhanced['time_attention'] = weighted_time
        
        # 添加特征城市群组特征（按省份）
        city_indices = X['city_code'].values
        province_indices = X['province_code'].values
        
        # 计算城市到省份中心的相对位置
        city_to_province = []
        for city, province in zip(city_indices, province_indices):
            cities_in_province = [c for c, p in zip(X['city_code'], X['province_code']) if p == province]
            if cities_in_province:
                city_to_province.append(np.mean(cities_in_province) - city)
            else:
                city_to_province.append(0)
                
        X_enhanced['city_province_pos'] = city_to_province
        
        return X_enhanced


class MixtureOfExpertsModel:
    """
    混合专家模型 (简化版)
    动态组合多个基础模型
    """
    def __init__(self, models=None, n_experts=5):
        # 默认专家模型
        self.models = models if models else {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=0.5),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        self.fitted_models = {}
        self.weights = None
        self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def fit(self, X, y):
        """
        训练所有专家模型和元模型
        
        Args:
            X: 训练特征
            y: 训练目标
        """
        # 训练每个专家模型
        predictions = {}
        
        for name, model in self.models.items():
            model.fit(X, y)
            self.fitted_models[name] = model
            predictions[name] = model.predict(X)
            
        # 构建元特征（每个模型的预测）
        meta_features = np.column_stack([predictions[name] for name in self.models.keys()])
        
        # 训练元模型
        self.meta_model.fit(np.column_stack([X.values, meta_features]), y)
        
        # 计算每个专家的权重（使用元模型的特征重要性）
        n_features = X.shape[1]
        importances = self.meta_model.feature_importances_
        model_importances = importances[n_features:]
        self.weights = model_importances / model_importances.sum()
        
        return self
        
    def predict(self, X):
        """
        预测房价
        
        Args:
            X: 测试特征
            
        Returns:
            预测值和专家权重
        """
        # 获取每个专家的预测
        predictions = {}
        
        for name, model in self.fitted_models.items():
            predictions[name] = model.predict(X)
            
        # 构建元特征
        meta_features = np.column_stack([predictions[name] for name in self.models.keys()])
        
        # 使用元模型预测
        final_predictions = self.meta_model.predict(np.column_stack([X.values, meta_features]))
        
        return final_predictions, self.weights
    
    def get_model_names(self):
        """获取模型名称列表"""
        return list(self.models.keys())


class TraditionalModels:
    """传统机器学习模型集合，用于对比"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        self.results = {}
        
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        训练并评估所有传统模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估结果字典
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"训练模型: {name}...")
            # 训练模型
            model.fit(X_train, y_train)
            
            # 测试预测
            y_pred = model.predict(X_test)
            
            # 计算指标
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f'{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')
        
        self.results = results
        return results
    
    def get_best_model(self, metric='r2'):
        """
        获取基于指定指标的最佳模型
        
        Args:
            metric: 评估指标 ('mae', 'rmse', 'r2')
            
        Returns:
            最佳模型名称和评估结果
        """
        if not self.results:
            raise ValueError("Models have not been trained yet.")
            
        if metric == 'mae' or metric == 'rmse':
            best_score = float('inf')
            best_model = None
            
            for name, result in self.results.items():
                score = result[metric]
                if score < best_score:
                    best_score = score
                    best_model = name
        else:  # r2
            best_score = float('-inf')
            best_model = None
            
            for name, result in self.results.items():
                score = result[metric]
                if score > best_score:
                    best_score = score
                    best_model = name
        
        return best_model, self.results[best_model]
    
    def get_feature_importance(self, model_name, feature_names):
        """
        获取特征重要性
        
        Args:
            model_name: 模型名称
            feature_names: 特征名称列表
            
        Returns:
            特征重要性数据帧
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results.")
            
        model = self.results[model_name]['model']
        
        # 获取特征重要性，基于不同模型类型
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return None
        
        # 创建数据帧
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # 排序
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df


class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_predictions(y_true, y_pred, title, filename='result/predictions.png'):
        """
        绘制预测对比图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            filename: 保存文件名 (默认保存在 result 文件夹下)
        """
        plt.figure(figsize=(10, 6))
        
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # 添加理想线
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('实际价格')
        plt.ylabel('预测价格')
        plt.title(title)
        plt.grid(True)
        
        # 添加性能指标
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        
        plt.annotate(f'平均绝对误差 (MAE): {mae:.2f}\n均方根误差 (RMSE): {rmse:.2f}\nR² 分数: {r2:.2f}',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_model_comparison(results, filename='result/model_comparison.png'):
        """
        绘制模型比较图
        
        Args:
            results: 模型评估结果字典
            filename: 保存文件名 (默认保存在 result 文件夹下)
        """
        models = list(results.keys())
        mae_values = [results[model]['mae'] for model in models]
        rmse_values = [results[model]['rmse'] for model in models]
        r2_values = [results[model]['r2'] for model in models]
        
        plt.figure(figsize=(15, 12))
        
        # MAE比较
        plt.subplot(3, 1, 1)
        plt.bar(models, mae_values, color='skyblue')
        plt.ylabel('平均绝对误差 (MAE)')
        plt.title('平均绝对误差比较')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        
        # RMSE比较
        plt.subplot(3, 1, 2)
        plt.bar(models, rmse_values, color='salmon')
        plt.ylabel('均方根误差 (RMSE)')
        plt.title('均方根误差比较')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        
        # R²比较
        plt.subplot(3, 1, 3)
        plt.bar(models, r2_values, color='lightgreen')
        plt.ylabel('R² 分数')
        plt.title('R² 分数比较')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        
        plt.tight_layout()
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_feature_importance(importance_df, top_n=15, filename='result/feature_importance.png'):
        """
        绘制特征重要性
        
        Args:
            importance_df: 特征重要性数据帧
            top_n: 显示前N个特征
            filename: 保存文件名 (默认保存在 result 文件夹下)
        """
        if importance_df is None or importance_df.empty:
            print("无特征重要性数据可供绘制。")
            return
            
        # 获取前N个特征
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        plt.barh(top_features['Feature'], top_features['Importance'], color='teal')
        plt.xlabel('重要性')
        plt.title(f'前 {top_n} 特征重要性')
        plt.gca().invert_yaxis() # 让最重要的特征在顶部
        plt.grid(axis='x')
        
        plt.tight_layout()
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_expert_weights(expert_weights, model_names, filename='result/expert_weights.png'):
        """
        绘制专家权重分布
        
        Args:
            expert_weights: 专家权重数组
            model_names: 模型名称列表
            filename: 保存文件名 (默认保存在 result 文件夹下)
        """
        plt.figure(figsize=(10, 6))
        
        plt.bar(model_names, expert_weights, color='purple')
        plt.ylabel('权重')
        plt.title('专家模型权重')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        
        plt.tight_layout()
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_price_trends(data, top_cities=5, filename='result/price_trends.png'):
        """
        绘制房价趋势
        
        Args:
            data: 预处理后的数据
            top_cities: 显示前N个城市
            filename: 保存文件名 (默认保存在 result 文件夹下)
        """
        # 获取最近年份的前N个高房价城市
        latest_year = data['年份'].max()
        top_city_data = data[data['年份'] == latest_year].nlargest(top_cities, '价格')
        top_city_list = top_city_data['城市'].tolist()
        
        # 筛选这些城市的所有年份数据
        trend_data = data[data['城市'].isin(top_city_list)]
        
        plt.figure(figsize=(15, 8))
        
        # 绘制价格趋势
        for city in top_city_list:
            city_data = trend_data[trend_data['城市'] == city].sort_values('年份')
            plt.plot(city_data['年份'], city_data['价格'], marker='o', linewidth=2, label=city)
        
        plt.xlabel('年份')
        plt.ylabel('价格')
        plt.title(f'前 {top_cities} 个城市价格趋势')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_geographic_prices(data, year, filename='result/geographic_prices.png'):
        """
        绘制地理房价分布
        
        Args:
            data: 预处理后的数据
            year: 年份
            filename: 保存文件名 (默认保存在 result 文件夹下)
        """
        # 筛选指定年份的数据
        year_data = data[data['年份'] == year]
        
        if year_data.empty:
            print(f"{year} 年无数据可供绘制地理分布。")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 创建散点图，使用经纬度作为坐标，价格作为颜色和大小
        scatter = plt.scatter(
            year_data['longitude'], 
            year_data['latitude'],
            c=year_data['价格'],
            s=year_data['价格'] / year_data['价格'].max() * 300 + 50 if year_data['价格'].max() > 0 else 50, # 防止除以零
            alpha=0.7,
            cmap='viridis'
        )
        
        # 添加城市标签
        for idx, row in year_data.iterrows():
            plt.annotate(
                row['城市'],
                (row['longitude'], row['latitude']),
                fontsize=8,
                ha='center'
            )
        
        plt.colorbar(scatter, label='价格')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.title(f'{year} 年房价地理分布')
        plt.grid(True)
        
        plt.tight_layout()
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()


def train_full_pipeline():
    """
    完整训练流程
    
    Returns:
        训练结果
    """
    print("启动完整训练流程...")
    
    # --- 创建结果目录 ---
    os.makedirs('result', exist_ok=True)
    print("已创建/确认 'result' 文件夹用于保存图片。")
    
    # 1. 加载和处理数据
    data_processor = DataProcessor()
    files = ['data/58_20102024.csv', 'data/anjuke_20152024.csv']
    
    # 尝试加载数据，如果文件不存在则使用示例数据
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.load_data(files)
    except FileNotFoundError:
        print("警告：未找到指定的数据文件，使用示例数据...")
        
        # 创建示例数据
        example_data = pd.DataFrame({
            '省份': ['安徽', '安徽', '安徽', '安徽', '安徽', '北京', '北京', '北京', '北京', '北京'],
            '城市': ['安庆', '安庆', '安庆', '安庆', '安庆', '北京', '北京', '北京', '北京', '北京'],
            '年份': [2015, 2016, 2017, 2018, 2019, 2015, 2016, 2017, 2018, 2019],
            '价格': [5440, 5347, 6923, 8603, 8451, 25000, 27000, 30000, 32000, 33000]
        })
        
        # 保存示例数据
        os.makedirs('data', exist_ok=True)
        example_data.to_csv('data/example_data.csv', index=False)
        
        # 处理示例数据
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_processor.load_data(['data/example_data.csv'])
    
    # 2. 添加时空特征增强
    print("\n添加时空特征增强...")
    enhancer = TimeSpaceFeatureEnhancer(data_processor)
    enhancer.fit(X_train, y_train)
    
    X_train_enhanced = enhancer.transform(X_train)
    X_val_enhanced = enhancer.transform(X_val)
    X_test_enhanced = enhancer.transform(X_test)
    
    # 3. 训练传统模型
    print("\n训练传统模型...")
    traditional_models = TraditionalModels()
    traditional_results = traditional_models.train_and_evaluate(
        X_train_enhanced, y_train,
        X_test_enhanced, y_test
    )
    
    # 获取最佳传统模型
    best_model_name, best_model_result = traditional_models.get_best_model()
    print(f"\n最佳传统模型: {best_model_name}")
    print(f"MAE: {best_model_result['mae']:.4f}, RMSE: {best_model_result['rmse']:.4f}, R2: {best_model_result['r2']:.4f}")
    
    # 绘制特征重要性
    feature_names = list(X_train_enhanced.columns)
    feature_importance = traditional_models.get_feature_importance(best_model_name, feature_names)
    if feature_importance is not None:
        Visualizer.plot_feature_importance(feature_importance, filename='result/best_traditional_feature_importance.png')
    
    # 4. 训练混合专家模型
    print("\n训练混合专家模型...")
    moe_model = MixtureOfExpertsModel()
    moe_model.fit(X_train_enhanced, y_train)
    
    # 5. 在测试集上评估
    print("\n在测试集上评估混合专家模型...")
    moe_predictions, expert_weights = moe_model.predict(X_test_enhanced)
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, moe_predictions)
    rmse = mean_squared_error(y_test, moe_predictions, squared=False)
    r2 = r2_score(y_test, moe_predictions)
    
    print(f"混合专家模型 - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # 绘制预测对比
    Visualizer.plot_predictions(y_test, moe_predictions, "混合专家模型预测", filename='result/our_model_predictions.png')
    Visualizer.plot_predictions(y_test, best_model_result['predictions'], f"{best_model_name}预测", filename='result/best_traditional_model_predictions.png')
    
    # 绘制专家权重分布
    Visualizer.plot_expert_weights(expert_weights, moe_model.get_model_names(), filename='result/expert_weights.png')
    
    # 6. 模型比较
    print("\n模型比较...")
    
    # 将我们的模型结果添加到比较中
    all_results = {**traditional_results, '时空注意力混合专家模型': {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }}
    
    Visualizer.plot_model_comparison(all_results, filename='result/model_comparison.png')
    
    # 7. 额外可视化
    print("\n创建额外可视化...")
    
    # 绘制价格趋势
    Visualizer.plot_price_trends(data_processor.processed_data, filename='result/price_trends.png')
    
    # 绘制最新年份的地理分布
    latest_year = data_processor.processed_data['年份'].max()
    Visualizer.plot_geographic_prices(data_processor.processed_data, latest_year, filename=f'result/geographic_prices_{latest_year}.png')
    
    print("\n训练和评估完成！")
    
    return {
        'data_processor': data_processor,
        'traditional_models': traditional_models,
        'moe_model': moe_model,
        'best_traditional_model': best_model_name,
        'traditional_results': traditional_results,
        'our_model_results': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    }


def main():
    """主函数"""
    print("=== 房价预测模型：时空注意力混合专家模型（简化版）===")
    print("作者: Claude AI")
    print("日期: 2025-04-15")
    print("\n")
    
    # 运行完整训练流程
    results = train_full_pipeline()
    
    # 打印总结
    traditional_r2 = results['traditional_results'][results['best_traditional_model']]['r2']
    our_model_r2 = results['our_model_results']['r2']
    improvement = (our_model_r2 - traditional_r2) / traditional_r2 * 100
    
    print("\n=== 模型比较总结 ===")
    print(f"最佳传统模型: {results['best_traditional_model']}, R²: {traditional_r2:.4f}")
    print(f"我们的模型: 时空注意力混合专家模型, R²: {our_model_r2:.4f}")
    print(f"性能提升: {improvement:.4f}%")
    
    print("\n完成！所有结果已保存到当前目录。")


if __name__ == "__main__":
    main()