#asset_[step_name].py
 
# -*- coding: utf-8 -*-
from datetime import datetime, timezone
import os
import random
import sys
from alolib.asset import Asset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, pairwise_distances_argmin_min, pairwise_distances
from sklearn.cluster import KMeans
from collections import Counter
#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class UserAsset(Asset):
    def __init__(self, asset_structure):
        super().__init__(asset_structure)
        self.args       = self.asset.load_args()
        self.config     = self.asset.load_config() # config from input asset
        self.data       = self.asset.load_data() # data from input asset 
 
    @Asset.decorator_run
    def run(self):

        base_input_path = self.asset.get_input_path() 
        csv_list = find_csv_files(base_input_path)

        
        
        
        
        
        assert len(csv_list) == 1
        self.asset.save_info(f'input csv files: {csv_list}')        
        for idx, input_file in enumerate(csv_list):
            df = pd.read_csv(input_file)
            self.data[f'dataframe{idx}'] = df
            self.asset.save_info(f"Loaded dataframe{idx}") # info logging
            self.asset.save_info(f'read dataframe from <<< {input_file}')
#        self.config['x_columns'] = self.args['x_columns']
#        self.config['y_column'] = self.args['y_column']




        df = self.data['dataframe0']        
        # train_df = pd.read_csv('renamed_train_df.csv')
        # train_df = pd.read_csv('train_df.csv')

        test_df =  df.copy()


        # train_df['기준일자'] = pd.to_datetime(train_df['기준일자'],format='%Y-%m-%d')
        test_df['기준일자'] = pd.to_datetime(test_df['기준일자'],format='%Y-%m-%d')

        # test_df에서 '최근 구매 제품 0번째' 가 notna()인 행만 필터
        test_purchase_df = test_df[test_df['최근 구매 제품 0번째'].notna()]

        # '기준일자'가 2024년 2월 1일 이전인 경우 buy_df
        buy_df = test_purchase_df[test_purchase_df['기준일자'] < pd.Timestamp('2024-02-01')]

        # '기준일자'가 2024년 2월 1일 이후이거나 같은 경우 buy_df_test
        buy_df_test = test_purchase_df[test_purchase_df['기준일자'] >= pd.Timestamp('2024-02-01')]

        # 기존 df 로드 및 데이터 분리
        df = buy_df.copy()

        excluded_columns = [
            'cust_id', 'train/test', '기준일자',
            '최근 구독 제품 0번째', '최근 구독 제품 1번째', '최근 구독 제품 2번째',
            '최근 구독 제품 3번째', '최근 구독 제품 4번째',
            '최근 구매 제품 0번째', '최근 구매 제품 1번째', '최근 구매 제품 2번째',
            '최근 구매 제품 3번째', '최근 구매 제품 4번째'
        ]
        x_columns = [col for col in df.columns if col not in excluded_columns and col != '가전구독여부']

        X = df[x_columns]
        y = df['가전구독여부']

        # 결측값 처리
        X = X.fillna(X.mean(numeric_only=True))

        # 수치형, 범주형 컬럼 분리
        cat_cols = X.select_dtypes(include='object').columns
        num_cols = X.select_dtypes(exclude='object').columns

        X_num = X[num_cols]
        X_cat = X[cat_cols]

        # One-Hot Encoding
        ohe = OneHotEncoder(handle_unknown='ignore')
        X_cat_ohe = ohe.fit_transform(X_cat)

        # TruncatedSVD를 통해 차원 축소
        svd = TruncatedSVD(n_components=10, random_state=41)
        X_cat_embedded = svd.fit_transform(X_cat_ohe)

        X_embedded = np.hstack([X_num.values, X_cat_embedded])

        # 스케일링
        scaler_1 = StandardScaler()
        X_scaled = scaler_1.fit_transform(X_embedded)

        pca_scaled = PCA(n_components=0.95, random_state=42)
        X_reduced_scaled = pca_scaled.fit_transform(X_scaled)


        selected_columns = ['cust_id', '최근 구매 제품 0번째', '최근 구매 제품 1번째', 
                            '최근 구매 제품 2번째', '최근 구매 제품 3번째', '최근 구매 제품 4번째']
        buy_df_selected = buy_df[selected_columns]

        cust_purchase_dict = buy_df_selected.set_index('cust_id').apply(lambda row: row.tolist(), axis=1).to_dict()

        all_products = set(product for purchases in cust_purchase_dict.values() for product in purchases if pd.notna(product))

        all_products_list = list(all_products)


        cust_purchase_frequency_vector = {
            cust_id: [purchases.count(product) for product in all_products_list]
            for cust_id, purchases in cust_purchase_dict.items()
        }

        cust_purchase_frequency_vector

        # 1. Transition Matrix 생성
        product_to_index = {product: i for i, product in enumerate(all_products_list)}
        num_products = len(all_products_list)

        cust_transition_matrices = {}

        for cust_id, purchases in cust_purchase_dict.items():
            # 전이행렬 초기화
            transition_matrix = np.zeros((num_products, num_products))

            # purchases가 예: [p1, p2, p3, ...] 라면
            # p1 -> p2 전이, p2 -> p3 전이, ... 를 카운트
            for i in range(len(purchases)-1):
                if pd.notna(purchases[i]) and pd.notna(purchases[i+1]):
                    from_idx = product_to_index[purchases[i]]
                    to_idx = product_to_index[purchases[i+1]]
                    transition_matrix[from_idx, to_idx] += 1


            cust_transition_matrices[cust_id] = transition_matrix

        # 2. 각 고객의 전이 행렬을 벡터로 펼친다 (Flatten)
        cust_ids = list(cust_transition_matrices.keys())
        transition_vectors = np.array([cust_transition_matrices[cust_id].flatten() for cust_id in cust_ids])

        # 3. PCA를 통해 전이 행렬 벡터를 10차원으로 축소
        pca_for_transition = PCA(n_components=10, random_state=42)
        transition_reduced = pca_for_transition.fit_transform(transition_vectors)

        
        frequency_vector = np.array([cust_purchase_frequency_vector[cust_id] for cust_id in cust_ids])

        # 전이행렬 특징(10차원)과 구매 빈도 벡터를 병합
        # 두 벡터를 가로 concat
        final_features = np.hstack([transition_reduced, frequency_vector])
        final_features

        X_combined = np.hstack([X_reduced_scaled, final_features])

        scaler_2 = StandardScaler()
        X_combined_scaled = scaler_2.fit_transform(X_combined)

        customer_ids = list(cust_purchase_frequency_vector.keys())        

        
        
        
        
        
        
        
        
        
        
        
        sample_fraction = 0.001

        best_k = None
        best_score = -1

        for k in range(10, 90):
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            clusters_temp = kmeans_temp.fit_predict(X_combined_scaled)

            # 데이터 샘플링
            num_samples = int(X_combined_scaled.shape[0] * sample_fraction)
            sample_indices = np.random.choice(X_combined_scaled.shape[0], size=num_samples, replace=False)
            X_sample = X_combined_scaled[sample_indices]
            clusters_sample = clusters_temp[sample_indices]

            # 샘플에 대해 실루엣 스코어 계산
            score = silhouette_score(X_sample, clusters_sample)
            print('k:', k, 'score:', score)

            if score > best_score:
                best_score = score
                best_k = k

        print(f"Best k: {best_k} with a silhouette score of {best_score:.4f}")
        
        
        
        
        
        
        
        
        
        
        
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        clusters = kmeans.fit_predict(X_combined_scaled)

        customer_clusters = {customer_id: cluster for customer_id, cluster in zip(customer_ids, clusters)}
        cluster_centers = kmeans.cluster_centers_



        closest_indices, _ = pairwise_distances_argmin_min(cluster_centers, X_combined_scaled)

        closest_customer_ids = [customer_ids[idx] for idx in closest_indices]


        cluster_counts = Counter(clusters)

        cluster_counts_dict = dict(cluster_counts)

        cluster_to_cust_ids = {cluster: [] for cluster in set(clusters)}

        for cust_id, cluster in zip(customer_ids, clusters):
            cluster_to_cust_ids[cluster].append(cust_id)


        if '가전구독여부' in buy_df.columns:
            buy_df['Cluster'] = buy_df['cust_id'].map(customer_clusters)

            cluster_subscription_stats = buy_df.groupby('Cluster')['가전구독여부'].value_counts(normalize=True).unstack().fillna(0)

            cluster_counts_dict = buy_df['Cluster'].value_counts().to_dict()
            cluster_subscription_stats['Cluster_Point_Count'] = cluster_subscription_stats.index.map(cluster_counts_dict)

        buy_df_selected = df.set_index('cust_id').loc[closest_customer_ids].reset_index()

        combined_df = pd.concat([cluster_subscription_stats.reset_index(drop=True), 
                                 buy_df_selected[selected_columns].reset_index(drop=True), 
                                 buy_df_selected.drop(columns=selected_columns).reset_index(drop=True)], axis=1)
        combined_df.insert(0, 'Cluster', combined_df.index)
        combined_df.insert(3, '구독지수', combined_df[1]/buy_df['가전구독여부'].mean())
        combined_df       

        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # 클러스터 간 거리 행렬 계산
        dist_mat = pairwise_distances(cluster_centers, cluster_centers)

        subscription_index = combined_df['구독지수'].values
        cluster_ids = combined_df['Cluster'].values

        closest_clusters_dict = {}

        for i, cid in enumerate(cluster_ids):
            current_sub_index = subscription_index[i]

            if current_sub_index < 1:
                # 구독지수가 1 이상인 클러스터 후보들
                candidates = [j for j, s in enumerate(subscription_index) if s >= 1 and j != i]
            else:
                # 구독지수가 현재 클러스터보다 높은 클러스터 후보들
                candidates = [j for j, s in enumerate(subscription_index) if s > current_sub_index and j != i]

            if candidates:
                # 후보들 중 가장 가까운 클러스터 찾기
                closest_candidate = min(candidates, key=lambda x: dist_mat[i, x])
                closest_clusters_dict[cid] = {
                    'closest_cluster': int(closest_candidate), 
                    'closest_subscription_score': float(subscription_index[closest_candidate])
                }
            else:
                closest_clusters_dict[cid] = None

        closest_clusters_dict
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # 'Closest_Cluster' 열 추가
        combined_df['Closest_Cluster'] = combined_df['Cluster'].map(
            lambda x: closest_clusters_dict[x]['closest_cluster'] if closest_clusters_dict[x] is not None else None
        )

        # 'Closest_Subscription_Score' 열 추가
        combined_df['Closest_Subscription_Score'] = combined_df['Cluster'].map(
            lambda x: closest_clusters_dict[x]['closest_subscription_score'] if closest_clusters_dict[x] is not None else None
        )

        combined_df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        model_path = self.asset.get_model_path()  
        try: 
            with open(model_path + 'svd.pkl', 'wb') as file:
                pickle.dump(svd, file)
        
        
        model_path = self.asset.get_model_path()  
        try: 
            with open(model_path + 'scaler_1.pkl', 'wb') as file:
                pickle.dump(scaler_1, file)  
        
        model_path = self.asset.get_model_path()  
        try: 
            with open(model_path + 'scaler_2.pkl', 'wb') as file:
                pickle.dump(scaler_2, file) 
        
        model_path = self.asset.get_model_path()  
        try: 
            with open(model_path + 'pca_for_transition.pkl', 'wb') as file:
                pickle.dump(pca_for_transition, file) 

        model_path = self.asset.get_model_path()  
        try: 
            with open(model_path + 'kmeans.pkl', 'wb') as file:
                pickle.dump(kmeans, file)                
              
            
            

            
            
            
            
            
            
            
            

        X_test = buy_df_test[[col for col in buy_df_test.columns if col not in excluded_columns and col != '가전구독여부']]
        X_test = X_test.fillna(X_test.mean(numeric_only=True))

        cat_cols_test = X_test.select_dtypes(include='object').columns
        num_cols_test = X_test.select_dtypes(exclude='object').columns

        X_test_num = X_test[num_cols_test]
        X_test_cat = X_test[cat_cols_test]

        X_test_cat_ohe = ohe.transform(X_test_cat) 
        X_test_cat_embedded = svd.transform(X_test_cat_ohe)  # 훈련 때 fit한 svd 객체 사용

        # 수치형 + 임베딩된 범주형 변수 결합
        X_test_embedded = np.hstack([X_test_num.values, X_test_cat_embedded])

        # 훈련 시 fit한 scaler_1 사용
        X_test_scaled = scaler_1.transform(X_test_embedded)

        # 훈련 시 fit한 PCA(pca_scaled) 사용
        X_test_reduced_scaled = pca_scaled.transform(X_test_scaled)

        buy_df_test_selected = buy_df_test[selected_columns]
        cust_purchase_dict_test = buy_df_test_selected.set_index('cust_id').apply(lambda row: row.tolist(), axis=1).to_dict()

        cust_purchase_frequency_vector_test = {
            cust_id: [purchases.count(product) for product in all_products_list]
            for cust_id, purchases in cust_purchase_dict_test.items()
        }
        frequency_vector_test = np.array([cust_purchase_frequency_vector_test[cust_id] for cust_id in cust_purchase_dict_test.keys()])

        test_cust_transition_matrices = {}
        for cust_id, purchases in cust_purchase_dict_test.items():
            transition_matrix = np.zeros((num_products, num_products))
            for i in range(len(purchases)-1):
                if pd.notna(purchases[i]) and pd.notna(purchases[i+1]):
                    from_idx = product_to_index[purchases[i]]
                    to_idx = product_to_index[purchases[i+1]]
                    transition_matrix[from_idx, to_idx] += 1
            test_cust_transition_matrices[cust_id] = transition_matrix

        test_cust_ids = list(test_cust_transition_matrices.keys())
        test_transition_vectors = np.array([test_cust_transition_matrices[cid].flatten() for cid in test_cust_ids])

        test_transition_reduced = pca_for_transition.transform(test_transition_vectors)


        test_final_features = np.hstack([X_test_reduced_scaled, test_transition_reduced, frequency_vector_test])

        X_test_combined_scaled = scaler_2.transform(test_final_features)

        clusters_test = kmeans.predict(X_test_combined_scaled)

        buy_df_test['Cluster'] = clusters_test

        cluster_subscription_stats_test = buy_df_test.groupby('Cluster')['가전구독여부'].value_counts(normalize=True).unstack().fillna(0)
        cluster_subscription_stats_test['Cluster_Point_Count'] = buy_df_test['Cluster'].value_counts()

        combined_df_tmp = combined_df.copy()
        combined_df_tmp.insert(4, 'test_set_구독지수', cluster_subscription_stats_test.reset_index()[1]/
                               buy_df_test['가전구독여부'].mean())
        combined_df_tmp['Cluster_Point_Count'] = cluster_subscription_stats_test.reset_index(drop = True)['Cluster_Point_Count']
        
        test_data_cluster_info = combined_df_tmp.copy()
        train_data_cluster_info = combined_df.copy()

        
        
        
        
        
        self.asset.save_data(test_data_cluster_info)
        self.asset.save_data(train_data_cluster_info)
        self.asset.save_data(buy_df)
        self.asset.save_data(buy_df_test)

#         self.asset.save_data(buy_df)
        self.asset.save_config(self.config)        
        
        
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()

