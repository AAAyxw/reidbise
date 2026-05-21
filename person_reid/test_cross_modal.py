import sys
import cv2
import numpy as np
sys.path.append('d:/reidbise-yolov12/person_reid')

from Algorithm.libs.extract.reid_extract import CrossModalReIdExtract
from Algorithm.libs.search.search_engine import SearchEngine
import Algorithm.libs.config.model_cfgs as cfgs

def test_cross_modal_extraction():
    print("=== 测试跨模态特征提取 ===")
    
    extractor = CrossModalReIdExtract(
        extract_class="person",
        providers=['CPUExecutionProvider']
    )
    
    print(f"模型输入数量: {len(extractor.model_inputs)}")
    print(f"融合模式: {extractor.fusion_mode}")
    print(f"双通道输入支持: {extractor._dual_input_onnx}")
    
    vis_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    ir_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    
    feat_vis = extractor.extract_visible(vis_img)
    print(f"可见光特征维度: {feat_vis.shape}")
    
    feat_ir = extractor.extract_ir(ir_img)
    print(f"红外特征维度: {feat_ir.shape}")
    
    feat_fusion = extractor(vis_img, ir_img)
    print(f"融合特征维度: {feat_fusion.shape}")
    
    return extractor

def test_search_engine():
    print("\n=== 测试搜索引擎 ===")
    
    num_samples = 100
    dims = cfgs.DIMS
    
    base_feat_lists = []
    base_idx_lists = []
    modalities = []
    
    for i in range(num_samples):
        feat = np.random.randn(dims).astype(np.float32)
        feat = feat / np.linalg.norm(feat)
        base_feat_lists.append(feat)
        base_idx_lists.append(f"person_{i:04d}")
        modalities.append("vis" if i % 2 == 0 else "ir")
    
    search_engine = SearchEngine(base_feat_lists, base_idx_lists, dims=dims, modalities=modalities)
    
    query_feat = np.random.randn(dims).astype(np.float32)
    query_feat = query_feat / np.linalg.norm(query_feat)
    
    labels, dists = search_engine.search(query_feat, top_k=5)
    print(f"搜索结果: {labels}")
    print(f"距离: {dists}")
    
    if len(labels) > 0 and len(dists) > 0 and len(dists[0]) > 0:
        reranked_labels, reranked_dists = search_engine.rerank(query_feat, labels, dists[0], query_modality="vis")
        print(f"重排序结果: {reranked_labels}")
        print(f"重排序距离: {reranked_dists}")
    
    return search_engine

if __name__ == "__main__":
    extractor = test_cross_modal_extraction()
    search_engine = test_search_engine()
    
    print("\n=== 测试完成 ===")
    print("跨模态行人重识别系统改造完成！")