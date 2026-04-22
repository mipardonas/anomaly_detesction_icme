import os
import json
import numpy as np
from collections import defaultdict

"""
    STrack1 = 0.2 × Sloc + 0.2 × Scls + 0.6 × Sgrade

"""

def get_class_name(class_id, class_file="/public/home/nc260224a/rt-trv4/ICME/track1/class-name.txt"):
    with open(class_file, 'r') as f:
        lines = f.readlines()
    if 0 <= class_id < len(lines):
        return lines[class_id].strip()
    else:
        return f"Unknown({class_id})"

def convert_mask2bbox(points):
    if len(points) % 2 != 0:
        raise ValueError("points 数量不是偶数，无法组成 (x,y) 对")

    npoints = np.zeros((2, int(len(points)/2)), dtype=float)

    for i in range(0, int(len(points)/2)):
        npoints[:, i] = np.array([points[2*i], points[2*i+1]])
    
    xmin, xmax = float(min(npoints[0,:])), float(max(npoints[0,:]))
    ymin, ymax = float(min(npoints[1,:])), float(max(npoints[1, :]))

    # 外接矩形：左上角 (xmin, ymin)，宽度 w，高度 h
    w = xmax - xmin
    h = ymax - ymin
    return [xmin, ymin, w, h]

def xywh_to_xyxy(box):
    """
    将 [x, y, w, h] 转为 [xmin, ymin, xmax, ymax]
    假设 x, y 为框中心坐标（若本来就是左上角坐标，请改这里）
    """
    x, y, w, h = box
    xmin = x - w / 2.0
    ymin = y - h / 2.0
    xmax = x + w / 2.0
    ymax = y + h / 2.0
    return [xmin, ymin, xmax, ymax]

def bbox_iou(box1, box2):
    """
    计算两个 box 的 IoU
    box: [x_min, y_min, x_max, y_max]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 交集
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # 并集
    area1 = max(0.0, (x1_max - x1_min)) * max(0.0, (y1_max - y1_min))
    area2 = max(0.0, (x2_max - x2_min)) * max(0.0, (y2_max - y2_min))
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

class CaculateMetric:
    def __init__(self):
        self.gt_data = {}
        self.pred_data = {}
        self.grade_map = ['Acceptable', 'Marginal NG', 'NG', 'Gross NG']

    def read_data(self, img_dir, txt_dir, txt_shuffix=".json"):
        all_data = {}
        classes_index = []
        for img_name in os.listdir(img_dir):
            if img_name.split(".")[-1] not in ["bmp", "jpg", "png","jpeg"]:
                continue
            txt_name = img_name.split(".")[0] + txt_shuffix
            txt_path = os.path.join(txt_dir, txt_name)
            if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
                all_data[img_name.split(".")[0]] = []
                continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                if txt_shuffix == ".txt":
                    lines = f.readlines()
                    data = []
                    for line in lines:
                        line = line.strip().split(" ")
                        bbox = convert_mask2bbox(line[1:])
                        data.append({"cls": int(line[0]), "bbox":bbox})
                        if int(line[0]) not in classes_index:
                            classes_index.append(int(line[0]))
                    all_data[img_name.split(".")[0]] = data
                elif txt_shuffix == ".json":
                    lines = json.load(f)
                    data = []
                    for line in lines:
                        points = [float(x.strip()) for x in line["points"].split(',')]
                        bbox = convert_mask2bbox(points)  
                        if "severity" in list(line.keys()):
                            grade = self.grade_map.index(line["severity"])
                        else:
                            grade = None
                        data.append({"cls": int(line["class"]), "bbox": bbox, "grade":grade}) 
                        if int(line["class"]) not in classes_index:
                            classes_index.append(int(line["class"]))
                    all_data[img_name.split(".")[0]] = data    
        return all_data, classes_index

    
    def read_pred_data(self, img_dir, txt_dir, txt_shuffix=".txt"):
        """
        读取预测数据（YOLO 检测格式：xywh）
        每行格式：cls x_center y_center width height [confidence]
        """
        all_data = {}
        classes_index = []

        for img_name in os.listdir(img_dir):
            if img_name.split(".")[-1] not in ["bmp", "jpg", "png", "jpeg"]:
                continue

            txt_name = img_name.split(".")[0] + txt_shuffix
            txt_path = os.path.join(txt_dir, txt_name)

            if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
                all_data[img_name.split(".")[0]] = []
                continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                data = []
                for line in lines:
                    parts = line.strip().split(" ")
                    cls_id = int(parts[0])

                    # 检测模型输出：xywh 格式（中心坐标）
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    xmin = x_center - width / 2.0
                    ymin = y_center - height / 2.0

                    bbox = [xmin, ymin, width, height]

                    # 如果有置信度
                    confidence = float(parts[5]) if len(parts) > 5 else None

                    item = {"cls": cls_id, "bbox": bbox}
                    if confidence is not None:
                        item["conf"] = confidence

                    data.append(item)

                    if cls_id not in classes_index:
                        classes_index.append(cls_id)

                all_data[img_name.split(".")[0]] = data

        return all_data, classes_index

    def caculate_screen(self):
        """
            True Positive(TP): 被模型预测为正类的正样本
            True Negative(TN): 被模型预测为负类的负样本
            False Positive(FP): 被模型预测为正类的负样本
            False Negative(FN): 被模型预测为负类的正样本
        """
        tp_img = dict.fromkeys(self.classes_index, 0)
        fp_img = dict.fromkeys(self.classes_index, 0) 
        fn_img = dict.fromkeys(self.classes_index, 0)
        tp_img_cnt = 0 
        tn_img_cnt = 0
        fp_img_cnt = 0
        fn_img_cnt = 0
        for img_name, data in self.gt_data.items():
            if len(data) == 0 and (img_name not in list(self.pred_data.keys()) or len(self.pred_data[img_name]) == 0):
                tn_img_cnt += 1
            elif len(data) == 0 and len(self.pred_data[img_name]) != 0:
                for v in self.pred_data[img_name]:
                    cls = v["cls"]
                    fp_img[cls] += 1
                fp_img_cnt += 1
            elif len(data) != 0 and (img_name not in list(self.pred_data.keys()) or len(self.pred_data[img_name]) == 0):
                for v in self.gt_data[img_name]:
                    cls = v['cls']
                    fn_img[cls] += 1
                fn_img_cnt += 1
            elif len(data) != 0 and len(self.pred_data[img_name]) != 0:
                # 至少有一个标签被被正确预测出
                gt_match = 0
                for i, gd in enumerate(data):
                    for pd in self.pred_data[img_name]:
                        bbox1 = xywh_to_xyxy(gd["bbox"])
                        bbox2 = xywh_to_xyxy(pd["bbox"])
                        if bbox_iou(bbox1, bbox2) > 0.25 and gd["cls"] == pd["cls"]:
                            gt_match += 1
                            cls = gd['cls']
                            tp_img[cls] += 1
                            break
                if gt_match > 0:
                    tp_img_cnt += 1
        
        # 构建每个类别的详细指标
        per_class_screen = {}
        SscreenDict = defaultdict(float)
        for cls in self.classes_index:
            cls_recall_img = tp_img[cls] / (tp_img[cls] + fn_img[cls]+10e-6)
            cls_specificity_img = tn_img_cnt / (tn_img_cnt + fn_img[cls]+10e-6)
            cls_sscreen = 0.5 * cls_recall_img + 0.5 * cls_specificity_img
            SscreenDict[cls] = cls_sscreen
            per_class_screen[cls] = {
            "recall_img": cls_recall_img,
            "specificity_img": cls_specificity_img,
            "tp_img": tp_img[cls],
            "fn_img": fn_img[cls],
            "tn_img_cnt": tn_img_cnt,   # 图像级别的真负例总数
            "fp_img_cnt": fp_img_cnt,   # 图像级假正例总数
            "fn_img_cnt": fn_img_cnt    # 图像级假负例总数
            }
        Recall_img = tp_img_cnt / (tp_img_cnt + fn_img_cnt+10e-6)
        Specificity_img = tn_img_cnt / (tn_img_cnt + fn_img_cnt+10e-6)
        Sscreen = 0.5 * Recall_img + 0.5 * Specificity_img
        SscreenDict = {cls: 0.5 * per_class_screen[cls]["recall_img"] + 
                        0.5 * per_class_screen[cls]["specificity_img"] 
                   for cls in self.classes_index}
        SscreenDict['all'] = Sscreen
        return SscreenDict, per_class_screen, Recall_img, Specificity_img

    def caculate_Sfine(self, iou_thresh=0.25):
        """
            True Positive(TP): 被模型预测为正类的正样本
            True Negative(TN): 被模型预测为负类的负样本
            False Positive(FP): 被模型预测为正类的负样本
            False Negative(FN): 被模型预测为负类的正样本
        """
        # 统计所有类别的 TP, FP, FN
        TP = dict.fromkeys(self.classes_index, 0) 
        FP = dict.fromkeys(self.classes_index, 0) 
        FN = dict.fromkeys(self.classes_index, 0) 

        for img_name, gt_data in self.gt_data.items():
            if len(gt_data) == 0 and (img_name not in list(self.pred_data.keys()) or len(self.pred_data[img_name]) == 0):
                continue
            elif len(gt_data) == 0 and len(self.pred_data[img_name]) != 0:
                for pd in self.pred_data[img_name]:
                    cls = pd["cls"]
                    FP[cls] += 1
            elif len(gt_data) != 0 and (img_name not in list(self.pred_data.keys()) or len(self.pred_data[img_name]) == 0):
                for gd in gt_data:
                    cls = gd["cls"]
                    FN[cls] += 1
            elif len(gt_data) != 0 and len(self.pred_data[img_name]) != 0: 
                for i, gd in enumerate(gt_data):
                    gt_match = 0
                    for pd in self.pred_data[img_name]:
                        bbox1 = xywh_to_xyxy(gd["bbox"])
                        bbox2 = xywh_to_xyxy(pd["bbox"])
                        if bbox_iou(bbox1, bbox2) > 0.25 and gd["cls"] == pd["cls"]:
                            gt_match += 1
                            break
                    if gt_match > 0:
                        TP[gd["cls"]] += 1
                    else:
                        FN[gd["cls"]] += 1

        # 计算每个类别的 Precision, Recall, F1
        per_class_f1 = {}
        sfine_sum = 0.0
        SfineDict = defaultdict(float)
        for cls in self.classes_index:
            tp = TP[cls]
            fp = FP[cls]
            fn = FN[cls]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            per_class_f1[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
            SfineDict[cls] = f1
            sfine_sum += f1

        # S_fine = Macro-F1
        Sfine = sfine_sum / len(self.classes_index)
        SfineDict['all'] = Sfine
        return SfineDict, per_class_f1

    def caculate_cls(self):
        SfineDict, per_class_f1 = self.caculate_Sfine()
        SscreenDict = self.caculate_screen()
        SclsDict = defaultdict(float)
        for cls in self.classes_index:
            SclsDict[cls] = 0.5 * SfineDict[cls] + 0.5 * SscreenDict[cls]
        SclsDict['all'] = 0.5 * SfineDict['all'] + 0.5 * SscreenDict['all']
        return SclsDict
       
    
    def caculate_loc(self):

        from collections import Counter
        gt_cls_cnt = Counter()
        for v in self.gt_data.values():
            for inst in v:
                gt_cls_cnt[inst["cls"]] += 1
        pred_cls_cnt = Counter()
        for v in self.pred_data.values():
            for inst in v:
                pred_cls_cnt[inst["cls"]] += 1
        print("GT 类别计数:", dict(gt_cls_cnt))
        print("Pred 类别计数:", dict(pred_cls_cnt))

        ious_dict = {}
        SlocDict = dict.fromkeys(self.classes_index, 0) 
        for file_name, gt_data in self.gt_data.items():
            if file_name not in list(self.pred_data.keys()):
                continue
            pt_data = self.pred_data[file_name]
            if len(gt_data) == 0 or len(pt_data) == 0:
                continue
            for g in gt_data:
                for p in pt_data:
                    if g["cls"] != p["cls"]:
                        continue
                    bbox1 = xywh_to_xyxy(g["bbox"])
                    bbox2 = xywh_to_xyxy(p["bbox"])
                    iou = bbox_iou(bbox1, bbox2)
                    if iou <= 0:
                        continue
                    if g['cls'] not in list(ious_dict.keys()):
                        ious_dict[g['cls']] = [iou]
                    else:
                        ious_dict[g['cls']].append(iou)
        ious = []
        for cls in self.classes_index:
            cls_name = get_class_name(cls)
            print(f"类别 {cls} ({cls_name})")
            print(f"ious_dict 中的键: {list(ious_dict.keys())}")
            v = ious_dict[cls]
            cls_sloc = float(np.mean(v))
            ious.extend(v)
            SlocDict[cls] = cls_sloc
        Sloc = float(np.mean(ious))
        SlocDict['all'] = Sloc
        return SlocDict
    
    def severity_grading_from_confmat(self, conf_mat):
        """
        conf_mat: (K, K) numpy 数组
                conf_mat[i, j] = 真实等级为 i，预测等级为 j 的样本数
                下标 i,j 为 0..K-1
        """
        conf_mat = np.asarray(conf_mat, dtype=float)
        K = conf_mat.shape[0]
        assert conf_mat.shape[0] == conf_mat.shape[1], "conf_mat 必须是 K×K 方阵"

        N = conf_mat.sum()
        if N == 0:
            return np.nan  # 没有样本

        # W_ij = ((i-j)^2) / (K-1)^2
        idx = np.arange(K)
        I, J = np.meshgrid(idx, idx, indexing="ij")
        W = ((I - J) ** 2) / float((K - 1) ** 2)

        # 分子
        num = N * np.sum(W * conf_mat)

        # 边缘分布
        n_i_dot = conf_mat.sum(axis=1)
        n_dot_j = conf_mat.sum(axis=0)

        # 期望矩阵
        expected = np.outer(n_i_dot, n_dot_j)

        # 分母
        denom = np.sum(W * expected)
        if denom == 0:
            return 0

        return 1.0 - num / denom

    def collect_triplets(self, gt_dict, pred_dict, iou_thres=0.25):
        """
        返回一个列表 triplets: [(cls, gt_grade, pred_grade), ...]
        当前假设：同一图像里，gt_dict[img][k] 与 pred_dict[img][k]
        对应同一个缺陷实例。
        """
        triplets = []
        common_imgs = set(gt_dict.keys()) & set(pred_dict.keys())
        if not common_imgs:
            raise ValueError("gt_dict 和 pred_dict 没有共同的图像名")

        for img in common_imgs:
            gt_instances = gt_dict[img]
            pred_instances = pred_dict[img]
            n = min(len(gt_instances), len(pred_instances))

            for k in range(n):
                gt_ins = gt_instances[k]
                pred_ins = pred_instances[k]
                bbox1 = xywh_to_xyxy(gt_ins["bbox"])
                bbox2 = xywh_to_xyxy(pred_ins["bbox"])
                if gt_ins["cls"] == pred_ins["cls"] and bbox_iou(bbox1, bbox2) > iou_thres:
                    cls_id = int(gt_ins["cls"])          # 以 GT 的 cls 为准
                    gt_g = int(gt_ins["grade"])
                    pred_g = int(pred_ins["grade"])
                    triplets.append((cls_id, gt_g, pred_g))
                    break

        if not triplets:
            raise ValueError("没有匹配到任何实例，请检查数据和匹配逻辑")

        return triplets

    def caculate_grade(self, K=4):
        SgradeDict = defaultdict(float)
        triplets = self.collect_triplets(self.gt_data, self.pred_data)

        # 提取所有 grade（用于自动确定 K）
        all_gt = np.array([g for _, g, _ in triplets], dtype=int)
        all_pred = np.array([p for _, _, p in triplets], dtype=int)

        # 自动判断等级是否从 0 或 1 开始
        if K is None:
            K = max(all_gt.max(), all_pred.max()) + 1  # 假设 0..K-1
        # 如果是 1..K，统一减 1
        if all_gt.min() == 1:
            all_gt -= 1
            all_pred -= 1
            # 同时对 triplets 中的 grade 做同样转换
            triplets = [(c, g - 1, p - 1) for (c, g, p) in triplets]

        # ---------- 3.1 构造“总”的混淆矩阵 ----------
        overall_conf = np.zeros((K, K), dtype=float)
        for g, p in zip(all_gt, all_pred):
            overall_conf[g, p] += 1
        overall_s_grade = self.severity_grading_from_confmat(overall_conf)
        SgradeDict['all'] = overall_s_grade

        # ---------- 3.2 按 cls 构造混淆矩阵 ----------
        # cls_id -> list of (gt_grade, pred_grade)
        per_cls_pairs = defaultdict(list)
        for cls_id, gt_g, pred_g in triplets:
            per_cls_pairs[cls_id].append((gt_g, pred_g))

        for cls_id, pairs in per_cls_pairs.items():
            conf = np.zeros((K, K), dtype=float)
            for gt_g, pred_g in pairs:
                conf[gt_g, pred_g] += 1
            SgradeDict[cls_id] = self.severity_grading_from_confmat(conf)

        return SgradeDict

    def read_classes(self, class_txt_dir): 
        with open(class_txt_dir, 'r') as f:
            classes_list = [i.strip() for i in f.readlines()]
        return classes_list

    def process_data(self, gt_img_dir, gt_txt_dir, pred_img_dir, pred_txt_dir, class_txt_dir, txt_shuffix,type='point',S=2):
        classes_list = self.read_classes(class_txt_dir)
        self.gt_data, gt_classes = self.read_data(gt_img_dir, gt_txt_dir, txt_shuffix=txt_shuffix)
        # self.pred_data, pt_classes = self.read_data(pred_img_dir, pred_txt_dir, txt_shuffix=txt_shuffix)
        if type == 'point':
            self.pred_data, pt_classes = self.read_data(pred_img_dir, pred_txt_dir, txt_shuffix=txt_shuffix)
        else:
            self.pred_data, pt_classes = self.read_pred_data(pred_img_dir, pred_txt_dir, txt_shuffix=txt_shuffix)
        self.classes_index = list(set(gt_classes) | set(pt_classes))
        SscreenDict, per_class_screen, Recall_img, Specificity_img = self.caculate_screen()
        SfineDict, per_class_fine = self.caculate_Sfine()
        SclsDict = defaultdict(float)     
        SclsDict['all'] = 0.5 * SfineDict['all'] + 0.5 * SscreenDict['all']
        # SclsDict = self.caculate_cls()
        SlocDict = self.caculate_loc()
        
        if S == 1:
            S1Dict = defaultdict(float)

            # 计算整体 precision 和 recall（微平均）
            total_tp = sum(per_class_fine[cls]['tp'] for cls in self.classes_index)
            total_fp = sum(per_class_fine[cls]['fp'] for cls in self.classes_index)
            total_fn = sum(per_class_fine[cls]['fn'] for cls in self.classes_index)
            precision_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall_all = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

            # 打印表头
            print("class" + " " * 5 + "Sloc" + " " * 5 + "0.5*Sscreen" + " " * 5 + "0.5*Sfine" + " " * 5 + "Scls" + " " * 5 + "precisionInSfine" + " " * 5 + "recallInSfine" + " " * 5 + "Sscreen" + " " * 5 + "Recall_imgInSscreen" + " " * 5 + "Specificity_imgInSscreen" + " " * 5 + "Strack1")

            # 打印 all 行
            S1 = 0.3 * SlocDict['all'] + 0.3 * SclsDict['all'] + 0.4 * SscreenDict['all']
            print(f"all       {SlocDict['all']:.3f}     {SscreenDict['all']*0.5:.3f}         {SfineDict['all']*0.5:.3f}      {SclsDict['all']:.3f}      {precision_all:.3f}               {recall_all:.3f}            {SscreenDict['all']:.3f}                {Recall_img:.3f}                      {Specificity_img:.3f}                   {S1:.3f}")

            # 打印每个类的行
            for cls_idx in self.classes_index:
                cls_sloc = SlocDict.get(cls_idx, 0.0)
                cls_sfine = SfineDict.get(cls_idx, 0.0)
                cls_screen = SscreenDict.get(cls_idx, 0.0)
                cls_scls = 0.5 * cls_sfine + 0.5 * cls_screen

                # 从详细字典中获取 precision、recall、recall_img、specificity_img
                cls_precision = per_class_fine[cls_idx]['precision'] if cls_idx in per_class_fine else 0.0
                cls_recall = per_class_fine[cls_idx]['recall'] if cls_idx in per_class_fine else 0.0
                cls_recall_img = per_class_screen[cls_idx]['recall_img'] if cls_idx in per_class_screen else 0.0
                cls_specificity_img = per_class_screen[cls_idx]['specificity_img'] if cls_idx in per_class_screen else 0.0

                cls_s1 = 0.3 * cls_sloc + 0.3 * cls_scls + 0.4 * cls_screen

                print(f"{classes_list[cls_idx]:<10} {cls_sloc:.3f}     {cls_screen*0.5:.3f}         {cls_sfine*0.5:.3f}      {cls_scls:.3f}      {cls_precision:.3f}               {cls_recall:.3f}            {cls_screen:.3f}                {cls_recall_img:.3f}                      {cls_specificity_img:.3f}                   {cls_s1:.3f}")

                S1Dict[cls_idx] = cls_s1
                S1Dict['all'] += cls_s1
        
        if S == 2:
            SgradeDict = self.caculate_grade(K=4)
            S2Dict = dict.fromkeys(self.classes_index, 0)
            S2 = 0.2 * SlocDict['all'] + 0.2 * SclsDict['all'] + 0.6 * SgradeDict['all']

            S2Dict['all'] = S2
            print("class" + " " * 5 + "Sloc" + " " * 5 +"0.5*Sscreen" + " " * 5 +"0.5*Sfine" + " " * 5 +"Scls" + " " * 5 + "Sgrade" + " " * 5 + "Strack2")
            print("all" + " " * 5 + f"{SlocDict['all']:.3f}" + " " * 5 + f"{SscreenDict['all']*0.5:.3f}"  +" " * 5 + f"{SfineDict['all']*0.5:.3f}" + " " * 5 + f"{SclsDict['all']:.3f}" + " " * 5 + f"{SgradeDict['all']:.3f}" + " " * 5 + f"{S2:.3f}")
       
            for cls_idx in self.classes_index:
                cls_sloc = SlocDict[cls_idx] if cls_idx in list(SlocDict.keys()) else 0
                cls_scls = SclsDict[cls_idx] if cls_idx in list(SclsDict.keys()) else 0
                cls_sfine = SfineDict[cls_idx] if cls_idx in list(SfineDict.keys()) else 0
                cls_screen = SscreenDict[cls_idx] if cls_idx in list(SscreenDict.keys()) else 0
                cls_scls = 0.5 * cls_sfine + 0.5 * cls_screen 
                SclsDict[cls_idx] = cls_scls
                cls_grade = SgradeDict[cls_idx] if cls_idx in list(SgradeDict.keys()) else 0
                cls_s2 = 0.2 * cls_sloc + 0.2 * cls_scls + 0.6 * cls_grade
                S2Dict[cls_idx] = cls_s2
                print(classes_list[cls_idx] + " " * 3 + f"{cls_sloc:.3f}" + " " * 5 + f"{cls_screen*0.5:.3f}" + " " * 5 + f"{cls_sfine*0.5:.3f}"  + " " * 5 + f"{cls_scls:.3f}"+ " " * 5 + f"{cls_grade:.3f}" + " " * 5 + f"{cls_s2:.3f}")
            return S2Dict
        return None
        
if __name__ == '__main__':
    # gt_img_dir = "/public/home/nc260224a/rt-trv4/ICME/track1hybrid/test/images"
    # gt_txt_dir = "/public/home/nc260224a/rt-trv4/ICME/track1hybrid/test/labels"
    # gt_img_dir = "/data/processed/ICME/track1/NG_1154/images/test"
    # gt_txt_dir = "/data/processed/ICME/track1/NG_1154/labels/test"
    # data_dir = "/data/processed/ICME/track1/NG_1154/dataset.yaml"
    gt_img_dir ="/public/home/nc260224a/rt-trv4/ICME/track1compose/images/test"
    gt_txt_dir ="/public/home/nc260224a/rt-trv4/ICME/track1compose/labels/test"
    # data_dir = "/public/home/nc260224a/rt-trv4/ICME/track1compose/dataset1.yaml"
    # pt_img_dir = "/public/home/nc260224a/rt-trv4/ICME/track1hybrid/test/images"
    pt_img_dir = "/public/home/nc260224a/rt-trv4/ICME/track1compose/images/test"
    # pt_txt_dir = "/data/usrs/lnj/icme/Track1/result/v26anomaly/predict/labels"
    # pt_txt_dir = "/public/home/nc260224a/rt-trv4/RT-DETRv4/result/rtv4_hgnetv2_s_anomaly_guided_test/labels"
    # pt_txt_dir ="/public/home/nc260224a/rt-trv4/RT-DETRv4/outputs/stage1_x_coco_det_test_hybrid/labels"
    pt_txt_dir="/public/home/nc260224a/rt-trv4/RT-DETRv4/result/rtv4_hgnetv2_s_7classes_anomaly_guided_onlydecoder/labels"
    # pt_txt_dir="../Test/Test/labels"
    cls_txt_dir = "/public/home/nc260224a/rt-trv4/ICME/track1/class-name.txt"
    txt_shuffix = '.txt'
    cm = CaculateMetric()
    SDict = cm.process_data(gt_img_dir, gt_txt_dir, pt_img_dir, pt_txt_dir, cls_txt_dir, txt_shuffix, type='nopoint', S=1)
    print("caculate finished!")