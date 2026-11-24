PROFILES = {
    "D1-A1": {
        "title": "Rendering Artifact / 局部渲染错误",
        "scope": "只判断局部外观异常。改变实体部件结构、连接、对象关系或正常风格的现象不得纳入。",
        "labels": {
            "boundary_blending": "本应可分的区域在边界异常融合",
            "local_smearing": "局部内容异常拉伸、扩散或抹开",
            "texture_discontinuity": "同一表面纹理方向、周期或形态突变",
            "texture_repetition": "相同纹理块机械复制",
            "detail_loss": "局部退化为无稳定结构的模糊或杂乱内容",
            "color_corruption": "局部颜色无法由材质、光照或遮挡解释",
        },
        "boundary": "正常景深、运动模糊、压缩噪声和明确艺术风格不是错误。",
    },
    "D1-A2": {
        "title": "Structural Artifact / 实体结构错误",
        "scope": "判断单个人物、动物或物体内部的部件数量、形状、位置和连接。",
        "labels": {
            "extra_part": "出现不应存在的额外部件",
            "missing_part": "无遮挡时缺少构成实体所需部件",
            "duplicated_part": "同一部件异常复制",
            "fused_parts": "两个本应独立的部件融合",
            "split_part": "一个部件异常分叉或分裂",
            "deformed_part": "部件形状、长度、宽度或轮廓明显异常",
            "misplaced_part": "部件存在但位于实体内部错误位置",
            "misattached_part": "部件连接到错误位置或错误结构",
            "disconnected_part": "本应连接的相邻部件之间出现空隙",
            "articulation_error": "关节、转轴或铰接结构的位置及运动关系异常",
            "proportion_error": "同一实体内部部件相对尺寸明显异常",
        },
        "boundary": "普通实体部件断开属于 disconnected_part；只有线稿、流程图等显式连接结构才属于拓扑错误。",
    },
    "D1-A3": {
        "title": "Topology Artifact / 拓扑连接错误",
        "scope": "只用于线稿、流程图、示意图、图标、技术图和显式连接结构。",
        "labels": {
            "broken_path": "同一连续线或路径内部有缺口",
            "missing_connection": "两个本应连接的独立元素没有连接",
            "wrong_connection": "连接存在但连到错误元素",
            "unintended_connection": "本应分离的元素被错误连接",
            "branching_error": "分支数量、位置或层级错误",
            "unclosed_shape": "本应闭合的轮廓有缺口",
            "unintended_closure": "本应开放的结构被误连成闭合图形",
            "crossing_state_error": "交叉点的连接或非连接状态错误",
            "arrow_error": "箭头头部、方向或连接关系异常",
            "symbol_deformation": "明确图标或符号失去基本结构",
        },
        "boundary": "手指粘连、肢体断开等普通实体问题必须退回 D1-A2，不得用 A3 凑数。",
    },
    "D1-A4": {
        "title": "Relational and Physical Artifact / 关系与物理错误",
        "scope": "判断多个对象之间或对象与场景之间的空间、交互和物理一致性。",
        "labels": {
            "interpenetration": "两个独立实体不合理占据同一空间",
            "occlusion_error": "对象前后遮挡关系矛盾",
            "contact_error": "应有接触未成立或接触位置错误",
            "support_error": "缺少有效支撑或支撑结构不可维持状态",
            "scale_inconsistency": "同一场景对象相对尺寸明显冲突",
            "occluded_structure_misalignment": "同一对象在遮挡两侧无法连续对齐",
            "perspective_inconsistency": "投影或消失点关系互相冲突",
            "illumination_inconsistency": "受光或阴影无法由统一光源解释",
            "reflection_inconsistency": "反射与原对象的数量、姿态或几何不一致",
            "pose_infeasibility": "人或动物姿态无法由关节和支撑关系实现",
            "material_behavior_inconsistency": "材料表现与基本物理性质冲突",
        },
        "boundary": "必须指出可见物理证据；不能只因原标签写着 physics 或 commonsense 就判定。",
    },
    "D2-A": {
        "title": "Text Visual Integrity / 文字视觉完整性",
        "scope": "只根据图片本身判断文字视觉完整性，不比较目标字符串。",
        "labels": {
            "glyph_unrecognizable": "单个字形无法稳定对应任何有效字符",
            "glyph_malformed": "字符身份基本可确定但字形结构明显异常",
            "character_fusion": "相邻字符粘成整体，边界无法区分",
            "character_overlap": "多个字符空间重叠但仍可看到多个字符结构",
            "spacing_error": "字符间距破坏字符或词语边界",
            "alignment_error": "字符或文本行无法形成稳定基线和排列",
            "orientation_error": "字符异常旋转、翻转或方向不一致",
            "text_blur": "模糊导致文字无法稳定识别",
            "text_occlusion": "其他图像元素覆盖文字并影响识别",
            "text_clipping": "文字被图像边界或容器截断",
        },
        "boundary": "身份不确定用 glyph_unrecognizable；身份基本明确但笔画异常用 glyph_malformed。没有目标字符串时绝不判断 D2-B。",
    },
    "D2-B": {
        "title": "Text Content Fidelity / 文字内容保真度",
        "scope": "仅在给出明确目标字符串和现成错误输出时，比较图中可读文字与目标内容。",
        "labels": {
            "text_absent": "目标文本完全没有有效内容出现",
            "text_incomplete": "目标文本只有部分内容出现",
            "extra_text": "出现目标文本之外的字符、词语或区域",
            "text_identity_mismatch": "对应位置文字身份与目标不同",
            "text_order_mismatch": "目标单元基本存在但顺序错误",
            "orthographic_mismatch": "大小写、标点、空格、附加符号或字符变体不符",
            "text_region_mismatch": "多个目标文本与指定区域的对应关系错误",
        },
        "boundary": "文字不可读时不能臆测内容错误；text_region_mismatch 要求多个明确目标区域。",
    },
}


def render_prompt(profile, dataset, metadata):
    spec = PROFILES[profile]
    labels = "\n".join(f"- {key}: {definition}" for key, definition in spec["labels"].items())
    return f"""你是图像错误 benchmark 的严谨预标员。本次只允许判断一个大类，不能输出其他大类标签。

数据集：{dataset}
当前大类：{profile} {spec['title']}
范围：{spec['scope']}
边界：{spec['boundary']}

可选叶子标签：
{labels}

原数据元信息如下。它只用于定位和路由，不能覆盖可见证据：
{metadata}

任务：
1. 只报告图片中清楚可见、符合当前大类的错误。
2. 可输出多个标签，但必须按视觉突出程度从高到低排序并逐一给 1-5 分。
3. primary_leaf 必须等于 ranked_labels 第一项；只有第一项得分至少 4 才可 eligible=true。
4. visible_evidence 必须具体描述对象、部位、位置和异常，禁止只复述标签。
5. 正常风格、正常模糊、遮挡造成的不可见、纯审美偏好和无法确认的猜测一律拒绝。
6. confidence 为 0 到 1。若没有合格错误，primary_leaf=null、ranked_labels=[]、eligible=false。
7. 输出中文证据，严格只输出 JSON，不要 Markdown。
"""
