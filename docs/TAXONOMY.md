# Taxonomy
## 1. 总体结构

```text
U1 Image Critic
│
├── D1. Artifact｜非文字图像错误
│   ├── A1. Rendering Artifact｜局部渲染错误
│   ├── A2. Structural Artifact｜实体结构错误
│   ├── A3. Topology Artifact｜拓扑连接错误
│   └── A4. Relational and Physical Artifact｜关系与物理错误
│
└── D2. Text Correctness｜文字正确性
│   ├── A1. Text Visual Integrity｜文字视觉完整性
│   └── A2. Text Content Fidelity｜文字内容保真度
│
├── D3. Instruction Compliance｜指令遵循（后续阶段）
│
└── D4. Aesthetic Quality｜美学质量（后续阶段）
```

美学质量和指令遵循暂不纳入当前 taxonomy，后续作为独立评价维度扩展。

Target-known 与 Target-unknown 的关系
This content is only supported in a Feishu Docs
Target-known 是在 Target-unknown 的视觉检查基础上，增加与目标字符串的内容比对。
---

# 2. D1：Artifact｜非文字图像错误

## A1. Rendering Artifact｜局部渲染错误

只描述局部外观异常。若错误已经改变实体结构、连接关系或对象间关系，应分别归入 A2、A3 或 A4。

| English key             | 中文名称   | 定义                   | 示例           |
| ----------------------- | ------ | -------------------- | ------------ |
| `boundary_blending`     | 边界异常混合 | 两个本应可区分的区域在边界处异常融合   | 头发与背景局部融为一体  |
| `local_smearing`        | 局部涂抹   | 局部内容被异常拉伸、扩散或抹开      | 人脸边缘出现不合理拖影  |
| `texture_discontinuity` | 纹理不连续  | 同一表面的纹理方向、周期或形态异常突变  | 衣服条纹在肩部突然错位  |
| `texture_repetition`    | 纹理重复   | 相同纹理块出现明显的机械复制       | 背景树叶以相同形态重复  |
| `detail_loss`           | 局部细节丢失 | 局部区域退化为无稳定结构的模糊或杂乱内容 | 手指区域变成模糊色块   |
| `color_corruption`      | 局部颜色异常 | 局部颜色无法由材质、光照或遮挡解释    | 人脸局部出现突兀绿色区域 |

正常景深、运动模糊、压缩噪声和明确的艺术风格不应自动标记为 Artifact。

---

## A2. Structural Artifact｜实体结构错误

适用于人物、动物及普通物体，评价单个实体内部的部件数量、形状、位置和连接关系。

| English key          | 中文名称      | 定义                   | 示例                 |
| -------------------- | --------- | -------------------- | ------------------ |
| `extra_part`         | 多余部件      | 实体出现不应存在的额外部件        | 六根手指、三只手臂、汽车出现五个车轮 |
| `missing_part`       | 部件缺失      | 在无遮挡情况下缺少构成实体所需的部件   | 手缺少一根手指、自行车缺少踏板    |
| `duplicated_part`    | 部件重复      | 同一部件被异常复制            | 面部出现两组耳朵           |
| `fused_parts`        | 部件融合      | 两个本应独立的部件连接为单一结构     | 两根手指粘连             |
| `split_part`         | 部件分裂      | 一个部件异常分叉或分裂为多个结构     | 一根手指末端分成两个指尖       |
| `deformed_part`      | 部件变形      | 部件的形状、长度、宽度或轮廓明显异常   | 手指严重扭曲             |
| `misplaced_part`     | 部件错位      | 部件存在，但位于实体内部的错误位置    | 眼睛位于额头中央           |
| `misattached_part`   | 部件连接错误    | 部件连接到错误位置或错误结构       | 杯把连接在杯口内部          |
| `disconnected_part`  | 部件断开      | 本应连接的相邻部件之间出现空隙      | 手掌与手腕断开            |
| `articulation_error` | 关节或活动连接错误 | 关节、转轴或铰接结构的位置及运动关系异常 | 手肘反向弯折、剪刀转轴错误      |
| `proportion_error`   | 内部比例错误    | 同一实体内部各部件的相对尺寸明显异常   | 手指远长于手臂、车轮大于车身     |

---

## A3. Topology Artifact｜拓扑连接错误

主要适用于线稿、流程图、示意图、图标、技术图和显式连接结构。

| English key             | 中文名称    | 定义                 | 示例             |
| ----------------------- | ------- | ------------------ | -------------- |
| `broken_path`           | 路径中断    | 同一条连续线或路径内部出现缺口    | 流程图连线中间断开      |
| `missing_connection`    | 连接缺失    | 两个本应连接的独立元素没有建立连接  | 导线未接触电路元件      |
| `wrong_connection`      | 连接目标错误  | 连接存在，但连接到了错误元素     | 箭头连到错误流程节点     |
| `unintended_connection` | 非预期连接   | 本应分离的元素被错误连接       | 两个独立图标被线条连接    |
| `branching_error`       | 分支结构错误  | 分支数量、位置或层级关系错误     | 单一路径出现额外分叉     |
| `unclosed_shape`        | 图形未闭合   | 本应闭合的轮廓存在缺口        | 圆形边缘没有闭合       |
| `unintended_closure`    | 图形误闭合   | 本应开放的结构被错误连接为闭合图形  | U 形线条变成封闭区域    |
| `crossing_state_error`  | 交叉点状态错误 | 线条交叉处的连接或非连接状态错误   | 两条跨线被错误画成连接节点  |
| `arrow_error`           | 箭头错误    | 箭头头部、方向或连接关系异常     | 箭头头部缺失、方向反向    |
| `symbol_deformation`    | 图形符号变形  | 具有明确结构的图标或符号失去基本形态 | 播放按钮不再形成可识别三角形 |

### A3 标签边界

| 情况           | 标签                      |
| ------------ | ----------------------- |
| 同一条线内部断裂     | `broken_path`           |
| 两个元素之间缺少连接   | `missing_connection`    |
| 连接到了错误节点     | `wrong_connection`      |
| 本不应连接的元素发生连接 | `unintended_connection` |
| 普通实体的部件断开    | `disconnected_part`     |

---

## A4. Relational and Physical Artifact｜关系与物理错误

评价多个对象之间，或对象与场景之间的空间关系、交互关系和物理一致性。

| English key                       | 中文名称     | 定义                    | 示例             |
| --------------------------------- | -------- | --------------------- | -------------- |
| `interpenetration`                | 对象穿透     | 两个独立实体不合理地占据同一空间      | 手指穿过杯壁         |
| `occlusion_error`                 | 遮挡关系错误   | 对象之间的前后遮挡关系发生矛盾       | 后方对象局部覆盖前方对象   |
| `contact_error`                   | 接触关系错误   | 应发生的接触没有成立，或接触位置错误    | 手未接触杯子却表现为握持   |
| `support_error`                   | 支撑关系错误   | 对象缺少有效支撑，或支撑结构无法维持其状态 | 杯子悬浮、椅腿无法支撑椅面  |
| `scale_inconsistency`             | 相对尺度不一致  | 同一场景中对象的相对尺寸明显冲突      | 普通杯子大于人物       |
| `occluded_structure_misalignment` | 遮挡两侧结构错位 | 同一对象在遮挡区域两侧无法形成连续结构   | 栏杆在人物两侧无法对齐    |
| `perspective_inconsistency`       | 透视不一致    | 同一场景中的投影和消失点关系互相冲突    | 桌子的边指向不同消失点    |
| `illumination_inconsistency`      | 光照或阴影不一致 | 受光方向、亮度或阴影无法由统一光源解释   | 多个对象的阴影方向相反    |
| `reflection_inconsistency`        | 反射不一致    | 反射内容与原对象在数量、姿态或几何上不一致 | 镜中人物多出一只手      |
| `pose_infeasibility`              | 姿态不可实现   | 人或动物的姿态无法由关节及支撑关系实现   | 人物在无支撑下保持不可能平衡 |
| `material_behavior_inconsistency` | 材料行为不一致  | 材料表现与其基本物理性质冲突        | 刚性金属无原因地像液体融合  |

---

# 3. D2-A：Text Visual Integrity｜文字视觉完整性

该类别只基于图像本身进行判断，不依赖目标文本。

| English key            | 中文名称   | 定义                         | 示例                  |
| ---------------------- | ------ | -------------------------- | ------------------- |
| `glyph_unrecognizable` | 字符不可识别 | 单个字形无法稳定对应任何有效字符           | 图形类似文字，但无法判断具体字符    |
| `glyph_malformed`      | 字形畸变   | 字符身份基本可以确定，但字形结构明显异常       | 可以辨认为 `B`，但内部结构严重扭曲 |
| `character_fusion`     | 字符融合   | 相邻字符粘连为一个整体，无法明确区分字符边界     | `rn` 融合成类似 `m` 的结构  |
| `character_overlap`    | 字符重叠   | 多个字符在空间上互相覆盖，但仍可观察到多个字符结构  | 两个汉字明显叠在一起          |
| `spacing_error`        | 字符间距错误 | 字符间距过大或过小，破坏字符或词语边界        | 一个单词被异常拆开或连成一块      |
| `alignment_error`      | 文字排列错位 | 同一文本单元中的字符或文本行无法形成稳定的基线和排列 | 一行文字中的字符无规律上下错位     |
| `orientation_error`    | 文字方向异常 | 同一文本单元中的字符出现异常旋转、翻转或方向不一致  | 一行文字中某个字符倒置         |
| `text_blur`            | 文字模糊   | 模糊导致部分或全部文字无法稳定识别          | 海报副标题整体模糊           |
| `text_occlusion`       | 文字遮挡   | 文字被其他图像元素覆盖，影响识别           | 图标遮住标题中部            |
| `text_clipping`        | 文字裁切   | 文字被图像边界或容器截断               | 标题最后两个字符超出画面        |

## 标签边界

| 情况               | 使用标签                   |
| ---------------- | ---------------------- |
| 完全无法确定字符身份       | `glyph_unrecognizable` |
| 字符身份基本明确，但结构异常   | `glyph_malformed`      |
| 相邻字符已经粘成单一结构     | `character_fusion`     |
| 多个字符仍可区分，但空间重叠   | `character_overlap`    |
| 问题主要来自字符间距       | `spacing_error`        |
| 问题主要来自基线、行列或相对位置 | `alignment_error`      |
| 问题主要来自旋转、翻转或方向异常 | `orientation_error`    |
| 无法识别主要由模糊造成      | `text_blur`            |
| 无法识别主要由其他元素遮挡造成  | `text_occlusion`       |
| 无法识别主要由区域边界裁切造成  | `text_clipping`        |

笔画缺失、多余、断裂、粘连和碎片化不再单独设置标签：

* 字符仍可辨认时，标记为 `glyph_malformed`；
* 字符无法辨认时，标记为 `glyph_unrecognizable`。

---

# 4. D2-B：Text Content Fidelity｜文字内容保真度

该类别仅适用于已知目标文本的场景，评价图中可读文字与目标内容是否一致。

| English key              | 中文名称     | 定义                                      | 示例                              |
| ------------------------ | -------- | --------------------------------------- | ------------------------------- |
| `text_absent`            | 目标文字完全缺失 | 目标文本没有任何有效内容出现在图中                       | 要求生成标题，但图中没有标题                  |
| `text_incomplete`        | 目标文字不完整  | 目标文本只有部分内容出现                            | `SUMMER SALE` 只生成 `SUMMER`      |
| `extra_text`             | 出现额外文字   | 图中出现目标文本之外的字符、词语或文本区域                   | 标题后多出随机字符或重复单词                  |
| `text_identity_mismatch` | 文字内容不匹配  | 对应位置存在可读文字，但字符或词语身份与目标不同                | `IMAGE` 生成成 `IMACE`             |
| `text_order_mismatch`    | 文字顺序不匹配  | 目标字符、词语或文本单元基本存在，但排列顺序错误                | `NEW PRODUCT` 生成成 `PRODUCT NEW` |
| `orthographic_mismatch`  | 书写形式不匹配  | 内容身份基本一致，但大小写、标点、空格、附加符号、书写系统或字符变体不符合目标 | `OpenAI` 生成成 `OPENAI`           |
| `text_region_mismatch`   | 文字区域映射错误 | 多个目标文本与指定区域之间的对应关系错误                    | 标题和副标题内容互换                      |

## 标签边界

| 情况               | 使用标签                     |
| ---------------- | ------------------------ |
| 整段目标文字完全没有生成     | `text_absent`            |
| 只生成目标文字的一部分      | `text_incomplete`        |
| 生成了目标之外的附加内容     | `extra_text`             |
| 对应位置生成了其他可读字符或词语 | `text_identity_mismatch` |
| 目标内容基本都存在，但顺序错误  | `text_order_mismatch`    |
| 内容身份正确，但规范书写形式错误 | `orthographic_mismatch`  |
| 多个目标文本与区域之间对应错误  | `text_region_mismatch`   |

### 特殊说明

* 字符重复产生了目标之外的内容时，标记为 `extra_text`。
* 字符缺失导致目标内容不完整时，标记为 `text_incomplete`。
* 字符被其他字符替代时，标记为 `text_identity_mismatch`。
* 不再单独设置 insertion、deletion、substitution、repetition 等编辑操作标签。
* `text_region_mismatch` 仅在任务明确提供多个文字区域及其目标内容时使用；单纯未满足 Prompt 中的位置要求属于后续的 Instruction Compliance。

---

# 5. Text Visual Integrity 与 Text Content Fidelity 的关系

| 实际情况              | Text Visual Integrity | Text Content Fidelity    |
| ----------------- | --------------------- | ------------------------ |
| 字符清晰，但拼写错误        | 正常                    | `text_identity_mismatch` |
| 字符变形，但仍可辨认为正确字符   | `glyph_malformed`     | 正常                       |
| 字符变形且内容也错误        | `glyph_malformed`     | `text_identity_mismatch` |
| 整个文字区域模糊不可读       | `text_blur`           | 不评价                      |
| 两个字符融合，无法判断实际内容   | `character_fusion`    | 不评价                      |
| 目标标题完全没有生成        | 无对应文字视觉错误             | `text_absent`            |
| 标题被画面边缘裁掉一部分      | `text_clipping`       | `text_incomplete`        |
| 标题与副标题内容互换，字符本身清晰 | 正常                    | `text_region_mismatch`   |

同一文字区域可以同时具有视觉完整性错误和内容保真度错误，但两个标签必须分别有独立证据。
