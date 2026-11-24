const RECORDS = window.OFFLINE_RECORDS || [];
const CONFIG = window.OFFLINE_CONFIG || {};
const PLAN = CONFIG.plan || {};
const STORAGE_KEY = "critic-benchmark-manual-review-v1";
let PAGE = 1;
let PAGES = 1;
let CURRENT_ITEMS = [];

const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
const pretty = value => JSON.stringify(value ?? {}, null, 2);

function loadReviews() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return {reviewer: parsed.reviewer || "", annotations: parsed.annotations || {}};
  } catch (_) {
    return {reviewer: "", annotations: {}};
  }
}

let REVIEW = loadReviews();

function persistReviews() {
  REVIEW.reviewer = document.getElementById("reviewer").value.trim();
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(REVIEW));
  } catch (error) {
    alert("浏览器本地存储失败，请立即导出人工标注 JSON：" + error);
  }
  renderReviewMetrics();
}

function renderReviewMetrics() {
  const values = Object.values(REVIEW.annotations);
  const count = status => values.filter(item => item.status === status).length;
  document.getElementById("reviewed").textContent = values.filter(item => item.status && item.status !== "unreviewed").length.toLocaleString();
  document.getElementById("approved").textContent = count("approved").toLocaleString();
  document.getElementById("corrected").textContent = count("corrected").toLocaleString();
  document.getElementById("rejected").textContent = count("rejected").toLocaleString();
}

function setOptions(element, values, label) {
  const current = element.value;
  element.innerHTML = `<option value="">${label}</option>` + values.map(value => `<option value="${esc(value)}">${esc(value)}</option>`).join("");
  element.value = current;
}

function refreshLeafOptions() {
  const group = document.getElementById("group").value;
  const leaves = group ? Object.keys(PLAN.leaf_quotas?.[group] || {}) : Object.values(PLAN.leaf_quotas || {}).flatMap(item => Object.keys(item));
  setOptions(document.getElementById("leaf"), leaves, "全部叶子");
}

function initOverview() {
  const state = CONFIG.state || {};
  const counts = state.counts || {};
  document.getElementById("accepted").textContent = Number(CONFIG.accepted_count || 0).toLocaleString();
  document.getElementById("records").textContent = Number(CONFIG.record_count || RECORDS.length).toLocaleString();
  document.getElementById("stamp").innerHTML = `导出时间 ${esc(CONFIG.exported_at || "-")}<br>图片 ${(Number(CONFIG.image_bytes || 0) / 1024 ** 3).toFixed(3)} GiB`;
  let html = "";
  for (const [group, leaves] of Object.entries(PLAN.leaf_quotas || {})) {
    const target = Object.values(leaves).reduce((sum, value) => sum + value, 0);
    const current = Object.keys(leaves).reduce((sum, leaf) => sum + (counts[group]?.[leaf] || 0), 0);
    html += `<div class="group"><div class="group-head"><span>${esc(group)}</span><span>${current} / ${target}</span></div>`;
    for (const [leaf, quota] of Object.entries(leaves)) {
      const value = counts[group]?.[leaf] || 0;
      const percent = quota ? Math.min(100, value / quota * 100) : 0;
      html += `<button class="leaf" data-group="${esc(group)}" data-leaf="${esc(leaf)}"><span>${esc(leaf)}</span><span class="bar"><span class="fill" style="width:${percent}%"></span></span><span class="count">${value} / ${quota}</span></button>`;
    }
    html += "</div>";
  }
  document.getElementById("overview").innerHTML = html;
  document.querySelectorAll(".leaf").forEach(button => button.onclick = () => {
    document.getElementById("group").value = button.dataset.group;
    refreshLeafOptions();
    document.getElementById("leaf").value = button.dataset.leaf;
    document.getElementById("vlm_status").value = "accepted";
    PAGE = 1;
    renderRecords();
    document.querySelector(".title-row").scrollIntoView({behavior: "smooth"});
  });
}

function filteredRecords() {
  const group = document.getElementById("group").value;
  const leaf = document.getElementById("leaf").value;
  const dataset = document.getElementById("dataset").value;
  const vlmStatus = document.getElementById("vlm_status").value;
  const reviewStatus = document.getElementById("review_status").value;
  const query = document.getElementById("search").value.trim().toLowerCase();
  return RECORDS.filter(record => {
    const decision = record.decision || {};
    const annotation = REVIEW.annotations[record.item_id];
    const currentReview = annotation?.status || "unreviewed";
    if (group && record.profile !== group) return false;
    if (leaf && decision.primary_leaf !== leaf) return false;
    if (dataset && record.dataset !== dataset) return false;
    if (vlmStatus !== "all" && record.status !== vlmStatus) return false;
    if (reviewStatus !== "all" && currentReview !== reviewStatus) return false;
    if (query) {
      const haystack = [record.item_id, record.source_prompt, decision.visible_evidence, pretty(record.source_metadata)].join(" ").toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    return true;
  });
}

function reviewGroupOptions(selected) {
  return Object.keys(PLAN.leaf_quotas || {}).map(group => `<option value="${esc(group)}" ${group === selected ? "selected" : ""}>${esc(group)}</option>`).join("");
}

function reviewLeafOptions(group, selected) {
  return Object.keys(PLAN.leaf_quotas?.[group] || {}).map(leaf => `<option value="${esc(leaf)}" ${leaf === selected ? "selected" : ""}>${esc(leaf)}</option>`).join("");
}

function cardHtml(record) {
  const decision = record.decision || {};
  const ranked = decision.ranked_labels || [];
  const annotation = REVIEW.annotations[record.item_id] || {};
  const reviewStatus = annotation.status || "unreviewed";
  const correctedGroup = annotation.corrected_profile || record.profile || Object.keys(PLAN.leaf_quotas || {})[0];
  const correctedLeaf = annotation.corrected_leaf || decision.primary_leaf || Object.keys(PLAN.leaf_quotas?.[correctedGroup] || {})[0] || "";
  const statusClass = record.accepted ? "" : (record.status === "quota_full" ? "warn" : "reject");
  const source = record.source_prompt || "原数据未提供可物化的生成 prompt；请查看 metadata 和来源记录。";
  const itemKey = encodeURIComponent(record.item_id);
  return `<article class="sample" data-item="${itemKey}"><a class="image-link" href="${esc(record.offline_image)}" target="_blank"><img loading="lazy" src="${esc(record.offline_image)}" alt="${esc(record.item_id)}"></a><div class="body"><span class="tag ${statusClass}">${esc(record.status)}</span><span class="tag">${esc(record.dataset)}</span><span class="tag">${esc(record.profile)}</span>${record.manual_review ? '<span class="tag warn">规则要求复核</span>' : ""}${reviewStatus !== "unreviewed" ? `<span class="tag">人工 ${esc(reviewStatus)}</span>` : ""}<div class="primary">${esc(decision.primary_leaf || "无主标签")} · confidence ${esc(decision.confidence ?? "-")}</div><div class="ranked">${ranked.map(item => `<span class="rank">${esc(item.leaf)} ${esc(item.score)}</span>`).join("")}</div><div class="evidence">${esc(decision.visible_evidence || record.error || "")}</div><div class="label">原 prompt / 原说明</div><div class="prompt">${esc(source)}</div><div class="meta">${esc(record.item_id)}<br>本地：${esc(record.offline_image)}<br>原路径：${esc(record.original_image_path)}<br>SHA-256：${esc(record.copied_image_sha256)}<br>来源 run：${esc(record.source_run)} · line ${esc(record.source_line)}</div><div class="review-box"><select class="review-status"><option value="unreviewed" ${reviewStatus === "unreviewed" ? "selected" : ""}>未复核</option><option value="approved" ${reviewStatus === "approved" ? "selected" : ""}>人工通过 VLM 标签</option><option value="corrected" ${reviewStatus === "corrected" ? "selected" : ""}>人工修正标签</option><option value="rejected" ${reviewStatus === "rejected" ? "selected" : ""}>人工拒绝样本</option><option value="followup" ${reviewStatus === "followup" ? "selected" : ""}>需进一步复核</option></select><select class="corrected-group">${reviewGroupOptions(correctedGroup)}</select><select class="corrected-leaf">${reviewLeafOptions(correctedGroup, correctedLeaf)}</select><span class="saved">${annotation.updated_at ? `上次保存 ${esc(annotation.updated_at)}` : "尚未人工保存"}</span><textarea class="review-note" placeholder="人工证据、修正原因或问题备注">${esc(annotation.note || "")}</textarea><button class="save-review">保存本条人工标注</button></div><details><summary>原数据 metadata</summary><pre>${esc(pretty(record.source_metadata))}</pre></details><details><summary>完整 VLM 分类决策</summary><pre>${esc(pretty(decision))}</pre></details><details><summary>实际分类 Prompt</summary><pre>${esc(record.classification_prompt || "")}</pre></details><details><summary>来源与 token</summary><pre>${esc(pretty({usage: record.usage, attempt_count: record.attempt_count, finish_reasons: record.finish_reasons, source_log: record.source_log, source_line: record.source_line}))}</pre></details><a href="${esc(record.raw_record_path)}" target="_blank">打开完整原始审计 JSON</a></div></article>`;
}

function bindReviewCards() {
  document.querySelectorAll(".sample").forEach(card => {
    const itemId = decodeURIComponent(card.dataset.item);
    const groupSelect = card.querySelector(".corrected-group");
    const leafSelect = card.querySelector(".corrected-leaf");
    groupSelect.onchange = () => {
      leafSelect.innerHTML = reviewLeafOptions(groupSelect.value, "");
    };
    card.querySelector(".save-review").onclick = () => {
      const record = RECORDS.find(item => item.item_id === itemId);
      const status = card.querySelector(".review-status").value;
      if (status === "unreviewed" && !card.querySelector(".review-note").value.trim()) {
        delete REVIEW.annotations[itemId];
      } else {
        REVIEW.annotations[itemId] = {
          item_id: itemId,
          status,
          corrected_profile: groupSelect.value,
          corrected_leaf: leafSelect.value,
          note: card.querySelector(".review-note").value.trim(),
          reviewer: document.getElementById("reviewer").value.trim(),
          updated_at: new Date().toISOString(),
          vlm_profile: record.profile,
          vlm_primary_leaf: record.decision?.primary_leaf || null,
          image_sha256: record.copied_image_sha256,
          original_image_path: record.original_image_path
        };
      }
      persistReviews();
      renderRecords();
    };
  });
}

function renderRecords() {
  const records = filteredRecords();
  const pageSize = Number(document.getElementById("page_size").value);
  PAGES = Math.max(1, Math.ceil(records.length / pageSize));
  PAGE = Math.min(Math.max(1, PAGE), PAGES);
  const start = (PAGE - 1) * pageSize;
  CURRENT_ITEMS = records.slice(start, start + pageSize);
  document.getElementById("result_count").textContent = `共 ${records.length} 条 · 第 ${PAGE}/${PAGES} 页`;
  document.getElementById("page_info").textContent = `${PAGE} / ${PAGES}`;
  document.getElementById("prev").disabled = PAGE <= 1;
  document.getElementById("next").disabled = PAGE >= PAGES;
  document.getElementById("active_filter").textContent = [document.getElementById("group").value, document.getElementById("leaf").value, document.getElementById("dataset").value].filter(Boolean).join(" / ") || "全部分类";
  document.getElementById("cards").innerHTML = CURRENT_ITEMS.length ? CURRENT_ITEMS.map(cardHtml).join("") : '<div class="empty">当前筛选条件没有数据</div>';
  bindReviewCards();
}

function exportReviews() {
  persistReviews();
  const payload = {
    format: STORAGE_KEY,
    exported_at: new Date().toISOString(),
    source_export: CONFIG.exported_at,
    source_run_chain: CONFIG.run_chain,
    reviewer: REVIEW.reviewer,
    annotations: REVIEW.annotations
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `critic_manual_reviews_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
}

function importReviews(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const payload = JSON.parse(reader.result);
      if (payload.format !== STORAGE_KEY || typeof payload.annotations !== "object") throw Error("文件格式不匹配");
      REVIEW = {reviewer: payload.reviewer || REVIEW.reviewer, annotations: {...REVIEW.annotations, ...payload.annotations}};
      document.getElementById("reviewer").value = REVIEW.reviewer;
      persistReviews();
      renderRecords();
      alert(`已导入 ${Object.keys(payload.annotations).length} 条人工标注`);
    } catch (error) {
      alert("导入失败：" + error);
    }
  };
  reader.readAsText(file);
}

function init() {
  setOptions(document.getElementById("group"), Object.keys(PLAN.leaf_quotas || {}), "全部大类");
  refreshLeafOptions();
  setOptions(document.getElementById("dataset"), [...new Set(RECORDS.map(item => item.dataset))].sort(), "全部数据集");
  document.getElementById("reviewer").value = REVIEW.reviewer;
  initOverview();
  renderReviewMetrics();
  renderRecords();
}

document.getElementById("group").onchange = refreshLeafOptions;
document.getElementById("apply").onclick = () => {PAGE = 1; renderRecords();};
document.getElementById("page_size").onchange = () => {PAGE = 1; renderRecords();};
document.getElementById("prev").onclick = () => {if (PAGE > 1) {PAGE--; renderRecords(); document.querySelector(".title-row").scrollIntoView();}};
document.getElementById("next").onclick = () => {if (PAGE < PAGES) {PAGE++; renderRecords(); document.querySelector(".title-row").scrollIntoView();}};
document.getElementById("search").onkeydown = event => {if (event.key === "Enter") {PAGE = 1; renderRecords();}};
document.getElementById("reviewer").onchange = persistReviews;
document.getElementById("export_reviews").onclick = exportReviews;
document.getElementById("import_reviews").onchange = event => {if (event.target.files[0]) importReviews(event.target.files[0]);};
init();
