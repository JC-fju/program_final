// ════════════════════════════════════════════════════
// 動態手勢判斷架構（Prototype）
// 適用於「動作型」手語詞彙（有位移/軌跡，非單一靜態手形）
// ════════════════════════════════════════════════════
//
// 使用方式：
//   1. 每一幀 MediaPipe onResults 拿到 landmarks 後，呼叫 pushHistory(landmarks)
//   2. 每一幀都呼叫 checkDynamicGestures(getHistory()) 檢查是否符合任何動態手勢
//   3. 符合時回傳詞彙字串，並且應該把 history 清空，避免同一動作被重複判斷
//
// 之後每支新影片分析完，只要新增一個 isXxxDynamic(history) 函式，
// 並加進 checkDynamicGestures() 的判斷清單即可，不用改動整體架構。
//
// 【改版重點】改成「以時間為窗口」而非「固定幀數」：
// 原本 HISTORY_LENGTH=20 是指定幀數，在 FPS 不穩定（尤其筆電內顯、
// modelComplexity 較高時 FPS 可能只有 1~2）的情況下，蒐集滿 20 幀
// 可能要等 10~20 秒，導致動態手勢判斷「反應超慢」。
// 改成時間窗口後，不管實際 FPS 多少，都是抓「最近 WINDOW_MS 毫秒內」
// 的資料來判斷，跟 FPS 沒有直接綁定關係。

const WINDOW_MS = 900;        // 動作偵測窗口長度（毫秒），可依實測調整
const MIN_SAMPLES = 6;        // 窗口內至少要有幾筆資料才夠判斷，避免 FPS 太低時資料太少誤判
let _landmarkHistory = [];    // 每筆為 { lm: landmarks, t: timestamp(ms) }

/**
 * 每一幀呼叫一次，把當幀 21 個關節點連同時間戳記存入歷史緩衝區
 * @param {Array} landmarks - MediaPipe 單手 21 點 landmarks
 */
function pushHistory(landmarks) {
  if (!landmarks) return;
  const now = performance.now();
  _landmarkHistory.push({ lm: landmarks, t: now });

  // 只保留最近 WINDOW_MS 毫秒內的資料，跟幀數無關
  const cutoff = now - WINDOW_MS;
  while (_landmarkHistory.length > 0 && _landmarkHistory[0].t < cutoff) {
    _landmarkHistory.shift();
  }
}

function getHistory() {
  return _landmarkHistory;
}

function clearHistory() {
  _landmarkHistory = [];
}

// 目前窗口涵蓋的實際時間長度（毫秒），可用來在畫面上顯示進度
function getHistoryTimeSpan() {
  if (_landmarkHistory.length < 2) return 0;
  return _landmarkHistory[_landmarkHistory.length - 1].t - _landmarkHistory[0].t;
}

// ── 輔助：拇指是否朝上伸直（沿用 gesture_rules.js 的手形判斷邏輯風格）──
function isThumbUpShape(lm) {
  // 拇指尖 y 座標明顯高於（小於）拇指根部與掌心，其餘四指彎曲收起
  const thumbUp = lm[4].y < lm[3].y && lm[4].y < lm[2].y;
  const othersCurled =
    lm[8].y  > lm[6].y &&
    lm[12].y > lm[10].y &&
    lm[16].y > lm[14].y &&
    lm[20].y > lm[18].y;
  return thumbUp && othersCurled;
}

// ── 「謝謝」動態判斷 ──
// 動作特徵：拇指朝上手形，且手部（以手腕 lm[0] 為代表點）
// 在近期時間窗口內「由下往上抬起，之後停留」
function isXieXieDynamic(history) {
  if (history.length < MIN_SAMPLES) return false;
  if (getHistoryTimeSpan() < WINDOW_MS * 0.8) return false; // 窗口時間還沒蒐集夠，先不判斷

  const lms = history.map(h => h.lm);

  // 不要求「每一筆」都完美符合拇指朝上手形，
  // 只要 60% 以上符合就算（容忍手指角度的抖動雜訊）
  const shapeMatchCount = lms.filter(isThumbUpShape).length;
  const shapeOK = (shapeMatchCount / lms.length) > 0.6;
  if (!shapeOK) return false;

  const n = lms.length;
  const earlySlice = lms.slice(0, Math.max(1, Math.floor(n * 0.3)));
  const lateSlice  = lms.slice(Math.floor(n * 0.6));

  const avgY = (slice) => slice.reduce((s, lm) => s + lm[0].y, 0) / slice.length;
  const earlyY = avgY(earlySlice);
  const lateY  = avgY(lateSlice);

  // 用「相對手部尺寸」的位移比例，取代絕對座標差，
  // 避免因為每個人手離鏡頭遠近不同、絕對座標門檻不合用的問題
  const avgHandSize = lms.reduce((s, lm) => s + dist(lm[0], lm[9]), 0) / lms.length;
  const riseRatio = (earlyY - lateY) / avgHandSize;
  const roseUp = riseRatio > 0.25; // 手腕上升幅度達手長的 25% 以上

  // 停留判斷：後段幀的 y 變化幅度要小（相對比例）
  const lateYs = lateSlice.map(lm => lm[0].y);
  const lateRange = (Math.max(...lateYs) - Math.min(...lateYs)) / avgHandSize;
  const settled = lateRange < 0.15;

  return roseUp && settled;
}

// ── 統一檢查入口：依序檢查所有已定義的動態手勢 ──
// 之後新增動態詞彙，只要在這個陣列加一行即可
const DYNAMIC_GESTURES = [
  { label: '謝謝', check: isXieXieDynamic },
  // { label: '下一個詞', check: isNextWordDynamic },
];

function checkDynamicGestures(history) {
  for (const g of DYNAMIC_GESTURES) {
    if (g.check(history)) {
      return g.label;
    }
  }
  return null;
}
