// ════════════════════════════════════════════════════
// 動態手勢判斷架構（Prototype）
// 適用於「動作型」手語詞彙（有位移/軌跡，非單一靜態手形）
// ════════════════════════════════════════════════════
//
// 使用方式：
//   1. 每一幀 MediaPipe onResults 拿到 multiHandLandmarks 後，
//      呼叫 pushHistory(handsArray)（handsArray 是這一幀偵測到的
//      所有手，可能 0~2 個，每個元素是 21 點 landmarks）
//   2. 每一幀都呼叫 checkDynamicGestures(getHistory()) 檢查是否符合任何動態手勢
//   3. 符合時回傳詞彙字串，並且應該把 history 清空，避免同一動作被重複判斷
//
// 之後每支新影片分析完，只要新增一個 isXxxDynamic(history) 函式，
// 並加進 checkDynamicGestures() 的判斷清單即可，不用改動整體架構。
//
// 【時間窗口】不用固定幀數，改抓「最近 WINDOW_MS 毫秒內」的資料，
// 跟實際 FPS 沒有直接綁定關係，避免 FPS 低的時候判斷變超慢。

const WINDOW_MS = 900;        // 動作偵測窗口長度（毫秒），可依實測調整
const MIN_SAMPLES = 6;        // 窗口內至少要有幾筆資料才夠判斷，避免 FPS 太低時資料太少誤判
let _landmarkHistory = [];    // 每筆為 { hands: [landmarks, ...], t: timestamp(ms) }

/**
 * 每一幀呼叫一次，把當幀所有偵測到的手（1 或 2 隻）連同時間戳記存入歷史緩衝區
 * @param {Array} handsArray - MediaPipe multiHandLandmarks，每個元素是單手 21 點 landmarks
 */
function pushHistory(handsArray) {
  if (!handsArray || handsArray.length === 0) return;
  const now = performance.now();
  _landmarkHistory.push({ hands: handsArray, t: now });

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

// ── 輔助手形判斷 ──

// 拇指朝上伸直，其餘四指彎（用於「謝謝」）
function isThumbUpShape(lm) {
  const thumbUp = lm[4].y < lm[3].y && lm[4].y < lm[2].y;
  const othersCurled =
    lm[8].y  > lm[6].y &&
    lm[12].y > lm[10].y &&
    lm[16].y > lm[14].y &&
    lm[20].y > lm[18].y;
  return thumbUp && othersCurled;
}

// 握拳：四指全彎（用於「拜託」）
function isFistShape(lm) {
  return lm[8].y  > lm[6].y
      && lm[12].y > lm[10].y
      && lm[16].y > lm[14].y
      && lm[20].y > lm[18].y;
}

// ── 「謝謝」動態判斷（單手）──
// 動作特徵：拇指朝上手形，且手部（以手腕 lm[0] 為代表點）
// 在近期時間窗口內「由下往上抬起，之後停留」
function isXieXieDynamic(history) {
  if (history.length < MIN_SAMPLES) return false;
  if (getHistoryTimeSpan() < WINDOW_MS * 0.8) return false;

  // 只取每幀的第一隻手來判斷（謝謝是單手動作）
  const lms = history.map(h => h.hands[0]).filter(Boolean);
  if (lms.length < MIN_SAMPLES) return false;

  const shapeMatchCount = lms.filter(isThumbUpShape).length;
  const shapeOK = (shapeMatchCount / lms.length) > 0.6;
  if (!shapeOK) return false;

  const n = lms.length;
  const earlySlice = lms.slice(0, Math.max(1, Math.floor(n * 0.3)));
  const lateSlice  = lms.slice(Math.floor(n * 0.6));

  const avgY = (slice) => slice.reduce((s, lm) => s + lm[0].y, 0) / slice.length;
  const earlyY = avgY(earlySlice);
  const lateY  = avgY(lateSlice);

  const avgHandSize = lms.reduce((s, lm) => s + dist(lm[0], lm[9]), 0) / lms.length;
  const riseRatio = (earlyY - lateY) / avgHandSize;
  const roseUp = riseRatio > 0.25;

  const lateYs = lateSlice.map(lm => lm[0].y);
  const lateRange = (Math.max(...lateYs) - Math.min(...lateYs)) / avgHandSize;
  const settled = lateRange < 0.15;

  return roseUp && settled;
}

// ── 「拜託」動態判斷（雙手）──
// 動作特徵：雙手握拳互扣（手腕距離近且穩定），合併中心點上下反覆擺動
function isBaituoDynamic(history) {
  if (history.length < MIN_SAMPLES) return false;
  if (getHistoryTimeSpan() < WINDOW_MS * 0.8) return false;

  // 只取「這一幀確實偵測到兩隻手」的樣本
  const validFrames = history.filter(h => h.hands && h.hands.length === 2);
  if (validFrames.length / history.length < 0.6) return false; // 大部分時間都要偵測到雙手
  if (validFrames.length < MIN_SAMPLES) return false;

  // 手形：兩手都要是握拳
  const fistMatchCount = validFrames.filter(h => isFistShape(h.hands[0]) && isFistShape(h.hands[1])).length;
  if (fistMatchCount / validFrames.length < 0.6) return false;

  // 互扣：兩手手腕距離要夠近（用第一幀手的尺寸當參考，避免絕對座標問題）
  const avgHandSize = validFrames.reduce((s, h) => s + dist(h.hands[0][0], h.hands[0][9]), 0) / validFrames.length;
  const wristDists = validFrames.map(h => dist(h.hands[0][0], h.hands[1][0]));
  const avgWristDist = wristDists.reduce((a, b) => a + b, 0) / wristDists.length;
  const clasped = (avgWristDist / avgHandSize) < 1.5; // 兩手腕距離小於約1.5倍手長，視為扣在一起
  if (!clasped) return false;

  // 擺動：合併中心點（兩手腕平均）的 y 座標方向要反覆改變
  const centersY = validFrames.map(h => (h.hands[0][0].y + h.hands[1][0].y) / 2);
  const noiseThreshold = 0.003;
  let directionChanges = 0;
  let prevDir = null;
  for (let i = 1; i < centersY.length; i++) {
    const diff = centersY[i] - centersY[i - 1];
    if (Math.abs(diff) < noiseThreshold) continue; // 忽略微小雜訊震動
    const dir = diff > 0 ? 1 : -1;
    if (prevDir !== null && dir !== prevDir) directionChanges++;
    prevDir = dir;
  }

  return directionChanges >= 2; // 至少一上一下算一次完整擺動
}

// ── 統一檢查入口：依序檢查所有已定義的動態手勢 ──
// 之後新增動態詞彙，只要在這個陣列加一行即可
const DYNAMIC_GESTURES = [
  { label: '謝謝', check: isXieXieDynamic },
  { label: '拜託', check: isBaituoDynamic },
];

function checkDynamicGestures(history) {
  for (const g of DYNAMIC_GESTURES) {
    if (g.check(history)) {
      return g.label;
    }
  }
  return null;
}
