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

const HISTORY_LENGTH = 20; // 約 0.6~0.8 秒（依前端偵測 FPS 而定），可依實測調整
let _landmarkHistory = [];

/**
 * 每一幀呼叫一次，把當幀 21 個關節點存入歷史緩衝區
 * @param {Array} landmarks - MediaPipe 單手 21 點 landmarks
 */
function pushHistory(landmarks) {
  if (!landmarks) return;
  _landmarkHistory.push(landmarks);
  if (_landmarkHistory.length > HISTORY_LENGTH) {
    _landmarkHistory.shift();
  }
}

function getHistory() {
  return _landmarkHistory;
}

function clearHistory() {
  _landmarkHistory = [];
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
// 動作特徵：拇指朝上手形，且手部（以手腕 lm[0] 或掌心 lm[9] 為代表點）
// 在近期歷史中「由下往上抬起，之後停留」
function isXieXieDynamic(history) {
  if (history.length < HISTORY_LENGTH) return false;

  // 放寬：不要求「每一幀」都完美符合拇指朝上手形，
  // 只要 60% 以上的幀數符合就算（容忍手指角度的抖動雜訊）
  const shapeMatchCount = history.filter(isThumbUpShape).length;
  const shapeOK = (shapeMatchCount / history.length) > 0.6;
  if (!shapeOK) return false;

  const n = history.length;
  const earlySlice = history.slice(0, Math.floor(n * 0.3));
  const lateSlice  = history.slice(Math.floor(n * 0.6));

  const avgY = (slice) => slice.reduce((s, lm) => s + lm[0].y, 0) / slice.length;
  const earlyY = avgY(earlySlice);
  const lateY  = avgY(lateSlice);

  // 放寬：改用「相對手部尺寸」的位移比例，取代絕對座標差，
  // 避免因為每個人手離鏡頭遠近不同、絕對座標門檻不合用的問題
  const avgHandSize = history.reduce((s, lm) => s + dist(lm[0], lm[9]), 0) / history.length;
  const riseRatio = (earlyY - lateY) / avgHandSize;
  const roseUp = riseRatio > 0.25; // 手腕上升幅度達手長的 25% 以上（原本是絕對值0.06，明顯更寬鬆）

  // 放寬：停留判斷也改用相對比例，並放寬允許的晃動範圍
  const lateYs = lateSlice.map(lm => lm[0].y);
  const lateRange = (Math.max(...lateYs) - Math.min(...lateYs)) / avgHandSize;
  const settled = lateRange < 0.15; // 原本 0.03（絕對值）太嚴格，改成手長比例且放寬

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
