// ================================================================
// 台灣手語數字 MediaPipe 關節點判斷函式
// 使用方式：傳入 MediaPipe Hands 的 landmarks 陣列（21個點）
// 每個點有 x, y, z 屬性（0~1 正規化座標）
// ================================================================

// 共用輔助函式
const fingerUp   = (tip, pip) => tip.y < pip.y;   // 手指伸直
const thumbRight = (tip, ip)  => tip.x > ip.x;    // 拇指往右伸（右手）
const thumbLeft  = (tip, ip)  => tip.x < ip.x;    // 拇指往左伸（右手比法）
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// 拇指是否張開：改用「拇指尖到小指根部（掌心對側）的距離」判斷，
// 不受手心/手背朝向鏡頭翻轉的影響（原本用 x 座標左右比較，手一翻面就會判斷錯誤）
const isThumbOut = (lm) => dist(lm[4], lm[17]) > 0.15;

// ── 零（○）：全部手指彎曲，拇指與食指圍成圓形 ──
function isZero(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      && dist(lm[4], lm[8]) < 0.08;   // 拇指尖靠近食指尖
}

// ── 一（一）：只有食指伸直 ──
function isOne(lm) {
  return  fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18]);
}

// ── 二（二）：食指 + 中指伸直，其餘彎曲 ──
function isTwo(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18]);
}

// ── 三（三）：食指 + 中指 + 無名指伸直 ──
function isThree(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      &&  fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18]);
}

// ── 四（四）：四根手指伸直（食中無小），拇指彎 ──
function isFour(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      &&  fingerUp(lm[16], lm[14])
      &&  fingerUp(lm[20], lm[18])
      && !isThumbOut(lm);   // 拇指收進來
}

// ── 五（五）：五根手指全部張開 ──
function isFive(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      &&  fingerUp(lm[16], lm[14])
      &&  fingerUp(lm[20], lm[18])
      &&  isThumbOut(lm);   // 拇指也張開
}

// ── 六（六）：拇指 + 小指伸直，食中無彎曲 ──
function isSix(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      &&  fingerUp(lm[20], lm[18])
      &&  isThumbOut(lm);   // 拇指伸出
}

// ── 七（七）：拇指 + 食指 + 中指伸出，無名小指彎 ──
function isSeven(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  isThumbOut(lm);
}

// ── 八（八）：拇指 + 食指伸出，形成 L 型，中無小彎 ──
function isEight(lm) {
  return  fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  isThumbOut(lm);
}

// ── 九（九）：四指橫向伸直（手掌向側），拇指彎 ──
// 九的特徵：四指伸直但手腕偏轉，用四指伸直 + 拇指彎來近似
function isNine(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      &&  fingerUp(lm[16], lm[14])
      &&  fingerUp(lm[20], lm[18])
      && dist(lm[4], lm[0]) < 0.35; // 拇指靠近手腕（未完全張開）
}

// ── 十（十）：全握拳，拇指彎曲扣在四指上 ──
function isTen(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      && !isThumbOut(lm)
      && dist(lm[4], lm[8]) > 0.08; // 拇指沒有圍圓（區分零）
}

// ── 廿（廿）：拇指 + 食指彎曲呈鉤狀，形成「廿」形 ──
function isTwenty(lm) {
  return !fingerUp(lm[8],  lm[6])   // 食指彎
      && !fingerUp(lm[12], lm[10])  // 中指彎
      && !fingerUp(lm[16], lm[14])  // 無名彎
      && !fingerUp(lm[20], lm[18])  // 小指彎
      &&  lm[4].y < lm[3].y;        // 拇指尖高於拇指第一關節（拇指彎鉤朝上）
}

// ── 百（百）：食指 + 中指合併伸直，其餘彎 ──
function isHundred(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      && dist(lm[8], lm[12]) < 0.05; // 食指中指靠攏
}

// ── 千（千）：拇指伸直向上，四指彎曲水平 ──
function isThousand(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  lm[4].y < lm[3].y          // 拇指尖高於第一關節
      &&  lm[4].y < lm[0].y;         // 拇指比手腕高（朝上）
}

// ── 萬（萬）：四指向下彎曲，拇指縮入，手背朝上 ──
function isWan(lm) {
  // 萬：四指彎曲向下，手掌朝下（指尖y > pip y，因手倒過來）
  return lm[8].y  > lm[6].y
      && lm[12].y > lm[10].y
      && lm[16].y > lm[14].y
      && lm[20].y > lm[18].y;
}

// ── 手（手）：四指伸直合攏，拇指彎曲貼掌 ──
function isHand(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      &&  fingerUp(lm[16], lm[14])
      &&  fingerUp(lm[20], lm[18])
      && !isThumbOut(lm)    // 拇指不張開
      && dist(lm[8], lm[12]) < 0.06  // 四指合攏
      && dist(lm[12], lm[16]) < 0.06;
}

// ── 女（女）：只有小指伸直，其餘彎曲 ──
function isWoman(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      &&  fingerUp(lm[20], lm[18]);
}

// ── 四十（四十）：食中無小指彎曲呈爪狀，拇指彎 ──
function isForty(lm) {
  // 四十：四指第一節彎，指尖朝向手心但未完全握拳
  return lm[8].y  > lm[5].y    // 食指尖低於掌根（彎曲）
      && lm[12].y > lm[9].y
      && lm[16].y > lm[13].y
      && lm[20].y > lm[17].y
      && lm[8].y  < lm[0].y;   // 但還沒到手腕（非全握）
}

// ── 八十（八十）：食中無小指彎成爪，拇指側張 ──
function isEighty(lm) {
  return lm[8].y  > lm[5].y
      && lm[12].y > lm[9].y
      && lm[16].y > lm[13].y
      && lm[20].y > lm[17].y
      && isThumbOut(lm);   // 拇指側張（與四十區分）
}

// ================================================================
// 第二批手語詞彙函式
// ================================================================

// ── 副（副）：全握拳，拇指微翹但不完全伸直 ──
function isFu(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  lm[4].y < lm[3].y          // 拇指尖比第一關節高（微翹）
      &&  lm[4].y > lm[0].y;         // 但沒有完全伸直到手腕以上
}

// ── 童（童）：中指+無名指+小指伸直，食指彎，拇指扣住食指 ──
function isTong(lm) {
  return !fingerUp(lm[8],  lm[6])    // 食指彎
      &&  fingerUp(lm[12], lm[10])   // 中指伸
      &&  fingerUp(lm[16], lm[14])   // 無名伸
      &&  fingerUp(lm[20], lm[18])   // 小指伸
      &&  dist(lm[4], lm[8]) < 0.07; // 拇指靠近食指
}

// ── 棕（棕）：食指+中指伸直靠攏，其餘彎，拇指扣 ──
function isZong(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  dist(lm[8], lm[12]) < 0.04  // 食指中指靠很近
      &&  dist(lm[4], lm[6])  < 0.08; // 拇指扣住食指根部
}

// ── 很（很）：拇指橫向伸出（指向左），四指彎握 ──
function isHen(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  lm[4].x < lm[3].x          // 拇指往左伸出
      &&  lm[4].y > lm[0].y;         // 拇指高度低（水平方向）
}

// ── 隻（隻）：拇指+食指伸出呈 L 型，手掌偏轉，中無小彎 ──
function isZhi(lm) {
  return  fingerUp(lm[8],  lm[6])    // 食指伸
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  lm[4].y < lm[0].y          // 拇指比手腕高（朝上）
      &&  lm[4].x < lm[3].x;         // 拇指往側邊伸
}

// ── 姊（姊）：只有無名指伸直，其餘四指彎 ──
function isJie(lm) {
  return !fingerUp(lm[8],  lm[6])    // 食指彎
      && !fingerUp(lm[12], lm[10])   // 中指彎
      &&  fingerUp(lm[16], lm[14])   // 無名指伸直
      && !fingerUp(lm[20], lm[18]);
}

// ── 果（果）：四指第一節彎曲成爪，指尖朝掌心，拇指對握 ──
function isGuo(lm) {
  // 四指彎但指尖在掌心附近（抓握狀）
  return dist(lm[8],  lm[0]) < 0.35
      && dist(lm[12], lm[0]) < 0.35
      && dist(lm[16], lm[0]) < 0.35
      && dist(lm[20], lm[0]) < 0.35
      && dist(lm[4],  lm[8]) < 0.10; // 拇指靠近食指
}

// ── 拳（拳）：標準握拳，四指彎曲包住拇指 ──
function isQuan(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  dist(lm[4], lm[9]) < 0.15; // 拇指在手掌中間區域（被包住）
}

// ── 飛機（飛機）：拇指+食指+小指伸出，中指+無名指彎 ──
function isAirplane(lm) {
  return  fingerUp(lm[8],  lm[6])    // 食指伸
      && !fingerUp(lm[12], lm[10])   // 中指彎
      && !fingerUp(lm[16], lm[14])   // 無名彎
      &&  fingerUp(lm[20], lm[18])   // 小指伸
      &&  lm[4].y < lm[0].y;         // 拇指伸出
}

// ── 借（借）：食指伸直，中指+無名指靠攏半彎，拇指+小指收 ──
function isJie2(lm) {
  return  fingerUp(lm[8],  lm[6])    // 食指伸直（主導）
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  lm[12].y > lm[9].y         // 中指半彎（在掌根以下）
      &&  dist(lm[4], lm[0]) < 0.3;  // 拇指收進來
}

// ── 同（同）：五指微彎朝下，手掌朝下傾斜 ──
function isTong2(lm) {
  // 手掌朝下，五指輕微彎曲
  return lm[8].y  > lm[5].y
      && lm[12].y > lm[9].y
      && lm[16].y > lm[13].y
      && lm[20].y > lm[17].y
      && lm[0].y  < lm[9].y;   // 手腕比掌心高（手掌朝下）
}

// ── 呂（呂）：拇指+食指圍成圓形（OK手勢），其他指彎 ──
function isLu(lm) {
  return dist(lm[4], lm[8]) < 0.05   // 拇指食指靠很近形成圓
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18]);
}

// ── 胡（胡）：食+中+無+小四指伸直，拇指彎扣掌，類似「手」但角度不同 ──
function isHu(lm) {
  return  fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      &&  fingerUp(lm[16], lm[14])
      &&  fingerUp(lm[20], lm[18])
      &&  dist(lm[4], lm[9]) < 0.2;  // 拇指靠近掌心（收起）
}

// ── 虎（虎）：拇指+食指張開呈虎口，中無小微彎 ──
function isHu2(lm) {
  return  lm[4].y < lm[3].y          // 拇指尖向上
      &&  lm[8].y < lm[5].y          // 食指也向上
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  dist(lm[4], lm[8]) > 0.12; // 拇食指張開（虎口）
}

// ── 句（句）：拇指+食指+中指捏合（三指捏，其餘彎） ──
function isJu(lm) {
  return dist(lm[4], lm[8])  < 0.06  // 拇指靠食指
      && dist(lm[4], lm[12]) < 0.08  // 拇指靠中指
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18]);
}

// ── 守（守）：食指+拇指+小指伸出，中+無彎（類似 ILY 手勢） ──
function isShou(lm) {
  return  fingerUp(lm[8],  lm[6])    // 食指伸
      && !fingerUp(lm[12], lm[10])   // 中指彎
      && !fingerUp(lm[16], lm[14])   // 無名彎
      &&  fingerUp(lm[20], lm[18])   // 小指伸
      &&  lm[4].y < lm[2].y;         // 拇指伸出（比拇指根高）
}

// ── 男（男）：只有拇指豎起（讚），四指握拳 ──
function isMan(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18])
      &&  lm[4].y < lm[2].y          // 拇指尖比拇指根高（朝上）
      &&  lm[4].y < lm[0].y;         // 拇指比手腕高
}

// ── 方（方）：四指彎曲朝側邊，拇指張開，像「ㄈ」形 ──
function isFang(lm) {
  return lm[8].y  > lm[5].y          // 四指向下彎
      && lm[12].y > lm[9].y
      && lm[16].y > lm[13].y
      && lm[20].y > lm[17].y
      && lm[4].x  < lm[3].x;         // 拇指往側邊張開
}

// ── 民（民）：拇指+小指伸出，食中無彎（Shaka/hang loose） ──
function isMin(lm) {
  return !fingerUp(lm[8],  lm[6])
      && !fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      &&  fingerUp(lm[20], lm[18])   // 小指伸
      &&  lm[4].y < lm[2].y;         // 拇指伸出
}

// ── 兄（兄）：只有中指伸直（與姊相同手形，由動作或方向區分） ──
// 注意：兄和姊手形相同，實際手語由動作方向區分
// 這裡只能做靜態辨識，需配合動態規則才能完全區分
function isXiong(lm) {
  return !fingerUp(lm[8],  lm[6])
      &&  fingerUp(lm[12], lm[10])
      && !fingerUp(lm[16], lm[14])
      && !fingerUp(lm[20], lm[18]);
  // ⚠️ 與 isJie() 手形相同，需靠動態方向區分
}

// ================================================================
// 更新主分類器（加入第二批）
// ================================================================
function classifyGesture(lm) {
  if (!lm || lm.length < 21) return null;

  // ── 靜態手勢（數字/基本手形）──
  if (isZero(lm))     return "零";
  if (isOne(lm))      return "一";
  if (isTwo(lm))      return "二";
  if (isThree(lm))    return "三";
  if (isHand(lm))     return "手";
  if (isFour(lm))     return "四";
  if (isFive(lm))     return "五";
  if (isSix(lm))      return "六";
  if (isSeven(lm))    return "七";
  if (isEight(lm))    return "八";
  if (isNine(lm))     return "九";
  if (isTen(lm))      return "十";
  if (isTwenty(lm))   return "廿";
  if (isHundred(lm))  return "百";
  if (isThousand(lm)) return "千";
  if (isWan(lm))      return "萬";
  if (isWoman(lm))    return "女";
  if (isForty(lm))    return "四十";
  if (isEighty(lm))   return "八十";

  // ── 詞彙手形 ──
  if (isMan(lm))      return "男";
  if (isAirplane(lm)) return "飛機";
  if (isShou(lm))     return "守";
  if (isMin(lm))      return "民";
  if (isHen(lm))      return "很";
  if (isZhi(lm))      return "隻";
  if (isJie(lm))      return "姊";    // ⚠️ 與兄相同，需動態區分
  if (isXiong(lm))    return "兄";
  if (isTong(lm))     return "童";
  if (isZong(lm))     return "棕";
  if (isJie2(lm))     return "借";
  if (isFu(lm))       return "副";
  if (isQuan(lm))     return "拳";
  if (isGuo(lm))      return "果";
  if (isHu2(lm))      return "虎";
  if (isHu(lm))       return "胡";
  if (isTong2(lm))    return "同";
  if (isFang(lm))     return "方";
  if (isLu(lm))       return "呂";
  if (isJu(lm))       return "句";

  return null;
}

// ================================================================
// 動態手勢歷史追蹤（留空，供後續填入）
// ================================================================

// let gestureHistory = [];
// const HISTORY_SIZE = 30;   // 追蹤幀數
//
// function updateHistory(lm) {
//   gestureHistory.push(lm);
//   if (gestureHistory.length > HISTORY_SIZE) gestureHistory.shift();
// }
//
// // ── TODO：在這裡加入動態手勢判斷 ──
// // 範例：偵測拇指反覆上下 → 謝謝
// // function isThankYou() {
// //   if (gestureHistory.length < 20) return false;
// //   const tips = gestureHistory.map(lm => lm[4].y);
// //   let changes = 0;
// //   for (let i = 1; i < tips.length; i++) {
// //     if (Math.abs(tips[i] - tips[i-1]) > 0.02) changes++;
// //   }
// //   return changes > 8;
// // }
