/**
 * お店ごとに変える設定はこのファイルにまとめています。
 * 別のお店・別のサイトで使うときは、まずここを書き換えてください。
 * （見出しや紹介文などの文章は index.html、色やフォントは style.css の :root で変更します）
 */
export const CONFIG = {
  // ブランド（ヘッダーのロゴ文字・フッター・結果画像に使われます）
  brand: {
    name: "Cafuné",                       // ロゴの大きい文字
    sub: "body face care",                // ロゴの下の小さい文字
    fullName: "Cafuné body&face care",    // フッター・結果画像の正式名
    siteUrl: "https://www.cafune-biyouseitai.com/", // ロゴ下・フッターのリンク先
  },

  // 「WEB予約」ボタンのリンク先（予約ページのURLがあればここへ）
  bookingUrl: "https://www.cafune-biyouseitai.com/",

  // ダウンロードする結果画像
  report: {
    title: "Cafuné セルフ姿勢チェック 結果",
    accentColor: "#46b5d6",
  },

  // 履歴の保存名（ブラウザ内）。お店ごとに別の名前にすると履歴が混ざりません
  historyKey: "cafune-posture-history-v1",

  // 判定のしきい値：g 以下＝良好、m 以下＝軽度、それより大きい＝要注意
  // 単位：° は角度、% は身長（または脚の長さ）に対する割合
  thresholds: {
    shoulder: { g: 1.5, m: 3 },   // 肩の高さの左右差（°）
    pelvis:   { g: 1.5, m: 3 },   // 骨盤の左右の傾き（°）
    head:     { g: 2,   m: 4 },   // 頭の傾き（°）
    shift:    { g: 1.5, m: 3 },   // 上半身の左右シフト（%）
    pshift:   { g: 1.5, m: 3 },   // 骨盤の左右シフト（%）
    knee:     { g: 2.5, m: 5 },   // O脚・X脚（%）
    fhp:      { g: 7,   m: 15 },  // 頭の前方位（°、+3°までは許容）
    trunk:    { g: 4,   m: 8 },   // 上体の前後傾（°）
    hipshift: { g: 3,   m: 6 },   // 骨盤の前後位置（°、+1.5°を基準）
    kneeSide: { g: 4,   m: 9 },   // 膝の曲がり・反り（°）
  },

  // 総合スコアでの各項目の重み（大きいほど影響が大きい）
  weights: { shoulder: 1.2, pelvis: 1.5, head: 0.8, shift: 0.8, pshift: 1, knee: 0.8, fhp: 1.4, trunk: 1.2, hipshift: 1.2, kneeSide: 0.7 },
};
