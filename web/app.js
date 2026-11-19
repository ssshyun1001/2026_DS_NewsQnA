/* ===========================================================================
   QuPicT 뉴스 Q&A — 화면 동작 (web/index.html이 불러 쓴다)

   서버(web.py)가 이 파일을 /static/app.js 주소로 내보낸다.
   index.html의 맨 아래에서 불러오므로, 이 파일이 실행될 때는 화면의 칸 · 버튼이 이미 다 만들어져 있다.

   화면 흐름 (카드 번호 = index.html의 카드)
     1) 검색어 + 기간 → [의미 찾기]           POST /api/senses       의미 후보 받기
     2) 의미 고르기 → [이 의미로 뉴스 모으기]  POST /api/prepare      준비 작업 시작 (서버 백그라운드)
     3) 진행 상황                              GET  /api/jobs/{id}    몇 초마다 확인해서 로그 · 상태 갱신
     ★) 많이 보도된 기사 Top 5                 (3의 응답에 같이 옴)   준비 완료 뒤 채움
     4) 질문 → [질문]                          POST /api/ask/stream   답변을 만들어지는 대로 한 줄씩 받기

   준비하는 동안에도 질문을 적을 수 있다. 적어 둔 질문은 답할 수 있게 되면 적은 순서대로 하나씩 답한다
   (아래 state.ready · state.fresh · state.chain).
     - DB에 예전에 저장한 기사가 있으면 준비 중에도 그 기사로 먼저 답한다 (답변 위에 "언제 저장된 기사인지" 표시)
     - 없으면 새 기사를 모아 저장한 뒤에 답한다. 먼저 답하는 중에 근거가 없는 질문도 저장이 끝난 뒤에 답한다

   주의: index.html의 id(f-query, b-sense, s-job …)를 이 파일이 찾아 쓴다.
         index.html에서 id를 지우거나 바꾸면 이 파일도 같이 고쳐야 한다.
   =========================================================================== */


/* ---------------------------------------------------------------------------
   0. 공통 도구 · 화면 상태
   --------------------------------------------------------------------------- */

// id로 화면 요소 찾기: $("query") = document.getElementById("query")
const $ = (id) => document.getElementById(id);

// 화면 상태 (페이지 하나에 하나)
//   query     지금 검색어 (의미 후보를 받은 검색어. 준비를 시작할 때 서버에 다시 보낸다)
//   sessionId 질문을 보낼 세션 id. 서버가 "질문 받을 준비 됨"을 알려 주면 채워진다
//   jobId     지금 보고 있는 준비 작업 id. 새 검색어를 넣으면 바뀌어서, 이전 작업 확인(poll)은 멈춘다
//   ready     질문을 받을 수 있게 되면 세션 id로 풀리는 약속(Promise). DB에 저장된 기사가 있으면 준비 중에,
//             없으면 3번 과정이 모두 끝난 뒤에 풀린다. 준비 중에 적은 질문은 이것을 기다렸다가 보낸다
//   fresh     3번 과정(수집 · 필터링 · 인덱싱 · 이미지 글자 읽기)이 모두 끝나 새 기사까지 들어가면 풀리는 약속.
//             먼저 답하는 중에 "DB에 근거 없음"이 온 질문은 이것을 기다렸다가 다시 묻는다
//   chain     질문을 적은 순서대로 하나씩 답하게 잇는 줄 (앞 질문의 답이 끝나야 다음 질문을 보낸다)
const state = { query: "", sessionId: null, jobId: null, ready: null, fresh: null, chain: Promise.resolve() };

// 약속(Promise) 하나 만들기: resolve(값)을 부르면 기다리던 쪽이 이어서 돌고, reject(오류)를 부르면 실패한다
function deferred() {
  let resolve, reject;
  const p = new Promise((res, rej) => { resolve = res; reject = rej; });
  p.catch(() => {});                  // 아무도 기다리지 않을 때 실패가 콘솔 오류로 남지 않게
  p.resolve = resolve; p.reject = reject;
  return p;
}

// 새 준비를 시작할 때마다 ready · fresh(약속)와 chain(질문 줄)을 새로 만든다.
// ready.resolve(세션 id)를 부르면 기다리던 질문들이 나가고, reject(오류)를 부르면 "답할 수 없습니다"가 뜬다.
function newReady() {
  state.ready = deferred();
  state.fresh = deferred();
  state.chain = Promise.resolve();
  state.sessionId = null;
}

// 준비가 실패하면: 아직 답하지 못한 질문에 "답할 수 없습니다"를 띄운다 (먼저 답하던 중이었으면 그 질문들만)
function failReady(err) {
  state.ready.reject(err);
  state.fresh.reject(err);
}

// 서버 API 부르기. body를 주면 POST(JSON), 안 주면 GET.
// 서버가 오류(4xx · 5xx)를 돌려주면 서버의 오류 문구(detail)로 예외를 던진다 → 부른 쪽에서 화면에 보여 준다.
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;   // FastAPI 오류: 문자열이거나, 입력 검사 실패면 [{msg: …}] 목록
    throw new Error(typeof d === "string" ? d : (Array.isArray(d) ? d.map(x => x.msg).join(", ") : `오류 ${res.status}`));
  }
  return data;
}

// 글자를 HTML에 넣기 전에 특수문자(< > & " ')를 바꾼다.
// 기사 제목 · 답변처럼 서버에서 온 글을 innerHTML로 넣을 때 꼭 거친다 (제목 속 "<"가 태그로 해석되지 않게)
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// 상태 줄 바꾸기: busy = 도는 동그라미 + 글, setStatus = 글만 (kind: "ok" 초록 / "err" 빨강 / "" 기본)
const busy = (el, text) => { el.className = "status"; el.innerHTML = `<span class="spinner"></span> ${esc(text)}`; };
const setStatus = (el, text, kind = "") => { el.className = "status " + kind; el.textContent = text; };

// 지금 작업 · 질문 세션을 서버 메모리에서 지워 달라고 알린다 (POST /api/leave).
// 페이지를 나갈 때와, 새 검색어 · 새 준비로 바꿔서 이전 것을 더 쓰지 않을 때 부른다.
// sendBeacon: 페이지가 닫히는 중에도 끝까지 보내 주는 방법 (fetch는 페이지가 닫히면 끊길 수 있다).
// DB에 모은 기사는 지우지 않는다 → 다음에 같은 검색어를 찾으면 빨리 준비된다.
function release() {
  if (!state.jobId && !state.sessionId) return;
  const body = JSON.stringify({ job_id: state.jobId || "", session_id: state.sessionId || "" });
  navigator.sendBeacon("/api/leave", new Blob([body], { type: "application/json" }));
  state.jobId = null; state.sessionId = null;
}
// 탭을 닫거나 · 다른 주소로 가거나 · 새로고침할 때 (pagehide). 다른 탭으로 잠깐 옮기는 것은 나간 것이 아니다
window.addEventListener("pagehide", release);

// 페이지를 열면 바로: 서버가 쓰는 LLM 모델 이름을 제목 아래에 보여 준다 (실패해도 화면은 그대로)
api("/api/info").then(i => {
  $("info").textContent = `LLM 모델: ${i.chat_model}`;
  // "DB 없이 메모리에서" 체크박스는 화면에서 뺐다. DB가 없으면 서버가 알아서 메모리 버전으로 돌린다 (web.py)
}).catch(() => {});


/* ---------------------------------------------------------------------------
   1. 검색어 → 의미 후보  ([의미 찾기] 버튼, 1번 카드)
   --------------------------------------------------------------------------- */

$("f-query").addEventListener("submit", async (e) => {
  e.preventDefault();                 // 폼 기본 동작(페이지 새로고침) 막기
  const query = $("query").value.trim();
  if (!query) return;
  release();                          // 이전 검색어의 작업 · 대화는 서버에서 지운다 (jobId도 비워서 확인을 멈춘다)
  state.query = query; state.sessionId = null; state.jobId = null;
  // 아래 카드(2 · 3 · ★ · 4)는 새 검색어로 다시 시작하므로 모두 숨긴다
  ["c-sense", "c-job", "c-hot", "c-qa"].forEach(id => $(id).classList.add("hidden"));
  $("b-query").disabled = true;       // 기다리는 동안 버튼을 두 번 누르지 않게
  busy($("s-query"), "의미 후보를 찾는 중…");
  try {
    const r = await api("/api/senses", { query });
    setStatus($("s-query"), "");
    renderSenses(r);
  } catch (err) {
    setStatus($("s-query"), err.message, "err");
  } finally { $("b-query").disabled = false; }
});

// 의미 후보를 2번 카드에 라디오 버튼으로 보여 준다.
// r = { query, source, senses: [{ name, definition, aliases, related_terms }] }
// 맨 아래에 항상 "기타 (직접 입력)"을 붙인다. 후보가 없으면 그것이 미리 골라져 있다.
function renderSenses(r) {
  $("sense-src").textContent = r.senses.length
    ? (r.senses.length === 1 ? "후보가 하나라 미리 골라 두었습니다." : "")
    : "의미 후보를 만들지 못했습니다. 찾고 싶은 대상을 직접 설명해 주세요.";
  const items = r.senses.map((s, i) => `
    <label class="sense"><input type="radio" name="sense" value="${i}" ${i === 0 ? "checked" : ""}>
      <span class="name">${esc(s.name)}</span>
      <div class="def">${esc(s.definition)}</div>
      <div class="tags">${[...s.aliases.map(a => "별칭 " + a), ...s.related_terms].map(t => `<span class="tag">${esc(t)}</span>`).join("")}</div>
    </label>`);
  items.push(`<label class="sense"><input type="radio" name="sense" value="custom" ${r.senses.length ? "" : "checked"}>
      <span class="name">기타 (직접 입력)</span></label>`);
  $("senses").innerHTML = items.join("");
  // "기타"를 골랐을 때만 설명 입력칸(custom-box)을 보인다. 라디오를 바꿀 때마다 다시 확인한다
  const sync = () => $("custom-box").classList.toggle("hidden",
    document.querySelector("input[name=sense]:checked")?.value !== "custom");
  $("senses").querySelectorAll("input").forEach(el => el.addEventListener("change", sync));
  sync();
  $("c-sense").classList.remove("hidden");
}


/* ---------------------------------------------------------------------------
   2. 의미 선택 → 준비 작업 시작  ([이 의미로 뉴스 모으기] 버튼, 2번 카드)
   --------------------------------------------------------------------------- */

$("f-sense").addEventListener("submit", async (e) => {
  e.preventDefault();
  // 고른 라디오의 값: 후보 번호("0", "1", …) 또는 "custom"(직접 입력)
  const v = document.querySelector("input[name=sense]:checked")?.value;
  const body = {
    query: state.query, period_days: Number($("days").value), max_articles: Number($("maxn").value),
    sense_index: v === "custom" ? null : Number(v), custom_description: v === "custom" ? $("custom").value : "",
  };
  // "기타"인데 설명을 안 적었으면 보내지 않고 입력칸으로 커서를 옮긴다
  if (v === "custom" && !body.custom_description.trim()) { $("custom").focus(); return; }
  $("b-sense").disabled = true;
  $("c-job").classList.remove("hidden");
  $("log").textContent = "";
  busy($("s-job"), "작업을 시작하는 중…");
  release();                          // 같은 검색어로 다시 준비하면 이전 작업 · 대화는 서버에서 지운다
  // 질문 칸은 준비가 끝나기 전부터 연다: DB에 저장된 기사가 있으면 바로, 없으면 3번 과정이 끝나는 대로 차례로 답한다
  newReady();
  $("chat").innerHTML = "";
  $("qa-label").textContent = "";
  $("qa-note").textContent = "DB에 저장된 기사가 있는지 확인하는 중… 질문은 지금 적어 두어도 됩니다.";
  $("c-qa").classList.remove("hidden");
  // 많이 보도된 기사 카드도 질문 칸처럼 미리 연다. 내용은 준비가 끝난 뒤 채운다 (renderHot)
  $("hot").innerHTML = `<div class="muted">⏳ 뉴스 준비가 끝나면 같은 내용을 다룬 기사가 많은 순서로 보여 드립니다.</div>`;
  $("c-hot").classList.remove("hidden");
  try {
    const { job_id, label } = await api("/api/prepare", body);
    state.jobId = job_id;
    $("qa-label").textContent = label;
    poll(job_id);                     // 이제부터 진행 상황을 몇 초마다 확인한다
  } catch (err) {
    setStatus($("s-job"), err.message, "err");
    $("b-sense").disabled = false;
    failReady(err);                   // 적어 둔 질문에 "답할 수 없습니다"를 띄운다
    hotFailed();
  }
});

// 뉴스 준비가 실패하면 미리 열어 둔 많이 보도된 기사 카드에도 알린다
function hotFailed() {
  $("hot").innerHTML = `<div class="status err">뉴스 준비에 실패해 보여 드릴 기사가 없습니다.</div>`;
}


/* ---------------------------------------------------------------------------
   3. 진행 상황 확인  (3번 카드, 서버의 준비 작업을 몇 초마다 확인)
   --------------------------------------------------------------------------- */

// 서버 응답 j = { status, logs, session_id, label, error, elapsed, images, hot, hot_articles, db_articles, basis }
//   status   "running"(준비 중) / "done"(완료) / "error"(실패)
//   session_id  질문을 받을 수 있게 되면 채워진다: DB에 저장된 기사가 있으면 준비 중에, 없으면 3번 과정이 모두 끝난 뒤
//   db_articles 준비 전에 DB에 있던 질문 가능 기사 수 (null 확인 중, 0 없음 → 새 기사를 저장한 뒤에 답함)
//   basis    세션의 답변 기준: stage "db"(저장해 둔 기사로 먼저 답하는 중) / "fresh"(새 기사까지 모두 들어감)
//   images   준비 완료 뒤 이미지 속 글자 읽기(OCR): "" 안 함 / "running" / "done"
//   hot      준비 완료 뒤 많이 보도된 기사 정리: "" / "running" / "done" / "error"
// 다시 확인하는 간격: 준비 중 1.5초, Top 5 정리 중 1초, 이미지 읽는 중 · 새 기사 반영 전 3초. 모두 끝나면 멈춘다.
async function poll(jobId) {
  if (jobId !== state.jobId) return;          // 새 검색어·새 준비를 시작했으면 이전 작업은 더 보지 않는다
  try {
    const j = await api(`/api/jobs/${jobId}`);
    if (jobId !== state.jobId) return;        // 기다리는 사이에 새 작업이 시작된 경우
    $("log").textContent = j.logs.join("\n");
    $("log").scrollTop = $("log").scrollHeight; // 로그는 항상 맨 아래(최신)가 보이게
    const early = j.basis && j.basis.stage === "db";   // DB에 저장해 둔 기사로 먼저 답하는 중
    // 새 기사까지 모두 들어감 (메모리 버전은 basis가 없어서, 세션이 열리면 바로 이 상태)
    const fresh = j.basis ? j.basis.stage === "fresh" : !!j.session_id;
    // 질문을 받을 수 있게 되면 한 번만: 미리 적어 둔 질문이 이제 차례로 나간다
    if (j.session_id && state.sessionId !== j.session_id) {
      state.sessionId = j.session_id;
      $("qa-label").textContent = j.label;
      state.ready.resolve(j.session_id);
      $("question").focus();
    }
    if (fresh) state.fresh.resolve();         // 새 기사까지 들어감: "근거 없음"으로 미룬 질문을 다시 묻는다
    $("qa-note").textContent = fresh ? ""
      : early ? `DB에 저장된 기사 ${j.db_articles}건으로 먼저 답합니다. 새 기사 수집 · 정리가 끝나면 그 기사까지 넣어 답합니다.`
      : j.db_articles === 0 ? "DB에 저장된 관련 기사가 없어서, 새 기사를 모아 저장한 뒤에 답합니다. 질문은 지금 적어 두어도 됩니다."
      : $("qa-note").textContent;
    if (j.status === "running") {
      busy($("s-job"), early
        ? "새 기사를 모으는 중… 그동안 DB에 저장된 기사로 먼저 답합니다."
        : "처리 중… 기사 수에 따라 몇 분 걸릴 수 있습니다. 그동안 아래에서 질문을 적어 둘 수 있어요.");
      setTimeout(() => poll(jobId), 1500);
      return;
    }
    $("b-sense").disabled = false;            // 준비가 끝났으니(완료든 실패든) 다시 준비할 수 있게
    if (j.status === "done") {
      renderHot(j);                            // 많이 보도된 기사 Top 5 (준비 완료 뒤 서버가 이어서 만든다)
      const took = `준비 완료 · ${j.elapsed}초`;
      // 이미지 속 글자 읽기도 준비 완료 뒤 서버에서 이어서 한다. 이게 끝나야 새 기사 기준으로 답한다 → 끝날 때까지 가끔 확인
      const after = early ? "새 기사까지 넣어 답합니다" : "질문에 답합니다";
      if (!fresh) {
        busy($("s-job"), j.images === "running"
          ? `${took} · 이미지 속 글자를 읽는 중… 끝나면 ${after}.`
          : `${took} · 마무리하는 중…`);
        if (!j.session_id) $("qa-note").textContent = "이미지 속 글자 읽기가 끝나면 답합니다. 질문은 지금 적어 두어도 됩니다.";
      } else {
        setStatus($("s-job"), j.images === "done" ? `${took} · 이미지 속 글자까지 반영됨 · 이제 질문할 수 있습니다` : `${took} · 이제 질문할 수 있습니다`, "ok");
      }
      if (j.hot === "running") setTimeout(() => poll(jobId), 1000);
      else if (!fresh) setTimeout(() => poll(jobId), 3000);
    } else {
      const msg = j.error || "작업이 실패했습니다.";
      setStatus($("s-job"), msg, "err");
      $("qa-note").textContent = early
        ? "새 기사 준비에 실패했습니다. 질문은 DB에 저장된 기사로만 답합니다."
        : "뉴스 준비에 실패해 질문에 답할 수 없습니다. 의미·기간을 바꿔 다시 준비해 주세요.";
      failReady(new Error(msg));               // 적어 둔 질문에 "답할 수 없습니다"를 띄운다 (먼저 답하던 세션은 그대로)
      hotFailed();
    }
  } catch (err) {                              // 서버에 연결이 안 되는 경우 등
    setStatus($("s-job"), err.message, "err");
    $("b-sense").disabled = false;
    failReady(err);
    hotFailed();
  }
}


/* ---------------------------------------------------------------------------
   ★. 많이 보도된 기사 Top 5  (3번과 4번 카드 사이)
   --------------------------------------------------------------------------- */

// 중복이 많이 묶인 기사 순으로 제목(원문 링크) · 한 줄 요약 · 언론사 · 날짜 · 비슷한 기사 수.
// 한 줄 요약은 LLM이 쓴 글이 아니라, 묶인 기사들이 공통으로 쓴 기사 속 문장 그대로다 (qupict/hot.py). 없으면 줄을 뺀다.
// j.hot_articles = [{ title, link, press, pub_date, duplicates, summary, similar: [{ title, link, press, pub_date }] }]
function renderHot(j) {
  $("c-hot").classList.remove("hidden");
  if (j.hot === "running") {
    $("hot").innerHTML = `<div class="status"><span class="spinner"></span> 많이 보도된 기사를 정리하는 중…</div>`;
  } else if (j.hot === "error") {
    $("hot").innerHTML = `<div class="status err">많이 보도된 기사를 정리하지 못했습니다. 질문은 그대로 할 수 있습니다.</div>`;
  } else if (!j.hot_articles.length) {
    $("hot").innerHTML = `<div class="muted">같은 내용을 다룬 기사(중복)가 없어 보여 드릴 기사가 없습니다.</div>`;
  } else {
    $("hot").innerHTML = `<ol class="hot">` + j.hot_articles.map((h, i) => `
      <li><span class="rank">${i + 1}</span>
        <div><a href="${esc(h.link)}" target="_blank" rel="noopener">${esc(h.title)}</a>
          ${h.summary ? `<div class="sum">${esc(h.summary)}</div>` : ""}
          <div class="meta">${esc(h.press)}${h.pub_date ? " · " + esc(h.pub_date) : ""} ·
            ${similarToggle(h)}</div></div></li>`).join("") + `</ol>`;
  }
}

// "비슷한 기사 N건 ▸"을 누르면 묶인 기사들이 펼쳐진다 (최신순, 기사마다 원문이 새 탭에서 열림).
// <details>/<summary>를 써서 열고 닫는 동작은 브라우저가 해 준다 (모양은 style.css의 7번).
// 목록을 못 받았으면(예전 서버 · 메모리 버전에서 본문을 모르는 기사) 글자만 보여 준다.
function similarToggle(h) {
  const label = `<span class="dup">비슷한 기사 ${h.duplicates}건</span>`;
  const items = h.similar || [];
  if (!items.length) return label;
  const li = items.map(s => `<li><a href="${esc(s.link)}" target="_blank" rel="noopener">${esc(s.title)}</a>
      <span class="muted">· ${esc(s.press)}${s.pub_date ? " · " + esc(s.pub_date) : ""}</span></li>`).join("");
  return `<details><summary>${label}</summary><ul class="similar">${li}</ul></details>`;
}


/* ---------------------------------------------------------------------------
   4. 질문 · 답변  ([질문] 버튼, 4번 카드)
   --------------------------------------------------------------------------- */

// 적은 순서대로 하나씩 (준비 중이면 준비가 끝날 때까지 기다렸다가)
$("f-ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  if (!question || !state.ready) return;      // 아직 준비를 시작하지 않았으면 받지 않는다
  $("question").value = "";
  enqueueQuestion(question);
  $("question").focus();                      // 바로 다음 질문을 적을 수 있게
});

// 질문을 대화창에 올리고 차례 줄(chain)에 넣는다. 아직 질문을 받을 수 없으면 받을 수 있을 때까지 기다렸다가 답한다.
// q = 질문 말풍선, a = 답변 말풍선 (처음엔 "기다리는 중", 답이 오면 그 자리에 채운다)
function enqueueQuestion(question) {
  const q = document.createElement("div"); q.className = "q"; q.textContent = question;
  const a = document.createElement("div"); a.className = "a waiting";
  a.innerHTML = state.sessionId ? `<span class="spinner"></span> 차례를 기다리는 중…`
                                : `⏳ 답할 수 있게 되면 차례로 답합니다.`;
  $("chat").append(q, a);
  a.scrollIntoView({ behavior: "smooth", block: "nearest" });
  // 이 질문이 기다릴 약속은 "질문을 적은 때"의 것이다 (그 사이 새 준비를 시작해도 섞이지 않게)
  const ready = state.ready, fresh = state.fresh;
  state.chain = state.chain.then(async () => {
    let sessionId;
    try { sessionId = await ready; }
    catch (err) { cannotAnswer(a, err); return; }
    if (await askStream(sessionId, question, a) === "wait") holdUntilFresh(fresh, sessionId, question, a);
  });
}

function cannotAnswer(a, err) {
  a.className = "a";
  a.innerHTML = `<span class="status err">답할 수 없습니다: ${esc(err.message)}</span>`;
}

// 먼저 답하는 중에 "DB에 저장된 기사에 근거 없음"이 온 질문: 새 기사까지 저장되면(fresh) 다시 묻는다.
// 그동안 다른 질문은 막지 않는다 (차례 줄에서 빠져 있다가, 그때 줄 끝에 다시 선다)
function holdUntilFresh(fresh, sessionId, question, a) {
  fresh.then(() => {
    if (fresh !== state.fresh) return;          // 그 사이 새 준비를 시작했다 → 이전 질문은 버린다
    state.chain = state.chain.then(() => askStream(sessionId, question, a));
  }, (err) => cannotAnswer(a, err));
}

// 답변 속 "(기사 1)" "(기사 1, 3)" 표시를 강조색으로 (출처 번호와 맞춰 보기 쉽게)
const citeHtml = (text) => esc(text).replace(/\(기사\s*\d+(?:\s*,\s*\d+)*\)/g, m => `<span class="cite">${m}</span>`);

// 답변 맨 위에 붙는 "언제 저장된 기사를 참고한 답변인지" 한 줄 (서버의 web.Basis.payload).
// db = 예전에 DB에 저장해 둔 기사로 먼저 답한 것 (노란 막대), fresh = 이번에 새로 모은 기사까지 참고한 것 (남색 막대)
// 그림 문자(이모지)는 쓰지 않는다: 글꼴에 따라 네모로 깨져 보였다
function basisHtml(b) {
  if (!b || !b.note) return "";
  return `<div class="basis ${b.stage === "db" ? "db" : "fresh"}">${esc(b.note)}</div>`;
}

// 답변을 만들어지는 대로 받는다 (/api/ask/stream: 한 줄에 JSON 하나).
// 서버가 보내는 줄 종류
//   {"type": "basis", "stage", "note"} 맨 처음: 어느 때 저장된 기사를 참고하는지 → 답변 위에 작게 보여 준다
//   {"type": "token", "text": "…"}   답변 글 조각 → 이어 붙여 바로 보여 준다 (끝에 깜빡이는 커서)
//   {"type": "done", "answer", "sources", "unused", "basis"}   완성본 → 답변 전체 + 출처 목록으로 바꾼다
//   {"type": "wait", "message"}      먼저 답하는 중인데 DB에 근거가 없음 → "wait"를 돌려준다 (새 기사 저장 뒤 다시 묻기)
//   {"type": "error", "message"}     실패 → 빨간 글로 보여 준다
async function askStream(sessionId, question, a) {
  a.className = "a waiting";
  a.innerHTML = `<span class="spinner"></span> 근거를 찾고 답변을 만드는 중…`;
  try {
    const res = await fetch("/api/ask/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(typeof data.detail === "string" ? data.detail : `오류 ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    // buf: 아직 줄바꿈이 안 온 조각 (네트워크로 오는 덩어리가 줄 단위로 끊겨 오지 않으므로 모아 둔다)
    let buf = "", text = "", top = "", finished = false;
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {   // 완성된 줄마다 하나씩 처리
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "basis") {
          top = basisHtml(ev);
        } else if (ev.type === "token") {
          text += ev.text;
          a.className = "a";
          a.innerHTML = top + citeHtml(text) + `<span class="caret"></span>`;
        } else if (ev.type === "done") {
          a.className = "a";
          a.innerHTML = basisHtml(ev.basis) + citeHtml(ev.answer) + renderSources(ev);
          finished = true;
        } else if (ev.type === "wait") {
          a.className = "a waiting";
          a.innerHTML = `⏳ ${esc(ev.message)}`;
          return "wait";
        } else if (ev.type === "error") {
          throw new Error(ev.message);
        }
      }
    }
    if (!finished) throw new Error("답변 연결이 끊겼습니다. 다시 질문해 주세요.");
  } catch (err) {
    a.className = "a";
    a.innerHTML = `<span class="status err">${esc(err.message)}</span>`;
  }
}

// 답변 아래 출처 목록. r.sources = [{ n, title, link, press, pub_date, kind }] (n = 답변 속 "기사 n" 번호)
// r.unused = 검색은 됐지만 답변에 인용되지 않아 뺀 기사 수
function renderSources(r) {
  if (!r.sources.length) return `<div class="srcs muted">출처: 관련성 있는 기사를 찾지 못해 이번 답변에는 뉴스 출처가 없습니다.</div>`;
  const li = r.sources.map(s => `<li>[${s.n}] <a href="${esc(s.link)}" target="_blank" rel="noopener">${esc(s.title)}</a>
    <span class="muted">· ${esc(s.press)}${s.pub_date ? " · " + esc(s.pub_date) : ""}${s.kind ? " · " + esc(s.kind) : ""}</span></li>`).join("");
  const extra = r.unused > 0 ? `<div class="muted">검색은 됐지만 답변에 쓰이지 않은 기사 ${r.unused}건은 뺐습니다.</div>` : "";
  return `<div class="srcs">출처<ol>${li}</ol>${extra}</div>`;
}
