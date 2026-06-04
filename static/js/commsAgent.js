// commsAgent.js — Universal Communication Agent panel.
//
// Drives the Comms modal: identify a signal's modality (one-shot) and run the
// iterative protocol-discovery loop, streaming its progress over SSE. Mirrors
// the lightweight tool-modal pattern (toggle `.hidden`, same-origin fetch).

const $ = (id) => document.getElementById(id);

let _activeJob = null;
let _streamAbort = null;

function _open() {
  const modal = $('comms-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  const ta = $('comms-signal');
  if (ta) ta.focus();
}

function _close() {
  const modal = $('comms-modal');
  if (!modal) return;
  modal.classList.add('hidden');
  if (_streamAbort) { try { _streamAbort.abort(); } catch (_) {} }
}

function _setStatus(text) {
  const el = $('comms-status');
  if (el) el.textContent = text || '';
}

function _append(html) {
  const out = $('comms-output');
  if (!out) return;
  out.insertAdjacentHTML('beforeend', html);
  out.scrollTop = out.scrollHeight;
}

function _esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function _clearOutput() {
  const out = $('comms-output');
  if (out) out.textContent = '';
}

async function _identify() {
  const signal = ($('comms-signal') || {}).value || '';
  if (!signal.trim()) { _setStatus('Enter a signal first.'); return; }
  _clearOutput();
  _setStatus('Identifying…');
  try {
    const res = await fetch('/api/comms/identify', {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signal }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const p = await res.json();
    const hyps = (p.modality_hypotheses || []).map(_esc).join(', ');
    _append(
      `<b>Modality profile</b>\n` +
      `  hypotheses: ${hyps || '—'}\n` +
      `  length: ${p.length}   alphabet: ${p.alphabet_size}\n` +
      `  entropy: ${p.entropy_bits_per_symbol} bits/symbol   redundancy: ${p.redundancy}\n` +
      `  looks_binary=${p.looks_binary}  looks_hex=${p.looks_hex}  looks_morse=${p.looks_morse}\n\n`
    );
    _setStatus('');
  } catch (e) {
    _setStatus('Identify failed: ' + e.message);
  }
}

function _renderEvent(ev) {
  const phase = ev.phase || ev.status || '';
  switch (phase) {
    case 'planning':
      _append('Analyzing signal and planning…\n'); break;
    case 'proposing':
      _append(`\n<b>Round ${ev.round}</b> (${_esc(ev.mode)} mode): proposing methods…\n`); break;
    case 'attempting':
      _append(`  · trying <i>${_esc(ev.method)}</i>…\n`); break;
    case 'measured': {
      const ok = ev.established ? ' ✓ ESTABLISHED' : '';
      _append(`  · ${_esc(ev.method)}: NMI=${ev.nmi}${ok}\n`); break;
    }
    case 'established':
      _append(`\n<b>Channel established</b> via ${_esc(ev.method)} (NMI=${ev.nmi}).\n`); break;
    case 'exhausted':
      _append(`\nNo full channel. Best so far: ${_esc(ev.best_method)} (NMI=${ev.best_nmi}).\n`); break;
    default:
      break;
  }
}

function _renderFinal(result) {
  if (!result) { _setStatus('Done.'); return; }
  if (result.error) { _setStatus('Error: ' + result.error); return; }
  const verdict = result.established
    ? `Channel ESTABLISHED via "${_esc(result.best_method)}"`
    : `No channel established (best: "${_esc(result.best_method)}")`;
  _append(
    `\n<b>${verdict}</b>\n` +
    `  normalized mutual information: ${result.best_normalized_mutual_information} ` +
    `(threshold ${result.threshold})\n` +
    (result.best_invented ? `  method was INVENTED by the agent\n` : '') +
    (result.best_decoded ? `  recovered: ${_esc(result.best_decoded)}\n` : '') +
    `  rounds: ${result.rounds}   elapsed: ${result.elapsed_seconds}s\n`
  );
  if (result.dossier) {
    _append(`\n<b>Dossier</b>\n${_esc(result.dossier)}\n`);
  }
  _setStatus(result.established ? 'Established.' : 'Exhausted.');
}

async function _discover() {
  const signal = ($('comms-signal') || {}).value || '';
  const context = ($('comms-context') || {}).value || '';
  if (!signal.trim()) { _setStatus('Enter a signal first.'); return; }
  _clearOutput();
  _setStatus('Starting discovery…');
  $('comms-cancel-btn').style.display = '';
  try {
    const res = await fetch('/api/comms/discover', {
      method: 'POST', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signal, context }),
    });
    if (!res.ok) {
      const msg = await res.text();
      throw new Error(msg || ('HTTP ' + res.status));
    }
    const { job_id } = await res.json();
    _activeJob = job_id;
    await _stream(job_id);
  } catch (e) {
    _setStatus('Discovery failed: ' + e.message);
    $('comms-cancel-btn').style.display = 'none';
  }
}

async function _stream(jobId) {
  _setStatus('Discovering…');
  _streamAbort = new AbortController();
  const res = await fetch(`/api/comms/stream/${jobId}`, {
    credentials: 'same-origin', signal: _streamAbort.signal,
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const chunks = buf.split('\n\n');
    buf = chunks.pop();
    for (const chunk of chunks) {
      const line = chunk.split('\n').find((l) => l.startsWith('data: '));
      if (!line) continue;
      let ev;
      try { ev = JSON.parse(line.slice(6)); } catch (_) { continue; }
      if (ev.final) {
        _renderFinal(ev.result);
        $('comms-cancel-btn').style.display = 'none';
        return;
      }
      _renderEvent(ev);
    }
  }
  $('comms-cancel-btn').style.display = 'none';
}

async function _cancel() {
  if (!_activeJob) return;
  _setStatus('Cancelling…');
  try {
    await fetch(`/api/comms/cancel/${_activeJob}`, {
      method: 'POST', credentials: 'same-origin',
    });
  } catch (_) {}
}

function _init() {
  const rail = $('rail-comms');
  if (rail) rail.addEventListener('click', _open);
  const sidebar = $('tool-comms-btn');
  if (sidebar) sidebar.addEventListener('click', _open);
  const close = $('close-comms-modal');
  if (close) close.addEventListener('click', _close);
  const idBtn = $('comms-identify-btn');
  if (idBtn) idBtn.addEventListener('click', _identify);
  const discBtn = $('comms-discover-btn');
  if (discBtn) discBtn.addEventListener('click', _discover);
  const cancelBtn = $('comms-cancel-btn');
  if (cancelBtn) cancelBtn.addEventListener('click', _cancel);
}

if (document.readyState !== 'loading') _init();
else document.addEventListener('DOMContentLoaded', _init);
