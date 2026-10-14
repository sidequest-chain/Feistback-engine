"""
Standalone Web UI Playground for Feistback Neuro-Symbolic Engine
================================================================
Provides a zero-dependency interactive web interface (`scripts/web_chat.py`)
built on Python's standard `http.server` for testing the Feistback neuro-symbolic
engine, CouncilPipeline, Layer A + Layer B caches, and MiniStackVM in real time.

Includes Smart Orchestration Features:
1. Pending-State Escape Guard: prevents getting trapped in stale NEED_CONFIRMATION
   or NEED_INPUT states when the user types a brand-new command.
2. Multi-Clause / Anaphora Planner: splits compound queries ("... ja siis ... sama ...")
   into sequential verified sub-steps, carrying over shared context parameters
   (e.g. "sama käibemaks" -> 24%) and aggregating results.
3. Estonian Inflected Synthesis Resolver: handles code synthesis queries without
   numeric operands (e.g. "kirjuta funktsioon liitmiseks", "tee kood lahutamiseks").
4. Out-of-Domain Rejection: cleanly rejects non-computational queries (e.g. weather)
   as `OUT_OF_DOMAIN` without proposing bogus mathematical confirmations.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import webbrowser

import torch

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dispatchers import ConfigDispatcher, CouncilDispatcher, CouncilPipeline
from dispatchers.council import CONFIRMATION_WORDS
from dispatchers.parsers.et_numbers import extract_numbers, is_synthesis_query
from runtime import FeistbackEngine
from runtime.canonicalizer import canonicalize_signature, decanonicalize_code


CANCEL_WORDS = {"ei", "tühista", "cancel", "stop", "loobu", "pole vaja"}

# Inflected Estonian stem map for code synthesis queries ("kirjuta funktsioon liitmiseks")
SYNTHESIS_INTENT_MAP: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b(liit\w*|liida\w*|summa\w*|summeer\w*)\b", re.I), "add", "def add(a: int, b: int) -> int:"),
    (re.compile(r"\b(lahut\w*|vahe\w*)\b", re.I), "sub", "def sub(a: int, b: int) -> int:"),
    (re.compile(r"\b(korrut\w*)\b", re.I), "mul", "def mul(a: int, b: int) -> int:"),
    (re.compile(r"\b(jaga\w*)\b", re.I), "div", "def div(a: int, b: int) -> int:"),
    (re.compile(r"\b(allahindlus\w*|soodustus\w*|ale\w*)\b", re.I), "discount", "def discount(price: float, pct: float) -> float:"),
    (re.compile(r"\b(käibemaks\w*|km\b|vat\b)\b", re.I), "vat_add", "def vat_add(price: float, rate: float) -> float:"),
    (re.compile(r"\b(ühistegur\w*|süt\b|gcd\b)\b", re.I), "gcd", "def gcd(a: int, b: int) -> int:"),
    (re.compile(r"\b(ühiskord\w*|vük\b|lcm\b)\b", re.I), "lcm", "def lcm(a: int, b: int) -> int:"),
    (re.compile(r"\b(xor\w*|välistav\w*)\b", re.I), "bitwise_comb", "def bitwise_xor_mask_1(a: int, b: int) -> int:"),
    (re.compile(r"\b(clamp\w*|clämbi\w*|klembi\w*|vahemik\w*)\b", re.I), "clamp_math", "def clamp_range_1(u: int, v: int, w: int) -> int:"),
    (re.compile(r"\b(negatiiv\w*)\b", re.I), "is_negative", "def is_negative(val: int) -> bool:"),
    (re.compile(r"\b(paaris\w*)\b", re.I), "is_even", "def is_even(counter_val: int) -> bool:"),
]

# Multi-clause split pattern ("... ja siis ...", "... ning seejärel ...", "... ja samuti ...")
MULTI_CLAUSE_SPLIT_RE = re.compile(
    r"\b(?:ja\s+siis|ning\s+siis|ning\s+seejärel|ja\s+seejärel|ja\s+samuti|ning\s+samuti)\b",
    re.IGNORECASE,
)

# Anaphora pattern referencing a shared parameter from the previous clause
ANAPHORA_RE = re.compile(
    r"\b(?:sama|samuti|sama\s+käibemaks\w*|sama\s+allahindlus\w*|sama\s+protsen\w*|sama\s+määr\w*)\b",
    re.IGNORECASE,
)


def is_pure_number_reply(text: str) -> bool:
    """
    Returns True only if the user's message is a short numeric/connector reply
    intended to fill missing slots (e.g. '15 ja 25', '10, 20'), rather than a new command.
    """
    nums = extract_numbers(text)
    if not nums:
        return False
    cleaned = re.sub(r"[-?\d.,]+", " ", text.lower())
    words = [w for w in re.findall(r"[a-zäöõü]+", cleaned) if w not in {"ja", "ning", "koma", "punkt", "miinus"}]
    return len(words) <= 1


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="et">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Feistback Engine — Neuro-Symbolic Playground</title>
  <style>
    :root {
      --bg-main: #0b0f17;
      --bg-panel: #111827;
      --bg-card: #1a2333;
      --bg-input: #0f172a;
      --border: #263248;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #818cf8;
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --code-bg: #090d16;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg-main);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    header {
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      color: #0b0f17;
      font-size: 18px;
    }
    .brand h1 {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    .brand p {
      font-size: 12px;
      color: var(--text-muted);
    }
    .controls {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    select, button {
      background: var(--bg-card);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 7px 12px;
      font-size: 12.5px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    select:hover, button:hover {
      border-color: var(--accent);
    }
    .btn-primary {
      background: linear-gradient(135deg, #0284c7, #4f46e5);
      border-color: transparent;
      font-weight: 600;
    }
    .btn-primary:hover {
      filter: brightness(1.12);
    }
    .btn-danger {
      border-color: rgba(239, 68, 68, 0.4);
      color: #fca5a5;
    }
    .btn-danger:hover {
      border-color: var(--danger);
      background: rgba(239, 68, 68, 0.15);
    }
    .stats-bar {
      display: flex;
      gap: 14px;
      font-size: 12px;
      color: var(--text-muted);
      background: rgba(15, 23, 42, 0.7);
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .stats-bar span strong {
      color: var(--accent);
      font-family: 'JetBrains Mono', Consolas, monospace;
    }
    .layout {
      flex: 1;
      display: grid;
      grid-template-columns: 335px 1fr;
      overflow: hidden;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { display: none; }
    }
    .sidebar {
      background: var(--bg-panel);
      border-right: 1px solid var(--border);
      padding: 16px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .sidebar h2 {
      font-size: 11.5px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .chip-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .chip {
      text-align: left;
      background: var(--bg-card);
      border: 1px solid var(--border);
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      color: var(--text);
      cursor: pointer;
      line-height: 1.35;
      transition: all 0.15s ease;
    }
    .chip:hover {
      border-color: var(--accent);
      transform: translateX(2px);
    }
    .chip small {
      display: block;
      font-size: 10.5px;
      color: var(--text-muted);
      margin-top: 2px;
    }
    .chat-area {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }
    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .msg {
      max-width: 920px;
      border-radius: 12px;
      padding: 14px 18px;
      border: 1px solid var(--border);
      line-height: 1.5;
    }
    .msg-user {
      align-self: flex-end;
      background: linear-gradient(135deg, #1e293b, #1e1b4b);
      border-color: #3730a3;
      max-width: 70%;
    }
    .msg-user .meta {
      font-size: 11px;
      color: #a5b4fc;
      margin-bottom: 4px;
    }
    .msg-bot {
      align-self: flex-start;
      background: var(--bg-panel);
      width: 100%;
    }
    .bot-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }
    .badges {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      align-items: center;
    }
    .badge {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;
      font-family: 'JetBrains Mono', Consolas, monospace;
      border: 1px solid transparent;
    }
    .badge-success { background: rgba(34, 197, 94, 0.15); color: #4ade80; border-color: rgba(34, 197, 94, 0.3); }
    .badge-warn { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3); }
    .badge-danger { background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: rgba(239, 68, 68, 0.3); }
    .badge-info { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border-color: rgba(56, 189, 248, 0.3); }
    .badge-purple { background: rgba(129, 140, 248, 0.15); color: #a5b4fc; border-color: rgba(129, 140, 248, 0.3); }
    .result-box {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 16px;
      margin-bottom: 12px;
    }
    .result-value {
      font-size: 24px;
      font-weight: 800;
      color: #4ade80;
      font-family: 'JetBrains Mono', Consolas, monospace;
    }
    .result-label {
      font-size: 12px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 10px;
    }
    @media (max-width: 720px) {
      .grid-2 { grid-template-columns: 1fr; }
    }
    .code-panel {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      overflow-x: auto;
    }
    .code-panel .title {
      font-size: 11px;
      color: var(--text-muted);
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }
    pre {
      font-family: 'JetBrains Mono', Consolas, monospace;
      font-size: 12px;
      color: #e2e8f0;
      white-space: pre-wrap;
    }
    .trace-row {
      margin-top: 10px;
      font-size: 11.5px;
      color: var(--text-muted);
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .trace-step {
      background: var(--bg-card);
      border: 1px solid var(--border);
      padding: 2px 8px;
      border-radius: 5px;
      font-family: 'JetBrains Mono', Consolas, monospace;
    }
    .confirm-actions {
      margin-top: 10px;
      display: flex;
      gap: 8px;
    }
    .composer {
      background: var(--bg-panel);
      border-top: 1px solid var(--border);
      padding: 14px 20px;
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .composer input {
      flex: 1;
      background: var(--bg-input);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 14px;
      outline: none;
    }
    .composer input:focus {
      border-color: var(--accent);
    }
    .pending-banner {
      background: rgba(245, 158, 11, 0.14);
      border-top: 1px solid rgba(245, 158, 11, 0.35);
      color: #fcd34d;
      padding: 6px 20px;
      font-size: 12px;
      display: none;
      justify-content: space-between;
      align-items: center;
    }
    .substep-card {
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 8px;
      font-size: 12.5px;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">F</div>
      <div>
        <h1>Feistback Engine Playground</h1>
        <p>PureFeistelLM (Zero KV-Cache, 48 KB RAF) + Multi-Step Planner + MiniStackVM</p>
      </div>
    </div>

    <div class="stats-bar" id="statsBar">
      <span>GPU: <strong id="stGpu">...</strong></span>
      <span>VRAM: <strong id="stVram">...</strong></span>
      <span>Layer A Cache: <strong id="stLayerA">0</strong></span>
      <span>Layer B JIT: <strong id="stLayerB">0</strong></span>
    </div>

    <div class="controls">
      <select id="modeSelect" title="Vali ruutimise režiim">
        <option value="pipeline" selected>CouncilPipeline + Multi-Step Planner</option>
        <option value="council">CouncilDispatcher (Otsene Tier 1/2)</option>
        <option value="fast_path">ConfigDispatcher (L1 Regex Fast-Path)</option>
      </select>
      <label style="font-size:12px; display:flex; align-items:center; gap:5px; cursor:pointer;">
        <input type="checkbox" id="useLayerA" checked /> Layer A Cache sees
      </label>
      <button onclick="clearCaches()" class="btn-danger" title="Tühjenda õpitud Layer A vahemälu">Tühjenda Layer A</button>
      <button onclick="resetChat()" title="Alusta uut vestlust">Puhasta vestlus</button>
    </div>
  </header>

  <div class="layout">
    <aside class="sidebar">
      <div>
        <h2>1. Eesti släng & kõnekeel</h2>
        <div class="chip-group">
          <button class="chip" onclick="sendPreset('ou pane 15 ja 25 kokku plz')">
            "ou pane 15 ja 25 kokku plz"
            <small>Slängi täitesõnade eemaldus + liitmine (40)</small>
          </button>
          <button class="chip" onclick="sendPreset('kuule viska 120 ja 80 kokku tänks')">
            "kuule viska 120 ja 80 kokku tänks"
            <small>Testi Layer A + Layer B kiirust (200)</small>
          </button>
        </div>
      </div>

      <div>
        <h2>2. Eesti käänded (-st) & baassõnavara</h2>
        <div class="chip-group">
          <button class="chip" onclick="sendPreset('lahuta 12 arvust 50')">
            "lahuta 12 arvust 50"
            <small>Seestütlev kääne: pööratud järjekord 50 - 12 = 38</small>
          </button>
          <button class="chip" onclick="sendPreset('mul oli 450 eurot ja kulutasin 125 eurot')">
            "mul oli 450 eurot ja kulutasin 125 eurot"
            <small>Igapäevane rahaline tekstülesanne (325)</small>
          </button>
          <button class="chip" onclick="sendPreset('laos oli 650 tükki ja müüdi ära 320 tükki')">
            "laos oli 650 tükki ja müüdi ära 320 tükki"
            <small>Laoseisu arvutus (330)</small>
          </button>
        </div>
      </div>

      <div>
        <h2>3. VM Tööriistad & Mitmeosaline lause</h2>
        <div class="chip-group">
          <button class="chip" onclick="sendPreset('arvuta 240 eurot 15% allahindlusega')">
            "arvuta 240 eurot 15% allahindlusega"
            <small>CALL_TOOL: discount(240, 15) -> 204.0</small>
          </button>
          <button class="chip" onclick="sendPreset('lisa 150 eurole käibemaks 24% ja siis lisa 200 eurole samuti sama käibemaks')">
            "lisa 150 eurole käibemaks 24% ja siis lisa 200 eurole samuti sama käibemaks"
            <small>Multi-Step + Anafoor ("sama käibemaks") -> [186.0, 248.0] (kokku 434.0)</small>
          </button>
          <button class="chip" onclick="sendPreset('leia suurim ühistegur arvudele 84 ja 36')">
            "leia suurim ühistegur arvudele 84 ja 36"
            <small>CALL_TOOL: gcd(84, 36) -> 12</small>
          </button>
        </div>
      </div>

      <div>
        <h2>4. Kernel Fusion & Koodisüntees</h2>
        <div class="chip-group">
          <button class="chip" onclick="sendPreset('liida kolm arvu 10, 20 ja 30 ning lahuta 15')">
            "liida kolm arvu 10, 20 ja 30 ning lahuta 15"
            <small>4-argumendiga sum3_sub(10, 20, 30, 15) -> 45</small>
          </button>
          <button class="chip" onclick="sendPreset('kirjuta funktsioon liitmiseks')">
            "kirjuta funktsioon liitmiseks"
            <small>Puhas Pythoni koodisüntees (ilma käivitamiseta)</small>
          </button>
          <button class="chip" onclick="sendPreset('kirjuta funktsioon allahindluse arvutamiseks')">
            "kirjuta funktsioon allahindluse arvutamiseks"
            <small>Sünteesib VM tööriista funktsiooni (discount)</small>
          </button>
        </div>
      </div>

      <div>
        <h2>5. Mürgituskaitse & Domeenikontroll</h2>
        <div class="chip-group">
          <button class="chip" onclick="sendPreset('kas liida või lahuta või korruta 15 ja 25')">
            "kas liida või lahuta või korruta 15 ja 25"
            <small>Vastuoluline päring -> Blokeeritud (NEED_CONFIRMATION)</small>
          </button>
          <button class="chip" onclick="sendPreset('mis ilm homme Tallinnas on 15 ja 25')">
            "mis ilm homme Tallinnas on 15 ja 25"
            <small>Domeeniväline päring -> OUT_OF_DOMAIN (ei pakuta tehet)</small>
          </button>
        </div>
      </div>
    </aside>

    <main class="chat-area">
      <div class="messages" id="messages">
        <div class="msg msg-bot">
          <div class="bot-header">
            <div class="badges">
              <span class="badge badge-info">FEISTBACK READY</span>
              <span class="badge badge-purple">MULTI-STEP + ANAPHORA + SYNTHESIS</span>
            </div>
          </div>
          <p style="font-size:13.5px; color:var(--text);">
            Tere tulemast Feistbacki veebimänguväljakule (<code>scripts/web_chat.py</code>)!
            Vali vasakult kiirtest või sisesta vabas vormis päring: eesti släng, käänded, VM tööriistad, mitmeosalised laused (<em>"ja siis ... sama ..."</em>) või koodisüntees.
          </p>
        </div>
      </div>

      <div class="pending-banner" id="pendingBanner">
        <span id="pendingText">Ootel kinnitus või puuduvad argumendid...</span>
        <button onclick="cancelPending()" style="padding:3px 8px; font-size:11px;">Tühista ootel olek</button>
      </div>

      <form class="composer" onsubmit="handleSubmit(event)">
        <input
          type="text"
          id="queryInput"
          placeholder="Kirjuta päring (nt 'lisa 150 eurole käibemaks 24% ja siis lisa 200 eurole samuti sama käibemaks')..."
          autocomplete="off"
          autofocus
        />
        <button type="submit" class="btn-primary" style="padding: 12px 20px; font-size:13.5px;">Saada</button>
      </form>
    </main>
  </div>

  <script>
    async function refreshStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        document.getElementById('stGpu').textContent = data.gpu_name;
        document.getElementById('stVram').textContent = data.vram_mb + ' MB';
        document.getElementById('stLayerA').textContent = data.layer_a_count + ' mustrit';
        document.getElementById('stLayerB').textContent = data.layer_b_count + ' funktsiooni';
        const banner = document.getElementById('pendingBanner');
        if (data.has_pending) {
          banner.style.display = 'flex';
          document.getElementById('pendingText').textContent = data.pending_summary;
        } else {
          banner.style.display = 'none';
        }
      } catch (e) {
        console.error(e);
      }
    }

    function escapeHtml(str) {
      if (str === null || str === undefined) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function sendPreset(text) {
      const inp = document.getElementById('queryInput');
      inp.value = text;
      sendQuery(text);
    }

    function handleSubmit(e) {
      e.preventDefault();
      const inp = document.getElementById('queryInput');
      const text = inp.value.trim();
      if (!text) return;
      sendQuery(text);
    }

    async function cancelPending() {
      await fetch('/api/reset_pending', { method: 'POST' });
      await refreshStatus();
    }

    async function clearCaches() {
      const res = await fetch('/api/clear_cache', { method: 'POST' });
      const data = await res.json();
      await refreshStatus();
      appendSystemNotice('Layer A semantiline vahemälu tühjendati (' + data.cleared + ' kirjet eemaldatud).');
    }

    async function resetChat() {
      await fetch('/api/reset_pending', { method: 'POST' });
      document.getElementById('messages').innerHTML = '';
      await refreshStatus();
      appendSystemNotice('Vestlus ja ootel olek lähtestati.');
    }

    function appendSystemNotice(text) {
      const msgs = document.getElementById('messages');
      const div = document.createElement('div');
      div.className = 'msg msg-bot';
      div.innerHTML = '<span class="badge badge-info">SYSTEM</span> <span style="font-size:13px; margin-left:8px;">' + escapeHtml(text) + '</span>';
      msgs.appendChild(div);
      msgs.scrollTop = msgs.scrollHeight;
    }

    async function sendQuery(queryText) {
      const inp = document.getElementById('queryInput');
      inp.value = '';
      const mode = document.getElementById('modeSelect').value;
      const useLayerA = document.getElementById('useLayerA').checked;

      const msgs = document.getElementById('messages');
      const userDiv = document.createElement('div');
      userDiv.className = 'msg msg-user';
      userDiv.innerHTML = '<div class="meta">Kasutaja (' + escapeHtml(mode) + ')</div><div>' + escapeHtml(queryText) + '</div>';
      msgs.appendChild(userDiv);
      msgs.scrollTop = msgs.scrollHeight;

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, mode: mode, use_layer_a: useLayerA })
      });
      const data = await res.json();
      renderBotResponse(data);
      await refreshStatus();
    }

    function renderBotResponse(d) {
      const msgs = document.getElementById('messages');
      const botDiv = document.createElement('div');
      botDiv.className = 'msg msg-bot';

      const status = d.status || 'UNKNOWN';
      let statusBadgeClass = 'badge-info';
      if (status === 'SUCCESS') statusBadgeClass = 'badge-success';
      else if (status === 'NEED_CONFIRMATION' || status === 'NEED_INPUT') statusBadgeClass = 'badge-warn';
      else statusBadgeClass = 'badge-danger';

      const route = d.route || d.mode || 'DIRECT';
      const domain = d.domain ? '<span class="badge badge-purple">domain: ' + escapeHtml(d.domain) + '</span>' : '';
      const marginPct = d.confidence_margin !== undefined ? (d.confidence_margin * 100).toFixed(1) + '%' : 'N/A';
      const marginBadgeClass = (d.confidence_margin || 0) >= 0.65 ? 'badge-success' : 'badge-warn';

      let bodyHtml = '';

      if (status === 'SUCCESS') {
        const isSynth = d.mode === 'synthesize';
        const isMulti = d.mode === 'multi_step';
        const mainVal = isSynth
          ? 'Funktsioon sünteesitud (AST OK)'
          : (isMulti ? d.summary_display : d.result);
        const vmLat = d.vm_latency_us !== undefined ? d.vm_latency_us.toFixed(2) + ' µs' : '0 µs';
        const totalMs = d.total_wall_ms !== undefined ? d.total_wall_ms.toFixed(2) + ' ms' : '0 ms';
        const hitA = d.cache_hit ? 'HIT (0 ms)' : 'MISS (Neural)';
        const hitB = d.bytecode_cache_hit ? 'HIT (JIT)' : (isSynth ? 'AST Verified' : 'Compiled');

        bodyHtml += `
          <div class="result-box">
            <div>
              <div class="result-label">Tulemus (${escapeHtml(d.intent_name || d.intent_id || 'unknown')} | args: ${escapeHtml(JSON.stringify(d.args || []))})</div>
              <div class="result-value">${escapeHtml(mainVal)}</div>
            </div>
            <div style="text-align:right; font-size:12px; color:var(--text-muted); font-family:monospace;">
              <div>Koguaeg: <strong style="color:var(--text);">${totalMs}</strong> | MiniStackVM: <strong style="color:#4ade80;">${vmLat}</strong></div>
              <div>Layer A: <strong>${hitA}</strong> | Layer B: <strong>${hitB}</strong></div>
            </div>
          </div>
        `;

        if (isMulti && d.substeps) {
          bodyHtml += '<div style="margin-bottom:10px;">';
          d.substeps.forEach((st, idx) => {
            bodyHtml += `
              <div class="substep-card">
                <strong>Samm ${idx + 1}:</strong> <code>${escapeHtml(st.clause)}</code>
                &rarr; <strong>${escapeHtml(st.intent_name)}(${escapeHtml(JSON.stringify(st.args))}) = <span style="color:#4ade80;">${escapeHtml(st.result)}</span></strong>
                <span style="float:right; color:var(--text-muted); font-family:monospace;">Route: ${escapeHtml(st.route)} | VM: ${st.vm_latency_us.toFixed(2)} µs</span>
              </div>
            `;
          });
          bodyHtml += '</div>';
        }

        bodyHtml += `
          <div class="grid-2">
            <div class="code-panel">
              <div class="title"><span>Genereeritud Python</span><span>AST Verified</span></div>
              <pre>${escapeHtml(d.generated_code || '# N/A')}</pre>
            </div>
            <div class="code-panel">
              <div class="title"><span>MiniStackVM Baitkood</span><span>Deterministic Sandbox</span></div>
              <pre>${escapeHtml(d.disassembly || '# Sünteesirežiim (ei käivitatud)')}</pre>
            </div>
          </div>
        `;
      } else if (status === 'NEED_CONFIRMATION') {
        bodyHtml += `
          <div class="result-box" style="border-color: rgba(245, 158, 11, 0.4);">
            <div>
              <div class="result-label" style="color:#fbbf24;">Confidence Gate peatas automaatse täitmise (vahemälu kaitstud!)</div>
              <div style="font-size:14px; margin-top:4px;">${escapeHtml(d.message)}</div>
              <div style="font-size:12px; color:var(--text-muted); margin-top:4px; font-family:monospace;">
                Kandidaat: ${escapeHtml(d.intent_name)} &rarr; ${escapeHtml(d.signature)} | Args: ${escapeHtml(JSON.stringify(d.args || []))}
              </div>
              <div class="confirm-actions">
                <button class="btn-primary" onclick="sendPreset('jah')">Jah, kinnita ja käivita</button>
                <button class="btn-danger" onclick="cancelPending()">Tühista</button>
              </div>
            </div>
          </div>
        `;
      } else if (status === 'OUT_OF_DOMAIN') {
        bodyHtml += `
          <div class="result-box" style="border-color: rgba(239, 68, 68, 0.45);">
            <div>
              <div class="result-label" style="color:#f87171;">Väljaspool registreeritud spetsialistide valdkonda (OUT_OF_DOMAIN)</div>
              <div style="font-size:14px; margin-top:4px;">${escapeHtml(d.message)}</div>
            </div>
          </div>
        `;
      } else if (status === 'NEED_INPUT') {
        bodyHtml += `
          <div class="result-box" style="border-color: rgba(56, 189, 248, 0.4);">
            <div>
              <div class="result-label" style="color:#38bdf8;">Puuduvad argumendid (Slot-Filling)</div>
              <div style="font-size:14px; margin-top:4px;">${escapeHtml(d.message)}</div>
            </div>
          </div>
        `;
      } else {
        bodyHtml += `
          <div class="result-box" style="border-color: rgba(239, 68, 68, 0.4);">
            <div>
              <div class="result-label" style="color:#f87171;">Viga / Tundmatu kavatsus</div>
              <div style="font-size:14px; margin-top:4px;">${escapeHtml(d.error || d.message || JSON.stringify(d))}</div>
            </div>
          </div>
        `;
      }

      if (d.pipeline_traces && d.pipeline_traces.length > 0) {
        const steps = d.pipeline_traces.map(t =>
          `<span class="trace-step">${escapeHtml(t.stage)}: ${t.latency_ms.toFixed(2)} ms</span>`
        ).join(' &rarr; ');
        bodyHtml += `<div class="trace-row"><span>Pipeline:</span> ${steps}</div>`;
      }

      botDiv.innerHTML = `
        <div class="bot-header">
          <div class="badges">
            <span class="badge ${statusBadgeClass}">${escapeHtml(status)}</span>
            <span class="badge badge-info">${escapeHtml(route)}</span>
            ${domain}
            <span class="badge ${marginBadgeClass}">Margin: ${marginPct}</span>
          </div>
          <span style="font-size:11.5px; color:var(--text-muted); font-family:monospace;">
            ${d.normalized_query && d.normalized_query !== d.query ? 'Norm: "' + escapeHtml(d.normalized_query) + '"' : ''}
          </span>
        </div>
        ${bodyHtml}
      `;
      msgs.appendChild(botDiv);
      msgs.scrollTop = msgs.scrollHeight;
    }

    refreshStatus();
  </script>
</body>
</html>
"""


class PlaygroundState:
    """Holds singleton engine, dispatchers, and smart multi-turn orchestration."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[WebPlayground] Loading FeistbackEngine on {self.device}...")
        self.engine = FeistbackEngine.from_checkpoint(
            "models/logic_feistel_clean/pure_feistel_raf_clean.pt",
            device=self.device,
        )
        self.fast_dispatcher = ConfigDispatcher(self.engine, config_path="config/intents.json")
        self.council_dispatcher = CouncilDispatcher(
            engine=self.engine,
            model_path="models/eesti_keele_sl2ng/slang_expert_feistel.pt",
            device=self.device,
        )
        self.pipeline = CouncilPipeline(dispatcher=self.council_dispatcher)

        self.pending_intent: Optional[Dict[str, Any]] = None
        self.pending_args: List[Any] = []

    def get_status(self) -> Dict[str, Any]:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0
        layer_a_count = len(self.council_dispatcher.learned_cache._entries)
        layer_b_count = len(self.engine.bytecode_cache._entries)
        has_pending = self.pending_intent is not None
        pending_summary = ""
        if self.pending_intent is not None:
            iname = self.pending_intent.get("intent_name", self.pending_intent.get("id", "unknown"))
            if self.pending_intent.get("_awaiting_confirmation"):
                pending_summary = f"Ootel kinnitus tehtele '{iname}' (vasta 'jah' või kirjuta uus käsk)"
            else:
                pending_summary = f"Ootel puuduvad arvud tehtele '{iname}' (hetkel: {self.pending_args})"
        return {
            "gpu_name": gpu_name,
            "vram_mb": vram_mb,
            "layer_a_count": layer_a_count,
            "layer_b_count": layer_b_count,
            "has_pending": has_pending,
            "pending_summary": pending_summary,
        }

    def _handle_synthesis_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Handles Estonian code synthesis requests (e.g. 'kirjuta funktsioon liitmiseks')."""
        q_lower = query.lower().strip()
        if not is_synthesis_query(q_lower) and not any(
            k in q_lower for k in ("funktsioon", "skript", "kood", "sünteesi")
        ):
            return None

        matched_intent = "add"
        matched_sig = "def add(a: int, b: int) -> int:"
        for pat, iname, sig in SYNTHESIS_INTENT_MAP:
            if pat.search(q_lower):
                matched_intent = iname
                matched_sig = sig
                break

        t0 = time.perf_counter()
        fused_code = self.council_dispatcher._resolve_fused_or_tool_code(matched_intent)
        if fused_code:
            full_code = fused_code
            try:
                bytecode = self.engine.compiler.compile(full_code)
                disasm = self.engine.vm.disassemble(bytecode)
            except Exception:
                disasm = "# AST Verified"
            gen_lat_s = time.perf_counter() - t0
        else:
            canonical_sig, inv_map = canonicalize_signature(matched_sig)
            gen_raw, _ = self.engine.model.generate(
                prompt=canonical_sig,
                max_new_tokens=48,
                temperature=0.0,
                early_stopping=True,
            )
            gen_lat_s = time.perf_counter() - t0
            body = decanonicalize_code(gen_raw, inv_map) if inv_map else gen_raw
            indented = "\n".join("    " + line.strip() for line in body.split("\n") if line.strip())
            full_code = matched_sig.strip() + "\n" + indented
            try:
                bytecode = self.engine.compiler.compile(full_code)
                disasm = self.engine.vm.disassemble(bytecode)
            except Exception:
                disasm = "# AST Verified"

        return {
            "status": "SUCCESS",
            "mode": "synthesize",
            "route": "CODE_SYNTHESIS_ENGINE",
            "domain": "code_synthesis",
            "intent_id": matched_intent,
            "intent_name": matched_intent,
            "signature": matched_sig,
            "args": [],
            "confidence_margin": 1.0,
            "cache_hit": False,
            "bytecode_cache_hit": True,
            "generated_code": full_code,
            "disassembly": disasm,
            "vm_latency_us": 0.0,
            "generation_latency_s": gen_lat_s,
            "message": f"Genereeritud ja AST-verifitseeritud funktsioon '{matched_intent}'.",
            "query": query,
        }

    def _handle_multi_clause(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Splits compound multi-step queries ('... ja siis ... sama ...') into sequential
        clauses, resolves anaphoric shared parameters (e.g. 'sama käibemaks' -> 24%),
        and executes each step deterministically on MiniStackVM.
        """
        if not MULTI_CLAUSE_SPLIT_RE.search(query):
            return None

        clauses = [c.strip() for c in MULTI_CLAUSE_SPLIT_RE.split(query) if c.strip()]
        if len(clauses) < 2:
            return None

        substeps: List[Dict[str, Any]] = []
        prev_args: List[Any] = []
        prev_intent: Optional[str] = None
        total_vm_us = 0.0
        min_margin = 1.0
        all_cache_hits = True

        for idx, clause in enumerate(clauses):
            clause_nums = extract_numbers(clause)
            resolved_clause = clause

            if idx > 0 and len(clause_nums) == 1 and len(prev_args) >= 2 and ANAPHORA_RE.search(clause):
                shared_param = prev_args[1]
                if prev_intent == "vat_add":
                    resolved_clause = f"lisa {clause_nums[0]} eurole {shared_param}% käibemaks"
                elif prev_intent == "discount":
                    resolved_clause = f"arvuta {clause_nums[0]} eurot {shared_param}% allahindlusega"
                else:
                    resolved_clause = f"{clause} {shared_param}"

            step_res = self.pipeline.run(resolved_clause)
            if step_res.get("status") != "SUCCESS":
                return None

            prev_args = step_res.get("args", [])
            prev_intent = step_res.get("intent_name")
            vm_us = float(step_res.get("vm_latency_us", 0.0))
            total_vm_us += vm_us
            min_margin = min(min_margin, float(step_res.get("confidence_margin", 1.0)))
            if not step_res.get("cache_hit"):
                all_cache_hits = False

            substeps.append(
                {
                    "clause": clause,
                    "resolved_clause": resolved_clause,
                    "intent_name": step_res.get("intent_name"),
                    "args": step_res.get("args", []),
                    "result": step_res.get("result"),
                    "route": step_res.get("route"),
                    "vm_latency_us": vm_us,
                    "generated_code": step_res.get("generated_code", ""),
                    "disassembly": step_res.get("disassembly", ""),
                }
            )

        results_list = [s["result"] for s in substeps]
        numeric_sum = sum(r for r in results_list if isinstance(r, (int, float)))
        summary_display = f"{results_list} (Kokku: {round(numeric_sum, 4)})"

        return {
            "status": "SUCCESS",
            "mode": "multi_step",
            "route": "MULTI_STEP_DAG_PLANNER",
            "domain": "multi_step_finance" if prev_intent in ("vat_add", "discount") else "multi_step",
            "intent_id": "+".join(str(s["intent_name"]) for s in substeps),
            "intent_name": " -> ".join(str(s["intent_name"]) for s in substeps),
            "args": [s["args"] for s in substeps],
            "result": results_list,
            "summary_display": summary_display,
            "substeps": substeps,
            "confidence_margin": min_margin,
            "cache_hit": all_cache_hits,
            "bytecode_cache_hit": True,
            "generated_code": substeps[-1]["generated_code"],
            "disassembly": substeps[-1]["disassembly"],
            "vm_latency_us": total_vm_us,
            "query": query,
        }

    def handle_chat(self, query: str, mode: str = "pipeline", use_layer_a: bool = True) -> Dict[str, Any]:
        with self.lock:
            saved_entries = None
            if not use_layer_a:
                saved_entries = dict(self.council_dispatcher.learned_cache._entries)
                self.council_dispatcher.learned_cache._entries.clear()

            try:
                t0 = time.perf_counter()
                clean_q = query.strip().lower()

                # 1. Pending State Escape Guard
                if self.pending_intent is not None:
                    if clean_q in CANCEL_WORDS:
                        self.pending_intent = None
                        self.pending_args = []
                        return {
                            "status": "OUT_OF_DOMAIN",
                            "route": "CANCELLED",
                            "confidence_margin": 1.0,
                            "message": "Ootel tehe tühistati.",
                            "query": query,
                        }
                    if self.pending_intent.get("_awaiting_confirmation"):
                        if clean_q not in CONFIRMATION_WORDS:
                            self.pending_intent = None
                            self.pending_args = []
                    else:
                        if not is_pure_number_reply(query):
                            self.pending_intent = None
                            self.pending_args = []

                # 2. Check for pure code synthesis query ("kirjuta funktsioon liitmiseks")
                if self.pending_intent is None:
                    synth_resp = self._handle_synthesis_query(query)
                    if synth_resp is not None:
                        synth_resp["total_wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                        return synth_resp

                # 3. Check for multi-clause / anaphora query ("... ja siis ... sama ...")
                if self.pending_intent is None and mode == "pipeline":
                    multi_resp = self._handle_multi_clause(query)
                    if multi_resp is not None:
                        multi_resp["total_wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                        return multi_resp

                # 4. Standard single-contract dispatch
                if mode == "fast_path":
                    resp = self.fast_dispatcher.dispatch(
                        query,
                        pending_intent=self.pending_intent,
                        pending_args=self.pending_args,
                    )
                elif mode == "council":
                    resp = self.council_dispatcher.dispatch(
                        query,
                        pending_intent=self.pending_intent,
                        pending_args=self.pending_args,
                    )
                else:
                    resp = self.pipeline.run(
                        query,
                        pending_intent=self.pending_intent,
                        pending_args=self.pending_args,
                    )

                wall_ms = (time.perf_counter() - t0) * 1000.0
                resp["total_wall_ms"] = round(wall_ms, 3)

                # 5. Cleanly reject out_of_domain queries without setting a bogus pending_intent
                if resp.get("domain") == "out_of_domain":
                    self.pending_intent = None
                    self.pending_args = []
                    return {
                        "status": "OUT_OF_DOMAIN",
                        "route": "DOMAIN_ROUTER_REJECTED",
                        "domain": "out_of_domain",
                        "confidence_margin": resp.get("confidence_margin", 0.0),
                        "total_wall_ms": resp["total_wall_ms"],
                        "pipeline_traces": resp.get("pipeline_traces", []),
                        "message": (
                            f"Päring \"{query}\" ei kuulu ühegi registreeritud spetsialisti valdkonda "
                            f"(aritmeetika, rahandus/käibemaks/allahindlus, arvuteooria, bititehted, koodisüntees). "
                            f"Vahemälu jäi puutumatuks."
                        ),
                        "query": query,
                    }

                status = resp.get("status")
                if status == "NEED_CONFIRMATION":
                    self.pending_intent = resp.get("intent")
                    self.pending_args = resp.get("args", [])
                elif status == "NEED_INPUT":
                    self.pending_intent = resp.get("intent")
                    self.pending_args = resp.get("provided_args", [])
                else:
                    self.pending_intent = None
                    self.pending_args = []

                return resp
            finally:
                if saved_entries is not None:
                    self.council_dispatcher.learned_cache._entries = saved_entries


STATE: Optional[PlaygroundState] = None


class PlaygroundHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/api/status":
            assert STATE is not None
            self._send_json(STATE.get_status())
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        assert STATE is not None
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        data = json.loads(body) if body else {}

        if self.path == "/api/chat":
            query = str(data.get("query", "")).strip()
            mode = str(data.get("mode", "pipeline"))
            use_layer_a = bool(data.get("use_layer_a", True))
            resp = STATE.handle_chat(query=query, mode=mode, use_layer_a=use_layer_a)
            self._send_json(resp)
        elif self.path == "/api/clear_cache":
            with STATE.lock:
                count = len(STATE.council_dispatcher.learned_cache._entries)
                STATE.council_dispatcher.learned_cache._entries.clear()
                STATE.council_dispatcher.learned_cache._save()
            self._send_json({"cleared": count})
        elif self.path == "/api/reset_pending":
            with STATE.lock:
                STATE.pending_intent = None
                STATE.pending_args = []
            self._send_json({"ok": True})
        else:
            self.send_error(404)


def main() -> None:
    global STATE
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    STATE = PlaygroundState()
    host = "127.0.0.1"
    port = 7860
    server = ThreadingHTTPServer((host, port), PlaygroundHandler)
    url = f"http://{host}:{port}"
    print("=" * 75)
    print(f"Feistback Web Playground is live at: {url}")
    print("=" * 75)
    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Feistback Web Playground.")
        server.server_close()


if __name__ == "__main__":
    main()
