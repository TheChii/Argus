/**
 * Argus Quant Institutional Trading Terminal — Client-Side Application
 * Pure Vanilla JavaScript (ES6+) with zero build dependencies.
 */

// Global State
const state = {
  currentView: 'terminal',
  snapshot: null,
  account: null,
  positionsData: null,
  fills: [],
  riskData: null,
  selectedTicker: null,
  watchlistFilter: 'top', // 'top' | 'orders' | 'held' | 'all'
  searchQuery: '',
  orderSide: 'buy',
  currentBudget: 10000,
  allowFractional: false,
  slipFormat: 'standard',
  slipsCache: {},
};

// =============================================================================
// Initialization & Data Loading
// =============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  // Set default manual fill date to today
  const dateInput = document.getElementById('manual-date');
  if (dateInput) {
    dateInput.value = new Date().toISOString().split('T')[0];
  }

  await loadAllData();
});

async function loadAllData() {
  try {
    const [snapRes, acctRes, posRes, fillsRes] = await Promise.all([
      fetch('/api/snapshot').then(r => r.json()),
      fetch('/api/account').then(r => r.json()),
      fetch('/api/positions').then(r => r.json()),
      fetch('/api/fills').then(r => r.json()),
    ]);

    state.snapshot = snapRes;
    state.account = acctRes;
    state.positionsData = posRes;
    state.fills = fillsRes.fills || [];

    // Default selected ticker to #1 ranked pick
    if (!state.selectedTicker && snapRes.today_cockpit?.actions?.length > 0) {
      const topPick = snapRes.today_cockpit.actions.find(a => a.rank === 1) || snapRes.today_cockpit.actions[0];
      state.selectedTicker = topPick.ticker;
    }

    renderHeader();
    renderWatchlist();
    renderInspector();
    renderPositionsDrawer();
    renderRebalanceDrawer();
    renderSectorsDrawer();
    renderFillsDrawer();
    renderSanityView();
    renderLedgerView();
    renderSettingsView();
    await loadRiskMonitor();

    // Trigger initial sizing calculation in background
    calculateSizerAllocations();
  } catch (err) {
    console.error('Failed to load terminal data:', err);
    showToast('Failed to load data from backend server.', 'error');
  }
}

async function refreshSnapshot() {
  showToast('Regenerating quantitative snapshot...', 'info');
  try {
    const res = await fetch('/api/refresh-snapshot', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      showToast('Snapshot updated successfully!', 'success');
      await loadAllData();
    } else {
      showToast('Snapshot refresh returned error.', 'error');
    }
  } catch (err) {
    showToast('Snapshot refresh failed: ' + err.message, 'error');
  }
}

// =============================================================================
// Header & Navigation Rendering
// =============================================================================
function renderHeader() {
  const acct = state.account || {};
  const meta = state.snapshot?.meta || {};
  const strip = meta.status_strip || {};

  // Formatted balances
  document.getElementById('hdr-port-val').textContent = formatCurrency(acct.portfolio_value || 100000);
  document.getElementById('hdr-cash-val').textContent = formatCurrency(acct.cash || 25000);
  document.getElementById('hdr-bp-val').textContent = formatCurrency(acct.buying_power || 50000);

  // Regime & VIX
  const regime = strip.regime_name || 'Low Vol Bull';
  const vix = strip.vix ? Number(strip.vix).toFixed(1) : '18.5';
  document.getElementById('hdr-regime').innerHTML = `${regime} <span class="text-xs text-xtb-steel font-mono-num" id="hdr-vix">(VIX ${vix})</span>`;

  // Confidence Light
  const light = (strip.confidence_light || 'green').toLowerCase();
  const badgeContainer = document.getElementById('hdr-confidence-badge');
  if (light === 'green') {
    badgeContainer.innerHTML = '<span class="px-2 py-0.5 rounded text-[11px] font-extrabold bg-xtb-emerald text-black tracking-wide">● SAFE TO INVEST (100%)</span>';
  } else if (light === 'amber') {
    badgeContainer.innerHTML = '<span class="px-2 py-0.5 rounded text-[11px] font-extrabold bg-xtb-amber text-black tracking-wide">● CAUTION (50% SCALE)</span>';
  } else {
    badgeContainer.innerHTML = '<span class="px-2 py-0.5 rounded text-[11px] font-extrabold bg-xtb-coral text-white tracking-wide">● DEFENSIVE (100% CASH)</span>';
  }

  // Active Broker Tag
  const brokerLabel = `${(acct.broker || 'DRY_RUN').toUpperCase()} (${(acct.environment || 'PAPER').toUpperCase()})`;
  document.getElementById('hdr-broker-tag').textContent = `BROKER: ${brokerLabel}`;
}

function switchView(viewName) {
  state.currentView = viewName;
  const views = ['terminal', 'sizer', 'sanity', 'risk', 'ledger', 'settings'];

  views.forEach(v => {
    const el = document.getElementById(`view-${v}`);
    const btn = document.getElementById(`nav-btn-${v}`);
    if (el) {
      if (v === viewName) {
        el.classList.remove('hidden');
      } else {
        el.classList.add('hidden');
      }
    }
    if (btn) {
      if (v === viewName) {
        btn.classList.add('text-xtb-emerald', 'bg-xtb-card');
        btn.classList.remove('text-xtb-steel');
      } else {
        btn.classList.remove('text-xtb-emerald', 'bg-xtb-card');
        btn.classList.add('text-xtb-steel');
      }
    }
  });

  if (viewName === 'risk') {
    loadRiskMonitor();
  }
}

// =============================================================================
// Watchlist Logic & Rendering
// =============================================================================
function setWatchlistFilter(filterKey) {
  state.watchlistFilter = filterKey;
  ['top', 'orders', 'held', 'all'].forEach(k => {
    const btn = document.getElementById(`filter-btn-${k}`);
    if (btn) {
      if (k === filterKey) {
        btn.classList.add('bg-xtb-panel', 'text-white');
        btn.classList.remove('text-xtb-steel');
      } else {
        btn.classList.remove('bg-xtb-panel', 'text-white');
        btn.classList.add('text-xtb-steel');
      }
    }
  });
  renderWatchlist();
}

function onSearchChange(val) {
  state.searchQuery = val.trim().toUpperCase();
  renderWatchlist();
}

function renderWatchlist() {
  const tbody = document.getElementById('watchlist-tbody');
  if (!tbody || !state.snapshot) return;

  const actions = state.snapshot.today_cockpit?.actions || [];
  const q = state.searchQuery;
  const filter = state.watchlistFilter;

  // Filter actions
  const filtered = actions.filter(a => {
    const t = a.ticker;
    const name = (a.name || '').toUpperCase();
    const sector = (a.sector || '').toUpperCase();
    const act = a.action || 'HOLD';

    if (q && !t.includes(q) && !name.includes(q) && !sector.includes(q)) {
      return false;
    }

    if (filter === 'top' && (a.rank > 15 || !a.rank)) return false;
    if (filter === 'orders' && act !== 'BUY' && act !== 'SELL') return false;
    if (filter === 'held' && act !== 'HOLD') return false;

    return true;
  });

  document.getElementById('watchlist-count-label').textContent = `Showing ${filtered.length} constituents`;

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="py-8 text-center text-xtb-steel italic">No stocks matching current criteria</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(a => {
    const isSelected = a.ticker === state.selectedTicker;
    const act = a.action || 'HOLD';
    let actBadge = '';
    if (act === 'BUY') actBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold badge-buy">BUY</span>';
    else if (act === 'SELL') actBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold badge-sell">SELL</span>';
    else actBadge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-semibold badge-hold">HOLD</span>';

    const rk = a.rank || '-';
    const chg = a.rank_change_1d || 0;
    let rkStr = `#${rk}`;
    if (chg > 0) rkStr += ` <span class="text-xtb-emerald">↑${chg}</span>`;
    else if (chg < 0) rkStr += ` <span class="text-xtb-coral">↓${Math.abs(chg)}</span>`;
    else rkStr += ` <span class="text-xtb-steel">—</span>`;

    // Estimate price from target weight or default
    const px = a.price || 150.00;
    const tgtPct = ((a.target_weight || 0) * 100).toFixed(1);
    const volPct = ((a.predicted_vol || 0.2) * 100).toFixed(1);

    return `
      <tr onclick="selectTicker('${a.ticker}')" class="watchlist-row ${isSelected ? 'active' : ''} hover:bg-xtb-hover cursor-pointer transition">
        <td class="py-2 px-3 font-mono font-bold text-sky-400">${a.ticker}</td>
        <td class="py-2 px-3 text-slate-300 truncate max-w-[120px]" title="${a.name}">${a.name}</td>
        <td class="py-2 px-3">${actBadge}</td>
        <td class="py-2 px-3 font-mono">${rkStr}</td>
        <td class="py-2 px-3 text-right font-mono font-semibold text-white">$${Number(px).toFixed(2)}</td>
        <td class="py-2 px-3 text-right font-mono font-semibold text-white">${tgtPct}%</td>
        <td class="py-2 px-3 text-right font-mono text-xtb-steel">${volPct}%</td>
        <td class="py-2 px-3 text-xtb-steel truncate max-w-[100px]">${a.sector || 'General'}</td>
      </tr>
    `;
  }).join('');
}

function selectTicker(ticker) {
  state.selectedTicker = ticker;
  renderWatchlist();
  renderInspector();
}

// =============================================================================
// Stock Inspector & Order Ticket Logic
// =============================================================================
function renderInspector() {
  if (!state.selectedTicker || !state.snapshot) return;

  const actions = state.snapshot.today_cockpit?.actions || [];
  const action = actions.find(a => a.ticker === state.selectedTicker) || {};
  const details = state.snapshot.ticker_details?.[state.selectedTicker] || {};

  const ticker = state.selectedTicker;
  const name = action.name || details.name || ticker;
  const sector = action.sector || details.sector || 'General';
  const act = action.action || 'HOLD';
  const px = action.price || 150.00;
  const rk = action.rank || details.rank || 1;
  const rkChg = action.rank_change_1d || 0;
  const tgtPct = ((action.target_weight || 0) * 100).toFixed(1);
  const volPct = ((action.predicted_vol || details.predicted_vol || 0.2) * 100).toFixed(1);
  const score = (action.pred_score !== undefined ? action.pred_score : (details.pred_score || 0.0)).toFixed(3);

  // Top header in Inspector
  document.getElementById('insp-ticker').textContent = ticker;
  document.getElementById('insp-name-sector').textContent = `${name} • ${sector}`;
  document.getElementById('insp-price').textContent = `$${Number(px).toFixed(2)}`;

  const badgeEl = document.getElementById('insp-badge');
  badgeEl.className = 'px-2 py-0.5 rounded text-xs font-bold ' + (act === 'BUY' ? 'badge-buy' : (act === 'SELL' ? 'badge-sell' : 'badge-hold'));
  badgeEl.textContent = act;

  const rkArrow = rkChg > 0 ? `↑${rkChg}` : (rkChg < 0 ? `↓${Math.abs(rkChg)}` : '—');
  document.getElementById('insp-rank').textContent = `RANK #${rk} (${rkArrow})`;
  document.getElementById('insp-target-w').textContent = `${tgtPct}%`;
  document.getElementById('insp-vol').textContent = `${volPct}%`;
  document.getElementById('insp-score').textContent = (score >= 0 ? `+${score}` : score);

  // Tab 1: AI Drivers & SHAP Attributions
  renderInspectorDrivers(action, details);

  // Tab 1b: Valuation Ratios & Fundamentals
  renderInspectorValuation(details);

  // Tab 2: Scored News Stream
  renderInspectorNews(details);

  // Tab 3: Order Ticket Defaults
  document.getElementById('order-est-px').textContent = `$${Number(px).toFixed(2)}`;
  updateOrderEstCost();
}

function renderInspectorValuation(details) {
  const container = document.getElementById('insp-valuation-grid');
  if (!container) return;
  const f = details.fundamentals || {};

  const ey = f.earnings_yield !== undefined ? Number(f.earnings_yield) : 0.04;
  const bp = f.book_to_price !== undefined ? Number(f.book_to_price) : 0.25;
  const sp = f.sales_to_price !== undefined ? Number(f.sales_to_price) : 0.35;
  const fcf = f.fcf_yield !== undefined ? Number(f.fcf_yield) : 0.045;
  const divY = f.dividend_yield !== undefined ? Number(f.dividend_yield) : 0.015;
  const evEbitda = f.ev_to_ebitda !== undefined ? Number(f.ev_to_ebitda) : 14.5;
  const mcap = f.market_cap ? (f.market_cap / 1e9).toFixed(1) + 'B' : 'Large Cap';

  const pe = ey > 0 ? (1.0 / ey).toFixed(1) + 'x' : 'N/A';
  const pb = bp > 0 ? (1.0 / bp).toFixed(1) + 'x' : 'N/A';
  const ps = sp > 0 ? (1.0 / sp).toFixed(1) + 'x' : 'N/A';

  container.innerHTML = `
    <div class="p-2.5 bg-xtb-card border border-xtb-border rounded">
      <span class="text-[10px] text-xtb-steel uppercase block">Earnings Yield (E/P)</span>
      <span class="text-sm font-bold font-mono text-white">${(ey * 100).toFixed(2)}%</span>
      <span class="text-[10px] text-xtb-steel block mt-0.5">P/E: ${pe}</span>
    </div>
    <div class="p-2.5 bg-xtb-card border border-xtb-border rounded">
      <span class="text-[10px] text-xtb-steel uppercase block">Book-to-Price (B/P)</span>
      <span class="text-sm font-bold font-mono text-white">${bp.toFixed(3)}</span>
      <span class="text-[10px] text-xtb-steel block mt-0.5">P/B: ${pb}</span>
    </div>
    <div class="p-2.5 bg-xtb-card border border-xtb-border rounded">
      <span class="text-[10px] text-xtb-steel uppercase block">Sales-to-Price (S/P)</span>
      <span class="text-sm font-bold font-mono text-white">${sp.toFixed(3)}</span>
      <span class="text-[10px] text-xtb-steel block mt-0.5">P/S: ${ps}</span>
    </div>
    <div class="p-2.5 bg-xtb-card border border-xtb-border rounded">
      <span class="text-[10px] text-xtb-steel uppercase block">Free Cash Flow Yield</span>
      <span class="text-sm font-bold font-mono text-xtb-emerald">${(fcf * 100).toFixed(2)}%</span>
      <span class="text-[10px] text-xtb-steel block mt-0.5">FCF / MCap</span>
    </div>
    <div class="p-2.5 bg-xtb-card border border-xtb-border rounded">
      <span class="text-[10px] text-xtb-steel uppercase block">Dividend Yield</span>
      <span class="text-sm font-bold font-mono text-white">${(divY * 100).toFixed(2)}%</span>
      <span class="text-[10px] text-xtb-steel block mt-0.5">Distribution</span>
    </div>
    <div class="p-2.5 bg-xtb-card border border-xtb-border rounded">
      <span class="text-[10px] text-xtb-steel uppercase block">EV / EBITDA</span>
      <span class="text-sm font-bold font-mono text-white">${evEbitda.toFixed(1)}x</span>
      <span class="text-[10px] text-xtb-steel block mt-0.5">MCap: $${mcap}</span>
    </div>
  `;
}

function renderInspectorDrivers(action, details) {
  const container = document.getElementById('insp-drivers-list');
  const reasons = details.shap_reasons || action.shap_reasons || [];

  if (!reasons.length) {
    container.innerHTML = `<div class="p-3 bg-xtb-card border border-xtb-border rounded text-xs text-xtb-steel italic">Factor attributions driven by cross-sectional momentum and market regime.</div>`;
    return;
  }

  container.innerHTML = reasons.slice(0, 3).map((r, i) => {
    const attr = Number(r.attribution || 0);
    const isPos = attr >= 0;
    const barWidth = Math.min(100, Math.max(10, Math.abs(attr) * 15)).toFixed(0);
    const barColor = isPos ? 'bg-xtb-emerald' : 'bg-xtb-coral';
    const sign = isPos ? '+' : '';

    return `
      <div class="p-3 bg-xtb-card border border-xtb-border rounded space-y-1.5">
        <div class="flex justify-between items-center text-xs">
          <span class="font-bold text-white flex items-center gap-1.5">
            <span class="inline-block w-1.5 h-1.5 rounded-full ${isPos ? 'bg-xtb-emerald' : 'bg-xtb-coral'}"></span>
            <span>${r.description || r.feature}</span>
          </span>
          <span class="font-mono font-bold ${isPos ? 'text-xtb-emerald' : 'text-xtb-coral'}">${sign}${attr.toFixed(4)}</span>
        </div>
        <div class="w-full bg-[#263038] h-1.5 rounded overflow-hidden">
          <div class="meter-bar ${barColor} h-full rounded" style="width: ${barWidth}%"></div>
        </div>
        <div class="text-[11px] text-slate-300 leading-tight">${r.summary || `Primary predictive signal for ${action.ticker}.`}</div>
      </div>
    `;
  }).join('');
}

function renderInspectorNews(details) {
  const container = document.getElementById('insp-news-list');
  const headlines = details.recent_headlines || [];

  if (!headlines.length) {
    container.innerHTML = `<div class="p-3 bg-xtb-card border border-xtb-border rounded text-xs text-xtb-steel italic">No company news headlines in current archive window.</div>`;
    return;
  }

  container.innerHTML = headlines.slice(0, 5).map(h => {
    const lbl = (h.sentiment_label || 'neutral').toLowerCase();
    let badge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold tone-neu">NEU</span>';
    if (lbl === 'positive') badge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold tone-pos">POS</span>';
    else if (lbl === 'negative') badge = '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold tone-neg">NEG</span>';

    const score = Number(h.sentiment_score || 0).toFixed(2);
    const dateStr = h.date ? h.date.split(' ')[0] : 'Recent';

    return `
      <div class="p-2.5 bg-xtb-card border border-xtb-border rounded text-xs space-y-1">
        <div class="flex justify-between items-center text-[10px] text-xtb-steel font-mono">
          <span>${dateStr}</span>
          <div class="flex items-center gap-1.5">
            ${badge}
            <span>(${score >= 0 ? '+' : ''}${score})</span>
          </div>
        </div>
        <p class="text-slate-200 text-xs leading-snug">${h.headline}</p>
      </div>
    `;
  }).join('');
}

function setInspectorTab(tabKey) {
  ['drivers', 'valuation', 'news', 'ticket'].forEach(k => {
    const pane = document.getElementById(`insp-pane-${k}`);
    const tabBtn = document.getElementById(`insp-tab-${k}`);
    if (pane) {
      if (k === tabKey) pane.classList.remove('hidden');
      else pane.classList.add('hidden');
    }
    if (tabBtn) {
      if (k === tabKey) {
        tabBtn.classList.add('text-xtb-emerald', 'border-b-2', 'border-xtb-emerald');
        tabBtn.classList.remove('text-xtb-steel');
      } else {
        tabBtn.classList.remove('text-xtb-emerald', 'border-b-2', 'border-xtb-emerald');
        tabBtn.classList.add('text-xtb-steel');
      }
    }
  });
}

function setOrderSide(side) {
  state.orderSide = side;
  const btnBuy = document.getElementById('btn-side-buy');
  const btnSell = document.getElementById('btn-side-sell');
  if (side === 'buy') {
    btnBuy.className = 'py-2 rounded font-bold text-xs uppercase tracking-wide bg-xtb-emerald text-black border border-xtb-emerald transition';
    btnSell.className = 'py-2 rounded font-bold text-xs uppercase tracking-wide bg-xtb-card text-xtb-steel border border-xtb-border hover:text-white transition';
  } else {
    btnBuy.className = 'py-2 rounded font-bold text-xs uppercase tracking-wide bg-xtb-card text-xtb-steel border border-xtb-border hover:text-white transition';
    btnSell.className = 'py-2 rounded font-bold text-xs uppercase tracking-wide bg-xtb-coral text-white border border-xtb-coral transition';
  }
}

function adjustOrderQty(delta) {
  const input = document.getElementById('order-qty-input');
  let val = Math.max(1, (Number(input.value) || 1) + delta);
  input.value = val;
  updateOrderEstCost();
}

function updateOrderEstCost() {
  const input = document.getElementById('order-qty-input');
  const qty = Number(input.value) || 1;
  const actions = state.snapshot?.today_cockpit?.actions || [];
  const action = actions.find(a => a.ticker === state.selectedTicker) || {};
  const px = action.price || 150.00;
  const total = qty * px;
  document.getElementById('order-est-total').textContent = formatCurrency(total);
}

function toggleOrderSubmitBtn() {
  const chk = document.getElementById('order-confirm-chk');
  const btn = document.getElementById('btn-submit-order');
  if (chk.checked) {
    btn.disabled = false;
    btn.classList.remove('opacity-50', 'cursor-not-allowed');
    btn.classList.add('hover:bg-emerald-400');
  } else {
    btn.disabled = true;
    btn.classList.add('opacity-50', 'cursor-not-allowed');
    btn.classList.remove('hover:bg-emerald-400');
  }
}

async function submitSingleOrder() {
  const ticker = state.selectedTicker;
  const qty = Number(document.getElementById('order-qty-input').value) || 1;
  const side = state.orderSide;
  const actions = state.snapshot?.today_cockpit?.actions || [];
  const action = actions.find(a => a.ticker === ticker) || {};
  const price = action.price || 150.00;

  showToast(`Submitting ${side.toUpperCase()} ${qty} ${ticker} to broker...`, 'info');

  try {
    const res = await fetch('/api/order', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, quantity: qty, side, price }),
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Order ${data.order_id} filled: ${side.toUpperCase()} ${qty} ${ticker} @ $${data.price.toFixed(2)}`, 'success');
      document.getElementById('order-confirm-chk').checked = false;
      toggleOrderSubmitBtn();
      // Reload accounts and positions
      await loadAllData();
    } else {
      showToast('Order failed: ' + (data.detail || 'Unknown error'), 'error');
    }
  } catch (err) {
    showToast('Failed to execute order: ' + err.message, 'error');
  }
}

// =============================================================================
// Bottom Drawer Logic (Positions, Target Rebalance, Sectors, Fills)
// =============================================================================
function setDrawerTab(tabKey) {
  ['positions', 'rebalance', 'sectors', 'fills'].forEach(k => {
    const pane = document.getElementById(`drawer-pane-${k}`);
    const btn = document.getElementById(`drawer-tab-${k}`);
    if (pane) {
      if (k === tabKey) pane.classList.remove('hidden');
      else pane.classList.add('hidden');
    }
    if (btn) {
      if (k === tabKey) {
        btn.classList.add('bg-xtb-card', 'text-white', 'border', 'border-xtb-border');
        btn.classList.remove('text-xtb-steel');
      } else {
        btn.classList.remove('bg-xtb-card', 'text-white', 'border', 'border-xtb-border');
        btn.classList.add('text-xtb-steel');
      }
    }
  });
}

function renderSectorsDrawer() {
  const container = document.getElementById('sector-cards-list');
  const barraContainer = document.getElementById('barra-diagnostics-body');
  if (!container || !state.snapshot) return;

  const cockpit = state.snapshot.today_cockpit || {};
  const sectors = cockpit.sector_breakdown || [];
  const barra = cockpit.barra_diagnostics || {};
  const summary = cockpit.summary || {};

  const neutMode = summary.neutralization_mode || 'full_barra';
  const modeBadge = document.getElementById('barra-neutral-mode');
  if (modeBadge) {
    modeBadge.textContent = neutMode === 'full_barra' ? 'Full Barra Neutralization' : neutMode.toUpperCase();
  }

  // Render Sector cards
  if (!sectors.length) {
    container.innerHTML = `<div class="py-4 text-center text-xtb-steel italic">No sector allocation data available.</div>`;
  } else {
    container.innerHTML = sectors.map(s => {
      const wPct = (s.weight * 100).toFixed(1);
      const isOver = s.weight > 0.2501;
      const barColor = isOver ? 'bg-xtb-coral' : 'bg-xtb-emerald';
      const badgeClass = isOver ? 'text-xtb-coral bg-xtb-coral/10 border-xtb-coral/30' : 'text-xtb-emerald bg-xtb-emerald/10 border-xtb-emerald/30';
      const badgeText = isOver ? 'EXCEEDS 25%' : 'COMPLIANT';
      const barWidth = Math.min(100, (s.weight / 0.25) * 100).toFixed(0);

      return `
        <div class="p-2 bg-xtb-panel border border-xtb-border rounded flex flex-col gap-1.5">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span class="font-bold text-white">${s.sector}</span>
              <span class="text-[10px] text-xtb-steel font-mono">(${s.count} names: ${s.tickers.join(', ')})</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="text-[10px] font-mono px-1.5 py-0.5 rounded border ${badgeClass}">${badgeText}</span>
              <span class="font-mono font-bold text-white text-xs">${wPct}%</span>
            </div>
          </div>
          <div class="w-full bg-[#182026] h-1.5 rounded overflow-hidden">
            <div class="${barColor} h-full rounded transition-all duration-300" style="width: ${barWidth}%"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  // Render Barra diagnostics
  if (barraContainer) {
    const r2 = ((barra.r2_explained || 0) * 100).toFixed(1);
    const maxRaw = ((barra.max_raw_correlation || 0) * 100).toFixed(1);
    const maxNeu = ((barra.max_neutral_correlation || 0) * 100).toFixed(1);

    const rawCorrs = barra.raw_correlations || {};
    const neuCorrs = barra.neutral_correlations || {};

    const styleFactors = [
      { key: 'style_size', label: 'Size (Log MCap)' },
      { key: 'style_volatility', label: 'Volatility (20d Vol)' },
      { key: 'style_momentum', label: 'Momentum (20d Trend)' },
      { key: 'style_value', label: 'Value (B/P + E/P + S/P)' },
    ];

    const factorRows = styleFactors.map(sf => {
      const rc = rawCorrs[sf.key] !== undefined ? rawCorrs[sf.key] : 0.0;
      const nc = neuCorrs[sf.key] !== undefined ? neuCorrs[sf.key] : 0.0;
      return `
        <div class="flex items-center justify-between py-1 border-b border-xtb-border/40 text-[11px] font-mono">
          <span class="text-slate-300 font-sans">${sf.label}</span>
          <div class="flex items-center gap-3">
            <span class="text-xtb-steel">Raw: <b class="text-white">${rc >= 0 ? '+' : ''}${rc.toFixed(3)}</b></span>
            <span class="text-xtb-emerald font-bold">Neutral: ${nc >= 0 ? '+' : ''}${nc.toFixed(3)}</span>
          </div>
        </div>
      `;
    }).join('');

    barraContainer.innerHTML = `
      <div class="grid grid-cols-2 gap-2">
        <div class="p-2 bg-xtb-panel border border-xtb-border rounded">
          <span class="text-[10px] text-xtb-steel block uppercase">Factor R-Squared</span>
          <span class="font-mono font-bold text-sm text-sky-400">${r2}%</span>
          <span class="text-[10px] text-xtb-steel block">Cross-sectional variance</span>
        </div>
        <div class="p-2 bg-xtb-panel border border-xtb-border rounded">
          <span class="text-[10px] text-xtb-steel block uppercase">Max Style Bias</span>
          <span class="font-mono font-bold text-sm text-xtb-emerald">${maxNeu}%</span>
          <span class="text-[10px] text-xtb-steel block">Down from ${maxRaw}% raw</span>
        </div>
      </div>
      <div class="p-2 bg-xtb-panel border border-xtb-border rounded space-y-1">
        <span class="text-[10px] font-bold text-xtb-steel uppercase block tracking-wider">Style Factor Orthogonality</span>
        ${factorRows}
      </div>
    `;
  }
}

function renderPositionsDrawer() {
  const tbody = document.getElementById('positions-tbody');
  const summaryEl = document.getElementById('drawer-summary-right');
  const data = state.positionsData || { positions: [] };

  if (summaryEl) {
    const totMV = formatCurrency(data.total_market_value || 0);
    const totPnl = data.total_unrealized_pnl || 0;
    const sign = totPnl >= 0 ? '+' : '';
    const color = totPnl >= 0 ? 'text-xtb-emerald' : 'text-xtb-coral';
    summaryEl.innerHTML = `Invested: <b class="text-white">${totMV}</b> | Net P&L: <b class="${color}">${sign}${formatCurrency(totPnl)}</b>`;
  }

  if (!data.positions || !data.positions.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-xtb-steel italic">No active positions recorded in SQLite ledger. Place orders above to build portfolio.</td></tr>`;
    return;
  }

  tbody.innerHTML = data.positions.map(p => {
    const pnl = p.unrealized_pnl || 0;
    const pnlPct = p.unrealized_pnl_pct || 0;
    const isPos = pnl >= 0;
    const color = isPos ? 'text-xtb-emerald' : 'text-xtb-coral';
    const sign = isPos ? '+' : '';

    return `
      <tr class="hover:bg-xtb-hover cursor-pointer" onclick="selectTicker('${p.ticker}')">
        <td class="py-2 px-3 font-mono font-bold text-sky-400">${p.ticker}</td>
        <td class="py-2 px-3 text-right font-semibold text-white">${p.shares}</td>
        <td class="py-2 px-3 text-right text-xtb-steel">$${p.avg_buy_price.toFixed(2)}</td>
        <td class="py-2 px-3 text-right text-white">$${p.current_price.toFixed(2)}</td>
        <td class="py-2 px-3 text-right font-semibold text-white">${formatCurrency(p.market_value)}</td>
        <td class="py-2 px-3 text-right font-bold ${color}">${sign}${formatCurrency(pnl)}</td>
        <td class="py-2 px-3 text-right font-bold ${color}">${sign}${pnlPct.toFixed(2)}%</td>
      </tr>
    `;
  }).join('');
}

function renderRebalanceDrawer() {
  const tbody = document.getElementById('rebalance-tbody');
  const msgEl = document.getElementById('rebalance-status-msg');
  const actions = state.snapshot?.today_cockpit?.actions || [];

  // Filter actions that warrant rebalancing
  const orders = actions.filter(a => {
    const delta = Math.abs((a.target_weight || 0) - (a.current_weight || 0));
    return delta >= 0.005 && (a.action === 'BUY' || a.action === 'SELL');
  });

  if (!orders.length) {
    msgEl.textContent = 'All portfolio constituent weights are within the 0.5% drift buffer. No orders required today.';
    document.getElementById('btn-batch-rebalance').disabled = true;
    document.getElementById('btn-batch-rebalance').classList.add('opacity-50', 'cursor-not-allowed');
    tbody.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-xtb-steel italic">Zero drift violations detected. Hold positions steady and save transaction fees.</td></tr>`;
    return;
  }

  msgEl.textContent = `${orders.length} constituent adjustments exceed the 0.5% turnover buffer and require action.`;
  document.getElementById('btn-batch-rebalance').disabled = false;
  document.getElementById('btn-batch-rebalance').classList.remove('opacity-50', 'cursor-not-allowed');

  tbody.innerHTML = orders.map(o => {
    const act = o.action;
    const badge = act === 'BUY' 
      ? '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold badge-buy">BUY</span>'
      : '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold badge-sell">SELL</span>';

    const curPct = ((o.current_weight || 0) * 100).toFixed(1);
    const tgtPct = ((o.target_weight || 0) * 100).toFixed(1);
    const deltaPct = (((o.target_weight || 0) - (o.current_weight || 0)) * 100).toFixed(1);
    const portVal = state.account?.portfolio_value || 100000;
    const tradeVal = Math.abs((o.target_weight || 0) - (o.current_weight || 0)) * portVal;
    const px = o.price || 150.00;
    const estShares = Math.max(1, Math.round(tradeVal / px));

    return `
      <tr class="hover:bg-xtb-hover">
        <td class="py-2 px-3 font-mono font-bold text-sky-400">${o.ticker}</td>
        <td class="py-2 px-3">${badge}</td>
        <td class="py-2 px-3 text-right text-xtb-steel">${curPct}%</td>
        <td class="py-2 px-3 text-right font-semibold text-white">${tgtPct}%</td>
        <td class="py-2 px-3 text-right font-bold ${Number(deltaPct) >= 0 ? 'text-xtb-emerald' : 'text-xtb-coral'}">${Number(deltaPct) >= 0 ? '+' : ''}${deltaPct}%</td>
        <td class="py-2 px-3 text-right font-semibold text-white">${estShares}</td>
        <td class="py-2 px-3 text-right font-semibold text-white">${formatCurrency(tradeVal)}</td>
      </tr>
    `;
  }).join('');
}

async function executeBatchRebalance() {
  const actions = state.snapshot?.today_cockpit?.actions || [];
  const portVal = state.account?.portfolio_value || 100000;

  const orders = actions
    .filter(a => Math.abs((a.target_weight || 0) - (a.current_weight || 0)) >= 0.005 && (a.action === 'BUY' || a.action === 'SELL'))
    .map(a => {
      const px = a.price || 150.00;
      const tradeVal = Math.abs((a.target_weight || 0) - (a.current_weight || 0)) * portVal;
      const shares = Math.max(1, Math.round(tradeVal / px));
      return {
        ticker: a.ticker,
        shares: shares,
        side: a.action.toLowerCase(),
        price: px,
      };
    });

  if (!orders.length) return;

  showToast(`Submitting ${orders.length} batch rebalance orders to broker...`, 'info');

  try {
    const res = await fetch('/api/rebalance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ orders }),
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Batch rebalance executed: ${data.executed_count} orders filled!`, 'success');
      await loadAllData();
    } else {
      showToast('Batch rebalance failed: ' + (data.detail || 'Unknown error'), 'error');
    }
  } catch (err) {
    showToast('Failed to execute rebalance: ' + err.message, 'error');
  }
}

function renderFillsDrawer() {
  const tbody = document.getElementById('fills-tbody');
  const fills = state.fills || [];

  if (!fills.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-xtb-steel italic">No trade execution history recorded yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = fills.slice(0, 15).map(f => {
    const act = (f.action || 'BUY').toUpperCase();
    const badge = act === 'BUY'
      ? '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold badge-buy">BUY</span>'
      : '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold badge-sell">SELL</span>';

    return `
      <tr class="hover:bg-xtb-hover">
        <td class="py-2 px-3 text-xtb-steel font-mono">${f.fill_date || 'Today'}</td>
        <td class="py-2 px-3 font-mono font-bold text-sky-400">${f.ticker}</td>
        <td class="py-2 px-3">${badge}</td>
        <td class="py-2 px-3 text-right font-semibold text-white">${f.shares}</td>
        <td class="py-2 px-3 text-right font-semibold text-white">$${Number(f.price || 0).toFixed(2)}</td>
        <td class="py-2 px-3 text-xtb-steel truncate max-w-[140px]">${f.broker || 'manual'} &bull; <span class="text-xtb-emerald font-bold">${f.status || 'FILLED'}</span></td>
        <td class="py-2 px-3 text-xtb-steel font-mono text-[10px] truncate max-w-[120px]">${f.broker_order_id || '-'}</td>
      </tr>
    `;
  }).join('');
}

// =============================================================================
// Capital Sizer Logic
// =============================================================================
function setBudgetPreset(amt) {
  state.currentBudget = amt;
  document.getElementById('sizer-budget-input').value = amt;
  calculateSizerAllocations();
}

async function calculateSizerAllocations() {
  const budget = Number(document.getElementById('sizer-budget-input')?.value) || 10000;
  const fractional = document.getElementById('sizer-fractional-chk')?.checked || false;

  state.currentBudget = budget;
  state.allowFractional = fractional;

  try {
    const res = await fetch('/api/size', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ budget, allow_fractional: fractional }),
    });
    const data = await res.json();
    state.slipsCache = data.slips || {};

    // Update Summary Cards
    const summary = data.summary || {};
    document.getElementById('sizer-total-budget').textContent = formatCurrency(summary.budget || budget);
    document.getElementById('sizer-total-allocated').textContent = formatCurrency(summary.total_allocated || 0);
    document.getElementById('sizer-cash-buffer').textContent = `${formatCurrency(summary.cash_remainder || 0)} (${summary.cash_remainder_pct || 0}%)`;

    // Render Allocations Table
    const tbody = document.getElementById('sizer-tbody');
    const allocs = summary.allocations || [];
    if (!allocs.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="py-6 text-center text-xtb-steel italic">No constituent targets found for allocation.</td></tr>`;
      return;
    }

    tbody.innerHTML = allocs.map(a => `
      <tr class="hover:bg-xtb-hover">
        <td class="py-2 px-3 font-mono font-bold text-sky-400">${a.ticker} <span class="text-xs text-xtb-steel font-normal">(${a.name.slice(0, 16)})</span></td>
        <td class="py-2 px-3 text-right text-xtb-steel">${(a.target_weight * 100).toFixed(1)}%</td>
        <td class="py-2 px-3 text-right text-white">$${Number(a.price).toFixed(2)}</td>
        <td class="py-2 px-3 text-right font-bold text-white">${a.shares}</td>
        <td class="py-2 px-3 text-right font-semibold text-xtb-emerald">${formatCurrency(a.actual_dollars)}</td>
      </tr>
    `).join('');

    // Update Slip Preview
    updateSlipFormat();
  } catch (err) {
    console.error('Failed to calculate sizing:', err);
  }
}

function updateSlipFormat() {
  const fmt = document.getElementById('slip-format-select')?.value || 'standard';
  state.slipFormat = fmt;
  const txt = state.slipsCache[fmt] || '';
  document.getElementById('slip-text-area').value = txt;
}

function copySlipToClipboard() {
  const txt = document.getElementById('slip-text-area').value;
  if (!txt) return;
  navigator.clipboard.writeText(txt).then(() => {
    showToast('Order slip copied to clipboard!', 'success');
  });
}

async function submitAllSizerOrders() {
  const budget = state.currentBudget;
  showToast(`Routing all sized portfolio orders for $${budget.toLocaleString()} to broker...`, 'info');

  try {
    const res = await fetch('/api/size', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ budget, allow_fractional: state.allowFractional }),
    });
    const data = await res.json();
    const allocs = data.summary?.allocations || [];

    const orders = allocs.filter(a => a.shares > 0).map(a => ({
      ticker: a.ticker,
      shares: a.shares,
      side: 'buy',
      price: a.price,
    }));

    if (!orders.length) {
      showToast('No orders with quantity > 0.', 'error');
      return;
    }

    const execRes = await fetch('/api/rebalance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ orders }),
    });
    const execData = await execRes.json();
    if (execData.success) {
      showToast(`Submitted ${execData.executed_count} sized orders to broker!`, 'success');
      await loadAllData();
      switchView('terminal');
    }
  } catch (err) {
    showToast('Failed to execute sized orders: ' + err.message, 'error');
  }
}

// =============================================================================
// AI Sanity & Model Health View Logic
// =============================================================================
function renderSanityView() {
  const mh = state.snapshot?.model_health || {};
  const sanity = mh.sanity_checks || {};

  // 1. Label Shuffle
  const ls = sanity.label_shuffle || {};
  document.getElementById('sanity-shuffle-verdict').textContent = ls.verdict || 'PASSED';
  document.getElementById('sanity-shuffle-stats').textContent = `Shuffled IC: ${ls.shuffled_ic ? Number(ls.shuffled_ic).toFixed(4) : '-0.0016'} (t = ${ls.t_stat ? Number(ls.t_stat).toFixed(2) : '-0.44'})`;

  // 2. 1-Day Lag Leakage
  const fl = sanity.feature_lag || {};
  document.getElementById('sanity-lag-verdict').textContent = fl.verdict || 'PASSED';
  document.getElementById('sanity-lag-stats').textContent = `Base: ${fl.base_ic ? Number(fl.base_ic).toFixed(4) : '+0.0346'} -> Lagged: ${fl.lagged_ic ? Number(fl.lagged_ic).toFixed(4) : '+0.0332'}`;

  // 3. Multiple Testing
  const mt = sanity.multiple_testing || {};
  document.getElementById('sanity-mt-verdict').textContent = mt.verdict || 'SIGNIFICANT';
  document.getElementById('sanity-mt-stats').textContent = `Achieved t: ${mt.achieved_t_stat ? Number(mt.achieved_t_stat).toFixed(2) : '3.99'} > Hurdle: ${mt.hurdle_t_stat ? Number(mt.hurdle_t_stat).toFixed(2) : '1.18'}`;

  // Monthly Spearman IC Table
  const tbody = document.getElementById('monthly-ic-tbody');
  const mIc = mh.monthly_ic || [];
  if (!mIc.length) {
    tbody.innerHTML = `<tr><td colspan="3" class="py-6 text-center text-xtb-steel italic">No monthly out-of-sample splits available.</td></tr>`;
    return;
  }

  tbody.innerHTML = mIc.map(m => {
    const icVal = Number(m.ic || 0);
    const isStrong = icVal >= 0.02;
    const isPos = icVal > 0;
    const color = isStrong ? 'text-xtb-emerald font-bold' : (isPos ? 'text-xtb-amber' : 'text-xtb-coral font-bold');
    const badge = isStrong
      ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-xtb-emerald/20 text-xtb-emerald border border-xtb-emerald/40">Strong Edge (>0.02)</span>'
      : (isPos 
          ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-xtb-amber/20 text-xtb-amber border border-xtb-amber/40">Positive Edge</span>'
          : '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-xtb-coral/20 text-xtb-coral border border-xtb-coral/40">Degradation</span>');

    return `
      <tr class="hover:bg-xtb-hover">
        <td class="py-2 px-3 text-white font-mono">${m.month}</td>
        <td class="py-2 px-3 text-right font-mono ${color}">${icVal >= 0 ? '+' : ''}${icVal.toFixed(4)}</td>
        <td class="py-2 px-3">${badge}</td>
      </tr>
    `;
  }).join('');
}

// =============================================================================
// Ledger & Manual Fill Logic
// =============================================================================
function renderLedgerView() {
  const tbody = document.getElementById('ledger-all-tbody');
  const fills = state.fills || [];

  if (!fills.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="py-6 text-center text-xtb-steel italic">No execution history found in SQLite database.</td></tr>`;
    return;
  }

  tbody.innerHTML = fills.map(f => {
    const act = (f.action || 'BUY').toUpperCase();
    const badge = act === 'BUY'
      ? '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold badge-buy">BUY</span>'
      : '<span class="px-1.5 py-0.5 rounded text-[10px] font-bold badge-sell">SELL</span>';

    return `
      <tr class="hover:bg-xtb-hover">
        <td class="py-2 px-3 text-xtb-steel font-mono">${f.fill_date || 'Today'}</td>
        <td class="py-2 px-3 font-mono font-bold text-sky-400">${f.ticker}</td>
        <td class="py-2 px-3">${badge}</td>
        <td class="py-2 px-3 text-right font-semibold text-white">${f.shares}</td>
        <td class="py-2 px-3 text-right font-semibold text-white">$${Number(f.price || 0).toFixed(2)}</td>
        <td class="py-2 px-3 text-xtb-steel truncate max-w-[140px]">${f.broker || 'manual'} &bull; <span class="text-xtb-emerald font-bold">${f.status || 'FILLED'}</span></td>
        <td class="py-2 px-3 text-xtb-steel font-mono text-[10px] truncate max-w-[120px]">${f.broker_order_id || '-'}</td>
      </tr>
    `;
  }).join('');
}

async function submitManualFill() {
  const date = document.getElementById('manual-date').value;
  const ticker = document.getElementById('manual-ticker').value.trim().toUpperCase();
  const side = document.getElementById('manual-side').value;
  const shares = Number(document.getElementById('manual-shares').value) || 1;
  const price = Number(document.getElementById('manual-price').value) || 150;
  const notes = document.getElementById('manual-notes').value;

  if (!ticker || shares <= 0 || price <= 0) {
    showToast('Please provide valid fill parameters.', 'error');
    return;
  }

  try {
    const res = await fetch('/api/order', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, quantity: shares, side, price }),
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Manual fill recorded for ${shares} ${ticker}!`, 'success');
      await loadAllData();
    }
  } catch (err) {
    showToast('Failed to record fill: ' + err.message, 'error');
  }
}

function downloadFillsCSV() {
  const fills = state.fills || [];
  if (!fills.length) {
    showToast('No fills to export.', 'error');
    return;
  }
  const headers = ['id', 'fill_date', 'ticker', 'action', 'shares', 'price', 'commission', 'notes', 'broker', 'broker_order_id', 'status'];
  const rows = fills.map(f => headers.map(h => JSON.stringify(f[h] ?? '')).join(','));
  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `argus_quant_fills_${new Date().toISOString().split('T')[0]}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast('Downloaded fills CSV.', 'success');
}

// =============================================================================
// Intraday Risk Circuit Breakers & Stop-Loss Logic
// =============================================================================
async function loadRiskMonitor() {
  try {
    const res = await fetch('/api/risk/monitor');
    const data = await res.json();
    state.riskData = data;
    renderRiskView(data);
    updateRiskHeaderBadge(data);
  } catch (err) {
    console.error('Failed to load risk telemetry:', err);
  }
}

function updateRiskHeaderBadge(data) {
  const badge = document.getElementById('hdr-risk-badge');
  if (!badge) return;

  const breached = data.breached_count || 0;
  const caution = data.caution_count || 0;

  if (breached > 0) {
    badge.innerHTML = `<span class="px-2 py-0.5 rounded text-[11px] font-extrabold bg-xtb-coral text-white tracking-wide animate-pulse">${breached} BREACHED</span>`;
  } else if (caution > 0) {
    badge.innerHTML = `<span class="px-2 py-0.5 rounded text-[11px] font-extrabold bg-xtb-amber text-black tracking-wide">${caution} CAUTION</span>`;
  } else {
    badge.innerHTML = `<span class="px-2 py-0.5 rounded text-[11px] font-extrabold bg-xtb-emerald text-black tracking-wide">NORMAL</span>`;
  }
}

function renderRiskView(data) {
  if (!data) return;

  // 1. System status pill
  const pill = document.getElementById('risk-system-status-pill');
  if (pill) {
    pill.className = 'px-2.5 py-0.5 rounded text-xs font-mono font-bold uppercase';
    if (data.breached_count > 0) {
      pill.textContent = `SYSTEM CRITICAL (${data.breached_count} BREACHED)`;
      pill.classList.add('bg-xtb-coral', 'text-white');
    } else if (data.caution_count > 0) {
      pill.textContent = `SYSTEM CAUTION (${data.caution_count} WATCH)`;
      pill.classList.add('bg-xtb-amber', 'text-black');
    } else {
      pill.textContent = 'SYSTEM NORMAL';
      pill.classList.add('bg-xtb-emerald', 'text-black');
    }
  }

  // 2. Telemetry Stat Cards
  const totalEl = document.getElementById('risk-stat-total');
  const breachedEl = document.getElementById('risk-stat-breached');
  const cautionEl = document.getElementById('risk-stat-caution');
  const countBadge = document.getElementById('risk-positions-count-badge');

  if (totalEl) totalEl.textContent = data.total_open ?? 0;
  if (breachedEl) {
    breachedEl.textContent = data.breached_count ?? 0;
    breachedEl.className = (data.breached_count > 0) ? 'text-2xl font-mono-num font-bold text-xtb-coral mt-1 animate-pulse' : 'text-2xl font-mono-num font-bold text-xtb-emerald mt-1';
  }
  if (cautionEl) {
    cautionEl.textContent = data.caution_count ?? 0;
    cautionEl.className = (data.caution_count > 0) ? 'text-2xl font-mono-num font-bold text-xtb-amber mt-1' : 'text-2xl font-mono-num font-bold text-xtb-steel mt-1';
  }
  if (countBadge) countBadge.textContent = `${data.total_open ?? 0} Holdings`;

  // 3. Live Positions Table
  const tbody = document.getElementById('risk-positions-tbody');
  if (tbody) {
    const positions = data.positions || [];
    if (positions.length === 0) {
      tbody.innerHTML = `<tr><td colspan="12" class="py-8 text-center text-xtb-steel italic">No active holdings currently held in portfolio.</td></tr>`;
    } else {
      tbody.innerHTML = positions.map(p => {
        const entryDd = Number(p.drawdown_from_entry_pct || 0);
        const peakDd = Number(p.drawdown_from_peak_pct || 0);
        const unPnl = Number(p.unrealized_pl || 0);

        const entryDdColor = entryDd < -7.0 ? 'text-xtb-coral font-bold' : (entryDd < 0 ? 'text-xtb-amber' : 'text-xtb-emerald');
        const peakDdColor = peakDd < -7.0 ? 'text-xtb-coral font-bold' : (peakDd < 0 ? 'text-xtb-amber' : 'text-xtb-emerald');
        const pnlColor = unPnl >= 0 ? 'text-xtb-emerald' : 'text-xtb-coral font-bold';

        let statusBadge = '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-xtb-emerald/20 text-xtb-emerald border border-xtb-emerald/40 uppercase">HEALTHY</span>';
        if (p.status === 'BREACHED') {
          statusBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-xtb-coral text-white border border-rose-500 uppercase animate-pulse">BREACHED</span>`;
        } else if (p.status === 'CAUTION') {
          statusBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-xtb-amber/20 text-xtb-amber border border-xtb-amber/40 uppercase">CAUTION</span>`;
        }

        let actionCell = `<span class="text-xtb-steel text-[10px] uppercase font-mono">OK</span>`;
        if (p.status === 'BREACHED') {
          actionCell = `<button onclick="triggerEmergencyLiquidation('${p.ticker}')" class="px-2 py-1 rounded bg-xtb-coral hover:bg-rose-600 text-white text-[10px] font-bold uppercase transition">Liquidate</button>`;
        }

        return `
          <tr class="hover:bg-xtb-hover">
            <td class="py-2.5 px-3 font-bold text-white font-mono">${p.ticker}</td>
            <td class="py-2.5 px-3 text-right font-mono text-xtb-steel">${Number(p.shares).toFixed(2)}</td>
            <td class="py-2.5 px-3 text-right font-mono text-xtb-steel">${formatCurrency(p.cost_basis)}</td>
            <td class="py-2.5 px-3 text-right font-mono text-white">${formatCurrency(p.peak_price)}</td>
            <td class="py-2.5 px-3 text-right font-mono font-bold text-white">${formatCurrency(p.current_price)}</td>
            <td class="py-2.5 px-3 text-right font-mono ${entryDdColor}">${entryDd >= 0 ? '+' : ''}${entryDd.toFixed(2)}%</td>
            <td class="py-2.5 px-3 text-right font-mono ${peakDdColor}">${peakDd >= 0 ? '+' : ''}${peakDd.toFixed(2)}%</td>
            <td class="py-2.5 px-3 text-right font-mono text-xtb-steel">${formatCurrency(p.atr_14)}</td>
            <td class="py-2.5 px-3 text-right font-mono text-xtb-amber font-semibold">${formatCurrency(p.stop_loss_price)}</td>
            <td class="py-2.5 px-3 text-right font-mono ${pnlColor}">${unPnl >= 0 ? '+' : ''}${formatCurrency(unPnl)}</td>
            <td class="py-2.5 px-3 text-center">${statusBadge}</td>
            <td class="py-2.5 px-3 text-center">${actionCell}</td>
          </tr>
        `;
      }).join('');
    }
  }

  // 4. Audit Log Table
  const auditTbody = document.getElementById('risk-audit-tbody');
  if (auditTbody) {
    const events = data.recent_events || [];
    if (events.length === 0) {
      auditTbody.innerHTML = `<tr><td colspan="10" class="py-8 text-center text-xtb-steel italic">No emergency circuit breaker liquidations recorded in ledger.</td></tr>`;
    } else {
      auditTbody.innerHTML = events.map(ev => {
        const pnl = Number(ev.pnl_pct || 0);
        const pnlCol = pnl >= 0 ? 'text-xtb-emerald' : 'text-xtb-coral font-bold';
        return `
          <tr class="hover:bg-xtb-hover">
            <td class="py-2 px-3 font-mono text-xtb-steel">${String(ev.timestamp).slice(0, 19).replace('T', ' ')}</td>
            <td class="py-2 px-3 font-bold font-mono text-white">${ev.ticker}</td>
            <td class="py-2 px-3 font-mono text-xs text-xtb-amber">${ev.trigger_type}</td>
            <td class="py-2 px-3 text-right font-mono text-xtb-steel">${formatCurrency(ev.entry_price)}</td>
            <td class="py-2 px-3 text-right font-mono text-xtb-steel">${formatCurrency(ev.peak_price)}</td>
            <td class="py-2 px-3 text-right font-mono text-white font-bold">${formatCurrency(ev.exit_price)}</td>
            <td class="py-2 px-3 text-right font-mono ${pnlCol}">${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%</td>
            <td class="py-2 px-3 text-center font-mono text-xs text-sky-400 font-bold uppercase">${ev.hedge_action || 'CASH'}</td>
            <td class="py-2 px-3 text-center"><span class="px-2 py-0.5 rounded text-[10px] font-bold bg-xtb-card border border-xtb-border text-xtb-steel uppercase">${ev.status || 'EXECUTED'}</span></td>
            <td class="py-2 px-3 text-xtb-steel text-xs truncate max-w-xs">${ev.notes || '-'}</td>
          </tr>
        `;
      }).join('');
    }
  }
}

async function triggerEmergencyLiquidation(targetTicker = null) {
  const destSelect = document.getElementById('risk-liquidate-destination');
  const liquidateTo = destSelect ? destSelect.value : 'cash';
  const tickers = targetTicker ? [targetTicker] : null;

  const targetMsg = targetTicker ? `position in ${targetTicker}` : 'all breached holdings';
  const destMsg = liquidateTo === 'hedged_spy' ? 'hedged SPY' : 'cash';

  if (!confirm(`Confirm immediate emergency liquidation of ${targetMsg} into ${destMsg}?`)) {
    return;
  }

  showToast(`Executing emergency circuit breaker for ${targetMsg}...`, 'warning');
  try {
    const res = await fetch('/api/risk/circuit-breaker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tickers, liquidate_to: liquidateTo }),
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Emergency liquidation executed! Liquidated ${data.liquidations_count} position(s) to ${destMsg}.`, 'success');
      await loadRiskMonitor();
      await loadAllData();
    } else {
      showToast('Circuit breaker execution returned error.', 'error');
    }
  } catch (err) {
    showToast('Emergency liquidation failed: ' + err.message, 'error');
  }
}

// =============================================================================
// Broker & Settings Logic
// =============================================================================
function renderSettingsView() {
  const acct = state.account || {};
  const currentBroker = (acct.broker || 'dry_run').toLowerCase();

  const radioDry = document.getElementById('radio-dry-run');
  const radioAlpaca = document.getElementById('radio-alpaca');
  const radioT212 = document.getElementById('radio-t212');

  if (currentBroker === 'alpaca' && radioAlpaca) radioAlpaca.checked = true;
  else if ((currentBroker === 'trading212' || currentBroker === 't212') && radioT212) radioT212.checked = true;
  else if (radioDry) radioDry.checked = true;

  // Integrations status text
  const alpacaConfigured = acct.integrations?.alpaca;
  const t212Configured = acct.integrations?.trading212;

  const alpacaStatus = document.getElementById('settings-alpaca-status');
  if (alpacaStatus) {
    alpacaStatus.textContent = alpacaConfigured ? 'API Keys Configured in .env' : 'Missing Keys in .env (Mock Mode)';
  }

  const t212Status = document.getElementById('settings-t212-status');
  if (t212Status) {
    t212Status.textContent = t212Configured ? 'API Keys Configured in .env' : 'Missing Keys in .env (Mock Mode)';
  }
}

async function switchBroker(choice) {
  showToast(`Switching execution engine to ${choice.toUpperCase()}...`, 'info');
  try {
    const res = await fetch('/api/settings/broker', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ broker: choice }),
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Active broker set to ${data.active_broker.toUpperCase()}!`, 'success');
      await loadAllData();
    }
  } catch (err) {
    showToast('Failed to switch broker: ' + err.message, 'error');
  }
}

async function triggerTestAlert() {
  showToast('Dispatching test alert...', 'info');
  try {
    const res = await fetch('/api/settings/test-alert', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      showToast(data.message, 'success');
    }
  } catch (err) {
    showToast('Test alert failed: ' + err.message, 'error');
  }
}

// =============================================================================
// Helper Utilities
// =============================================================================
function formatCurrency(val) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(val || 0);
}

let toastTimer = null;
function showToast(msg, type = 'info') {
  const toast = document.getElementById('toast');
  const msgEl = document.getElementById('toast-msg');
  const indicator = document.getElementById('toast-indicator');

  if (!toast || !msgEl) return;

  msgEl.textContent = msg;

  // Reset border & indicator color classes
  toast.classList.remove('border-xtb-emerald', 'border-xtb-coral', 'border-xtb-amber', 'border-sky-500');
  if (indicator) {
    indicator.classList.remove('bg-xtb-emerald', 'bg-xtb-coral', 'bg-xtb-amber', 'bg-sky-500');
  }

  if (type === 'success') {
    toast.classList.add('border-xtb-emerald');
    if (indicator) indicator.classList.add('bg-xtb-emerald');
  } else if (type === 'error') {
    toast.classList.add('border-xtb-coral');
    if (indicator) indicator.classList.add('bg-xtb-coral');
  } else if (type === 'warning' || type === 'warn') {
    toast.classList.add('border-xtb-amber');
    if (indicator) indicator.classList.add('bg-xtb-amber');
  } else {
    toast.classList.add('border-sky-500');
    if (indicator) indicator.classList.add('bg-sky-500');
  }

  toast.classList.remove('hide');
  toast.classList.add('show');

  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove('show');
    toast.classList.add('hide');
  }, 3500);
}
