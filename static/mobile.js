/**
 * mobile.js - Touch-optimized Socket.IO client for the Mobile SOC Remote Controller.
 */

document.addEventListener("DOMContentLoaded", () => {
  const socket = io();

  let totalFlows = 0;
  let totalAttacks = 0;
  let recentWindow = [];
  const WINDOW_SIZE = 25;

  const dot = document.getElementById("mobile-status-dot");
  const statusText = document.getElementById("mobile-status-text");

  const threatBadge = document.getElementById("mobile-threat-badge");
  const threatScore = document.getElementById("mobile-threat-score");
  const threatDesc = document.getElementById("mobile-threat-desc");
  const threatBar = document.getElementById("mobile-threat-bar");

  const statTotal = document.getElementById("m-stat-total");
  const statAttacks = document.getElementById("m-stat-attacks");
  const statBlocked = document.getElementById("m-stat-blocked");

  const ipsToggle = document.getElementById("m-ips-toggle");
  const quarantineList = document.getElementById("m-quarantine-list");
  const quarantineCount = document.getElementById("m-quarantine-count");
  const flowFeed = document.getElementById("m-flow-feed");
  const feedbackPill = document.getElementById("m-sim-feedback");

  // 1. Initial State Fetch
  fetch("/api/status")
    .then(r => r.json())
    .then(data => {
      if (ipsToggle) ipsToggle.checked = !!data.ips_enabled;
      if (statBlocked) statBlocked.textContent = data.blocked_count || 0;
      if (quarantineCount) quarantineCount.textContent = `${data.blocked_count || 0} Blocked`;
    })
    .catch(console.error);

  fetchQuarantine();

  // 2. Socket.IO Connection Handlers
  socket.on("connect", () => {
    dot.className = "status-dot connected";
    statusText.textContent = "Live Remote";
  });

  socket.on("disconnect", () => {
    dot.className = "status-dot disconnected";
    statusText.textContent = "Disconnected";
  });

  // 3. New Flow Ingestion
  socket.on("new_flow", (flow) => {
    totalFlows++;
    statTotal.textContent = totalFlows.toLocaleString();

    const isAttack = flow.predicted_class !== "Benign";
    if (isAttack) {
      totalAttacks++;
      statAttacks.textContent = totalAttacks.toLocaleString();

      // Haptic phone vibration on threat detection!
      if (navigator.vibrate) {
        navigator.vibrate([80, 40, 80]);
      }
    }

    // Threat Index Window
    recentWindow.push(isAttack ? 1 : 0);
    if (recentWindow.length > WINDOW_SIZE) recentWindow.shift();
    updateThreatIndex();

    // Append to live feed
    appendFlowCard(flow);
  });

  // 4. IPS Events
  socket.on("ip_blocked", (record) => {
    fetchQuarantine();
  });

  socket.on("ip_unblocked", () => {
    fetchQuarantine();
  });

  function updateThreatIndex() {
    if (recentWindow.length === 0) return;
    const attacksInWindow = recentWindow.reduce((a, b) => a + b, 0);
    const score = Math.round((attacksInWindow / recentWindow.length) * 100);

    threatScore.textContent = `${score}%`;
    threatBar.style.width = `${score}%`;

    if (score === 0) {
      threatBadge.className = "threat-badge badge-secure";
      threatBadge.textContent = "SECURE";
      threatDesc.textContent = "All traffic normal";
    } else if (score < 50) {
      threatBadge.className = "threat-badge badge-elevated";
      threatBadge.textContent = "ELEVATED";
      threatDesc.textContent = "Anomalies detected";
    } else {
      threatBadge.className = "threat-badge badge-critical";
      threatBadge.textContent = "CRITICAL";
      threatDesc.textContent = "High attack frequency";
    }
  }

  function appendFlowCard(flow) {
    const isAttack = flow.predicted_class !== "Benign";
    const card = document.createElement("div");
    card.className = `m-flow-card ${flow.predicted_class}`;
    const srcDev = flow.src_device_name ? flow.src_device_name.split(' (')[0] : `${flow.src_ip}:${flow.src_port}`;
    const appTag = flow.application_name ? (flow.application_name.split(' ')[0] + " " + flow.application_name.split('(')[0].trim()) : (flow.protocol || "TCP");

    card.innerHTML = `
      <div>
        <div style="font-weight:700; font-size:12px; display:flex; align-items:center; gap:4px;">
          <span>${escapeHtml(srcDev)}</span>
        </div>
        <div style="font-size:10.5px; color:var(--text-muted);">${escapeHtml(appTag)} • ${flow.timestamp ? flow.timestamp.split(" ")[1] || flow.timestamp : "now"}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-weight:800; color:${isAttack ? 'var(--ddos)' : 'var(--benign)'};">${flow.predicted_class}</div>
        <div class="mono" style="font-size:10px; color:var(--text-secondary);">${flow.confidence}%</div>
      </div>
    `;

    flowFeed.prepend(card);
    while (flowFeed.children.length > 20) {
      flowFeed.removeChild(flowFeed.lastChild);
    }
  }

  function fetchQuarantine() {
    fetch("/api/ips/status")
      .then(r => r.json())
      .then(data => {
        const list = data.blocked_ips || [];
        if (statBlocked) statBlocked.textContent = list.length;
        if (quarantineCount) quarantineCount.textContent = `${list.length} Blocked`;

        if (!quarantineList) return;
        if (list.length === 0) {
          quarantineList.innerHTML = `<div class="m-empty-text">No active threats in quarantine.</div>`;
          return;
        }

        quarantineList.innerHTML = list.map(item => `
          <div class="m-quarantine-item">
            <div>
              <div class="mono" style="font-weight:700; color:var(--ddos);">${escapeHtml(item.ip)}</div>
              <div style="font-size:10.5px; color:var(--text-muted);">${escapeHtml(item.reason)} (${item.confidence}%)</div>
            </div>
            <button class="m-unblock-btn" onclick="unblockMobileIp('${escapeHtml(item.ip)}')">Unblock</button>
          </div>
        `).join("");
      })
      .catch(console.error);
  }

  // 5. IPS Master Toggle
  if (ipsToggle) {
    ipsToggle.addEventListener("change", () => {
      fetch("/api/ips/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: ipsToggle.checked })
      })
      .then(r => r.json())
      .then(res => {
        showFeedback(res.enabled ? "🛡️ Active IPS Defense Enabled" : "⚠️ IPS Auto-Defense Disabled");
      })
      .catch(console.error);
    });
  }

  // Global Helpers for Mobile Actions
  window.triggerMobileAttack = function(type, count = 1) {
    if (navigator.vibrate) navigator.vibrate(50);
    fetch("/api/simulate-attack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, count })
    })
    .then(r => r.json())
    .then(res => {
      showFeedback(`⚡ Injected ${res.count}x ${type.toUpperCase()} flow(s) to SOC!`);
    })
    .catch(err => {
      showFeedback(`❌ Simulation error: ${err.message}`);
    });
  };

  window.unblockMobileIp = function(ip) {
    fetch("/api/ips/unblock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ip })
    })
    .then(r => r.json())
    .then(res => {
      showFeedback(`✅ Released ${ip} from quarantine.`);
      fetchQuarantine();
    })
    .catch(console.error);
  };

  function showFeedback(msg) {
    if (!feedbackPill) return;
    feedbackPill.textContent = msg;
    feedbackPill.style.display = "block";
    setTimeout(() => {
      feedbackPill.style.display = "none";
    }, 2500);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/[&<>"']/g, m => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
    })[m]);
  }
});
