// =========================================================================
// IoT Intrusion Detection & Prevention System — SOC Live Client Engine
// =========================================================================

const socket = io();

// UI Elements
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const tableBody = document.getElementById("flow-table-body");
const flowDisplayCounter = document.getElementById("flow-display-counter");
const totalFlowsStat = document.getElementById("total-flows-stat");

// Threat Gauge Elements
const threatScoreElem = document.getElementById("threat-score");
const threatDescElem = document.getElementById("threat-desc");
const threatBadgeElem = document.getElementById("threat-badge");
const threatBarFill = document.getElementById("threat-bar-fill");

// Control Buttons
const btnPauseStream = document.getElementById("btn-pause-stream");
const pauseIcon = document.getElementById("pause-icon");
const pauseText = document.getElementById("pause-text");
const btnAudioToggle = document.getElementById("btn-audio-toggle");
const audioIcon = document.getElementById("audio-icon");
const audioText = document.getElementById("audio-text");
const btnExportPdf = document.getElementById("btn-export-pdf");
const btnExportCsv = document.getElementById("btn-export-csv");
const btnClearTable = document.getElementById("btn-clear-table");
const searchInput = document.getElementById("flow-search-input");
const filterTabsContainer = document.getElementById("filter-tabs");

// Inspection Modal Elements
const flowModal = document.getElementById("flow-modal");
const modalCloseBtn = document.getElementById("modal-close-btn");
const modalClassBadge = document.getElementById("modal-class-badge");
const modalSrc = document.getElementById("modal-src");
const modalDst = document.getElementById("modal-dst");
const modalProto = document.getElementById("modal-proto");
const modalProbBars = document.getElementById("modal-prob-bars");
const modalFeaturesGrid = document.getElementById("modal-features-grid");
const modalXaiInsights = document.getElementById("modal-xai-insights");
const modalXaiBars = document.getElementById("modal-xai-bars");
const toastContainer = document.getElementById("toast-container");

// Threat Simulator Elements
const btnOpenSim = document.getElementById("btn-open-sim");
const simModal = document.getElementById("sim-modal");
const simCloseBtn = document.getElementById("sim-close-btn");
const simStatusBox = document.getElementById("sim-status-box");
const simStatusText = document.getElementById("sim-status-text");

// Mobile Alerts Modal Elements
const btnOpenAlerts = document.getElementById("btn-open-alerts");
const alertsModal = document.getElementById("alerts-modal");
const alertsCloseBtn = document.getElementById("alerts-close-btn");
const cfgMasterEnabled = document.getElementById("cfg-master-enabled");
const cfgTgEnabled = document.getElementById("cfg-tg-enabled");
const cfgTgToken = document.getElementById("cfg-tg-token");
const cfgTgChatId = document.getElementById("cfg-tg-chat-id");
const cfgDcEnabled = document.getElementById("cfg-dc-enabled");
const cfgDcWebhook = document.getElementById("cfg-dc-webhook");
const btnSaveAlerts = document.getElementById("btn-save-alerts");
const btnTestAlert = document.getElementById("btn-test-alert");
const alertsFeedbackBox = document.getElementById("alerts-feedback-box");

// Active IPS & Quarantine Elements
const btnToggleIPS = document.getElementById("btn-toggle-ips");
const ipsToggleText = document.getElementById("ips-toggle-text");
const btnOpenQuarantine = document.getElementById("btn-open-quarantine");
const quarantineBtnText = document.getElementById("quarantine-btn-text");
const quarantineModal = document.getElementById("quarantine-modal");
const quarantineCloseBtn = document.getElementById("quarantine-close-btn");
const quarantineTableBody = document.getElementById("quarantine-table-body");
const quarantineModalCounter = document.getElementById("quarantine-modal-counter");
const manualBlockInput = document.getElementById("manual-block-ip");

// State
const counts = { Benign: 0, DDoS: 0, DoS: 0, Recon: 0 };
let totalFlows = 0;
let totalAttacks = 0;
let allFlows = [];           // Array of all session flow objects
let isPaused = false;
let audioEnabled = true;
let activeFilter = "all";
let searchTerm = "";
let ipsActive = true;
let quarantinedIPsList = [];
const MAX_ROWS = 100;

// =========================================================================
// 1. Chart.js Initialization
// =========================================================================
let timelineChart = null;
let donutChart = null;

const TIMELINE_POINTS = 20;
const timelineLabels = Array(TIMELINE_POINTS).fill("");
const timelineTrafficData = Array(TIMELINE_POINTS).fill(0);
const timelineAttackData = Array(TIMELINE_POINTS).fill(0);

let currentIntervalTraffic = 0;
let currentIntervalAttacks = 0;

function initCharts() {
  const ctxTimeline = document.getElementById("trafficTimelineChart").getContext("2d");
  timelineChart = new Chart(ctxTimeline, {
    type: "line",
    data: {
      labels: timelineLabels,
      datasets: [
        {
          label: "Total Flows",
          data: timelineTrafficData,
          borderColor: "#0284c7",
          backgroundColor: "rgba(2, 132, 199, 0.12)",
          borderWidth: 2,
          fill: true,
          tension: 0.35,
          pointRadius: 2,
        },
        {
          label: "Flagged Attacks",
          data: timelineAttackData,
          borderColor: "#ef4444",
          backgroundColor: "rgba(239, 68, 68, 0.2)",
          borderWidth: 2,
          fill: true,
          tension: 0.35,
          pointRadius: 3,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b", font: { size: 10 } }
        },
        y: {
          beginAtZero: true,
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b", font: { size: 10 }, precision: 0 }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#181b26",
          borderColor: "#2b3145",
          borderWidth: 1,
        }
      }
    }
  });

  const ctxDonut = document.getElementById("threatDonutChart").getContext("2d");
  donutChart = new Chart(ctxDonut, {
    type: "doughnut",
    data: {
      labels: ["Benign", "DDoS", "DoS", "Recon"],
      datasets: [
        {
          data: [0, 0, 0, 0],
          backgroundColor: ["#10b981", "#ef4444", "#f97316", "#3b82f6"],
          borderColor: "#11131a",
          borderWidth: 3,
          hoverOffset: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "right",
          labels: {
            color: "#94a3b8",
            font: { size: 11 },
            boxWidth: 10,
            padding: 10
          }
        },
        tooltip: {
          backgroundColor: "#181b26",
          borderColor: "#2b3145",
          borderWidth: 1,
          callbacks: {
            label: function (ctx) {
              const val = ctx.raw || 0;
              const pct = totalFlows > 0 ? ((val / totalFlows) * 100).toFixed(1) : 0;
              return ` ${ctx.label}: ${val} (${pct}%)`;
            }
          }
        }
      }
    }
  });

  setInterval(() => {
    const timeStr = new Date().toLocaleTimeString([], { hour12: false });
    timelineLabels.shift();
    timelineLabels.push(timeStr);

    timelineTrafficData.shift();
    timelineTrafficData.push(currentIntervalTraffic);

    timelineAttackData.shift();
    timelineAttackData.push(currentIntervalAttacks);

    currentIntervalTraffic = 0;
    currentIntervalAttacks = 0;

    timelineChart.update("none");
  }, 2000);
}

// =========================================================================
// 2. Vis.js Real-Time Network Topology Engine
// =========================================================================
let topoNodes = null;
let topoEdges = null;
let topoNetwork = null;
let topoPhysicsEnabled = true;
let topologyStats = { nodes: 1, links: 0, threats: 0 };
const discoveredNodesMap = new Map();

function initTopologyNetwork() {
  const container = document.getElementById("topology-network-container");
  if (!container || typeof vis === "undefined") return;

  topoNodes = new vis.DataSet([
    {
      id: "gateway",
      label: "💻 IoT Gateway\n(IDS / IPS Server)",
      shape: "box",
      margin: 10,
      color: {
        background: "#0f172a",
        border: "#38bdf8",
        highlight: { background: "#1e293b", border: "#0284c7" }
      },
      font: { color: "#f8fafc", face: "Inter", size: 12, bold: true },
      shadow: { enabled: true, color: "rgba(56, 189, 248, 0.4)", size: 14 }
    }
  ]);

  discoveredNodesMap.set("gateway", { type: "gateway", isAttacker: false });

  topoEdges = new vis.DataSet([]);

  const data = { nodes: topoNodes, edges: topoEdges };
  const options = {
    physics: {
      enabled: true,
      solver: "forceAtlas2Based",
      forceAtlas2Based: {
        gravitationalConstant: -35,
        centralGravity: 0.008,
        springLength: 100,
        springConstant: 0.12,
        damping: 0.4
      },
      stabilization: { iterations: 60 }
    },
    nodes: {
      borderWidth: 2,
      borderWidthSelected: 3,
    },
    edges: {
      arrows: { to: { enabled: true, scaleFactor: 0.6 } },
      smooth: { type: "continuous" }
    },
    interaction: {
      hover: true,
      tooltipDelay: 100,
      zoomView: true,
      dragView: true
    }
  };

  topoNetwork = new vis.Network(container, data, options);

  topoNetwork.on("click", (params) => {
    if (params.nodes.length > 0) {
      const nodeId = params.nodes[0];
      if (nodeId !== "gateway") {
        searchTerm = nodeId;
        searchInput.value = nodeId;
        reRenderTable();
      }
    }
  });
}

function updateTopologyFromFlow(flow) {
  if (!topoNodes || !topoEdges) return;

  const isAttack = flow.predicted_class !== "Benign";
  const src = flow.src_ip;
  const dst = flow.dst_ip;

  // Identify external or local peer
  let peerIP = src;
  let isAttackerPeer = isAttack;

  if (!discoveredNodesMap.has(peerIP) && peerIP && peerIP !== "127.0.0.1") {
    let nodeColor = { background: "#064e3b", border: "#10b981" };
    let nodeLabel = `📱 ${peerIP}`;
    let shadowColor = "rgba(16, 185, 129, 0.3)";

    if (isAttackerPeer) {
      nodeColor = { background: "#7f1d1d", border: "#ef4444" };
      nodeLabel = `🚨 Attacker\n(${peerIP})`;
      shadowColor = "rgba(239, 68, 68, 0.6)";
      topologyStats.threats += 1;
    } else if (!peerIP.startsWith("192.168.") && !peerIP.startsWith("10.") && !peerIP.startsWith("172.")) {
      nodeColor = { background: "#1e1b4b", border: "#818cf8" };
      nodeLabel = `🌐 Remote\n(${peerIP})`;
      shadowColor = "rgba(129, 140, 248, 0.3)";
    }

    topoNodes.add({
      id: peerIP,
      label: nodeLabel,
      shape: isAttackerPeer ? "box" : "ellipse",
      color: nodeColor,
      font: { color: "#f8fafc", face: "Inter", size: 10, bold: isAttackerPeer },
      shadow: { enabled: true, color: shadowColor, size: 10 }
    });

    discoveredNodesMap.set(peerIP, { type: isAttackerPeer ? "attacker" : "host", isAttacker: isAttackerPeer });
    topologyStats.nodes += 1;

    // Connect to gateway
    const edgeId = `${peerIP}->gateway`;
    topoEdges.add({
      id: edgeId,
      from: peerIP,
      to: "gateway",
      color: isAttackerPeer ? { color: "#ef4444", highlight: "#f87171" } : { color: "#10b981", highlight: "#34d399" },
      width: isAttackerPeer ? 3 : 1.5,
      dashes: isAttackerPeer,
      title: `${flow.protocol} | ${flow.predicted_class} (${flow.confidence}%)`
    });
    topologyStats.links += 1;

    updateTopologyStatsUI();
  } else if (discoveredNodesMap.has(peerIP) && isAttackerPeer && !discoveredNodesMap.get(peerIP).isAttacker) {
    // Escalate existing node to attacker
    topoNodes.update({
      id: peerIP,
      label: `🚨 Attacker\n(${peerIP})`,
      shape: "box",
      color: { background: "#7f1d1d", border: "#ef4444" },
      shadow: { enabled: true, color: "rgba(239, 68, 68, 0.6)", size: 14 }
    });
    discoveredNodesMap.get(peerIP).isAttacker = true;
    topologyStats.threats += 1;
    updateTopologyStatsUI();
  }
}

function updateTopologyStatsUI() {
  const nodeCountElem = document.getElementById("topo-nodes-count");
  const linksCountElem = document.getElementById("topo-links-count");
  const attackCountElem = document.getElementById("topo-attack-count");
  if (nodeCountElem) nodeCountElem.textContent = `Nodes: ${topologyStats.nodes}`;
  if (linksCountElem) linksCountElem.textContent = `Active Links: ${topologyStats.links}`;
  if (attackCountElem) attackCountElem.textContent = `Threats: ${topologyStats.threats}`;
}

window.fitTopologyMap = function() {
  if (topoNetwork) topoNetwork.fit({ animation: { duration: 500, easingFunction: "easeInOutQuad" } });
};

window.toggleTopologyPhysics = function() {
  if (!topoNetwork) return;
  topoPhysicsEnabled = !topoPhysicsEnabled;
  topoNetwork.setOptions({ physics: { enabled: topoPhysicsEnabled } });
  const btn = document.getElementById("btn-toggle-physics");
  if (btn) btn.innerHTML = `<i class="fa-solid fa-atom"></i> Physics: ${topoPhysicsEnabled ? 'On' : 'Off'}`;
};

window.resetTopologyMap = function() {
  if (!topoNodes || !topoEdges) return;
  topoNodes.clear();
  topoEdges.clear();
  discoveredNodesMap.clear();
  topologyStats = { nodes: 1, links: 0, threats: 0 };
  topoNodes.add({
    id: "gateway",
    label: "💻 IoT Gateway\n(IDS / IPS Server)",
    shape: "box",
    margin: 10,
    color: { background: "#0f172a", border: "#38bdf8" },
    font: { color: "#f8fafc", face: "Inter", size: 12, bold: true },
    shadow: { enabled: true, color: "rgba(56, 189, 248, 0.4)", size: 14 }
  });
  discoveredNodesMap.set("gateway", { type: "gateway", isAttacker: false });
  updateTopologyStatsUI();
};

window.switchSOCView = function(viewType) {
  const chartsContainer = document.getElementById("charts-view-container");
  const topoContainer = document.getElementById("topology-view-container");
  const tabCharts = document.getElementById("view-tab-charts");
  const tabTopo = document.getElementById("view-tab-topology");

  if (viewType === "topology") {
    chartsContainer.style.display = "none";
    topoContainer.style.display = "block";
    tabCharts.classList.remove("active");
    tabTopo.classList.add("active");
    setTimeout(() => { if (topoNetwork) topoNetwork.fit(); }, 200);
  } else {
    topoContainer.style.display = "none";
    chartsContainer.style.display = "grid";
    tabTopo.classList.remove("active");
    tabCharts.classList.add("active");
  }
};

// =========================================================================
// 3. Active IPS Auto-Defense & Quarantine Pool Manager
// =========================================================================
async function loadIPSStatus() {
  try {
    const res = await fetch("/api/ips/status");
    const data = await res.json();
    ipsActive = !!data.enabled;
    quarantinedIPsList = data.blocked_ips || [];

    updateIPSButtonsUI();
    renderQuarantineTable(quarantinedIPsList);
  } catch (e) {
    console.error("Error loading IPS status:", e);
  }
}

function updateIPSButtonsUI() {
  if (btnToggleIPS) {
    if (ipsActive) {
      btnToggleIPS.className = "btn btn-ips active";
      ipsToggleText.textContent = "IPS: Active (≥90%)";
    } else {
      btnToggleIPS.className = "btn btn-ips disabled";
      ipsToggleText.textContent = "IPS: Disabled";
    }
  }

  if (quarantineBtnText) {
    quarantineBtnText.textContent = `Quarantine (${quarantinedIPsList.length})`;
  }
  if (quarantineModalCounter) {
    quarantineModalCounter.textContent = `${quarantinedIPsList.length} Active Blocks`;
  }
}

if (btnToggleIPS) {
  btnToggleIPS.onclick = async () => {
    try {
      const res = await fetch("/api/ips/toggle", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: !ipsActive }) });
      const data = await res.json();
      ipsActive = data.enabled;
      updateIPSButtonsUI();
    } catch (e) {
      console.error("Failed to toggle IPS:", e);
    }
  };
}

if (btnOpenQuarantine) {
  btnOpenQuarantine.onclick = () => {
    loadIPSStatus();
    quarantineModal.classList.add("active");
  };
}
if (quarantineCloseBtn) {
  quarantineCloseBtn.onclick = () => quarantineModal.classList.remove("active");
}
if (quarantineModal) {
  quarantineModal.onclick = (e) => { if (e.target === quarantineModal) quarantineModal.classList.remove("active"); };
}

function renderQuarantineTable(list) {
  quarantineTableBody.innerHTML = "";
  if (!list || list.length === 0) {
    quarantineTableBody.innerHTML = `<tr class="empty-row"><td colspan="6">No IP addresses currently quarantined.</td></tr>`;
    return;
  }

  for (const item of list) {
    const tr = document.createElement("tr");
    const fwStatusClass = item.firewall_applied ? "active-firewall" : "safe-mode";
    const fwStatusText = item.firewall_applied ? "Firewall Dropped" : "Quarantine Logged";

    tr.innerHTML = `
      <td class="mono font-bold text-danger">${item.ip}</td>
      <td><span class="badge ${item.reason.split(' ')[0]}">${item.reason}</span></td>
      <td class="mono font-bold">${item.confidence}%</td>
      <td class="mono">${item.timestamp}</td>
      <td><span class="status-tag ${fwStatusClass}">${fwStatusText}</span></td>
      <td>
        <button class="btn-unblock" onclick="unblockAttackerIP('${item.ip}')">
          <i class="fa-solid fa-lock-open"></i> Unblock
        </button>
      </td>
    `;
    quarantineTableBody.appendChild(tr);
  }
}

window.unblockAttackerIP = async function(ip) {
  try {
    const res = await fetch("/api/ips/unblock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ip: ip })
    });
    const data = await res.json();
    if (data.success) {
      quarantinedIPsList = quarantinedIPsList.filter(item => item.ip !== ip);
      updateIPSButtonsUI();
      renderQuarantineTable(quarantinedIPsList);
    }
  } catch (e) {
    console.error("Failed to unblock IP:", e);
  }
};

window.triggerManualBlock = async function() {
  const ip = manualBlockInput.value.trim();
  if (!ip) return;

  try {
    const res = await fetch("/api/ips/block", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ip: ip, reason: "Manual Administrator Block" })
    });
    const data = await res.json();
    if (data.success) {
      manualBlockInput.value = "";
      loadIPSStatus();
    } else {
      alert(data.message || "Failed to block IP.");
    }
  } catch (e) {
    console.error("Manual block failed:", e);
  }
};

socket.on("ip_blocked", (record) => {
  if (!quarantinedIPsList.some(item => item.ip === record.ip)) {
    quarantinedIPsList.unshift(record);
    updateIPSButtonsUI();
    renderQuarantineTable(quarantinedIPsList);
  }
  showThreatToast({
    predicted_class: "IPS Firewall Block",
    src_ip: record.ip,
    dst_ip: "Firewall Rule Created",
    confidence: record.confidence
  });
});

socket.on("ip_unblocked", (data) => {
  quarantinedIPsList = quarantinedIPsList.filter(item => item.ip !== data.ip);
  updateIPSButtonsUI();
  renderQuarantineTable(quarantinedIPsList);
});

// =========================================================================
// QR Code Mobile Remote Pairing Modal
// =========================================================================
const qrModal = document.getElementById("qr-modal");
const btnOpenQR = document.getElementById("btn-open-qr");
const qrCloseBtn = document.getElementById("qr-close-btn");
const qrcodeBox = document.getElementById("qrcode-box");
const qrDirectLink = document.getElementById("qr-direct-link");
let qrcodeInstance = null;

if (btnOpenQR) {
  btnOpenQR.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/mobile-info");
      const info = await res.json();
      const mobileUrl = info.mobile_url;

      if (qrDirectLink) {
        qrDirectLink.href = mobileUrl;
        qrDirectLink.textContent = mobileUrl;
      }

      if (qrcodeBox) {
        qrcodeBox.innerHTML = "";
        if (window.QRCode) {
          qrcodeInstance = new QRCode(qrcodeBox, {
            text: mobileUrl,
            width: 180,
            height: 180,
            colorDark: "#090a0f",
            colorLight: "#ffffff",
            correctLevel: QRCode.CorrectLevel.H
          });
        }
      }

      if (qrModal) qrModal.classList.add("active");
    } catch (e) {
      console.error("Failed to load mobile info:", e);
    }
  });
}

if (qrCloseBtn) {
  qrCloseBtn.addEventListener("click", () => {
    if (qrModal) qrModal.classList.remove("active");
  });
}

if (qrModal) {
  qrModal.addEventListener("click", (e) => {
    if (e.target === qrModal) qrModal.classList.remove("active");
  });
}

// =========================================================================
// 4. Audio Alert Synthesizer
// =========================================================================
let audioCtx = null;

function playAlertChime(attackType) {
  if (!audioEnabled) return;
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = "sine";
    const freq = attackType === "DDoS" ? 880 : attackType === "DoS" ? 660 : 550;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(300, audioCtx.currentTime + 0.35);

    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 0.35);
  } catch (e) {}
}

// =========================================================================
// 5. Threat Gauge & Toast Notifications
// =========================================================================
function updateThreatGauge() {
  if (totalFlows === 0) {
    threatScoreElem.textContent = "0%";
    threatBarFill.style.width = "0%";
    threatBadgeElem.textContent = "SECURE";
    threatBadgeElem.className = "threat-status-badge badge-secure";
    threatDescElem.textContent = "Monitoring live network traffic";
    return;
  }

  const attackRatio = totalAttacks / totalFlows;
  let score = Math.round(attackRatio * 100);
  if (totalAttacks > 0 && score < 15) score = 15;
  score = Math.min(100, Math.max(0, score));

  threatScoreElem.textContent = `${score}%`;
  threatBarFill.style.width = `${score}%`;

  if (score >= 40) {
    threatBadgeElem.textContent = "CRITICAL THREAT";
    threatBadgeElem.className = "threat-status-badge badge-critical";
    threatDescElem.textContent = `${totalAttacks} attack flows intercepted`;
  } else if (score >= 10) {
    threatBadgeElem.textContent = "ELEVATED";
    threatBadgeElem.className = "threat-status-badge badge-elevated";
    threatDescElem.textContent = "Suspicious traffic patterns detected";
  } else {
    threatBadgeElem.textContent = "SECURE";
    threatBadgeElem.className = "threat-status-badge badge-secure";
    threatDescElem.textContent = "All traffic verified as benign";
  }
}

function showThreatToast(flow) {
  const toast = document.createElement("div");
  toast.className = `toast ${flow.predicted_class.replace(/\s+/g, '')}`;
  toast.innerHTML = `
    <div class="toast-icon"><i class="fa-solid fa-triangle-exclamation"></i></div>
    <div>
      <div class="toast-title">${flow.predicted_class} Detected!</div>
      <div class="toast-msg">${flow.src_ip} &rarr; ${flow.dst_ip} (${flow.confidence}%)</div>
    </div>
  `;
  toastContainer.prepend(toast);
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, 5000);
}

// =========================================================================
// 6. Socket.IO Live Stream Handlers
// =========================================================================
socket.on("connect", () => {
  statusDot.classList.add("connected");
  statusDot.classList.remove("disconnected");
  statusText.textContent = "Connected — watching live traffic";
  loadIPSStatus();
});

socket.on("system_status", (config) => {
  if (config && config.iface) {
    statusText.textContent = `Connected — watching live traffic on '${config.iface}'`;
  } else if (config && config.mode === "demo") {
    statusText.textContent = "Connected — demo mode (replaying test data)";
  }
});

socket.on("disconnect", () => {
  statusDot.classList.add("disconnected");
  statusDot.classList.remove("connected");
  statusText.textContent = "Disconnected";
});

socket.on("new_flow", (flow) => {
  allFlows.unshift(flow);
  if (allFlows.length > 500) allFlows.pop();

  totalFlows += 1;
  currentIntervalTraffic += 1;

  const isAttack = flow.predicted_class !== "Benign";
  if (isAttack) {
    totalAttacks += 1;
    currentIntervalAttacks += 1;
    playAlertChime(flow.predicted_class);
    if (flow.confidence >= 75) showThreatToast(flow);
  }

  // Update topology
  updateTopologyFromFlow(flow);

  // Update counts
  if (counts[flow.predicted_class] !== undefined) {
    counts[flow.predicted_class] += 1;
    document.getElementById(`count-${flow.predicted_class}`).textContent = counts[flow.predicted_class];
  }

  // Update percentages
  for (const [key, val] of Object.entries(counts)) {
    const pct = ((val / totalFlows) * 100).toFixed(1);
    const subElem = document.getElementById(`pct-${key}`);
    if (subElem) subElem.textContent = `${pct}% of total`;
  }

  // Update filter tab counts
  document.getElementById("tab-count-all").textContent = totalFlows;
  document.getElementById("tab-count-attacks").textContent = totalAttacks;
  totalFlowsStat.textContent = `Total: ${totalFlows} Flows`;

  // Update charts & threat gauge
  updateThreatGauge();
  donutChart.data.datasets[0].data = [counts.Benign, counts.DDoS, counts.DoS, counts.Recon];
  donutChart.update("none");

  // Render to table if stream is active and matches filter
  if (!isPaused && matchesFilter(flow)) {
    insertFlowRow(flow);
  }

  updateDisplayCounter();
});

function matchesFilter(flow) {
  if (activeFilter === "attacks" && flow.predicted_class === "Benign") return false;
  if (activeFilter !== "all" && activeFilter !== "attacks" && flow.predicted_class !== activeFilter) return false;

  if (searchTerm) {
    const query = searchTerm.toLowerCase();
    const str = `${flow.src_ip} ${flow.dst_ip} ${flow.src_port} ${flow.dst_port} ${flow.protocol} ${flow.predicted_class}`.toLowerCase();
    if (!str.includes(query)) return false;
  }
  return true;
}

function insertFlowRow(flow) {
  const emptyRow = tableBody.querySelector(".empty-row");
  if (emptyRow) emptyRow.remove();

  const row = document.createElement("tr");
  row.className = "new-row";
  row.onclick = () => openFlowModal(flow);

  const barColor = flow.predicted_class === "Benign" ? "var(--benign)" :
                   flow.predicted_class === "DDoS" ? "var(--ddos)" :
                   flow.predicted_class === "DoS" ? "var(--dos)" : "var(--recon)";

  const ipsBadge = flow.ips_blocked ? `<span class="status-tag active-firewall ml-1" title="Blocked by IPS"><i class="fa-solid fa-shield-halved"></i> Blocked</span>` : "";

  row.innerHTML = `
    <td>${flow.timestamp}</td>
    <td class="mono">${flow.src_ip}:${flow.src_port} ${ipsBadge}</td>
    <td class="mono">${flow.dst_ip}:${flow.dst_port}</td>
    <td><span class="proto-badge">${flow.protocol}</span></td>
    <td><span class="badge ${flow.predicted_class}">${flow.predicted_class}</span></td>
    <td>
      <div class="confidence-bar-wrap">
        <span class="confidence-val">${flow.confidence}%</span>
        <div class="mini-bar"><div class="mini-bar-fill" style="width: ${flow.confidence}%; background: ${barColor};"></div></div>
      </div>
    </td>
    <td><button class="btn-inspect" onclick="event.stopPropagation(); openFlowModal(${JSON.stringify(flow).replace(/"/g, '&quot;')})"><i class="fa-solid fa-magnifying-glass-plus"></i></button></td>
  `;

  tableBody.prepend(row);

  while (tableBody.rows.length > MAX_ROWS) {
    tableBody.deleteRow(tableBody.rows.length - 1);
  }
}

function reRenderTable() {
  tableBody.innerHTML = "";
  const filtered = allFlows.filter(matchesFilter).slice(0, MAX_ROWS);

  if (filtered.length === 0) {
    tableBody.innerHTML = `<tr class="empty-row"><td colspan="7">No flows match the current filter/search.</td></tr>`;
  } else {
    for (const flow of filtered) {
      insertFlowRow(flow);
    }
  }
  updateDisplayCounter();
}

function updateDisplayCounter() {
  const visible = tableBody.querySelectorAll("tr:not(.empty-row)").length;
  flowDisplayCounter.textContent = `${visible} of ${allFlows.length} flows`;
}

// =========================================================================
// 7. Flow Inspection Modal (with XAI Explainability)
// =========================================================================
window.openFlowModal = function(flow) {
  modalClassBadge.textContent = flow.predicted_class;
  modalClassBadge.className = `modal-badge badge ${flow.predicted_class}`;
  modalSrc.textContent = `${flow.src_ip}:${flow.src_port}`;
  modalDst.textContent = `${flow.dst_ip}:${flow.dst_port}`;
  modalProto.textContent = `${flow.protocol} | ${flow.timestamp}`;

  // 1. Classification Probabilities
  modalProbBars.innerHTML = "";
  const probs = flow.class_probs || { [flow.predicted_class]: flow.confidence };
  const classOrder = ["Benign", "DDoS", "DoS", "Recon"];
  const colorMap = { Benign: "var(--benign)", DDoS: "var(--ddos)", DoS: "var(--dos)", Recon: "var(--recon)" };

  for (const cls of classOrder) {
    const val = probs[cls] !== undefined ? probs[cls] : 0;
    const isWinner = flow.predicted_class === cls;
    const item = document.createElement("div");
    item.className = "prob-item";
    item.innerHTML = `
      <div class="prob-header">
        <span style="color: ${colorMap[cls]}">${cls} ${isWinner ? '★ (Selected)' : ''}</span>
        <span class="mono">${val}%</span>
      </div>
      <div class="prob-bar">
        <div class="prob-bar-fill" style="width: ${val}%; background: ${colorMap[cls]}"></div>
      </div>
    `;
    modalProbBars.appendChild(item);
  }

  // 2. Explainable AI (XAI)
  modalXaiInsights.innerHTML = "";
  modalXaiBars.innerHTML = "";
  const xai = flow.xai || {};
  const insights = xai.insights || [];
  const topFeatures = xai.top_features || [];

  if (insights.length === 0) {
    modalXaiInsights.innerHTML = `<div class="xai-insight-desc">No anomalous attribution flags found. Flow aligned with benign model baseline.</div>`;
  } else {
    for (const ins of insights) {
      const card = document.createElement("div");
      card.className = `xai-insight-card ${ins.type || 'threat'}`;
      card.innerHTML = `
        <div class="xai-insight-icon"><i class="fa-solid ${ins.type === 'benign' ? 'fa-circle-check' : 'fa-triangle-exclamation'}"></i></div>
        <div>
          <div class="xai-insight-title">${ins.title}</div>
          <div class="xai-insight-desc">${ins.desc}</div>
        </div>
      `;
      modalXaiInsights.appendChild(card);
    }
  }

  if (topFeatures.length > 0) {
    for (const feat of topFeatures) {
      const barItem = document.createElement("div");
      barItem.className = "xai-bar-item";
      const isThreat = flow.predicted_class !== "Benign";
      barItem.innerHTML = `
        <span class="xai-feat-name" title="${feat.name}">${feat.name}</span>
        <div class="xai-bar-bg">
          <div class="xai-bar-fill ${isThreat ? 'threat' : ''}" style="width: ${feat.score}%;"></div>
        </div>
        <span class="xai-bar-score">${feat.score}%</span>
      `;
      modalXaiBars.appendChild(barItem);
    }
  }

  // 3. Statistical Features
  modalFeaturesGrid.innerHTML = "";
  const m = flow.metrics || {};
  const featureList = [
    { label: "Flow Duration", val: m.duration_ms !== undefined ? `${m.duration_ms} ms` : "-" },
    { label: "Total Packets", val: m.total_pkts !== undefined ? `${m.total_pkts} pkts` : "-" },
    { label: "Fwd / Bwd Pkts", val: m.fwd_pkts !== undefined ? `${m.fwd_pkts} / ${m.bwd_pkts}` : "-" },
    { label: "Flow Rate", val: m.flow_pkts_s !== undefined ? `${m.flow_pkts_s} pkts/s` : "-" },
    { label: "Throughput", val: m.flow_byts_s !== undefined ? `${(m.flow_byts_s / 1024).toFixed(2)} KB/s` : "-" },
    { label: "Mean Pkt Size", val: m.pkt_len_mean !== undefined ? `${m.pkt_len_mean} B` : "-" },
    { label: "Max Pkt Size", val: m.pkt_len_max !== undefined ? `${m.pkt_len_max} B` : "-" },
    { label: "TCP Flags (S/F/R/A)", val: m.syn_flags !== undefined ? `${m.syn_flags} / ${m.fin_flags} / ${m.rst_flags} / ${m.ack_flags}` : "-" },
  ];

  for (const feat of featureList) {
    const box = document.createElement("div");
    box.className = "modal-metric";
    box.innerHTML = `<span class="metric-label">${feat.label}</span><span class="metric-val mono">${feat.val}</span>`;
    modalFeaturesGrid.appendChild(box);
  }

  flowModal.classList.add("active");
};

modalCloseBtn.onclick = () => flowModal.classList.remove("active");
flowModal.onclick = (e) => { if (e.target === flowModal) flowModal.classList.remove("active"); };

// =========================================================================
// 8. Threat Simulator & Push Alerts
// =========================================================================
if (btnOpenSim) btnOpenSim.onclick = () => simModal.classList.add("active");
if (simCloseBtn) simCloseBtn.onclick = () => simModal.classList.remove("active");
if (simModal) simModal.onclick = (e) => { if (e.target === simModal) simModal.classList.remove("active"); };

window.triggerSimulation = async function(attackType, count = 1) {
  if (simStatusBox) {
    simStatusBox.style.display = "flex";
    simStatusText.textContent = `Injecting simulated ${attackType.toUpperCase()} threat (${count} flow${count > 1 ? 's' : ''})...`;
  }

  try {
    const res = await fetch("/api/simulate-attack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: attackType, count: count })
    });
    const data = await res.json();
    if (data.status === "success") {
      setTimeout(() => {
        if (simStatusBox) simStatusBox.style.display = "none";
        simModal.classList.remove("active");
      }, 700);
    }
  } catch (err) {
    console.error("Simulation error:", err);
    if (simStatusBox) simStatusBox.style.display = "none";
  }
};

async function loadAlertsConfig() {
  try {
    const res = await fetch("/api/alerts-config");
    const cfg = await res.json();
    cfgMasterEnabled.checked = cfg.enabled !== false;
    cfgTgEnabled.checked = !!cfg.telegram_enabled;
    cfgTgToken.value = cfg.telegram_bot_token || "";
    cfgTgChatId.value = cfg.telegram_chat_id || "";
    cfgDcEnabled.checked = !!cfg.discord_enabled;
    cfgDcWebhook.value = cfg.discord_webhook_url || "";
  } catch (e) {
    console.error("Failed to load alert config:", e);
  }
}

if (btnOpenAlerts) {
  btnOpenAlerts.onclick = () => {
    loadAlertsConfig();
    alertsModal.classList.add("active");
  };
}
if (alertsCloseBtn) alertsCloseBtn.onclick = () => alertsModal.classList.remove("active");
if (alertsModal) alertsModal.onclick = (e) => { if (e.target === alertsModal) alertsModal.classList.remove("active"); };

if (btnSaveAlerts) {
  btnSaveAlerts.onclick = async () => {
    const newConfig = {
      enabled: cfgMasterEnabled.checked,
      telegram_enabled: cfgTgEnabled.checked,
      telegram_bot_token: cfgTgToken.value.trim(),
      telegram_chat_id: cfgTgChatId.value.trim(),
      discord_enabled: cfgDcEnabled.checked,
      discord_webhook_url: cfgDcWebhook.value.trim(),
    };

    try {
      const res = await fetch("/api/alerts-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newConfig)
      });
      const data = await res.json();
      if (data.status === "success") {
        alertsFeedbackBox.style.display = "block";
        alertsFeedbackBox.className = "alert-feedback-box success";
        alertsFeedbackBox.textContent = "Alert configuration saved successfully!";
        setTimeout(() => { alertsFeedbackBox.style.display = "none"; }, 3500);
      }
    } catch (e) {
      alertsFeedbackBox.style.display = "block";
      alertsFeedbackBox.className = "alert-feedback-box error";
      alertsFeedbackBox.textContent = "Failed to save configuration.";
    }
  };
}

if (btnTestAlert) {
  btnTestAlert.onclick = async () => {
    alertsFeedbackBox.style.display = "block";
    alertsFeedbackBox.className = "alert-feedback-box";
    alertsFeedbackBox.textContent = "Dispatching test alert ping to your mobile phone...";

    try {
      const res = await fetch("/api/test-alert", { method: "POST" });
      const data = await res.json();

      let msg = "";
      let isSuccess = true;

      if (data.telegram) {
        if (data.telegram.success) {
          msg += "✅ Telegram: Delivered to phone! ";
        } else {
          msg += "❌ Telegram: " + data.telegram.message + " ";
          isSuccess = false;
        }
      }
      if (data.discord) {
        if (data.discord.success) {
          msg += "✅ Discord: Delivered to channel! ";
        } else {
          msg += "❌ Discord: " + data.discord.message + " ";
          isSuccess = false;
        }
      }
      if (data.warning) {
        msg = "⚠️ " + data.warning;
        isSuccess = false;
      }

      alertsFeedbackBox.className = isSuccess ? "alert-feedback-box success" : "alert-feedback-box error";
      alertsFeedbackBox.textContent = msg || "Test alert completed.";
      setTimeout(() => { alertsFeedbackBox.style.display = "none"; }, 6000);
    } catch (e) {
      alertsFeedbackBox.className = "alert-feedback-box error";
      alertsFeedbackBox.textContent = "Test ping request failed: " + e.message;
    }
  };
}

// =========================================================================
// 9. Interactive Controls (Pause, Audio, Filters, Search, Export)
// =========================================================================
btnPauseStream.onclick = () => {
  isPaused = !isPaused;
  if (isPaused) {
    pauseIcon.className = "fa-solid fa-play";
    pauseText.textContent = "Resume";
    btnPauseStream.classList.add("btn-primary");
    btnPauseStream.classList.remove("btn-secondary");
  } else {
    pauseIcon.className = "fa-solid fa-pause";
    pauseText.textContent = "Pause";
    btnPauseStream.classList.remove("btn-primary");
    btnPauseStream.classList.add("btn-secondary");
    reRenderTable();
  }
};

btnAudioToggle.onclick = () => {
  audioEnabled = !audioEnabled;
  if (audioEnabled) {
    audioIcon.className = "fa-solid fa-volume-high";
    audioText.textContent = "Audio";
  } else {
    audioIcon.className = "fa-solid fa-volume-xmark";
    audioText.textContent = "Muted";
  }
};

btnClearTable.onclick = () => {
  allFlows = [];
  reRenderTable();
};

filterTabsContainer.addEventListener("click", (e) => {
  const tab = e.target.closest(".filter-tab");
  if (!tab) return;
  filterTabsContainer.querySelectorAll(".filter-tab").forEach(t => t.classList.remove("active"));
  tab.classList.add("active");
  activeFilter = tab.dataset.filter;
  reRenderTable();
});

searchInput.addEventListener("input", (e) => {
  searchTerm = e.target.value.trim();
  reRenderTable();
});

// CSV Export
btnExportCsv.onclick = () => {
  if (allFlows.length === 0) {
    alert("No flows to export yet.");
    return;
  }
  const headers = ["timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "predicted_class", "confidence"];
  const rows = allFlows.map(f => [
    f.timestamp, f.src_ip, f.src_port, f.dst_ip, f.dst_port, f.protocol, f.predicted_class, f.confidence
  ]);
  const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ids_flow_log_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "_")}.csv`;
  a.click();
  URL.revokeObjectURL(url);
};

// PDF Incident Report Export
btnExportPdf.onclick = () => {
  if (allFlows.length === 0) {
    alert("No flows to generate report from.");
    return;
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();

  doc.setFillColor(15, 23, 42);
  doc.rect(0, 0, 210, 38, "F");

  doc.setTextColor(255, 255, 255);
  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  doc.text("IoT Intrusion Detection & Prevention System — Report", 14, 18);

  doc.setFontSize(9.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(148, 163, 184);
  doc.text(`Generated: ${new Date().toLocaleString()} | Active IPS Defense: ${ipsActive ? 'ENABLED' : 'DISABLED'}`, 14, 28);

  doc.setTextColor(15, 23, 42);
  doc.setFontSize(13);
  doc.setFont("helvetica", "bold");
  doc.text("Executive Session Summary", 14, 48);

  const attackRatio = totalFlows > 0 ? ((totalAttacks / totalFlows) * 100).toFixed(1) : "0.0";
  const summaryRows = [
    ["Total Monitored Flows", totalFlows.toString(), "Benign Traffic", `${counts.Benign} (${totalFlows > 0 ? ((counts.Benign/totalFlows)*100).toFixed(1) : 0}%)`],
    ["Total Attacks Detected", totalAttacks.toString(), "DDoS Attacks", `${counts.DDoS} (${totalFlows > 0 ? ((counts.DDoS/totalFlows)*100).toFixed(1) : 0}%)`],
    ["Overall Threat Ratio", `${attackRatio}%`, "DoS Attacks", `${counts.DoS} (${totalFlows > 0 ? ((counts.DoS/totalFlows)*100).toFixed(1) : 0}%)`],
    ["Quarantined Attackers", quarantinedIPsList.length.toString(), "Recon / PortScans", `${counts.Recon} (${totalFlows > 0 ? ((counts.Recon/totalFlows)*100).toFixed(1) : 0}%)`],
  ];

  doc.autoTable({
    startY: 53,
    head: [["Metric", "Value", "Attack Category", "Count (% Total)"]],
    body: summaryRows,
    theme: "striped",
    headStyles: { fillColor: [30, 41, 59], textColor: [255, 255, 255], fontStyle: "bold" },
    styles: { fontSize: 9.5, cellPadding: 3.5 }
  });

  doc.setFontSize(13);
  doc.setFont("helvetica", "bold");
  doc.text("Recent Classified Network Flows & IPS Actions", 14, doc.lastAutoTable.finalY + 12);

  const tableData = allFlows.slice(0, 40).map(f => [
    f.timestamp,
    `${f.src_ip}:${f.src_port}`,
    `${f.dst_ip}:${f.dst_port}`,
    f.protocol,
    f.predicted_class + (f.ips_blocked ? " [BLOCKED]" : ""),
    `${f.confidence}%`
  ]);

  doc.autoTable({
    startY: doc.lastAutoTable.finalY + 16,
    head: [["Time", "Source", "Destination", "Proto", "Verdict", "Confidence"]],
    body: tableData,
    theme: "grid",
    headStyles: { fillColor: [15, 23, 42], textColor: [255, 255, 255] },
    styles: { fontSize: 8, cellPadding: 2.5 },
    columnStyles: {
      4: {
        fontStyle: "bold",
        textColor: (cell) => (cell.raw.includes("Benign") ? [16, 185, 129] : [239, 68, 68])
      }
    }
  });

  const pageCount = doc.internal.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFontSize(8);
    doc.setTextColor(148, 163, 184);
    doc.text(`IoT Intrusion Detection & Prevention System — Confidential SOC Report — Page ${i} of ${pageCount}`, 14, 290);
  }

  doc.save(`ips_incident_report_${new Date().toISOString().slice(0, 10)}.pdf`);
};

// Initialize everything on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  initTopologyNetwork();
  loadIPSStatus();
});
