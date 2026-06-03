/* DRL Adaptive Beamforming Dashboard Controller (Plotly.js + WebSockets) */

// Global State
let ws = null;
let isDemoMode = true;
let isLiveStream = false;
let demoTimer = null;
let demoStep = 0;
let reconnectDelay = 1000;
let demoPolicy = 'oracle'; // oracle | random | fixed

// Rolling metrics history (last 200 points)
const MAX_HISTORY = 200;
const metricsHistory = {
    steps: [],
    reward: [],
    error: [],
    gain: []
};

// Colors matching our CSS variables
const COLORS = {
    cyan: '#00f0ff',
    blue: '#3a86ff',
    success: '#00ff88',
    danger: '#ff4466',
    gridColor: 'rgba(255, 255, 255, 0.05)',
    textColor: '#8892a4',
    paperBg: 'rgba(0, 0, 0, 0)',
    plotBg: 'rgba(0, 0, 0, 0)'
};

// UI Element Handles
const connBadge = document.getElementById('connection-status');
const algoBadge = document.getElementById('algo-badge');
const epVal = document.getElementById('episode-num');
const stepVal = document.getElementById('step-num');
const btnDemo = document.getElementById('btn-demo');
const speedSlider = document.getElementById('speed-slider');
const speedVal = document.getElementById('speed-val');

const footerGain = document.getElementById('stat-gain');
const footerError = document.getElementById('stat-error');
const footerReward = document.getElementById('stat-reward');
const targetTelemetry = document.getElementById('target-telemetry');
const policyBtnGroup = document.getElementById('policy-btn-group');
const policyHint = document.getElementById('policy-hint');
const policyButtons = document.querySelectorAll('.btn-policy');

// Initialize Charts on Page Load
window.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setupEventListeners();
    setActivePolicyButton(demoPolicy);
    startDemoMode(); // Start demo mode by default until WS connects
    connectWebSocket();
});

// Setup Slider and Button Listeners
function setupEventListeners() {
    btnDemo.addEventListener('click', () => {
        if (isDemoMode) {
            stopDemoMode();
        } else {
            startDemoMode();
        }
    });

    policyButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            if (isLiveStream) return;
            demoPolicy = btn.dataset.policy;
            setActivePolicyButton(demoPolicy);
            demoStep = 0;
        });
    });

    speedSlider.addEventListener('input', (e) => {
        speedVal.innerText = `${e.target.value}%`;
    });
}

function setActivePolicyButton(policy) {
    policyButtons.forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.policy === policy);
    });
}

function setPolicyControlsLive(live) {
    isLiveStream = live;
    policyBtnGroup.classList.toggle('disabled', live);
    if (live) {
        policyHint.innerHTML = 'Live stream active — policy is set by <code>visualize_env.py</code> or <code>train.py</code>';
    } else {
        policyHint.innerHTML = 'Live runs: use <code>python scripts/visualize_env.py --policy oracle</code>';
    }
}

function updateDemoToggleUI() {
    if (isDemoMode) {
        btnDemo.classList.add('btn-demo-on');
        btnDemo.classList.remove('btn-demo-off');
        btnDemo.querySelector('.btn-demo-label').innerText = 'DEMO RUNNING';
        btnDemo.querySelector('.btn-demo-hint').innerText = 'Click to pause demo and wait for live Python stream';
        btnDemo.setAttribute('aria-pressed', 'true');
    } else {
        btnDemo.classList.remove('btn-demo-on');
        btnDemo.classList.add('btn-demo-off');
        btnDemo.querySelector('.btn-demo-label').innerText = 'WAITING FOR LIVE';
        btnDemo.querySelector('.btn-demo-hint').innerText = 'Run visualize_env.py — or click to restart built-in demo';
        btnDemo.setAttribute('aria-pressed', 'false');
    }
}

function updateTargetTelemetry(tgt, targetAngles) {
    if (!tgt || tgt.x === undefined) return;
    const x = tgt.x;
    const y = tgt.y;
    const z = tgt.z;
    const rangeM = Math.sqrt(x * x + y * y * z * z);
    const rangeKm = rangeM / 1000.0;
    let azDeg;
    if (targetAngles && targetAngles.length >= 2) {
        azDeg = (targetAngles[1] * 180.0 / Math.PI) % 360.0;
    } else {
        azDeg = (Math.atan2(y, x) * 180.0 / Math.PI + 360.0) % 360.0;
    }
    let elDeg = 0;
    if (targetAngles && targetAngles.length >= 1) {
        elDeg = targetAngles[0] * 180.0 / Math.PI;
    } else if (rangeM > 0) {
        elDeg = Math.acos(Math.min(1, Math.max(-1, z / rangeM))) * 180.0 / Math.PI;
    }
    targetTelemetry.innerText =
        `Range: ${rangeKm.toFixed(2)} km · Altitude: ${z.toFixed(0)} m · Azimuth: ${azDeg.toFixed(1)}° · Elevation: ${elDeg.toFixed(1)}°`;
}

function syncPolicyHighlightFromAlgorithm(algorithm) {
    const algo = (algorithm || '').toUpperCase();
    if (algo.includes('ORACLE')) setActivePolicyButton('oracle');
    else if (algo.includes('RANDOM')) setActivePolicyButton('random');
    else if (algo.includes('FIXED')) setActivePolicyButton('fixed');
}

// ── WebSocket Connection ────────────────────────────────────────────────
function connectWebSocket() {
    console.log("Connecting to WebSocket server...");
    
    // Check hostname, fallback if needed
    const host = window.location.hostname || 'localhost';
    ws = new WebSocket(`ws://${host}:8765`);

    ws.onopen = () => {
        console.log("WebSocket connected.");
        stopDemoMode(); // Disable demo when real connection works
        setPolicyControlsLive(true);

        connBadge.className = 'status-badge connected';
        connBadge.querySelector('.badge-text').innerText = 'CONNECTED';
        reconnectDelay = 1000; // Reset backoff
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'state_update') {
                syncPolicyHighlightFromAlgorithm(data.algorithm);
                updateDashboard(data);
            }
        } catch (err) {
            console.error("Error parsing WS message:", err);
        }
    };

    ws.onclose = () => {
        console.log("WebSocket disconnected.");
        connBadge.className = 'status-badge disconnected';
        connBadge.querySelector('.badge-text').innerText = 'DISCONNECTED';
        setPolicyControlsLive(false);

        // Try to reconnect with backoff
        ws = null;
        setTimeout(() => {
            if (!isDemoMode && !ws) {
                connectWebSocket();
            }
        }, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 30000); // Backoff limit 30s
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
    };
}

// ── Chart Initialization ───────────────────────────────────────────────
function initCharts() {
    // 1. 3D Beam Pattern Layout
    const beamLayout = {
        paper_bgcolor: COLORS.paperBg,
        plot_bgcolor: COLORS.plotBg,
        font: { color: COLORS.textColor, family: 'Outfit, sans-serif' },
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
            xaxis: { title: 'X', gridcolor: COLORS.gridColor, showbackground: false },
            yaxis: { title: 'Y', gridcolor: COLORS.gridColor, showbackground: false },
            zaxis: { title: 'Z', gridcolor: COLORS.gridColor, showbackground: false },
            camera: {
                eye: { x: 1.25, y: 1.25, z: 1.25 }
            }
        },
        showlegend: false
    };
    Plotly.newPlot('beam-pattern-chart', [], beamLayout, { displayModeBar: false });

    // 2. 3D Space Tracker Layout
    const trackerLayout = {
        paper_bgcolor: COLORS.paperBg,
        plot_bgcolor: COLORS.plotBg,
        font: { color: COLORS.textColor, family: 'Outfit, sans-serif' },
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
            xaxis: { title: 'East X (m)', range: [-12000, 12000], gridcolor: COLORS.gridColor, showbackground: false },
            yaxis: { title: 'North Y (m)', range: [-12000, 12000], gridcolor: COLORS.gridColor, showbackground: false },
            zaxis: { title: 'Up Z · radar at 0 (m)', range: [0, 12000], gridcolor: COLORS.gridColor, showbackground: false },
            camera: {
                eye: { x: 1.5, y: 1.5, z: 0.8 }
            },
            annotations: [{
                x: 0, y: 0, z: 0,
                text: 'RADAR',
                showarrow: false,
                font: { color: '#e6edf7', size: 11 },
                bgcolor: 'rgba(0,0,0,0.6)',
            }],
        },
        showlegend: false
    };
    Plotly.newPlot('target-tracker-chart', [], trackerLayout, { displayModeBar: false });

    // 3. Phase Heatmap Layout
    const heatmapLayout = {
        paper_bgcolor: COLORS.paperBg,
        plot_bgcolor: COLORS.plotBg,
        font: { color: COLORS.textColor, family: 'Outfit, sans-serif' },
        margin: { l: 40, r: 20, t: 15, b: 30 },
        xaxis: { gridcolor: COLORS.gridColor, tickmode: 'linear', showgrid: false },
        yaxis: { gridcolor: COLORS.gridColor, tickmode: 'linear', showgrid: false, autorange: 'reversed' },
    };
    Plotly.newPlot('phases-heatmap', [], heatmapLayout, { displayModeBar: false });

    // 4. Performance Curves Layout
    const perfLayout = {
        paper_bgcolor: COLORS.paperBg,
        plot_bgcolor: COLORS.plotBg,
        font: { color: COLORS.textColor, family: 'Outfit, sans-serif' },
        margin: { l: 50, r: 50, t: 15, b: 40 },
        xaxis: { title: 'Steps', gridcolor: COLORS.gridColor },
        yaxis: { title: 'Gain / Reward', range: [-0.5, 1.2], gridcolor: COLORS.gridColor },
        yaxis2: {
            title: 'Error (deg)',
            overlaying: 'y',
            side: 'right',
            range: [-5, 60],
            gridcolor: COLORS.gridColor,
            showgrid: false
        },
        legend: { x: 0.05, y: 1.0, orientation: 'h' }
    };
    Plotly.newPlot('performance-chart', [], perfLayout, { displayModeBar: false });
}

// ── Dashboard Live Update Handler ──────────────────────────────────────
function updateDashboard(data) {
    // 1. Text Indicators
    algoBadge.innerText = data.algorithm || 'SAC';
    epVal.innerText = data.episode || 0;
    stepVal.innerText = data.step || 0;

    const gain = data.metrics.gain;
    const error = data.metrics.error_deg;
    const reward = data.metrics.reward;

    footerGain.innerText = gain.toFixed(3);
    footerError.innerText = `${error.toFixed(2)}°`;
    footerReward.innerText = reward.toFixed(3);

    // Apply colored bounds styling to tracking error
    if (error < 5.0) {
        footerError.className = 'stat-val text-success';
    } else if (error < 15.0) {
        footerError.className = 'stat-val text-cyan';
    } else {
        footerError.className = 'stat-val text-danger';
    }

    // ── 2. Update Performance Curves ────────────────────────────────────
    updatePerformancePlot(data.step, reward, error, gain);

    // ── 3. Update Heatmap phases ────────────────────────────────────────
    if (data.phases) {
        const heatmapData = [{
            z: data.phases,
            type: 'heatmap',
            colorscale: 'HSV', // Circular phase scale
            zmin: -Math.PI,
            zmax: Math.PI,
            showscale: false
        }];
        Plotly.react('phases-heatmap', heatmapData, document.getElementById('phases-heatmap').layout);
    }

    // ── 4. Update Target & Vector Tracker ──────────────────────────────
    if (data.target) {
        updateTargetTracker(data);
        updateTargetTelemetry(data.target, data.target_angles);
    }

    // ── 5. Update 3D Beam Pattern ───────────────────────────────────────
    if (data.beam_pattern) {
        const bp = data.beam_pattern;
        
        // Reconstruct X, Y, Z matrices from spherical coordinates in JSON
        const theta = bp.theta; // (T,)
        const phi = bp.phi;     // (P,)
        const power = bp.power; // (T, P) grid

        const X = [];
        const Y = [];
        const Z = [];

        for (let i = 0; i < theta.length; i++) {
            const rowX = [];
            const rowY = [];
            const rowZ = [];
            const th = theta[i];
            
            for (let j = 0; j < phi.length; j++) {
                const ph = phi[j];
                const r = power[i][j]; // normalized gain radius

                // Cartesian projection
                rowX.push(r * Math.sin(th) * Math.cos(ph));
                rowY.push(r * Math.sin(th) * Math.sin(ph));
                rowZ.push(r * Math.cos(th));
            }
            X.push(rowX);
            Y.push(rowY);
            Z.push(rowZ);
        }

        // Draw a target direction marker dot in the beam pattern plot
        const targetRadius = 1.05; // draw target outside the pattern surface
        const t_theta = data.target_angles[0];
        const t_phi = data.target_angles[1];
        const tx_dir = [targetRadius * Math.sin(t_theta) * Math.cos(t_phi)];
        const ty_dir = [targetRadius * Math.sin(t_theta) * Math.sin(t_phi)];
        const tz_dir = [targetRadius * Math.cos(t_theta)];

        const beamData = [
            {
                x: X, y: Y, z: Z,
                type: 'surface',
                colorscale: 'Electric',
                showscale: false
            },
            {
                x: tx_dir, y: ty_dir, z: tz_dir,
                type: 'scatter3d',
                mode: 'markers',
                marker: { color: COLORS.cyan, size: 10, symbol: 'circle' }
            }
        ];
        Plotly.react('beam-pattern-chart', beamData, document.getElementById('beam-pattern-chart').layout);
    }
}

function updatePerformancePlot(step, reward, error, gain) {
    metricsHistory.steps.push(step);
    metricsHistory.reward.push(reward);
    metricsHistory.error.push(error);
    metricsHistory.gain.push(gain);

    // Keep history bounds
    if (metricsHistory.steps.length > MAX_HISTORY) {
        metricsHistory.steps.shift();
        metricsHistory.reward.shift();
        metricsHistory.error.shift();
        metricsHistory.gain.shift();
    }

    const data = [
        {
            x: metricsHistory.steps,
            y: metricsHistory.gain,
            type: 'scatter',
            mode: 'lines',
            name: 'Normalized Gain',
            line: { color: COLORS.success, width: 2 }
        },
        {
            x: metricsHistory.steps,
            y: metricsHistory.reward,
            type: 'scatter',
            mode: 'lines',
            name: 'Reward',
            line: { color: COLORS.cyan, width: 2 }
        },
        {
            x: metricsHistory.steps,
            y: metricsHistory.error,
            type: 'scatter',
            mode: 'lines',
            name: 'Error (deg)',
            yaxis: 'y2',
            line: { color: COLORS.danger, width: 2, dash: 'dot' }
        }
    ];

    Plotly.react('performance-chart', data, document.getElementById('performance-chart').layout);
}

function updateTargetTracker(data) {
    const tgt = data.target;
    const traj = tgt.trajectory || [];
    const tx = traj.map((p) => p[0]);
    const ty = traj.map((p) => p[1]);
    const tz = traj.map((p) => p[2]);

    const beamVectorLength = 8000;
    const bTipX = beamVectorLength * Math.sin(data.beam_angles[0]) * Math.cos(data.beam_angles[1]);
    const bTipY = beamVectorLength * Math.sin(data.beam_angles[0]) * Math.sin(data.beam_angles[1]);
    const bTipZ = beamVectorLength * Math.cos(data.beam_angles[0]);

    const trackerData = [
        {
            x: tx, y: ty, z: tz,
            type: 'scatter3d',
            mode: 'lines',
            line: { color: COLORS.blue, width: 5 },
            name: 'Flight path',
            hovertemplate: 'Trail<br>X:%{x:.0f} Y:%{y:.0f} Z:%{z:.0f}<extra></extra>',
        },
        {
            x: [0], y: [0], z: [0],
            type: 'scatter3d',
            mode: 'markers+text',
            marker: { color: '#e6edf7', size: 7, symbol: 'diamond' },
            text: ['RADAR'],
            textfont: { color: '#e6edf7', size: 12 },
            textposition: 'top center',
            name: 'Radar',
            hovertemplate: 'Phased array at origin<extra></extra>',
        },
        {
            x: [tgt.x], y: [tgt.y], z: [tgt.z],
            type: 'scatter3d',
            mode: 'markers+text',
            marker: { color: COLORS.cyan, size: 10, symbol: 'circle' },
            text: ['TARGET'],
            textfont: { color: COLORS.cyan, size: 12 },
            textposition: 'top center',
            name: 'Target',
            hovertemplate: 'Aircraft<br>Range %{customdata:.2f} km<extra></extra>',
            customdata: [Math.sqrt(tgt.x ** 2 + tgt.y ** 2 + tgt.z ** 2) / 1000],
        },
        {
            x: [0, bTipX], y: [0, bTipY], z: [0, bTipZ],
            type: 'scatter3d',
            mode: 'lines+text',
            line: { color: COLORS.success, width: 7, dash: 'dash' },
            text: ['', 'BEAM'],
            textfont: { color: COLORS.success, size: 12 },
            textposition: 'top center',
            name: 'Beam',
            hovertemplate: 'Beam pointing direction<extra></extra>',
        },
    ];
    Plotly.react('target-tracker-chart', trackerData, document.getElementById('target-tracker-chart').layout);
}

// ── local Demo Mode (Mock simulation loops for UI validation) ───────
function startDemoMode() {
    isDemoMode = true;
    setPolicyControlsLive(false);
    updateDemoToggleUI();

    // Clear old metric history
    metricsHistory.steps = [];
    metricsHistory.reward = [];
    metricsHistory.error = [];
    metricsHistory.gain = [];

    demoStep = 0;
    runDemoLoop();
}

function stopDemoMode() {
    isDemoMode = false;
    updateDemoToggleUI();

    if (demoTimer) {
        clearTimeout(demoTimer);
        demoTimer = null;
    }

    if (!ws) {
        connectWebSocket();
    }
}

function runDemoLoop() {
    if (!isDemoMode) return;

    demoStep++;
    
    // 1. Generate target angles along a complex trajectory
    const t_theta = 0.3 + 0.15 * Math.sin(demoStep * 0.05) + 0.05 * Math.cos(demoStep * 0.12);
    const t_phi = (demoStep * 0.02) % (2.0 * Math.PI);

    let b_theta;
    let b_phi;
    if (demoPolicy === 'oracle') {
        b_theta = t_theta;
        b_phi = t_phi;
    } else if (demoPolicy === 'fixed') {
        b_theta = 0.35;
        b_phi = 0.2;
    } else {
        b_theta = t_theta + 0.25 * Math.sin(demoStep * 0.11);
        b_phi = t_phi + 0.3 * Math.cos(demoStep * 0.09);
    }

    const angularError = Math.sqrt((t_theta - b_theta) ** 2 + (Math.sin(t_theta) * (t_phi - b_phi)) ** 2);
    const errorDeg = angularError * (180.0 / Math.PI);
    
    // Gain drops off rapidly away from main lobe center
    const normalizedGain = Math.max(0.01, Math.exp(-15.0 * (angularError**2)));
    
    const reward = normalizedGain - 0.3 * angularError + 0.01;

    // 2. Generate sinc-like beam pattern surface
    const thetaGrid = [];
    const phiGrid = [];
    const powerGrid = [];
    const gridRes = 25;

    for (let i = 0; i <= gridRes; i++) {
        thetaGrid.push((i / gridRes) * (Math.PI / 2.0));
    }
    for (let j = 0; j <= gridRes * 2; j++) {
        phiGrid.push((j / (gridRes * 2)) * (2.0 * Math.PI));
    }

    for (let i = 0; i < thetaGrid.length; i++) {
        const row = [];
        const th = thetaGrid[i];
        for (let j = 0; j < phiGrid.length; j++) {
            const ph = phiGrid[j];
            
            // Angular distance between grid point (th, ph) and steered beam peak (b_theta, b_phi)
            const dot = Math.sin(th) * Math.sin(b_theta) * Math.cos(ph - b_phi) + Math.cos(th) * Math.cos(b_theta);
            const dist = Math.acos(Math.min(1.0, Math.max(-1.0, dot)));
            
            // Simulates narrow main lobe (8x8 elements) + side lobes
            const x_sinc = 7.0 * dist;
            const AF_power = x_sinc === 0 ? 1.0 : (Math.sin(x_sinc) / x_sinc)**2;
            const gainVal = 0.1 * AF_power + 0.9 * Math.exp(-2.0 * dist);
            row.push(gainVal);
        }
        powerGrid.push(row);
    }

    // 3. Spiral target trajectory (helix)
    const trajectory = [];
    const helixRadius = 6000;
    for (let i = Math.max(0, demoStep - 50); i <= demoStep; i++) {
        const angle = i * 0.05;
        trajectory.push([
            helixRadius * Math.sin(angle),
            helixRadius * Math.cos(angle),
            4000 + i * 25
        ]);
    }
    const currentPos = trajectory[trajectory.length - 1];

    // 4. Heatmap Phase Weights
    const phaseMatrix = [];
    for (let m = 0; m < 8; m++) {
        const row = [];
        for (let n = 0; n < 8; n++) {
            // Planar progressive phase profile steered towards b_theta, b_phi
            // beta = -k * d * (m*sin(th)*cos(ph) + n*sin(th)*sin(ph))
            const phase = -Math.PI * (m * Math.sin(b_theta) * Math.cos(b_phi) + n * Math.sin(b_theta) * Math.sin(b_phi));
            // Wrap to [-pi, pi]
            const phaseWrapped = (phase + Math.PI) % (2.0 * Math.PI) - Math.PI;
            row.push(phaseWrapped);
        }
        phaseMatrix.push(row);
    }

    // Assemble and render demo frame
    const policyLabels = { oracle: 'DEMO · ORACLE', random: 'DEMO · RANDOM', fixed: 'DEMO · FIXED' };
    const demoData = {
        type: 'state_update',
        algorithm: policyLabels[demoPolicy] || 'DEMO',
        episode: Math.floor(demoStep / 500) + 1,
        step: demoStep % 500,
        metrics: {
            gain: normalizedGain,
            error_deg: errorDeg,
            reward: reward
        },
        phases: phaseMatrix,
        beam_angles: [b_theta, b_phi],
        target_angles: [t_theta, t_phi],
        target: {
            x: currentPos[0],
            y: currentPos[1],
            z: currentPos[2],
            trajectory: trajectory
        },
        beam_pattern: {
            theta: thetaGrid,
            phi: phiGrid,
            power: powerGrid
        }
    };

    updateDashboard(demoData);

    // Schedule next frame based on speed slider
    const speed = speedSlider.value; // 1 to 100
    const delay = Math.max(10, 200 - (speed * 1.9)); // map to 10ms - 200ms
    demoTimer = setTimeout(runDemoLoop, delay);
}
