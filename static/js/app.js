document.addEventListener('DOMContentLoaded', () => {
    // ─── FFmpeg availability check ───
    (async () => {
        try {
            const res = await fetch('/api/ffmpeg-status');
            const status = await res.json();
            if (!status.ffmpeg || !status.ffprobe) {
                const banner = document.createElement('div');
                banner.style.cssText = `
                    position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
                    background: linear-gradient(90deg, #7f1d1d, #991b1b);
                    color: #fecaca; padding: 12px 24px;
                    display: flex; align-items: center; justify-content: space-between;
                    font-size: 0.9rem; font-family: 'Inter', sans-serif; font-weight: 500;
                    border-bottom: 1px solid #ef4444; box-shadow: 0 4px 20px rgba(239,68,68,0.3);
                `;
                banner.innerHTML = `
                    <div style="display:flex;align-items:center;gap:12px">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f87171" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                        <span><strong>FFmpeg not found on PATH.</strong> Frame extraction will fail — videos won't be fingerprinted. 
                        Add FFmpeg to your system PATH (found at <code style="background:rgba(0,0,0,0.3);padding:1px 6px;border-radius:3px">${status.ffmpeg_path}</code>).</span>
                    </div>
                    <button onclick="this.parentElement.remove()" style="background:none;border:none;color:#fca5a5;font-size:1.3rem;cursor:pointer;padding:0 4px;line-height:1">&times;</button>
                `;
                document.body.prepend(banner);
                // Push header down
                document.querySelector('.app-header').style.marginTop = '50px';
            } else {
                console.log(`✓ FFmpeg OK: ${status.ffmpeg_path}`);
            }
        } catch (e) {
            console.warn('Could not check FFmpeg status:', e);
        }
    })();
    // ─── State ───
    let folders = [];
    let pollInterval = null;
    let currentResults = null;
    let currentView = 'groups';

    // ─── DOM Elements ───
    const panels = {
        setup: document.getElementById('panel-setup'),
        progress: document.getElementById('panel-progress'),
        results: document.getElementById('panel-results')
    };

    // Setup
    const folderInput = document.getElementById('folder-input');
    const folderList = document.getElementById('folder-list');
    const btnAddFolder = document.getElementById('btn-add-folder');
    const btnAddFolderInput = document.getElementById('btn-add-folder-input');
    const btnStartScan = document.getElementById('btn-start-scan');

    // Settings
    const inputs = {
        threshold: document.getElementById('threshold'),
        numFrames: document.getElementById('num-frames'),
        batchSize: document.getElementById('batch-size')
    };
    const values = {
        threshold: document.getElementById('threshold-value'),
        numFrames: document.getElementById('num-frames-value'),
        batchSize: document.getElementById('batch-size-value')
    };

    // Progress
    const progTitle = document.getElementById('progress-title');
    const progMsg = document.getElementById('progress-message');
    const progFill = document.getElementById('progress-fill');
    const progCount = document.getElementById('progress-count');
    const progElapsed = document.getElementById('progress-elapsed');
    const progFile = document.getElementById('progress-current-file');
    const btnPauseScan = document.getElementById('btn-pause-scan');
    const txtPauseScan = document.getElementById('txt-pause-scan');
    const btnAbortScan = document.getElementById('btn-abort-scan');

    // Results
    const resultsStats = document.getElementById('results-stats');
    const resultsContent = document.getElementById('results-content');
    const btnNewScan = document.getElementById('btn-new-scan');

    // Modals
    const modalSystem = document.getElementById('modal-system');
    const modalPreview = document.getElementById('modal-preview');
    const btnSystemInfo = document.getElementById('btn-system-info');
    const btnHardReset = document.getElementById('btn-hard-reset');

    // ─── Utility Functions ───

    const showPanel = (panelName) => {
        Object.values(panels).forEach(p => p.classList.remove('active'));
        panels[panelName].classList.add('active');
    };

    const formatDuration = (sec) => {
        const h = Math.floor(sec / 3600);
        const m = Math.floor((sec % 3600) / 60);
        const s = Math.floor(sec % 60);
        return h > 0 ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
            : `${m}:${s.toString().padStart(2, '0')}`;
    };

    // ─── Setup Logic ───

    const addFolder = (path) => {
        path = path.trim();
        if (!path || folders.includes(path)) return;
        folders.push(path);
        renderFolders();
        btnStartScan.disabled = folders.length === 0;
        folderInput.value = '';
    };

    const removeFolder = (index) => {
        folders.splice(index, 1);
        renderFolders();
        btnStartScan.disabled = folders.length === 0;
    };

    const renderFolders = () => {
        if (folders.length === 0) {
            folderList.innerHTML = `
                <div class="folder-empty">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.4"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
                    <p>No folders added yet. Type a folder path below.</p>
                </div>
            `;
            return;
        }

        folderList.innerHTML = folders.map((f, i) => `
            <div class="folder-item">
                <div class="folder-path">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
                    ${f}
                </div>
                <button class="btn-remove-folder" onclick="removeFolder(${i})" title="Remove">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                </button>
            </div>
        `).join('');
    };

    window.removeFolder = removeFolder; // Expose to inline onclick

    // ─── Native folder picker (pywebview api or Tkinter fallback) ───
    const pickFolderNative = async () => {
        // 1. pywebview desktop mode → call Python API directly
        if (window.pywebview && window.pywebview.api && window.pywebview.api.pick_folder) {
            try {
                const path = await window.pywebview.api.pick_folder();
                if (path) addFolder(path);
                return;
            } catch (e) {
                console.warn('pywebview pick_folder failed, falling back', e);
            }
        }
        // 2. Browser mode → ask Flask to open Tkinter dialog on server side
        try {
            const res = await fetch('/api/pick-folder', { method: 'POST' });
            const data = await res.json();
            if (data.path) addFolder(data.path);
        } catch (e) {
            console.error('Folder picker error', e);
        }
    };

    btnAddFolder.addEventListener('click', () => pickFolderNative());
    btnAddFolderInput.addEventListener('click', () => addFolder(folderInput.value));
    folderInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') addFolder(folderInput.value);
    });

    // Settings Listeners
    inputs.threshold.addEventListener('input', (e) => {
        values.threshold.textContent = `${Math.round(e.target.value * 100)}%`;
    });
    inputs.numFrames.addEventListener('input', (e) => {
        values.numFrames.textContent = e.target.value;
    });
    inputs.batchSize.addEventListener('input', (e) => {
        values.batchSize.textContent = e.target.value;
    });

    // ─── Scanning Logic ───

    btnStartScan.addEventListener('click', async () => {
        try {
            btnStartScan.disabled = true;
            btnStartScan.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spinning"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Starting...`;

            const res = await fetch('/api/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    folders: folders,
                    threshold: parseFloat(inputs.threshold.value),
                    num_frames: parseInt(inputs.numFrames.value),
                    batch_size: parseInt(inputs.batchSize.value)
                })
            });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.error || 'Failed to start scan');
            }

            // Transition to progress panel
            showPanel('progress');
            startPolling();

        } catch (err) {
            alert(`Error: ${err.message}`);
            btnStartScan.disabled = false;
            btnStartScan.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Start Scan`;
        }
    });

    const startPolling = () => {
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(async () => {
            try {
                const res = await fetch('/api/status');
                const state = await res.json();
                updateProgressUI(state);

                if (state.status === 'done' || state.status === 'error' || state.status === 'aborted') {
                    clearInterval(pollInterval);
                    if (state.status === 'done') {
                        setTimeout(loadResults, 1000);
                    } else if (state.status === 'aborted') {
                        resetScan();
                    } else {
                        alert(`Scan error: ${state.error}`);
                        resetScan();
                    }
                }
            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 1000);
    };

    // Pause/Resume Logic
    if (btnPauseScan) {
        btnPauseScan.addEventListener('click', async () => {
            if (txtPauseScan.textContent === 'Pause') {
                await fetch('/api/pause', { method: 'POST' });
                // We don't change UI manually, polling will pick up status="paused"
            } else {
                // Resume
                try {
                    const res = await fetch('/api/scan', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ resume: true })
                    });
                    if (!res.ok) {
                        const data = await res.json();
                        alert(`Cannot resume: ${data.error}`);
                    }
                } catch (e) {
                    console.error('Resume error', e);
                }
            }
        });
    }

    if (btnAbortScan) {
        btnAbortScan.addEventListener('click', async () => {
            if (confirm("Are you sure you want to completely abort this scan?")) {
                await fetch('/api/abort', { method: 'POST' });
            }
        });
    }

    const updateProgressUI = (state) => {
        progTitle.textContent = state.status === 'done' ? 'Complete!' :
            state.status.charAt(0).toUpperCase() + state.status.slice(1) + '...';
        progMsg.textContent = state.message;

        // GPU Cat logic
        const gpuCat = document.getElementById('gpu-cat');
        const gpuText = document.getElementById('gpu-cat-text');
        if (['done', 'idle', 'aborted', 'error'].includes(state.status)) {
            if (gpuCat) gpuCat.style.display = 'none';
            if (gpuText) gpuText.style.display = 'none';
        } else {
            if (gpuCat) gpuCat.style.display = 'block';
            if (gpuText) {
                gpuText.style.display = 'inline';
                const util = state.gpu_util || 0;
                gpuText.textContent = `${Math.round(util)}%`;

                gpuCat.classList.remove('cat-slow', 'cat-medium', 'cat-fast');
                // <50 slow run, >=51 medium run >75 fast  run
                if (util > 75) {
                    gpuCat.classList.add('cat-fast');
                } else if (util >= 51) {
                    gpuCat.classList.add('cat-medium');
                } else {
                    gpuCat.classList.add('cat-slow');
                }
            }
        }

        let pct = 0;
        if (state.total > 0) {
            pct = (state.progress / state.total) * 100;
        } else if (state.status === 'done') {
            pct = 100;
        }

        progFill.style.width = `${pct}%`;
        progCount.textContent = `${state.progress} / ${state.total || '?'}`;
        progElapsed.textContent = formatDuration(state.elapsed || 0);
        progFile.textContent = state.current_file || '—';

        if (state.status === 'paused') {
            btnPauseScan.classList.replace('btn-outline', 'btn-primary');
            txtPauseScan.textContent = 'Resume';
            btnPauseScan.querySelector('.icon-pause').style.display = 'none';
            btnPauseScan.querySelector('.icon-resume').style.display = 'block';
            progTitle.textContent = 'Paused';
        } else {
            btnPauseScan.classList.replace('btn-primary', 'btn-outline');
            txtPauseScan.textContent = 'Pause';
            btnPauseScan.querySelector('.icon-pause').style.display = 'block';
            btnPauseScan.querySelector('.icon-resume').style.display = 'none';
        }

        // Update phases
        const phases = ['scanning', 'extracting', 'hashing', 'processing', 'comparing'];
        let currentIndex = phases.indexOf(state.status);
        if (state.status === 'done') currentIndex = 4;

        phases.forEach((phase, idx) => {
            const el = document.getElementById(`phase-${phase}`);
            if (!el) return;

            el.classList.remove('active', 'completed');
            if (idx < currentIndex) el.classList.add('completed');
            else if (idx === currentIndex) el.classList.add('active');
        });
    };

    // ─── Results Logic ───

    const loadResults = async () => {
        try {
            const res = await fetch('/api/results');
            currentResults = await res.json();
            renderResults();
            showPanel('results');
        } catch (err) {
            alert(`Error loading results: ${err.message}`);
        }
    };

    const renderResults = () => {
        const { stats, groups } = currentResults;

        // Render Stats
        resultsStats.innerHTML = `
            <div class="stat-box">
                <div class="stat-value highlight">${stats.duplicate_groups}</div>
                <div class="stat-label">Duplicate Groups</div>
            </div>
            <div class="stat-box">
                <div class="stat-value highlight">${stats.total_duplicates}</div>
                <div class="stat-label">Extra Files</div>
            </div>
            <div class="stat-box">
                <div class="stat-value highlight">${stats.potential_savings}</div>
                <div class="stat-label">Recoverable Space</div>
            </div>
            <div class="stat-box">
                <div class="stat-value text-muted">${stats.total_videos}</div>
                <div class="stat-label">Total Videos Scanned</div>
            </div>
        `;

        if (groups.length === 0) {
            resultsContent.innerHTML = `
                <div style="text-align:center; padding: 60px 20px; color: var(--text-muted);">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" style="margin-bottom:20px; color: var(--success);"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                    <h2 style="color:var(--text-primary); margin-bottom:10px;">No duplicates found!</h2>
                    <p>Your video collection is perfectly clean.</p>
                </div>
            `;
            return;
        }

        if (currentView === 'all') {
            const allVideos = state.videos || [];
            resultsContent.innerHTML = `
                <div class="group-card">
                    <div class="group-header">
                        <div class="group-title">
                            All Scanned Videos
                            <span class="group-badge">${allVideos.length} Files</span>
                        </div>
                    </div>
                    <div class="video-list">
                        ${allVideos.map((v, i) => renderVideoItem(v, true, null)).join('')}
                    </div>
                </div>
            `;
            return;
        }

        // Render Groups
        resultsContent.innerHTML = groups.map((g, idx) => {
            const maxSim = g.max_similarity || (g.videos[1] ? g.videos[1].similarity : 0) || 0;
            const simStr = isNaN(maxSim) ? '0.00' : (maxSim * 100).toFixed(2);

            return `
                <div class="group-card" style="animation-delay: ${idx * 0.1}s">
                    <div class="group-header">
                        <div class="group-title">
                            Group ${idx + 1} 
                            <span class="group-badge">${g.videos.length} Files</span>
                        </div>
                        <div class="group-sim">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                            ${simStr}% Match
                        </div>
                    </div>
                    <div class="video-list">
                        ${g.videos.map((v, vidx) => renderVideoItem(v, vidx === 0, g.group_id)).join('')}
                    </div>
                </div>
            `;
        }).join('');
    };

    const renderVideoItem = (v, isOriginal, groupId) => {
        return `
            <div class="video-item ${isOriginal ? 'original' : ''}" id="vid-${v.index}">
                <div class="video-thumb-cont" onclick="previewVideo('${v.path.replace(/\\/g, '\\\\')}')">
                    <img src="/api/thumbnail/${v.index}" class="video-thumb" onload="this.classList.add('loaded')" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100%\\' height=\\'100%\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'%23334155\\' stroke-width=\\'1\\'%3E%3Crect x=\\'2\\' y=\\'2\\' width=\\'20\\' height=\\'20\\' rx=\\'2.18\\' ry=\\'2.18\\'/%3E%3Cline x1=\\'7\\' y1=\\'2\\' x2=\\'7\\' y2=\\'22\\'/%3E%3Cline x1=\\'17\\' y1=\\'2\\' x2=\\'17\\' y2=\\'22\\'/%3E%3C/svg%3E'">
                    <div class="play-overlay">
                        <div class="play-btn">
                            <svg width="24" height="24" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                        </div>
                    </div>
                    <div class="video-duration">${v.duration_str}</div>
                </div>
                <div class="video-info">
                    <div class="video-name" title="${v.name}" onclick="openFile('${v.path.replace(/\\/g, '\\\\')}')">${v.name}</div>
                    <div class="video-meta">
                        <span class="meta-item text-primary">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/></svg>
                            ${v.resolution}
                        </span>
                        <span class="meta-item">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                            ${v.size_str}
                        </span>
                        ${!isOriginal ? `
                        <span class="meta-item text-warning">
                            Match: ${isNaN(v.similarity) ? '0.00' : (v.similarity * 100).toFixed(2)}%
                        </span>` : ''}
                    </div>
                    <div class="video-path" onclick="openFolder('${v.path.replace(/\\/g, '\\\\')}')" title="Open in Explorer">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
                        ${v.folder}
                    </div>
                </div>
                <div class="video-actions">
                    <button class="btn btn-outline btn-sm" onclick="openFile('${v.path.replace(/\\/g, '\\\\')}')">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                        Play
                    </button>
                    ${!isOriginal ? `
                    <button class="btn btn-danger btn-sm" onclick="deleteFile('${v.path.replace(/\\/g, '\\\\')}', ${v.index})">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
                        Delete
                    </button>
                    ` : ''}
                </div>
            </div>
        `;
    };

    btnNewScan.addEventListener('click', () => resetScan());

    const resetScan = async () => {
        try {
            await fetch('/api/reset', { method: 'POST' });
            if (pollInterval) clearInterval(pollInterval);
            currentResults = null;
            btnStartScan.disabled = folders.length === 0;
            btnStartScan.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Start Scan`;
            showPanel('setup');
        } catch (err) {
            console.error(err);
            showPanel('setup');
        }
    };

    // ─── File Operations (Globally exposed for onclick) ───

    window.openFolder = async (path) => {
        try {
            await fetch('/api/open-folder', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });
        } catch (err) {
            console.error('Error opening folder', err);
        }
    };

    window.openFile = async (path) => {
        try {
            await fetch('/api/open-file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });
        } catch (err) {
            console.error('Error opening file', err);
        }
    };

    window.deleteFile = async (path, vidIndex) => {
        if (!confirm('Are you sure you want to delete this video?\n(It will be moved to the Recycle Bin if possible)')) return;

        try {
            const res = await fetch('/api/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            });

            if (res.ok) {
                // Remove from UI
                const el = document.getElementById(`vid-${vidIndex}`);
                if (el) {
                    el.style.opacity = '0';
                    el.style.transform = 'scale(0.95)';
                    setTimeout(() => el.remove(), 300);
                }
            } else {
                const data = await res.json();
                alert(`Delete failed: ${data.error}`);
            }
        } catch (err) {
            alert(`Error deleting file: ${err.message}`);
        }
    };

    window.previewVideo = (path) => {
        // Can't easily preview arbitrary local files in browser due to security.
        // Easiest is just to open it in default local player.
        window.openFile(path);
    };

    // ─── Modal / System Info ───

    const openModal = (modal) => {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    };

    const closeModal = (modal) => {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    };

    document.querySelectorAll('.modal-close, .modal-backdrop').forEach(el => {
        el.addEventListener('click', function () {
            closeModal(this.closest('.modal'));
        });
    });

    btnSystemInfo.addEventListener('click', async () => {
        openModal(modalSystem);
        const body = document.getElementById('system-info-body');
        body.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';

        try {
            const res = await fetch('/api/system-info');
            const contentType = res.headers.get('content-type') || '';
            if (!contentType.includes('application/json')) {
                throw new Error(`Server returned non-JSON response (status ${res.status}). Check console.`);
            }
            const info = await res.json();

            let html = `<table class="sys-info-table"><tbody>`;
            html += `<tr><th>CUDA Available</th><td>${info.cuda_available ? '<span class="text-success">Yes</span>' : '<span class="text-danger">No (CPU Fallback)</span>'}</td></tr>`;
            html += `<tr><th>PyTorch Version</th><td>${info.torch_version || 'N/A'}</td></tr>`;

            if (info.cuda_available) {
                html += `<tr><th>CUDA Version</th><td>${info.cuda_version || 'N/A'}</td></tr>`;
                html += `<tr><th>cuDNN Version</th><td>${info.cudnn_version || 'N/A'}</td></tr>`;
                html += `<tr><th>Device Count</th><td>${info.device_count}</td></tr>`;
                html += `</tbody></table>`;

                info.devices.forEach((d, i) => {
                    html += `
                        <div class="gpu-card">
                            <div class="gpu-name">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>
                                GPU ${i}: ${d.name}
                            </div>
                            <table class="sys-info-table" style="background: transparent; border: none;">
                                <tr><th style="padding:4px 0; border:none; width: 50%;">VRAM</th><td style="padding:4px 0; border:none;">${d.total_memory}</td></tr>
                                <tr><th style="padding:4px 0; border:none;">Compute Capability</th><td style="padding:4px 0; border:none;">${d.compute_capability}</td></tr>
                            </table>
                        </div>
                    `;
                });
            } else {
                html += `</tbody></table>`;
                html += `<p class="text-warning" style="margin-top: 16px; font-size: 0.9rem;">No CUDA GPU detected. Processing will use the CPU — significantly slower for large libraries.</p>`;
            }

            body.innerHTML = html;

        } catch (err) {
            body.innerHTML = `<div class="text-danger" style="padding: 8px 0;">Error loading system info: ${err.message}</div>`;
        }
    });

    btnHardReset.addEventListener('click', async () => {
        if (!confirm('Are you absolutely sure? This will wipe your folders, settings, and all scan data.')) return;
        try {
            await fetch('/api/hard-reset', { method: 'POST' });
            folders = [];
            currentResults = null;
            if (pollInterval) clearInterval(pollInterval);
            renderFolders();
            btnStartScan.disabled = true;
            showPanel('setup');
            // Refresh settings to defaults
            inputs.threshold.value = "0.88";
            values.threshold.innerText = "88%";
            inputs.numFrames.value = "32";
            values.numFrames.innerText = "32";
        } catch (err) {
            console.error("Hard reset failed", err);
        }
    });

    const resultsTabs = document.querySelectorAll('.results-tabs .tab');
    resultsTabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            resultsTabs.forEach(t => t.classList.remove('active'));
            e.currentTarget.classList.add('active');
            currentView = e.currentTarget.dataset.view;
            if (currentResults) renderResults(currentResults);
        });
    });

    // Auto-resume from state on page load
    fetch('/api/status').then(res => res.json()).then(state => {
        if (['scanning', 'extracting', 'hashing', 'processing', 'comparing', 'paused'].includes(state.status)) {
            showPanel('progress');
            updateProgressUI(state);
            startPolling();
        } else {
            renderFolders();
        }
    }).catch(err => {
        console.error("Failed to load initial status", err);
        renderFolders();
    });
});
