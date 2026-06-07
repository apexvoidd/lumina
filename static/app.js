// API Base URL (enables double-clicking index.html local file to connect to server)
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:5000" : "";

// Application State variables
let isTraining = false;
let pollingInterval = null;
let datasetPollingInterval = null;
let lossChart = null;
let visualizerAnimationId = null;
let voices = [];
let speechUtterance = null;
let recognition = null;
let isListening = false;
let currentMood = "calm";
let currentAppState = "idle"; // idle, training, thinking, speaking, listening

// DOM elements
const el = {
    // Sliders & Values
    n_embd: document.getElementById("n_embd"),
    n_head: document.getElementById("n_head"),
    n_layer: document.getElementById("n_layer"),
    max_iters: document.getElementById("max_iters"),
    embdVal: document.getElementById("embd-val"),
    headVal: document.getElementById("head-val"),
    layerVal: document.getElementById("layer-val"),
    itersVal: document.getElementById("iters-val"),
    
    // Config/Training Buttons
    btnStartTrain: document.getElementById("btn-start-train"),
    btnStopTrain: document.getElementById("btn-stop-train"),
    btnExportModel: document.getElementById("btn-export-model"),
    selectDataset: document.getElementById("select-dataset"),
    chkStartFresh: document.getElementById("chk-start-fresh"),
    selectModel: document.getElementById("select-model"),
    txtNewModel: document.getElementById("txt-new-model"),
    btnCreateModel: document.getElementById("btn-create-model"),
    
    // Status Badges & Counters
    trainingStatus: document.getElementById("training-status"),
    metricLoss: document.getElementById("metric-loss"),
    metricProgress: document.getElementById("metric-progress"),
    metricSpeed: document.getElementById("metric-speed"),
    modelReadiness: document.getElementById("model-readiness"),
    aiMood: document.getElementById("ai-mood"),
    moodText: document.getElementById("mood-text"),
    glowOrb: document.getElementById("glow-orb"),
    
    // Web Speech & Voice
    voiceSelect: document.getElementById("voice-select"),
    temperature: document.getElementById("temperature"),
    tempVal: document.getElementById("temp-val"),
    toggleVoiceOut: document.getElementById("toggle-voice-out"),
    
    // Chat UI
    chatTranscript: document.getElementById("chat-transcript"),
    chatInput: document.getElementById("chat-input"),
    btnSend: document.getElementById("btn-send"),
    btnMic: document.getElementById("btn-mic"),
    waveVisualizer: document.getElementById("waveVisualizer"),
    
    // --- NEW: Dataset Elements ---
    datasetStatus: document.getElementById("dataset-load-status"),
    tabButtons: document.querySelectorAll(".tab-btn"),
    tabPanes: document.querySelectorAll(".tab-pane"),
    
    // HF Form Elements
    hfToken: document.getElementById("hf-token"),
    hfDatasetName: document.getElementById("hf-dataset-name"),
    btnFetchColumns: document.getElementById("btn-fetch-columns"),
    hfConfigRow: document.getElementById("hf-config-row"),
    hfSubsetSelect: document.getElementById("hf-subset-select"),
    btnFetchSubset: document.getElementById("btn-fetch-subset"),
    hfMappingOptions: document.getElementById("hf-mapping-options"),
    hfMappingType: document.getElementById("hf-mapping-type"),
    hfUserColumn: document.getElementById("hf-user-column"),
    hfAiColumn: document.getElementById("hf-ai-column"),
    hfListColumn: document.getElementById("hf-list-column"),
    hfRawColumn: document.getElementById("hf-raw-column"),
    hfDatasetLimit: document.getElementById("hf-dataset-limit"),
    btnLoadHfDataset: document.getElementById("btn-load-hf-dataset"),
    hfDataPreview: document.getElementById("hf-data-preview"),
    
    mappingPairsRow: document.getElementById("mapping-pairs-row"),
    mappingListRow: document.getElementById("mapping-list-row"),
    mappingRawRow: document.getElementById("mapping-raw-row"),
    
    // Local File Elements
    dropZone: document.getElementById("drop-zone"),
    localFileInput: document.getElementById("local-file-input")
};

// --- INITIALIZATION ---
document.addEventListener("DOMContentLoaded", () => {
    initSliders();
    initChart();
    initVoices();
    initSpeechRecognition();
    initVisualizer();
    initDatasetTabs();
    initHFLoader();
    initLocalUpload();
    
    refreshDatasetsList();
    refreshModelsList();
    checkModelStatus(); // Check initial status on load
    
    // Set up event listeners
    el.btnStartTrain.addEventListener("click", startTraining);
    el.btnStopTrain.addEventListener("click", stopTraining);
    el.btnExportModel.addEventListener("click", exportTrainedModel);
    
    el.selectModel.addEventListener("change", (e) => switchActiveModel(e.target.value));
    el.btnCreateModel.addEventListener("click", createNewModel);
    el.txtNewModel.addEventListener("keydown", (e) => {
        if (e.key === "Enter") createNewModel();
    });
    
    el.btnSend.addEventListener("click", sendChatMessage);
    el.chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !el.btnSend.disabled) sendChatMessage();
    });
    el.btnMic.addEventListener("click", toggleSpeechInput);
    el.toggleVoiceOut.addEventListener("click", () => {
        el.toggleVoiceOut.classList.toggle("active");
    });
    el.temperature.addEventListener("input", (e) => {
        el.tempVal.textContent = e.target.value;
    });
});

// --- SLIDERS & CONFIGS ---
function initHFLoader() {
    // Dynamic Mapping Format selection toggle
    el.hfMappingType.addEventListener("change", (e) => {
        const val = e.target.value;
        el.mappingPairsRow.classList.add("hidden");
        el.mappingListRow.classList.add("hidden");
        el.mappingRawRow.classList.add("hidden");
        if (val === "pairs") el.mappingPairsRow.classList.remove("hidden");
        else if (val === "list") el.mappingListRow.classList.remove("hidden");
        else if (val === "raw" || val === "plain") el.mappingRawRow.classList.remove("hidden");
    });

    // ── Step 1: Fetch configs/subsets list ──────────────────────────────────
    function fetchConfigs() {
        const datasetName = el.hfDatasetName.value.trim();
        const token       = el.hfToken.value.trim();
        if (!datasetName) {
            alert("Please enter a Hugging Face dataset name first!");
            return;
        }

        el.btnFetchColumns.disabled = true;
        el.btnFetchColumns.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Checking...`;
        el.hfConfigRow.classList.add("hidden");
        el.hfMappingOptions.classList.add("hidden");
        el.datasetStatus.className = "status-pill loading";
        el.datasetStatus.textContent = "Connecting...";

        fetch(API_BASE + "/api/dataset/hf/configs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_name: datasetName, token: token })
        })
        .then(r => r.json())
        .then(data => {
            el.btnFetchColumns.disabled = false;
            el.btnFetchColumns.innerHTML = `<i class="fa-solid fa-arrows-spin"></i> Fetch Columns`;

            if (data.error) {
                el.datasetStatus.className = "status-pill error";
                el.datasetStatus.textContent = "Error";
                alert(data.error);
                return;
            }

            const configs = data.configs || [];
            if (configs.length > 1) {
                // Multiple subsets — show selector
                el.hfSubsetSelect.innerHTML = "";
                configs.forEach(c => {
                    const opt = document.createElement("option");
                    opt.value = c; opt.textContent = c;
                    el.hfSubsetSelect.appendChild(opt);
                });
                el.hfConfigRow.classList.remove("hidden");
                el.datasetStatus.className = "status-pill loading";
                el.datasetStatus.textContent = "Select a subset";
                appendMessage("system", `Dataset '${datasetName}' has ${configs.length} subsets. Select one and click 'Fetch Subset'.`);
            } else {
                // Single (or no) config — go straight to preview
                const configName = configs.length === 1 ? configs[0] : "";
                fetchPreview(configName);
            }
        })
        .catch(() => {
            el.btnFetchColumns.disabled = false;
            el.btnFetchColumns.innerHTML = `<i class="fa-solid fa-arrows-spin"></i> Fetch Columns`;
            el.datasetStatus.className = "status-pill error";
            el.datasetStatus.textContent = "Failed";
            alert("Could not reach the server. Is it running?");
        });
    }

    // ── Step 2: Preview a single row for the chosen config ──────────────────
    function fetchPreview(configName = "") {
        const datasetName = el.hfDatasetName.value.trim();
        const token       = el.hfToken.value.trim();

        el.datasetStatus.className = "status-pill loading";
        el.datasetStatus.textContent = "Loading preview...";
        if (el.btnFetchSubset) {
            el.btnFetchSubset.disabled = true;
            el.btnFetchSubset.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Fetching...`;
        }

        fetch(API_BASE + "/api/dataset/hf/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset_name: datasetName, token: token, config: configName })
        })
        .then(r => r.json())
        .then(data => {
            if (el.btnFetchSubset) {
                el.btnFetchSubset.disabled = false;
                el.btnFetchSubset.innerHTML = `<i class="fa-solid fa-arrows-spin"></i> Fetch Subset`;
            }

            // Server detected that a subset/config must be chosen
            if (data.configs && data.configs.length > 0) {
                el.hfSubsetSelect.innerHTML = "";
                data.configs.forEach(c => {
                    const opt = document.createElement("option");
                    opt.value = c; opt.textContent = c;
                    el.hfSubsetSelect.appendChild(opt);
                });
                el.hfConfigRow.classList.remove("hidden");
                el.hfMappingOptions.classList.add("hidden");
                el.datasetStatus.className = "status-pill loading";
                el.datasetStatus.textContent = "Select a subset";
                appendMessage("system", `⚠️ Dataset has ${data.configs.length} subsets. Select one from the dropdown and click "Fetch Subset".`);
                return;
            }

            if (data.error) {
                el.datasetStatus.className = "status-pill error";
                el.datasetStatus.textContent = "Error";
                alert(data.error);
                return;
            }
            populateDropdowns(data.columns);
            el.hfDataPreview.textContent = JSON.stringify(data.preview, null, 2);
            el.hfMappingOptions.classList.remove("hidden");
            el.datasetStatus.className = "status-pill idle";
            el.datasetStatus.textContent = "Preview Loaded";
        })
        .catch(() => {
            if (el.btnFetchSubset) {
                el.btnFetchSubset.disabled = false;
                el.btnFetchSubset.innerHTML = `<i class="fa-solid fa-arrows-spin"></i> Fetch Subset`;
            }
            el.datasetStatus.className = "status-pill error";
            el.datasetStatus.textContent = "Failed";
            alert("Error fetching dataset preview.");
        });
    }

    // ── Wire up buttons ─────────────────────────────────────────────────────
    el.btnFetchColumns.addEventListener("click", () => fetchConfigs());
    el.btnFetchSubset.addEventListener("click",  () => fetchPreview(el.hfSubsetSelect.value));

    // ── Load & Format full dataset ──────────────────────────────────────────
    el.btnLoadHfDataset.addEventListener("click", () => {
        const datasetName  = el.hfDatasetName.value.trim();
        const token        = el.hfToken.value.trim();
        const config       = el.hfConfigRow.classList.contains("hidden") ? "" : el.hfSubsetSelect.value;
        const limit        = el.hfDatasetLimit.value;
        const mappingType  = el.hfMappingType.value;

        if (!datasetName) { alert("Enter a dataset name first!"); return; }

        let mappingConfig = { type: mappingType };
        if (mappingType === "pairs") {
            mappingConfig.user_col = el.hfUserColumn.value;
            mappingConfig.ai_col   = el.hfAiColumn.value;
        } else if (mappingType === "list") {
            mappingConfig.list_col = el.hfListColumn.value;
        } else if (mappingType === "raw" || mappingType === "plain") {
            mappingConfig.raw_col = el.hfRawColumn.value;
        }

        el.btnLoadHfDataset.disabled = true;
        el.btnLoadHfDataset.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Importing...`;

        fetch(API_BASE + "/api/dataset/hf/load", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                dataset_name:   datasetName,
                token:          token,
                config:         config,
                limit:          limit,
                mapping_config: mappingConfig
            })
        })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                el.btnLoadHfDataset.disabled = false;
                el.btnLoadHfDataset.innerHTML = `<i class="fa-solid fa-download"></i> Load & Format`;
            } else {
                startDatasetPolling();
            }
        })
        .catch(() => {
            alert("Error sending load dataset request.");
            el.btnLoadHfDataset.disabled = false;
            el.btnLoadHfDataset.innerHTML = `<i class="fa-solid fa-download"></i> Load & Format`;
        });
    });
}

function initSliders() {
    el.n_embd.addEventListener("input", (e) => el.embdVal.textContent = e.target.value);
    el.n_head.addEventListener("input", (e) => el.headVal.textContent = e.target.value);
    el.n_layer.addEventListener("input", (e) => el.layerVal.textContent = e.target.value);
    el.max_iters.addEventListener("input", (e) => el.itersVal.textContent = e.target.value);
}

// --- REAL-TIME CHART ---
function initChart() {
    const ctx = document.getElementById("lossChart").getContext("2d");
    
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: '#3b82f6',
                borderWidth: 2,
                backgroundColor: 'rgba(59, 130, 246, 0.05)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHitRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } }
                }
            }
        }
    });
}

function updateChartColor(color) {
    if (!lossChart) return;
    lossChart.data.datasets[0].borderColor = color;
    lossChart.data.datasets[0].backgroundColor = color.replace(')', ', 0.05)').replace('rgb', 'rgba');
    lossChart.update();
}

// --- STATE MANAGEMENT ---
function setAppState(state) {
    currentAppState = state;
    el.glowOrb.style.animationDuration = 
        state === "thinking" ? "0.6s" : 
        state === "speaking" ? "1.5s" : 
        state === "listening" ? "1s" : "3s";
        
    if (state === "thinking") {
        el.glowOrb.classList.add("spinning");
    } else {
        el.glowOrb.classList.remove("spinning");
    }
}

function checkModelStatus() {
    fetch(API_BASE + "/api/status")
        .then(res => res.json())
        .then(data => {
            updateModelBadge(data.has_weights, data.status);
            
            if (data.active_model && el.selectModel) {
                el.selectModel.value = data.active_model;
            }
            
            // Check if application is running in Standalone chat-only mode
            if (data.standalone) {
                document.body.classList.add("standalone-mode");
                // Remove dataset and training nodes to clean memory
                const dsNode = document.getElementById("dataset-panel-node");
                const trNode = document.getElementById("training-panel-node");
                if (dsNode) dsNode.remove();
                if (trNode) trNode.remove();
                
                // Add a welcome system message for standalone mode
                const welcomes = el.chatTranscript.querySelectorAll(".system-message");
                welcomes.forEach(w => w.remove());
                appendMessage("system", "Lumina custom GPT chatbot companion is active and loaded. Speak or type below to chat!");
            }
            
            // Restore training history and metrics if they exist
            if (data.loss_history && data.loss_history.length > 0 && !data.standalone) {
                lossChart.data.labels = data.loss_history.map(item => item.iter);
                lossChart.data.datasets[0].data = data.loss_history.map(item => item.loss);
                lossChart.update();
                
                el.metricLoss.textContent = data.loss ? data.loss.toFixed(4) : "--";
                el.metricSpeed.textContent = data.speed ? `${data.speed.toFixed(1)} it/s` : "--";
                
                const progress = data.max_iters > 0 ? Math.round((data.iteration / data.max_iters) * 100) : 0;
                el.metricProgress.textContent = `${progress}%`;
                
                el.trainingStatus.textContent = data.status;
                el.trainingStatus.className = `status-pill ${data.status}`;
            }
            
            // Restore Hyperparameter slider values to match the last run
            if (data.hyperparams && Object.keys(data.hyperparams).length > 0 && !data.standalone) {
                el.n_embd.value = data.hyperparams.n_embd;
                el.embdVal.textContent = data.hyperparams.n_embd;
                el.n_head.value = data.hyperparams.n_head;
                el.headVal.textContent = data.hyperparams.n_head;
                el.n_layer.value = data.hyperparams.n_layer;
                el.layerVal.textContent = data.hyperparams.n_layer;
                el.max_iters.value = data.hyperparams.max_iters;
                el.itersVal.textContent = data.hyperparams.max_iters;
            }
            
            if (data.status === "training" && !isTraining && !data.standalone) {
                isTraining = true;
                setTrainingUI(true);
                startPolling();
            }
            
            // Restore dataset loading status if in progress
            if (data.dataset_status && data.dataset_status.status === "loading" && !data.standalone) {
                startDatasetPolling();
            }
        })
        .catch(err => console.error("Error checking model status:", err));
}

function updateModelBadge(hasWeights, status) {
    if (status === "training") {
        el.modelReadiness.className = "model-badge training";
        el.modelReadiness.querySelector(".badge-text").textContent = "Training Model...";
    } else if (hasWeights) {
        el.modelReadiness.className = "model-badge ready";
        el.modelReadiness.querySelector(".badge-text").textContent = "Custom GPT Coded & Ready";
        el.chatInput.disabled = false;
        el.btnSend.disabled = false;
    } else {
        el.modelReadiness.className = "model-badge";
        el.modelReadiness.querySelector(".badge-text").textContent = "Untrained (Click Start Training)";
        el.chatInput.disabled = true;
        el.btnSend.disabled = true;
    }
}

// --- DATASET MANAGEMENT CONTROLS ---

function initDatasetTabs() {
    el.tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            // Remove active classes
            el.tabButtons.forEach(b => b.classList.remove("active"));
            el.tabPanes.forEach(p => p.classList.remove("active"));
            
            // Set active
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });
}

function populateDropdowns(columns) {
    const selects = [el.hfUserColumn, el.hfAiColumn, el.hfListColumn, el.hfRawColumn];
    selects.forEach(sel => {
        sel.innerHTML = "";
        columns.forEach(col => {
            const opt = document.createElement("option");
            opt.value = col;
            opt.textContent = col;
            sel.appendChild(opt);
        });
    });
    
    // Smart Mapping Auto-Select suggestions based on key names
    columns.forEach(col => {
        const c = col.toLowerCase();
        if (c.includes("prompt") || c.includes("context") || c.includes("user") || c.includes("question") || c.includes("input") || c.includes("text1")) {
            el.hfUserColumn.value = col;
        }
        if (c.includes("response") || c.includes("ai") || c.includes("answer") || c.includes("output") || c.includes("text2") || c.includes("reply")) {
            el.hfAiColumn.value = col;
        }
        if (c.includes("dialog") || c.includes("dialogue") || c.includes("conversation") || c.includes("turns")) {
            el.hfListColumn.value = col;
            el.hfMappingType.value = "list";
            // trigger change event
            el.hfMappingType.dispatchEvent(new Event("change"));
        }
        if (c === "text") {
            el.hfRawColumn.value = col;
        }
    });
}

function refreshDatasetsList(selectedName = "") {
    fetch(API_BASE + "/api/datasets")
        .then(res => res.json())
        .then(data => {
            el.selectDataset.innerHTML = "";
            
            if (!data.datasets || data.datasets.length === 0) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.textContent = "No datasets found (Load/upload one first)";
                el.selectDataset.appendChild(opt);
                return;
            }
            
            data.datasets.forEach(db => {
                const opt = document.createElement("option");
                opt.value = db;
                opt.textContent = db;
                // If the database name matches or contains our selected name, select it
                if (selectedName && (db === selectedName || db.toLowerCase().includes(selectedName.toLowerCase()))) {
                    opt.selected = true;
                }
                el.selectDataset.appendChild(opt);
            });
        })
        .catch(err => console.error("Error fetching datasets list:", err));
}

function refreshModelsList(activeName = "") {
    fetch(API_BASE + "/api/models")
        .then(res => res.json())
        .then(data => {
            el.selectModel.innerHTML = "";
            const current = activeName || data.active_model;
            
            data.models.forEach(name => {
                const opt = document.createElement("option");
                opt.value = name;
                opt.textContent = name;
                if (name === current) {
                    opt.selected = true;
                }
                el.selectModel.appendChild(opt);
            });
        })
        .catch(err => console.error("Error loading models list:", err));
}

function switchActiveModel(modelName) {
    if (!modelName) return;
    
    fetch(API_BASE + "/api/model/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName })
    })
    .then(res => res.json())
    .then(data => {
        appendMessage("system", `Switched active model configuration to "${modelName}".`);
        // Reload training state and chart
        checkModelStatus();
    })
    .catch(err => {
        alert("Failed to switch model.");
        console.error(err);
    });
}

function createNewModel() {
    const modelName = el.txtNewModel.value.trim().replace(/\s+/g, "_");
    if (!modelName) {
        alert("Please enter a valid model name!");
        return;
    }
    
    el.txtNewModel.value = "";
    
    fetch(API_BASE + "/api/model/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName })
    })
    .then(res => res.json())
    .then(data => {
        appendMessage("system", `Created and switched to new model database "${modelName}".`);
        refreshModelsList(modelName);
        checkModelStatus();
    })
    .catch(err => {
        alert("Failed to create model.");
        console.error(err);
    });
}

function startDatasetPolling() {
    if (datasetPollingInterval) clearInterval(datasetPollingInterval);

    el.datasetStatus.className = "status-pill loading";
    el.datasetStatus.textContent = "Downloading...";
    el.btnLoadHfDataset.disabled = true;

    datasetPollingInterval = setInterval(() => {
        fetch(API_BASE + "/api/status")
            .then(res => res.json())
            .then(data => {
                const s = data.dataset_status;
                if (s.status === "loading") {
                    const pct = s.total > 0 ? Math.round((s.progress / s.total) * 100) : 0;
                    el.datasetStatus.textContent = `Downloading ${s.progress}/${s.total} (${pct}%)`;
                    el.btnLoadHfDataset.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${pct}%`;
                } else {
                    // Load ended
                    clearInterval(datasetPollingInterval);
                    datasetPollingInterval = null;
                    el.btnLoadHfDataset.disabled = false;
                    el.btnLoadHfDataset.innerHTML = `<i class="fa-solid fa-download"></i> Load & Format`;

                    if (s.status === "completed") {
                        el.datasetStatus.className = "status-pill completed";
                        el.datasetStatus.textContent = "Ready";

                        // Auto-select the newly downloaded dataset in the Training panel
                        const newFile = s.filename || "";
                        refreshDatasetsList(newFile);

                        // Also directly set the dropdown value once the list is refreshed
                        if (newFile) {
                            setTimeout(() => {
                                if (el.selectDataset) {
                                    el.selectDataset.value = newFile;
                                    // If not found by exact match, pick the first option
                                    if (!el.selectDataset.value) {
                                        el.selectDataset.selectedIndex = 0;
                                    }
                                }
                            }, 300);
                        }

                        appendMessage("system",
                            `✅ Dataset "${newFile || "download"}" is ready!\n` +
                            `👉 It has been auto-selected in the Training Panel below. ` +
                            `Just click Start Training!`
                        );
                    } else if (s.status === "error") {
                        el.datasetStatus.className = "status-pill error";
                        el.datasetStatus.textContent = "Error";
                        appendMessage("system", `❌ Dataset loading failed: ${s.error_message}`);
                    }
                }
            })
            .catch(err => {
                console.error("Error checking dataset status:", err);
                clearInterval(datasetPollingInterval);
                datasetPollingInterval = null;
            });
    }, 500);
}

function initLocalUpload() {
    // Setup file drag-and-drop actions
    const zone = el.dropZone;
    
    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
    });
    
    zone.addEventListener("dragleave", () => {
        zone.classList.remove("dragover");
    });
    
    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    zone.addEventListener("click", () => {
        el.localFileInput.click();
    });
    
    el.localFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

function handleFileUpload(file) {
    if (!file.name.endsWith(".txt") && !file.name.toLowerCase().endsWith(".txt")) {
        alert("Only raw .txt files are supported for dataset uploading!");
        return;
    }
    
    el.datasetStatus.className = "status-pill loading";
    el.datasetStatus.textContent = "Uploading...";
    
    const formData = new FormData();
    formData.append("file", file);
    
    fetch(API_BASE + "/api/dataset/upload", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            el.datasetStatus.className = "status-pill error";
            el.datasetStatus.textContent = "Failed";
            alert(data.error);
        } else {
            el.datasetStatus.className = "status-pill completed";
            el.datasetStatus.textContent = "Ready";
            
            const count = Math.round(data.size_bytes / 1024);
            appendMessage("system", `Custom dataset text file uploaded successfully (${count} KB).`);
            refreshDatasetsList(file.name);
            if (!data.has_dialogue_format) {
                appendMessage("system", "Note: The uploaded file was raw text (no User/AI tags). The model will pair consecutive text blocks.");
            }
        }
    })
    .catch(err => {
        el.datasetStatus.className = "status-pill error";
        el.datasetStatus.textContent = "Failed";
        alert("Connection error occurred uploading dataset file.");
    });
}

// --- TRAINING RUNS ---
function startTraining() {
    const params = {
        n_embd: el.n_embd.value,
        n_head: el.n_head.value,
        n_layer: el.n_layer.value,
        max_iters: el.max_iters.value,
        temperature: el.temperature.value,
        dataset: el.selectDataset.value,
        start_fresh: el.chkStartFresh.checked
    };
    el.chkStartFresh.checked = false;

    // Disable Start button while we wait for server confirmation
    el.btnStartTrain.disabled = true;
    el.btnStartTrain.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Starting...`;

    fetch(API_BASE + "/api/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params)
    })
    .then(res => res.json())
    .then(data => {
        // Re-enable button either way until we know outcome
        el.btnStartTrain.innerHTML = `<i class="fa-solid fa-play"></i> Start Training`;

        if (data.error) {
            // Server rejected the request — stay in idle state
            el.btnStartTrain.disabled = false;
            el.trainingStatus.className = "status-pill error";
            el.trainingStatus.textContent = "Error";
            appendMessage("system", `\u26a0\ufe0f ${data.error}`);
        } else {
            // Server confirmed training started — now activate training UI
            setTrainingUI(true);
            isTraining = true;
            setAppState("training");

            // Clear old chart data
            lossChart.data.labels = [];
            lossChart.data.datasets[0].data = [];
            lossChart.update();

            appendMessage("system", "Starting custom GPT model compile and training loop in PyTorch...");
            startPolling();
        }
    })
    .catch(err => {
        el.btnStartTrain.disabled = false;
        el.btnStartTrain.innerHTML = `<i class="fa-solid fa-play"></i> Start Training`;
        el.trainingStatus.className = "status-pill error";
        el.trainingStatus.textContent = "Error";
        appendMessage("system", "Could not reach the server. Is it running?");
        console.error("Error launching training:", err);
    });
}

function stopTraining() {
    fetch(API_BASE + "/api/train/stop", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            appendMessage("system", "Halting training early. Saving weights at current epoch...");
        })
        .catch(err => console.error("Error stopping training:", err));
}

function exportTrainedModel() {
    // Check status first to see if weights exist
    fetch(API_BASE + "/api/status")
        .then(res => res.json())
        .then(data => {
            if (!data.has_weights) {
                alert("Please train the model first before exporting it!");
                return;
            }
            // Start download attachment
            appendMessage("system", "Packaging model, weights, code, and static files into a standalone ZIP archive...");
            window.location.href = API_BASE + "/api/model/export";
        })
        .catch(err => {
            alert("Error preparing model export.");
        });
}

function setTrainingUI(active) {
    el.btnStartTrain.disabled = active;
    el.btnStopTrain.disabled = !active;
    el.btnExportModel.disabled = active;
    
    el.n_embd.disabled = active;
    el.n_head.disabled = active;
    el.n_layer.disabled = active;
    el.max_iters.disabled = active;
    
    if (active) {
        el.trainingStatus.className = "status-pill training";
        el.trainingStatus.textContent = "Training";
        el.modelReadiness.className = "model-badge training";
        el.modelReadiness.querySelector(".badge-text").textContent = "Training...";
        el.chatInput.disabled = true;
        el.btnSend.disabled = true;
    } else {
        el.btnStopTrain.disabled = true;
    }
}

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    
    pollingInterval = setInterval(() => {
        fetch(API_BASE + "/api/status")
            .then(res => res.json())
            .then(data => {
                // Update stats
                el.metricLoss.textContent = data.loss ? data.loss.toFixed(4) : "--";
                el.metricSpeed.textContent = data.speed ? `${data.speed.toFixed(1)} it/s` : "--";
                
                const progress = data.max_iters > 0 ? Math.round((data.iteration / data.max_iters) * 100) : 0;
                el.metricProgress.textContent = `${progress}%`;
                
                // Update chart values
                if (data.loss_history && data.loss_history.length > 0) {
                    const chartLabels = data.loss_history.map(item => item.iter);
                    const chartData = data.loss_history.map(item => item.loss);
                    
                    lossChart.data.labels = chartLabels;
                    lossChart.data.datasets[0].data = chartData;
                    lossChart.update();
                }
                
                // Check if ended
                if (data.status !== "training") {
                    clearInterval(pollingInterval);
                    isTraining = false;
                    setTrainingUI(false);
                    setAppState("idle");
                    
                    updateModelBadge(data.has_weights, data.status);
                    
                    if (data.status === "completed") {
                        el.trainingStatus.className = "status-pill completed";
                        el.trainingStatus.textContent = "Completed";
                        appendMessage("system", "Training complete! The model weights have been compiled and loaded successfully. You can now chat below!");
                    } else if (data.status === "interrupted") {
                        el.trainingStatus.className = "status-pill interrupted";
                        el.trainingStatus.textContent = "Stopped";
                        appendMessage("system", "Training stopped by user. Current model weights loaded.");
                    } else if (data.status === "error") {
                        el.trainingStatus.className = "status-pill interrupted";
                        el.trainingStatus.textContent = "Error";
                        appendMessage("system", `Training halted due to error: ${data.error_message}`);
                    }
                }
            })
            .catch(err => {
                console.error("Error polling training status:", err);
                clearInterval(pollingInterval);
            });
    }, 500);
}

// --- CONVERSATION / INFERENCE ---
function sendChatMessage() {
    const text = el.chatInput.value.trim();
    if (!text) return;
    
    cancelSpeech(); // Stop speaking if talking
    
    appendMessage("user", text);
    el.chatInput.value = "";
    
    setAppState("thinking");
    
    fetch(API_BASE + "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            message: text,
            temperature: el.temperature.value
        })
    })
    .then(res => res.json())
    .then(data => {
        setAppState("idle");
        
        if (data.error) {
            appendMessage("system", `Inference Error: ${data.error}`);
        } else {
            appendMessage("ai", data.response);
            setMood(data.mood);
            
            // Speak response if voice-out is active
            if (el.toggleVoiceOut.classList.contains("active")) {
                speakText(data.response);
            }
        }
    })
    .catch(err => {
        setAppState("idle");
        appendMessage("system", "Error connecting to model server.");
    });
}

function appendMessage(sender, text) {
    const bubble = document.createElement("div");
    bubble.className = `message ${sender}`;
    
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = text;
    
    bubble.appendChild(content);
    el.chatTranscript.appendChild(bubble);
    
    // Auto-scroll transcript
    el.chatTranscript.scrollTop = el.chatTranscript.scrollHeight;
}

function setMood(mood) {
    currentMood = mood;
    el.moodText.textContent = mood;
    
    document.body.className = `mood-${mood}`;
    if (document.body.classList.contains("standalone-mode")) {
        document.body.classList.add("standalone-mode");
    }
    
    let color = 'rgb(59, 130, 246)'; // calm blue
    if (mood === 'excited') color = 'rgb(217, 70, 239)'; // pink/purple
    else if (mood === 'thoughtful') color = 'rgb(245, 158, 11)'; // amber/yellow
    else if (mood === 'empathetic') color = 'rgb(16, 185, 129)'; // green
    
    updateChartColor(color);
}

// --- TEXT-TO-SPEECH (TTS) ---
function initVoices() {
    if (!('speechSynthesis' in window)) {
        el.voiceSelect.innerHTML = "<option>Not supported in browser</option>";
        el.voiceSelect.disabled = true;
        return;
    }
    
    function populateVoiceList() {
        voices = window.speechSynthesis.getVoices();
        let filteredVoices = voices.filter(v => v.lang.includes("en-") || v.lang.includes("en_"));
        if (filteredVoices.length === 0) filteredVoices = voices;
        
        el.voiceSelect.innerHTML = "";
        
        if (filteredVoices.length === 0) {
            el.voiceSelect.innerHTML = "<option value=''>No system voices found</option>";
            return;
        }
        
        filteredVoices.forEach(voice => {
            const option = document.createElement("option");
            option.textContent = `${voice.name} (${voice.lang})`;
            option.value = voice.name;
            if (voice.name.includes("Google") || voice.name.includes("Natural") || voice.name.includes("Zira") || voice.name.includes("David")) {
                option.selected = true;
            }
            el.voiceSelect.appendChild(option);
        });
    }
    
    populateVoiceList();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = populateVoiceList;
    }
}

function speakText(text) {
    if (!('speechSynthesis' in window)) return;
    
    cancelSpeech(); // Cancel any current utterance
    
    setTimeout(() => {
        speechUtterance = new SpeechSynthesisUtterance(text);
        
        const selectedVoiceName = el.voiceSelect.value;
        const selectedVoice = voices.find(v => v.name === selectedVoiceName);
        if (selectedVoice) {
            speechUtterance.voice = selectedVoice;
        }
        
        if (currentMood === "excited") {
            speechUtterance.rate = 1.15;
            speechUtterance.pitch = 1.1;
        } else if (currentMood === "thoughtful") {
            speechUtterance.rate = 0.85;
            speechUtterance.pitch = 0.95;
        } else if (currentMood === "empathetic") {
            speechUtterance.rate = 0.9;
            speechUtterance.pitch = 1.0;
        } else {
            speechUtterance.rate = 1.0;
            speechUtterance.pitch = 1.0;
        }
        
        speechUtterance.onstart = () => {
            setAppState("speaking");
        };
        
        speechUtterance.onend = () => {
            setAppState("idle");
        };
        
        speechUtterance.onerror = () => {
            setAppState("idle");
        };
        
        window.speechSynthesis.speak(speechUtterance);
    }, 50);
}

function cancelSpeech() {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
    if (currentAppState === "speaking") {
        setAppState("idle");
    }
}

// --- SPEECH-TO-TEXT (STT) ---
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        el.btnMic.disabled = true;
        el.btnMic.title = "Speech Recognition not supported in this browser (Use Chrome)";
        console.warn("Speech Recognition not supported");
        return;
    }
    
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    
    recognition.onstart = () => {
        isListening = true;
        el.btnMic.className = "btn-mic listening";
        setAppState("listening");
        el.chatInput.placeholder = "Listening...";
        cancelSpeech();
    };
    
    recognition.onend = () => {
        isListening = false;
        el.btnMic.className = "btn-mic";
        if (currentAppState === "listening") setAppState("idle");
        el.chatInput.placeholder = "Type a message or click mic to talk...";
    };
    
    recognition.onresult = (event) => {
        const transcriptText = event.results[0][0].transcript;
        el.chatInput.value = transcriptText;
        sendChatMessage();
    };
    
    recognition.onerror = (e) => {
        console.error("Speech Recognition error:", e);
        isListening = false;
        el.btnMic.className = "btn-mic";
        setAppState("idle");
    };
}

function toggleSpeechInput() {
    if (!recognition) return;
    
    if (isListening) {
        recognition.stop();
    } else {
        recognition.start();
    }
}

// --- CANVAS GLOWING WAVE VISUALIZER ---
function initVisualizer() {
    const canvas = el.waveVisualizer;
    const ctx = canvas.getContext("2d");
    
    function resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        canvas.width = canvas.parentElement.clientWidth * dpr;
        canvas.height = canvas.parentElement.clientHeight * dpr;
        ctx.scale(dpr, dpr);
    }
    
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    
    let waveOffset = 0;
    
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const width = canvas.width / (window.devicePixelRatio || 1);
        const height = canvas.height / (window.devicePixelRatio || 1);
        const centerY = height / 2;
        
        let baseColor = '59, 130, 246';
        if (currentMood === 'excited') baseColor = '217, 70, 239';
        else if (currentMood === 'thoughtful') baseColor = '245, 158, 11';
        else if (currentMood === 'empathetic') baseColor = '16, 185, 129';
        
        let waveCount = 3;
        let speed = 0.04;
        let baseAmplitude = 8;
        
        if (currentAppState === "speaking") {
            waveCount = 4;
            speed = 0.12;
            baseAmplitude = 25;
        } else if (currentAppState === "listening") {
            waveCount = 3;
            speed = 0.08;
            baseAmplitude = 18;
        } else if (currentAppState === "thinking") {
            waveCount = 5;
            speed = 0.18;
            baseAmplitude = 12;
        } else if (currentAppState === "training") {
            waveCount = 4;
            speed = 0.15;
            baseAmplitude = 15;
        } else {
            waveCount = 2;
            speed = 0.03;
            baseAmplitude = 5;
        }
        
        waveOffset += speed;
        
        for (let i = 0; i < waveCount; i++) {
            ctx.beginPath();
            const opacity = (1 - (i / waveCount)) * 0.45;
            ctx.strokeStyle = `rgba(${baseColor}, ${opacity})`;
            ctx.lineWidth = i === 0 ? 3 : 1.5;
            
            const frequency = 0.008 + (i * 0.003);
            const phase = i * (Math.PI / 4) + waveOffset;
            const amplitude = baseAmplitude * (1 - (i * 0.2));
            
            for (let x = 0; x < width; x++) {
                const edgeSoftener = Math.sin((x / width) * Math.PI);
                const y = centerY + Math.sin(x * frequency + phase) * amplitude * edgeSoftener;
                
                if (x === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }
        visualizerAnimationId = requestAnimationFrame(draw);
    }
    draw();
}
