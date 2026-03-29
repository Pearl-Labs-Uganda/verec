"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  createFeedSocket,
  fetchPresets,
  fetchReport,
  fetchExportAll,
  downloadJson,
  type FeedFrame,
  type DetectionObject,
  type ReportResult,
} from "@/lib/api";

// ── Main page ────────────────────────────────────────────────────────────

export default function Home() {
  // Feed state
  const [connected, setConnected] = useState(false);
  const [aiFrame, setAiFrame] = useState<string | null>(null);
  const [rawFrame, setRawFrame] = useState<string | null>(null);
  const [caption, setCaption] = useState("");
  const [detectionInfo, setDetectionInfo] = useState("");
  const [log, setLog] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // Settings
  const [source, setSource] = useState<"local" | "ip">("local");
  const [presets, setPresets] = useState<Record<string, string>>({});
  const [selectedPreset, setSelectedPreset] = useState("");
  const [streamUrl, setStreamUrl] = useState("");
  const [conf, setConf] = useState(0.45);
  const [iou, setIou] = useState(0.45);
  const [vlmInterval, setVlmInterval] = useState(5);
  const [enableDet, setEnableDet] = useState(true);
  const [enableSeg, setEnableSeg] = useState(true);
  const [enableVlm, setEnableVlm] = useState(true);

  // Report
  const [report, setReport] = useState<ReportResult | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  // Tab
  const [tab, setTab] = useState<"feed" | "export">("feed");

  // Export
  const [exportData, setExportData] = useState<object | null>(null);

  // Load presets
  useEffect(() => {
    fetchPresets().then(setPresets).catch(() => {});
  }, []);

  // Start feed
  const startFeed = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    const ws = createFeedSocket({
      source: source === "local" ? "local" : "ip",
      url: streamUrl,
      conf,
      iou,
      vlm_interval: vlmInterval,
      enable_det: enableDet,
      enable_seg: enableSeg,
      enable_vlm: enableVlm,
    });

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (ev) => {
      try {
        const data: FeedFrame = JSON.parse(ev.data);
        if (data.ai_frame) setAiFrame(`data:image/jpeg;base64,${data.ai_frame}`);
        if (data.raw_frame) setRawFrame(`data:image/jpeg;base64,${data.raw_frame}`);
        if (data.caption) setCaption(data.caption);
        if (data.detection) {
          const d = data.detection;
          const objs = d.objects.map((o: DetectionObject) => o.class).join(", ");
          setDetectionInfo(`${d.count} objects | ${d.time_ms}ms | ${d.fps} FPS — ${objs || "none"}`);
          setLog((prev) => {
            const line = `[${new Date(data.timestamp).toLocaleTimeString()}] ${objs || "(nothing)"}`;
            const next = [...prev, line];
            return next.slice(-30);
          });
        }
      } catch {
        /* ignore parse errors */
      }
    };
    wsRef.current = ws;
  }, [source, streamUrl, conf, iou, vlmInterval, enableDet, enableSeg, enableVlm]);

  const stopFeed = useCallback(() => {
    if (wsRef.current) {
      try { wsRef.current.send(JSON.stringify({ action: "stop" })); } catch { /* ok */ }
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => () => { wsRef.current?.close(); }, []);

  // Generate report
  const handleReport = async () => {
    setReportLoading(true);
    try {
      const r = await fetchReport();
      setReport(r);
    } catch (e) {
      setReport({ timestamp: new Date().toISOString(), model: "error", report: String(e), detection_count: 0, caption_count: 0 });
    }
    setReportLoading(false);
  };

  // Export all
  const handleExport = async () => {
    try {
      const data = await fetchExportAll();
      setExportData(data);
      downloadJson(data, `verec-export-${Date.now()}.json`);
    } catch { /* ok */ }
  };

  return (
    <div className="min-h-screen bg-stone-950 text-stone-100">
      {/* Header */}
      <header className="border-b border-stone-800 px-6 py-4">
        <h1 className="text-3xl font-extrabold text-orange-500">VEREC</h1>
        <p className="text-sm text-stone-400">
          Video Recognition &amp; Reporting — real-time object detection, segmentation, VLM captioning &amp; LLM reports
        </p>
      </header>

      {/* Tab bar */}
      <nav className="flex gap-1 px-6 pt-4">
        <button
          onClick={() => setTab("feed")}
          className={`px-4 py-2 rounded-t text-sm font-medium ${
            tab === "feed" ? "bg-stone-800 text-orange-400" : "bg-stone-900 text-stone-500 hover:text-stone-300"
          }`}
        >
          Live Feed
        </button>
        <button
          onClick={() => setTab("export")}
          className={`px-4 py-2 rounded-t text-sm font-medium ${
            tab === "export" ? "bg-stone-800 text-orange-400" : "bg-stone-900 text-stone-500 hover:text-stone-300"
          }`}
        >
          Data Export
        </button>
      </nav>

      <main className="px-6 pb-8">
        {tab === "feed" && (
          <div className="bg-stone-800 rounded-b rounded-tr p-4 space-y-4">
            <div className="flex gap-4">
              {/* Big AI view */}
              <div className="flex-[3] space-y-2">
                <div className="relative aspect-video bg-stone-900 rounded overflow-hidden flex items-center justify-center">
                  {aiFrame ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={aiFrame} alt="AI View" className="w-full h-full object-contain" />
                  ) : (
                    <span className="text-stone-600">AI View — press Start</span>
                  )}
                  {/* Connection badge */}
                  <span
                    className={`absolute top-2 left-2 px-2 py-0.5 rounded text-xs font-bold ${
                      connected ? "bg-green-600/80 text-white" : "bg-stone-700 text-stone-400"
                    }`}
                  >
                    {connected ? "LIVE" : "IDLE"}
                  </span>
                </div>
                <div className="flex gap-2 text-xs text-stone-400">
                  <span className="bg-stone-900 px-2 py-1 rounded flex-1 truncate">
                    {detectionInfo || "Waiting…"}
                  </span>
                </div>
              </div>

              {/* Right panel */}
              <div className="flex-[2] space-y-3">
                {/* Small raw feed */}
                <div className="aspect-video bg-stone-900 rounded overflow-hidden flex items-center justify-center">
                  {rawFrame ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={rawFrame} alt="Raw Feed" className="w-full h-full object-contain" />
                  ) : (
                    <span className="text-stone-600 text-sm">Raw Feed</span>
                  )}
                </div>

                {/* Source controls */}
                <details open className="bg-stone-900 rounded p-3">
                  <summary className="text-sm font-semibold text-stone-300 cursor-pointer">Source</summary>
                  <div className="mt-2 space-y-2">
                    <div className="flex gap-2">
                      <button
                        onClick={() => setSource("local")}
                        className={`px-3 py-1 rounded text-xs ${source === "local" ? "bg-orange-600 text-white" : "bg-stone-800 text-stone-400"}`}
                      >
                        Local Camera
                      </button>
                      <button
                        onClick={() => setSource("ip")}
                        className={`px-3 py-1 rounded text-xs ${source === "ip" ? "bg-orange-600 text-white" : "bg-stone-800 text-stone-400"}`}
                      >
                        IP Camera
                      </button>
                    </div>
                    {source === "ip" && (
                      <>
                        <select
                          value={selectedPreset}
                          onChange={(e) => {
                            setSelectedPreset(e.target.value);
                            setStreamUrl(presets[e.target.value] || "");
                          }}
                          className="w-full bg-stone-800 text-stone-200 text-xs rounded px-2 py-1"
                        >
                          <option value="">Select preset…</option>
                          {Object.keys(presets).map((k) => (
                            <option key={k} value={k}>{k}</option>
                          ))}
                        </select>
                        <input
                          type="text"
                          value={streamUrl}
                          onChange={(e) => setStreamUrl(e.target.value)}
                          placeholder="YouTube / RTSP / MJPEG URL"
                          className="w-full bg-stone-800 text-stone-200 text-xs rounded px-2 py-1"
                        />
                      </>
                    )}
                    <div className="flex gap-2">
                      <button
                        onClick={startFeed}
                        className="flex-1 bg-orange-600 hover:bg-orange-700 text-white text-sm font-semibold py-1.5 rounded"
                      >
                        ▶ Start
                      </button>
                      <button
                        onClick={stopFeed}
                        className="flex-1 bg-stone-700 hover:bg-stone-600 text-stone-200 text-sm font-semibold py-1.5 rounded"
                      >
                        ⏹ Stop
                      </button>
                    </div>
                  </div>
                </details>

                {/* Model toggles */}
                <details className="bg-stone-900 rounded p-3">
                  <summary className="text-sm font-semibold text-stone-300 cursor-pointer">Detection Settings</summary>
                  <div className="mt-2 space-y-2">
                    <div className="flex gap-3 text-xs">
                      <label className="flex items-center gap-1">
                        <input type="checkbox" checked={enableDet} onChange={(e) => setEnableDet(e.target.checked)} className="accent-orange-500" />
                        Detection
                      </label>
                      <label className="flex items-center gap-1">
                        <input type="checkbox" checked={enableSeg} onChange={(e) => setEnableSeg(e.target.checked)} className="accent-orange-500" />
                        Segmentation
                      </label>
                      <label className="flex items-center gap-1">
                        <input type="checkbox" checked={enableVlm} onChange={(e) => setEnableVlm(e.target.checked)} className="accent-orange-500" />
                        VLM
                      </label>
                    </div>
                    <div className="space-y-1 text-xs text-stone-400">
                      <label className="block">Confidence: {conf.toFixed(2)}
                        <input type="range" min="0.1" max="1" step="0.05" value={conf} onChange={(e) => setConf(+e.target.value)} className="w-full accent-orange-500" />
                      </label>
                      <label className="block">IoU: {iou.toFixed(2)}
                        <input type="range" min="0.1" max="1" step="0.05" value={iou} onChange={(e) => setIou(+e.target.value)} className="w-full accent-orange-500" />
                      </label>
                      <label className="block">VLM Interval: {vlmInterval}s
                        <input type="range" min="3" max="15" step="1" value={vlmInterval} onChange={(e) => setVlmInterval(+e.target.value)} className="w-full accent-orange-500" />
                      </label>
                    </div>
                  </div>
                </details>

                {/* VLM caption */}
                <div className="bg-stone-900 rounded p-3">
                  <h4 className="text-xs font-semibold text-stone-400 mb-1">Scene Caption (VLM)</h4>
                  <p className="text-sm text-stone-200 min-h-[2em]">{caption || "—"}</p>
                </div>
              </div>
            </div>

            {/* Bottom: Log + Report */}
            <div className="flex gap-4">
              {/* Detection log */}
              <div className="flex-[3] bg-stone-900 rounded p-3">
                <h4 className="text-xs font-semibold text-stone-400 mb-1">Detection Log</h4>
                <pre className="text-xs text-stone-300 max-h-40 overflow-y-auto font-mono whitespace-pre-wrap">
                  {log.length ? log.join("\n") : "No detections yet."}
                </pre>
              </div>

              {/* Report */}
              <div className="flex-[2] bg-stone-900 rounded p-3 space-y-2">
                <h4 className="text-xs font-semibold text-stone-400">LLM Report</h4>
                <button
                  onClick={handleReport}
                  disabled={reportLoading}
                  className="w-full bg-orange-600 hover:bg-orange-700 disabled:bg-stone-700 text-white text-sm font-semibold py-1.5 rounded"
                >
                  {reportLoading ? "Generating…" : "📝 Generate Report"}
                </button>
                {report && (
                  <div className="text-xs text-stone-300 max-h-32 overflow-y-auto whitespace-pre-wrap">
                    {report.report}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {tab === "export" && (
          <div className="bg-stone-800 rounded-b rounded-tr p-6 space-y-4">
            <h2 className="text-lg font-bold text-stone-200">Data Export</h2>
            <p className="text-sm text-stone-400">
              Export all collected detection, VLM caption, and report data as JSON.
            </p>
            <button
              onClick={handleExport}
              className="bg-orange-600 hover:bg-orange-700 text-white font-semibold px-6 py-2 rounded"
            >
              ⬇ Download All Data (JSON)
            </button>
            {exportData && (
              <div className="bg-stone-900 rounded p-4 max-h-96 overflow-auto">
                <pre className="text-xs text-stone-300 font-mono whitespace-pre-wrap">
                  {JSON.stringify(exportData, null, 2).slice(0, 5000)}
                  {JSON.stringify(exportData, null, 2).length > 5000 ? "\n…(truncated)" : ""}
                </pre>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
