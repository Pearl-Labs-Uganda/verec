"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  createFeedSocket,
  fetchPresets,
  fetchReport,
  fetchExportAll,
  fetchSystemInfo,
  downloadJson,
  type FeedFrame,
  type DetectionObject,
  type ReportResult,
  type SystemInfo,
} from "@/lib/api";

/* ── helpers ─────────────────────────────────────────────────────────── */

function fmtTime(s: number) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return [h, m, sec].map((v) => String(v).padStart(2, "0")).join(":");
}

function fmtBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}

/* ── Icon components ─────────────────────────────────────────────────── */

function GearIcon({ className = "w-5 h-5" }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.5}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 010 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 010-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
      />
    </svg>
  );
}

function Spinner({ className = "w-5 h-5" }: { className?: string }) {
  return (
    <svg
      className={`animate-spin ${className}`}
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

/* ── Markdown prose wrapper ──────────────────────────────────────────── */

function MdReport({ text }: { text: string }) {
  return (
    <div
      className="prose prose-invert prose-xs max-w-none
      prose-headings:text-orange-400 prose-headings:text-xs prose-headings:font-bold prose-headings:mb-1 prose-headings:mt-2
      prose-p:text-[11px] prose-p:text-gray-300 prose-p:leading-relaxed prose-p:mb-1
      prose-li:text-[11px] prose-li:text-gray-300 prose-li:leading-relaxed
      prose-strong:text-gray-200 prose-ul:my-1 prose-ol:my-1"
    >
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  );
}

/* ── Generic expandable card ─────────────────────────────────────────── */

interface ExpandableCardProps {
  timestamp: string;
  preview: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
  meta?: React.ReactNode;
}

function ExpandableCard({
  timestamp,
  preview,
  defaultOpen,
  children,
  meta,
}: ExpandableCardProps) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  const time = new Date(timestamp).toLocaleTimeString();
  return (
    <div className="bg-gray-800/50 border border-gray-700/40 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-gray-800/80 transition-colors"
      >
        <div className="min-w-0 flex-1">
          <span className="text-[10px] text-gray-500 font-mono mr-2">
            {time}
          </span>
          <span className="text-[11px] text-gray-400 truncate">
            {preview.slice(0, 80)}
            {preview.length > 80 ? "…" : ""}
          </span>
        </div>
        <span className="text-gray-500 text-xs ml-2 shrink-0">
          {open ? "▾" : "▸"}
        </span>
      </button>
      {open && (
        <div className="px-3 pb-3 border-t border-gray-700/30">
          {children}
          {meta && (
            <div className="flex gap-3 mt-2 text-[9px] text-gray-600 font-mono">
              {meta}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────── */

export default function Home() {
  /* Feed state */
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [aiFrame, setAiFrame] = useState<string | null>(null);
  const [rawFrame, setRawFrame] = useState<string | null>(null);
  const [caption, setCaption] = useState("");
  const [captions, setCaptions] = useState<
    { text: string; timestamp: string }[]
  >([]);
  const [captionSearch, setCaptionSearch] = useState("");
  const [detectionInfo, setDetectionInfo] = useState("");
  const [log, setLog] = useState<string[]>([]);
  const [logSearch, setLogSearch] = useState("");
  const [objectCounts, setObjectCounts] = useState<Record<string, number>>({});
  const [currentAction, setCurrentAction] = useState<
    { label: string; confidence: number }[] | null
  >(null);
  const [poseInfo, setPoseInfo] = useState("");
  const wsRef = useRef<WebSocket | null>(null);

  /* Start dropdown */
  const [startOpen, setStartOpen] = useState(false);
  const [quickUrl, setQuickUrl] = useState("");

  /* Global search */
  const [globalSearch, setGlobalSearch] = useState("");
  const [globalSearchOpen, setGlobalSearchOpen] = useState(false);

  /* Bandwidth */
  const [bandwidth, setBandwidth] = useState(0);
  const bwAccum = useRef(0);
  const bwTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  /* Timer */
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* Settings panel */
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [source, setSource] = useState<"local" | "ip">("local");
  const [presets, setPresets] = useState<Record<string, string>>({});
  const [selectedPreset, setSelectedPreset] = useState("");
  const [streamUrl, setStreamUrl] = useState("");
  const [conf, setConf] = useState(0.45);
  const [iou, setIou] = useState(0.45);
  const [vlmInterval, setVlmInterval] = useState(5);
  const [enableDet, setEnableDet] = useState(true);
  const [enableVlm, setEnableVlm] = useState(true);
  const [enablePose, setEnablePose] = useState(true);

  /* Action log */
  const [actionLog, setActionLog] = useState<
    { actions: { label: string; confidence: number }[]; timestamp: string }[]
  >([]);
  const [actionSearch, setActionSearch] = useState("");

  const [autoReportInterval, setAutoReportInterval] = useState(60);

  /* Report */
  const [report, setReport] = useState<ReportResult | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reports, setReports] = useState<ReportResult[]>([]);
  const [reportSearch, setReportSearch] = useState("");
  const autoReportRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* System info */
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null);

  /* Tab */
  const [tab, setTab] = useState<"feed" | "export">("feed");
  const [exportData, setExportData] = useState<object | null>(null);

  /* ── Init ──────────────────────────────────────────────────────── */
  useEffect(() => {
    fetchPresets()
      .then(setPresets)
      .catch(() => {});
    fetchSystemInfo()
      .then(setSysInfo)
      .catch(() => {});
  }, []);

  /* ── Bandwidth meter ───────────────────────────────────────────── */
  useEffect(() => {
    bwTimer.current = setInterval(() => {
      setBandwidth(bwAccum.current);
      bwAccum.current = 0;
    }, 1000);
    return () => {
      if (bwTimer.current) clearInterval(bwTimer.current);
    };
  }, []);

  /* ── Auto-report timer ─────────────────────────────────────────── */
  const triggerReport = useCallback(async () => {
    try {
      const r = await fetchReport();
      setReport(r);
      setReports((prev) => [...prev, r].slice(-20));
    } catch (e) {
      const err: ReportResult = {
        timestamp: new Date().toISOString(),
        model: "error",
        report: String(e),
        detection_count: 0,
        caption_count: 0,
      };
      setReport(err);
      setReports((prev) => [...prev, err].slice(-20));
    }
  }, []);

  useEffect(() => {
    if (autoReportRef.current) clearInterval(autoReportRef.current);
    if (connected && autoReportInterval > 0) {
      autoReportRef.current = setInterval(() => {
        triggerReport();
      }, autoReportInterval * 1000);
    }
    return () => {
      if (autoReportRef.current) clearInterval(autoReportRef.current);
    };
  }, [connected, autoReportInterval, triggerReport]);

  /* ── Start / Stop feed ─────────────────────────────────────────── */
  const startFeed = useCallback(() => {
    wsRef.current?.close();
    setConnecting(true);
    const ws = createFeedSocket({
      source,
      url: streamUrl,
      conf,
      iou,
      vlm_interval: vlmInterval,
      enable_det: enableDet,
      enable_vlm: enableVlm,
      enable_pose: enablePose,
    });

    ws.onopen = () => {
      setConnecting(false);
      setConnected(true);
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((p) => p + 1), 1000);
    };
    ws.onclose = () => {
      setConnecting(false);
      setConnected(false);
      if (timerRef.current) clearInterval(timerRef.current);
    };
    ws.onerror = () => {
      setConnecting(false);
      setConnected(false);
      if (timerRef.current) clearInterval(timerRef.current);
    };

    ws.onmessage = (ev) => {
      try {
        const data: FeedFrame = JSON.parse(ev.data);
        if (data.frame_bytes) bwAccum.current += data.frame_bytes;
        if (data.ai_frame)
          setAiFrame(`data:image/jpeg;base64,${data.ai_frame}`);
        if (data.raw_frame)
          setRawFrame(`data:image/jpeg;base64,${data.raw_frame}`);
        if (data.caption) {
          const cap = data.caption;
          setCaption(cap);
          setCaptions((prev) =>
            [...prev, { text: cap, timestamp: data.timestamp }].slice(-50),
          );
        }
        if (data.detection) {
          const d = data.detection;
          const objs = d.objects
            .map(
              (o: DetectionObject) =>
                `${o.class} (${Math.round(o.confidence * 100)}%)`,
            )
            .join(", ");
          setDetectionInfo(`${d.count} obj | ${d.time_ms}ms | ${d.fps} FPS`);
          setLog((prev) =>
            [
              ...prev,
              `[${new Date(data.timestamp).toLocaleTimeString()}] ${objs || "(clear)"}`,
            ].slice(-50),
          );
        }
        if (data.object_counts) {
          setObjectCounts(data.object_counts);
        }
        if (data.pose) {
          const p = data.pose;
          setPoseInfo(
            `${p.count} person${p.count !== 1 ? "s" : ""} | ${p.time_ms}ms`,
          );
        }
        if (data.action) {
          setCurrentAction(data.action.actions);
          setActionLog((prev) =>
            [
              ...prev,
              { actions: data.action!.actions, timestamp: data.timestamp },
            ].slice(-50),
          );
        }
      } catch {
        /* skip */
      }
    };
    wsRef.current = ws;
  }, [
    source,
    streamUrl,
    conf,
    iou,
    vlmInterval,
    enableDet,
    enableVlm,
    enablePose,
  ]);

  const stopFeed = useCallback(() => {
    try {
      wsRef.current?.send(JSON.stringify({ action: "stop" }));
    } catch {
      /* ok */
    }
    wsRef.current?.close();
    wsRef.current = null;
    setConnecting(false);
    setConnected(false);
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  useEffect(
    () => () => {
      wsRef.current?.close();
      if (timerRef.current) clearInterval(timerRef.current);
    },
    [],
  );

  /* ── Manual report ─────────────────────────────────────────────── */
  const handleReport = async () => {
    setReportLoading(true);
    await triggerReport();
    setReportLoading(false);
  };

  /* ── Export ────────────────────────────────────────────────────── */
  const handleExport = async () => {
    try {
      const d = await fetchExportAll();
      setExportData(d);
      downloadJson(d, `verec-export-${Date.now()}.json`);
    } catch {
      /* ok */
    }
  };

  /* ── bandwidth color ───────────────────────────────────────────── */
  const bwColor =
    bandwidth === 0
      ? "text-gray-500"
      : bandwidth < 500_000
        ? "text-green-400"
        : bandwidth < 2_000_000
          ? "text-yellow-400"
          : "text-red-400";

  /* ── status label ──────────────────────────────────────────────── */
  const statusLabel = connecting ? "Connecting" : connected ? "Live" : "Idle";
  const statusClass = connecting
    ? "bg-yellow-500/20 text-yellow-400 ring-1 ring-yellow-500/30"
    : connected
      ? "bg-green-500/20 text-green-400 ring-1 ring-green-500/30"
      : "bg-gray-800 text-gray-500";

  /* ── Render ────────────────────────────────────────────────────── */
  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col overflow-hidden">
      {/* ═══ Header ═══════════════════════════════════════════════ */}
      <header className="shrink-0 border-b border-gray-800/60 bg-gray-900/80 backdrop-blur sticky top-0 z-30 px-3 sm:px-5 py-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <h1 className="text-xl sm:text-2xl font-black tracking-tight text-orange-500 shrink-0">
            VEREC
          </h1>
          <div className="hidden sm:flex items-center gap-0.5 ml-2">
            {(["feed", "export"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-3 py-1.5 rounded-full text-xs font-semibold transition-colors ${tab === t ? "bg-gray-700/80 text-orange-400" : "text-gray-500 hover:text-gray-300"}`}
              >
                {t === "feed" ? "Live Feed" : "Export"}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0 text-xs">
          <div
            className={`hidden md:flex items-center gap-1 font-mono px-2 py-1 rounded-full bg-gray-800/80 border border-gray-700/50 ${bwColor}`}
          >
            <span>↕</span> {fmtBytes(bandwidth)}/s
          </div>
          {sysInfo?.device && (
            <span className="hidden lg:inline px-2 py-1 rounded-full bg-gray-800/80 border border-gray-700/50 text-gray-400 font-mono uppercase">
              {sysInfo.device} · {sysInfo.machine}
            </span>
          )}
          {sysInfo?.local_ip && (
            <span className="hidden xl:inline px-2 py-1 rounded-full bg-gray-800/80 border border-gray-700/50 text-blue-400 font-mono text-[10px]">
              {sysInfo.local_ip}
            </span>
          )}
          <div className="font-mono tabular-nums px-2 py-1 rounded-full bg-gray-800/80 border border-gray-700/50">
            <span className={connected ? "text-green-400" : "text-gray-500"}>
              {fmtTime(elapsed)}
            </span>
          </div>
          <span
            className={`px-2 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider flex items-center gap-1 ${statusClass}`}
          >
            {connecting && <Spinner className="w-3 h-3" />}
            {statusLabel}
          </span>
          <button
            onClick={() => setSettingsOpen(!settingsOpen)}
            className={`p-1.5 rounded-full transition-colors ${settingsOpen ? "bg-orange-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"}`}
          >
            <GearIcon className="w-4 h-4" />
          </button>
          <button
            onClick={() => setGlobalSearchOpen(!globalSearchOpen)}
            className={`p-1.5 rounded-full transition-colors ${globalSearchOpen ? "bg-orange-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"}`}
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
              />
            </svg>
          </button>
        </div>
      </header>

      {/* ═══ Global search bar ═══════════════════════════════════ */}
      {globalSearchOpen && (
        <div className="shrink-0 px-3 py-2 bg-gray-900/90 backdrop-blur border-b border-gray-800/40">
          <div className="flex items-center gap-2 max-w-2xl mx-auto">
            <svg
              className="w-4 h-4 text-gray-500 shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
              />
            </svg>
            <input
              type="text"
              value={globalSearch}
              onChange={(e) => {
                const v = e.target.value;
                setGlobalSearch(v);
                setCaptionSearch(v);
                setLogSearch(v);
                setReportSearch(v);
              }}
              placeholder="Search all captions, detections, and reports…"
              autoFocus
              className="flex-1 bg-transparent text-gray-200 text-sm placeholder-gray-600 focus:outline-none"
            />
            {globalSearch && (
              <button
                onClick={() => {
                  setGlobalSearch("");
                  setCaptionSearch("");
                  setLogSearch("");
                  setReportSearch("");
                }}
                className="text-gray-500 hover:text-gray-300 text-xs"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      )}

      {/* ═══ Mobile tab bar ══════════════════════════════════════ */}
      <div className="sm:hidden flex border-b border-gray-800/40 bg-gray-900/60">
        {(["feed", "export"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 py-2 text-xs font-semibold text-center ${tab === t ? "text-orange-400 border-b-2 border-orange-500" : "text-gray-500"}`}
          >
            {t === "feed" ? "Live Feed" : "Export"}
          </button>
        ))}
      </div>

      {/* ═══ Settings slide-out panel ════════════════════════════ */}
      {settingsOpen && (
        <div className="absolute top-[52px] right-2 z-40 w-80 max-h-[calc(100vh-60px)] overflow-y-auto bg-gray-900/95 backdrop-blur-xl border border-gray-700/50 rounded-2xl shadow-2xl p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-gray-200">Settings</h3>
            <button
              onClick={() => setSettingsOpen(false)}
              className="text-gray-500 hover:text-gray-300 text-lg leading-none"
            >
              &times;
            </button>
          </div>

          {/* Source */}
          <div className="space-y-2">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
              Source
            </h4>
            <div className="flex gap-2">
              {(["local", "ip"] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setSource(s)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${source === s ? "bg-orange-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
                >
                  {s === "local" ? "Local Camera" : "IP / Stream"}
                </button>
              ))}
            </div>
            {source === "ip" && (
              <div className="space-y-2">
                <div className="flex flex-wrap gap-1">
                  {Object.keys(presets).map((k) => (
                    <button
                      key={k}
                      onClick={() => {
                        setSelectedPreset(k);
                        setStreamUrl(presets[k]);
                      }}
                      className={`px-2 py-1 rounded-full text-[10px] font-medium transition-colors ${selectedPreset === k ? "bg-orange-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700 border border-gray-700/50"}`}
                    >
                      {k.replace(/ \(.*\)/, "")}
                    </button>
                  ))}
                </div>
                <input
                  type="text"
                  value={streamUrl}
                  onChange={(e) => setStreamUrl(e.target.value)}
                  placeholder="YouTube / RTSP / MJPEG URL"
                  className="w-full bg-gray-800 text-gray-200 text-xs rounded-xl px-3 py-2 border border-gray-700/50 focus:outline-none focus:ring-1 focus:ring-orange-500/50"
                />
              </div>
            )}
          </div>

          {/* Model toggles */}
          <div className="space-y-2">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
              Models
            </h4>
            <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs">
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableDet}
                  onChange={(e) => setEnableDet(e.target.checked)}
                  className="accent-orange-500 w-3.5 h-3.5"
                />{" "}
                Detection
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableVlm}
                  onChange={(e) => setEnableVlm(e.target.checked)}
                  className="accent-orange-500 w-3.5 h-3.5"
                />{" "}
                VLM
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enablePose}
                  onChange={(e) => setEnablePose(e.target.checked)}
                  className="accent-purple-500 w-3.5 h-3.5"
                />{" "}
                Pose + Action
              </label>
            </div>
          </div>

          {/* Sliders */}
          <div className="space-y-2 text-xs text-gray-400">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
              Detection
            </h4>
            <label className="block">
              <span className="flex justify-between">
                <span>Confidence</span>
                <span className="text-orange-400 font-mono">
                  {conf.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={conf}
                onChange={(e) => setConf(+e.target.value)}
                className="w-full accent-orange-500 mt-1"
              />
            </label>
            <label className="block">
              <span className="flex justify-between">
                <span>IoU</span>
                <span className="text-orange-400 font-mono">
                  {iou.toFixed(2)}
                </span>
              </span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={iou}
                onChange={(e) => setIou(+e.target.value)}
                className="w-full accent-orange-500 mt-1"
              />
            </label>
            <label className="block">
              <span className="flex justify-between">
                <span>VLM Interval</span>
                <span className="text-orange-400 font-mono">
                  {vlmInterval}s
                </span>
              </span>
              <input
                type="range"
                min="3"
                max="15"
                step="1"
                value={vlmInterval}
                onChange={(e) => setVlmInterval(+e.target.value)}
                className="w-full accent-orange-500 mt-1"
              />
            </label>
          </div>

          {/* Auto-report */}
          <div className="space-y-2 text-xs text-gray-400">
            <h4 className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
              Auto Report
            </h4>
            <label className="block">
              <span className="flex justify-between">
                <span>Interval</span>
                <span className="text-orange-400 font-mono">
                  {autoReportInterval === 0 ? "Off" : `${autoReportInterval}s`}
                </span>
              </span>
              <input
                type="range"
                min="0"
                max="300"
                step="15"
                value={autoReportInterval}
                onChange={(e) => setAutoReportInterval(+e.target.value)}
                className="w-full accent-orange-500 mt-1"
              />
            </label>
            <p className="text-[10px] text-gray-600">
              0 = manual only. Reports use DeepSeek-R1 via Ollama.
            </p>
          </div>

          {/* System */}
          {sysInfo && (
            <div className="space-y-1 bg-gray-800/40 rounded-xl p-2.5 border border-gray-700/30">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                System
              </h4>
              <div className="text-[11px] text-gray-400 space-y-0.5 font-mono">
                <p>
                  {sysInfo.platform} {sysInfo.machine} · Python {sysInfo.python}
                </p>
                <p>
                  Device:{" "}
                  <span className="text-orange-400">{sysInfo.device}</span>
                </p>
                {sysInfo.local_ip && (
                  <p>
                    Local IP:{" "}
                    <span className="text-blue-400">{sysInfo.local_ip}</span>
                  </p>
                )}
                <p>Detector: {sysInfo.models?.detector ?? "–"}</p>
                <p>
                  Pose:{" "}
                  {(sysInfo.models as Record<string, string>)?.pose ?? "–"}
                </p>
                <p>
                  Action:{" "}
                  {(sysInfo.models as Record<string, string>)?.action ?? "–"}
                </p>
                <p>VLM: {sysInfo.models?.vlm ?? "–"}</p>
                <p>LLM: {sysInfo.models?.llm ?? "–"}</p>
                <p>
                  Logs: {sysInfo.log_counts?.detections ?? 0} det /{" "}
                  {sysInfo.log_counts?.captions ?? 0} cap /{" "}
                  {sysInfo.log_counts?.reports ?? 0} rpt
                </p>
              </div>
            </div>
          )}

          {/* Start/Stop */}
          <div className="flex gap-2 pt-1">
            <button
              onClick={() => {
                startFeed();
                setSettingsOpen(false);
              }}
              disabled={connecting}
              className="flex-1 bg-orange-600 hover:bg-orange-500 disabled:opacity-50 text-white text-sm font-bold py-2 rounded-xl transition-colors flex items-center justify-center gap-1.5"
            >
              {connecting ? (
                <>
                  <Spinner className="w-4 h-4" /> Connecting…
                </>
              ) : (
                "▶ Start"
              )}
            </button>
            <button
              onClick={stopFeed}
              className="flex-1 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm font-bold py-2 rounded-xl transition-colors"
            >
              ⏹ Stop
            </button>
          </div>
        </div>
      )}

      {/* ═══ Main content ═══════════════════════════════════════ */}
      <main className="flex-1 overflow-hidden">
        {tab === "feed" && (
          <div className="h-full flex flex-col">
            {/* ── Video area ───────────────────────────────────── */}
            <div className="flex-1 flex flex-col lg:flex-row min-h-0">
              {/* Big AI view */}
              <div className="flex-[3] min-h-0 flex flex-col">
                <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden rounded-2xl m-1">
                  {aiFrame ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={aiFrame}
                      alt="AI View"
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : (
                    <div className="text-center space-y-3 px-4">
                      {connecting ? (
                        <>
                          <Spinner className="w-10 h-10 text-orange-500 mx-auto" />
                          <p className="text-gray-400 text-sm">
                            Connecting to stream…
                          </p>
                          <p className="text-gray-600 text-[11px]">
                            Resolving URL and initializing models
                          </p>
                        </>
                      ) : (
                        <>
                          <div className="w-16 h-16 mx-auto rounded-2xl bg-gray-800/60 border border-gray-700/40 flex items-center justify-center">
                            <svg
                              className="w-8 h-8 text-gray-600"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                              strokeWidth={1.5}
                            >
                              <path
                                strokeLinecap="round"
                                d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"
                              />
                            </svg>
                          </div>
                          <p className="text-gray-500 text-sm">
                            Press{" "}
                            <span className="text-orange-500 font-semibold">
                              ▶ Start
                            </span>{" "}
                            or open{" "}
                            <GearIcon className="inline w-4 h-4 -mt-0.5 text-gray-400" />{" "}
                            settings
                          </p>
                        </>
                      )}
                    </div>
                  )}
                  {connected && (
                    <span className="absolute top-2 left-2 flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-red-600/90 text-white text-[10px] font-bold">
                      <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />{" "}
                      LIVE
                    </span>
                  )}
                  {/* Action overlay */}
                  {connected && currentAction && currentAction.length > 0 && (
                    <div className="absolute top-2 left-20 flex gap-1.5">
                      <span className="px-2 py-0.5 rounded-full bg-purple-600/90 backdrop-blur-sm text-white text-[10px] font-bold shadow-lg">
                        🏃 {currentAction[0].label}{" "}
                        <span className="opacity-75">
                          {Math.round(currentAction[0].confidence * 100)}%
                        </span>
                      </span>
                    </div>
                  )}
                  {/* Stats overlay */}
                  <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-3 py-2 flex items-end justify-between text-[11px] font-mono">
                    <span className="text-gray-300">
                      {detectionInfo || "Waiting…"}
                    </span>
                    <span className={bwColor}>{fmtBytes(bandwidth)}/s</span>
                  </div>
                  {/* Floating Start / Stop controls */}
                  <div className="absolute top-2 right-2 z-10">
                    {!connected && !connecting && (
                      <div className="relative">
                        <div className="flex">
                          <button
                            onClick={startFeed}
                            className="bg-orange-600 hover:bg-orange-500 text-white text-xs font-bold px-3 py-1.5 rounded-l-full shadow-lg transition-colors"
                          >
                            ▶ Start
                          </button>
                          <button
                            onClick={() => setStartOpen(!startOpen)}
                            className="bg-orange-700 hover:bg-orange-600 text-white text-xs font-bold px-2 py-1.5 rounded-r-full shadow-lg transition-colors border-l border-orange-500/40"
                          >
                            {startOpen ? "▴" : "▾"}
                          </button>
                        </div>
                        {startOpen && (
                          <div className="absolute top-full right-0 mt-1.5 w-72 bg-gray-900/95 backdrop-blur-xl border border-gray-700/50 rounded-xl shadow-2xl p-3 space-y-2">
                            <div className="flex gap-1">
                              {(["local", "ip"] as const).map((s) => (
                                <button
                                  key={s}
                                  onClick={() => setSource(s)}
                                  className={`px-2 py-1 rounded-full text-[10px] font-medium transition-colors ${source === s ? "bg-orange-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
                                >
                                  {s === "local" ? "Camera" : "Stream"}
                                </button>
                              ))}
                            </div>
                            {source === "ip" && (
                              <>
                                <input
                                  type="text"
                                  value={quickUrl}
                                  onChange={(e) => {
                                    setQuickUrl(e.target.value);
                                    setStreamUrl(e.target.value);
                                  }}
                                  placeholder="Paste YouTube / RTSP / MJPEG URL"
                                  className="w-full bg-gray-800 text-gray-200 text-xs rounded-lg px-3 py-2 border border-gray-700/50 focus:outline-none focus:ring-1 focus:ring-orange-500/50"
                                />
                                <div className="flex flex-wrap gap-1">
                                  {Object.keys(presets).map((k) => (
                                    <button
                                      key={k}
                                      onClick={() => {
                                        setSelectedPreset(k);
                                        setStreamUrl(presets[k]);
                                        setQuickUrl(presets[k]);
                                      }}
                                      className={`px-2 py-0.5 rounded-full text-[9px] font-medium transition-colors ${selectedPreset === k ? "bg-orange-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700 border border-gray-700/50"}`}
                                    >
                                      {k.replace(/ \(.*\)/, "")}
                                    </button>
                                  ))}
                                </div>
                              </>
                            )}
                            <button
                              onClick={() => {
                                startFeed();
                                setStartOpen(false);
                              }}
                              className="w-full bg-orange-600 hover:bg-orange-500 text-white text-xs font-bold py-1.5 rounded-lg transition-colors"
                            >
                              ▶ Start Stream
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                    {connecting && (
                      <span className="bg-yellow-600/90 text-white text-xs font-bold px-3 py-1.5 rounded-full shadow-lg flex items-center gap-1">
                        <Spinner className="w-3 h-3" /> Connecting…
                      </span>
                    )}
                    {connected && (
                      <button
                        onClick={stopFeed}
                        className="bg-gray-700/90 hover:bg-gray-600 text-gray-200 text-xs font-bold px-3 py-1.5 rounded-full shadow-lg transition-colors"
                      >
                        ⏹ Stop
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Right sidebar: raw feed + captions */}
              <div className="hidden lg:flex lg:w-72 xl:w-80 shrink-0 flex-col bg-gray-900/60 border-l border-gray-800/40">
                <div className="aspect-video bg-black flex items-center justify-center overflow-hidden shrink-0 rounded-2xl m-1">
                  {rawFrame ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={rawFrame}
                      alt="Raw Feed"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <span className="text-gray-700 text-[10px]">Raw Feed</span>
                  )}
                </div>
                <div className="px-3 py-2 border-t border-gray-800/40 flex-1 overflow-y-auto flex flex-col">
                  <div className="flex items-center justify-between mb-2 shrink-0">
                    <h4 className="text-[10px] font-semibold uppercase tracking-wider text-orange-400">
                      Scene Captions
                    </h4>
                    <span className="text-[8px] text-gray-600 font-mono">
                      {captions.length}
                    </span>
                  </div>
                  {captions.length > 3 && (
                    <input
                      type="text"
                      value={captionSearch}
                      onChange={(e) => setCaptionSearch(e.target.value)}
                      placeholder="Search captions…"
                      className="w-full mb-1.5 bg-gray-800/60 text-gray-300 text-[11px] rounded-lg px-2 py-1 border border-gray-700/40 focus:outline-none focus:ring-1 focus:ring-orange-500/40"
                    />
                  )}
                  {captions.length === 0 ? (
                    <p className="text-[11px] text-gray-600 italic">
                      {connected || connecting
                        ? "Waiting for captions…"
                        : "No captions yet."}
                    </p>
                  ) : (
                    <div className="space-y-1.5 overflow-y-auto max-h-[50vh]">
                      {[...captions]
                        .reverse()
                        .filter(
                          (c) =>
                            !captionSearch ||
                            c.text
                              .toLowerCase()
                              .includes(captionSearch.toLowerCase()),
                        )
                        .map((c, i) => (
                          <ExpandableCard
                            key={`${c.timestamp}-${i}`}
                            timestamp={c.timestamp}
                            preview={c.text}
                            defaultOpen={i === 0}
                          >
                            {c.text.split("\n").map((l, j) => (
                              <p
                                key={j}
                                className="text-[11px] text-gray-300 leading-relaxed"
                              >
                                {l}
                              </p>
                            ))}
                          </ExpandableCard>
                        ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* ── Bottom panel — independent scroll columns ──── */}
            <div
              className="shrink-0 border-t border-gray-800/40 flex flex-col md:flex-row"
              style={{ height: "40vh" }}
            >
              {/* Mobile: caption cards (hidden on lg, shown above in sidebar) */}
              <div className="lg:hidden px-3 py-2 border-b md:border-b-0 md:border-r border-gray-800/30 min-w-0 flex-1 overflow-y-auto">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-gray-600">
                    Scene Captions
                  </h4>
                  <span className="text-[9px] text-gray-600 font-mono">
                    {captions.length}
                  </span>
                </div>
                {captions.length > 3 && (
                  <input
                    type="text"
                    value={captionSearch}
                    onChange={(e) => setCaptionSearch(e.target.value)}
                    placeholder="Search captions…"
                    className="w-full mb-1.5 bg-gray-800/60 text-gray-300 text-[11px] rounded-lg px-2 py-1 border border-gray-700/40 focus:outline-none focus:ring-1 focus:ring-orange-500/40"
                  />
                )}
                {captions.length === 0 ? (
                  <p className="text-[11px] text-gray-600 italic">
                    {connected || connecting
                      ? "Waiting for captions…"
                      : "No captions yet."}
                  </p>
                ) : (
                  <div className="space-y-1.5">
                    {[...captions]
                      .reverse()
                      .filter(
                        (c) =>
                          !captionSearch ||
                          c.text
                            .toLowerCase()
                            .includes(captionSearch.toLowerCase()),
                      )
                      .map((c, i) => (
                        <ExpandableCard
                          key={`${c.timestamp}-${i}`}
                          timestamp={c.timestamp}
                          preview={c.text}
                          defaultOpen={i === 0}
                        >
                          <p className="text-[11px] text-gray-300 leading-relaxed">
                            {c.text}
                          </p>
                        </ExpandableCard>
                      ))}
                  </div>
                )}
              </div>

              {/* Detections — independent scroll */}
              <div className="flex-1 p-3 md:border-r border-gray-800/30 min-w-0 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between mb-1.5 shrink-0">
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-orange-400">
                    Detections
                  </h4>
                  <span className="text-[9px] text-gray-600 font-mono">
                    {log.length}
                  </span>
                </div>
                {/* Object frequency stats */}
                {Object.keys(objectCounts).length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-1.5 shrink-0">
                    {Object.entries(objectCounts)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 8)
                      .map(([cls, count]) => (
                        <span
                          key={cls}
                          className="px-1.5 py-0.5 rounded-full bg-gray-800/60 border border-gray-700/40 text-[9px] font-mono text-gray-400"
                        >
                          {cls} <span className="text-orange-400">{count}</span>
                        </span>
                      ))}
                    {Object.keys(objectCounts).length > 8 && (
                      <span className="text-[9px] text-gray-600">
                        +{Object.keys(objectCounts).length - 8} more
                      </span>
                    )}
                  </div>
                )}
                {log.length > 3 && (
                  <input
                    type="text"
                    value={logSearch}
                    onChange={(e) => setLogSearch(e.target.value)}
                    placeholder="Search detections…"
                    className="w-full mb-1.5 shrink-0 bg-gray-800/60 text-gray-300 text-[11px] rounded-lg px-2 py-1 border border-gray-700/40 focus:outline-none focus:ring-1 focus:ring-orange-500/40"
                  />
                )}
                <div className="flex-1 overflow-y-auto">
                  {log.length === 0 && !connected && !connecting ? (
                    <p className="text-[11px] text-gray-600 italic">
                      No detections yet.
                    </p>
                  ) : log.length === 0 && (connected || connecting) ? (
                    <div className="flex items-center gap-2 text-[11px] text-gray-500">
                      <Spinner className="w-3.5 h-3.5" /> Waiting for
                      detections…
                    </div>
                  ) : (
                    <pre className="text-[11px] text-gray-400 font-mono whitespace-pre-wrap leading-relaxed">
                      {(logSearch
                        ? log.filter((l) =>
                            l.toLowerCase().includes(logSearch.toLowerCase()),
                          )
                        : log
                      ).join("\n")}
                    </pre>
                  )}
                </div>
              </div>

              {/* Actions — independent scroll */}
              <div className="flex-1 p-3 md:border-r border-gray-800/30 min-w-0 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between mb-1.5 shrink-0">
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-purple-400">
                    Actions
                  </h4>
                  <span className="text-[9px] text-gray-600 font-mono">
                    {actionLog.length}
                  </span>
                </div>
                {/* Current action highlight */}
                {currentAction && currentAction.length > 0 && (
                  <div className="flex flex-wrap gap-1 mb-1.5 shrink-0">
                    <span className="text-[9px] text-purple-400 uppercase tracking-wider font-semibold mr-1 self-center">
                      Now
                    </span>
                    {currentAction.map((a, i) => (
                      <span
                        key={`bottom-${a.label}-${i}`}
                        className={`px-1.5 py-0.5 rounded-full text-[9px] font-mono border ${
                          i === 0
                            ? "bg-purple-900/40 border-purple-600/50 text-purple-300"
                            : "bg-gray-800/60 border-gray-700/40 text-gray-400"
                        }`}
                      >
                        {a.label}{" "}
                        <span
                          className={
                            i === 0 ? "text-purple-400" : "text-gray-500"
                          }
                        >
                          {Math.round(a.confidence * 100)}%
                        </span>
                      </span>
                    ))}
                  </div>
                )}
                {poseInfo && (
                  <div className="text-[9px] text-gray-500 font-mono mb-1 shrink-0">
                    🦴 {poseInfo}
                  </div>
                )}
                {actionLog.length > 3 && (
                  <input
                    type="text"
                    value={actionSearch}
                    onChange={(e) => setActionSearch(e.target.value)}
                    placeholder="Search actions…"
                    className="w-full mb-1.5 shrink-0 bg-gray-800/60 text-gray-300 text-[11px] rounded-lg px-2 py-1 border border-purple-700/40 focus:outline-none focus:ring-1 focus:ring-purple-500/40"
                  />
                )}
                <div className="flex-1 overflow-y-auto space-y-1">
                  {actionLog.length === 0 ? (
                    <p className="text-[11px] text-gray-600 italic">
                      {connected || connecting
                        ? enablePose
                          ? "Accumulating pose data (100 frames)…"
                          : "Pose + Action disabled in settings"
                        : "No actions captured yet."}
                    </p>
                  ) : (
                    [...actionLog]
                      .reverse()
                      .filter(
                        (a) =>
                          !actionSearch ||
                          a.actions.some((act) =>
                            act.label
                              .toLowerCase()
                              .includes(actionSearch.toLowerCase()),
                          ),
                      )
                      .map((a, i) => (
                        <div
                          key={`bl-${a.timestamp}-${i}`}
                          className="flex items-center gap-2 text-[11px]"
                        >
                          <span className="text-[9px] text-gray-600 font-mono shrink-0">
                            {new Date(a.timestamp).toLocaleTimeString()}
                          </span>
                          <div className="flex flex-wrap gap-1">
                            {a.actions.map((act, j) => (
                              <span
                                key={`${act.label}-${j}`}
                                className={`px-1.5 py-0.5 rounded-full text-[9px] font-mono ${
                                  j === 0
                                    ? "bg-purple-900/30 text-purple-300"
                                    : "text-gray-500"
                                }`}
                              >
                                {act.label} {Math.round(act.confidence * 100)}%
                              </span>
                            ))}
                          </div>
                        </div>
                      ))
                  )}
                </div>
              </div>

              {/* Report panel — independent scroll */}
              <div className="flex-1 p-3 min-w-0 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between shrink-0">
                  <h4 className="text-[10px] font-semibold uppercase tracking-wider text-gray-600">
                    LLM Reports
                  </h4>
                  <div className="flex items-center gap-2">
                    {autoReportInterval > 0 && connected && (
                      <span className="text-[9px] text-green-500/80 font-mono">
                        auto {autoReportInterval}s
                      </span>
                    )}
                    <span className="text-[9px] text-gray-600 font-mono">
                      {reports.length}
                    </span>
                  </div>
                </div>
                <button
                  onClick={handleReport}
                  disabled={reportLoading}
                  className="w-full mt-1.5 bg-orange-600 hover:bg-orange-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-xs font-bold py-1.5 rounded-xl transition-colors shrink-0 flex items-center justify-center gap-1.5"
                >
                  {reportLoading ? (
                    <>
                      <Spinner className="w-3.5 h-3.5" /> Generating…
                    </>
                  ) : (
                    "Generate Report"
                  )}
                </button>

                {reports.length > 3 && (
                  <input
                    type="text"
                    value={reportSearch}
                    onChange={(e) => setReportSearch(e.target.value)}
                    placeholder="Search reports…"
                    className="w-full mt-1.5 shrink-0 bg-gray-800/60 text-gray-300 text-[11px] rounded-lg px-2 py-1 border border-gray-700/40 focus:outline-none focus:ring-1 focus:ring-orange-500/40"
                  />
                )}

                {/* All reports as scrollable cards — latest expanded */}
                <div className="flex-1 overflow-y-auto mt-1.5 space-y-1.5">
                  {reports.length > 0 ? (
                    [...reports]
                      .reverse()
                      .filter(
                        (r) =>
                          !reportSearch ||
                          r.report
                            .toLowerCase()
                            .includes(reportSearch.toLowerCase()),
                      )
                      .map((r, i) => (
                        <ExpandableCard
                          key={`${r.timestamp}-${i}`}
                          timestamp={r.timestamp}
                          preview={
                            r.report
                              .split("\n")
                              .find((l) => l.trim().length > 0) ?? ""
                          }
                          defaultOpen={i === 0}
                          meta={
                            <>
                              <span>{r.model}</span>
                              <span>{r.detection_count} det</span>
                              <span>{r.caption_count} cap</span>
                            </>
                          }
                        >
                          <MdReport text={r.report} />
                        </ExpandableCard>
                      ))
                  ) : (
                    <p className="text-[11px] text-gray-600 italic">
                      No reports yet.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── Export tab ─────────────────────────────────────────── */}
        {tab === "export" && (
          <div className="p-4 sm:p-6 space-y-4 overflow-y-auto h-full">
            <h2 className="text-lg font-bold text-gray-200">Data Export</h2>
            <p className="text-sm text-gray-400">
              All data is automatically saved to{" "}
              <code className="text-orange-400/80">logs/</code> on disk.
              Download a combined JSON snapshot below.
            </p>
            <button
              onClick={handleExport}
              className="bg-orange-600 hover:bg-orange-500 text-white font-bold px-6 py-2.5 rounded-xl transition-colors"
            >
              Download All Data (JSON)
            </button>
            {sysInfo && (
              <div className="text-xs text-gray-500 font-mono">
                On disk: {sysInfo.log_counts?.detections ?? 0} detections ·{" "}
                {sysInfo.log_counts?.captions ?? 0} captions ·{" "}
                {sysInfo.log_counts?.reports ?? 0} reports
              </div>
            )}
            {exportData && (
              <div className="bg-gray-900 border border-gray-700/40 rounded-2xl p-4 max-h-80 overflow-auto">
                <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap">
                  {JSON.stringify(exportData, null, 2).slice(0, 5000)}
                  {JSON.stringify(exportData, null, 2).length > 5000
                    ? "\n…(truncated)"
                    : ""}
                </pre>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
