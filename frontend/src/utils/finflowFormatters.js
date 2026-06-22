export function getHomeRoute(role) {
  if (role === "manager" || role === "admin") return "/manager";
  return "/dashboard";
}

export function formatJobStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  if (normalized === "quarantined") return "Quarantined";
  if (normalized === "schema_review") return "Schema review";
  if (normalized === "clarification") return "Needs your input";
  if (normalized === "succeeded") return "Succeeded";
  if (normalized === "callback_failed") return "Callback failed";
  if (normalized === "declined") return "Declined";
  return String(status || "running")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatStepStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  if (normalized === "complete" || normalized === "succeeded") return "Complete";
  if (normalized === "running" || normalized === "queued" || normalized === "planning") return "Running";
  if (normalized === "blocked") return "Blocked";
  if (normalized === "failed" || normalized === "callback_failed" || normalized === "declined") return "Failed";
  return "Running";
}

export function formatDateTime(value) {
  if (!value) return "In progress";
  return new Date(value).toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function summarizeInstruction(instruction) {
  if (!instruction) return "";
  return instruction.length > 92 ? `${instruction.slice(0, 89)}...` : instruction;
}

export function formatAgentName(name) {
  const labels = {
    file_cleaning_agent: "Cleaner",
    chart_generation_agent: "Chart Builder",
    fx_conversion_agent: "FX Converter",
    ocr_extraction_agent: "OCR Extractor",
    cost_model_agent: "Cost Modeler",
  };

  if (labels[name]) return labels[name];

  return String(name || "")
    .replaceAll("_", " ")
    .replace(/\bfx\b/gi, "FX")
    .replace(/\bocr\b/gi, "OCR")
    .replace(/\bgl\b/gi, "GL")
    .replace(/\berp\b/gi, "ERP")
    .replace(/\bagent\b/gi, "Agent")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatCapabilityTag(tag) {
  const labels = {
    cleaning: "Cleaning",
    xlsx: "XLSX",
    csv: "CSV",
    "schema-normalisation": "Schema normalisation",
    charts: "Charts",
    "variance-analysis": "Variance analysis",
    "pdf-export": "PDF export",
    "fx-conversion": "FX conversion",
    treasury: "Treasury",
    reconciliation: "Reconciliation",
    ocr: "OCR",
    pdf: "PDF",
    "image-extraction": "Image extraction",
    headcount: "Headcount",
    "cost-model": "Cost model",
    forecasting: "Forecasting",
  };

  return labels[tag] || String(tag || "").replaceAll("-", " ");
}
