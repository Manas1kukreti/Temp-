import React, { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FiCheck, FiChevronDown, FiDownload, FiRefreshCw, FiX } from "react-icons/fi";
import { useParams } from "react-router-dom";
import DataTable from "../components/DataTable.jsx";
import ClarificationPanel from "../components/ClarificationPanel.jsx";
import {
  approveSchemaProposal,
  declineSchemaProposal,
  downloadJobOutput,
  fetchClarificationStatus,
  fetchJobDetail,
  retryJob,
  confirmExtraction,
} from "../api/finflow.js";
import { useLiveJobRefresh } from "../hooks/useLiveJobRefresh.js";
import { useClarificationSocket } from "../hooks/useClarificationSocket.js";
import {
  formatDateTime,
  formatJobStatus,
  formatStepStatus,
} from "../utils/finflowFormatters.js";

export default function AuditPage() {
  const { jobId } = useParams();
  const [openIndex, setOpenIndex] = useState(0);
  const queryClient = useQueryClient();
  const {
    data: job,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["jobs", "detail", jobId],
    queryFn: () => fetchJobDetail(jobId),
    enabled: Boolean(jobId),
  });
  useLiveJobRefresh(jobId);

  // Clarification WebSocket integration
  const {
    session: clarificationSession,
    isActive: clarificationActive,
    isResolved: clarificationResolved,
    isExpired: clarificationExpired,
    refreshSession: refreshClarificationSession,
    dismiss: dismissClarification,
  } = useClarificationSocket(jobId);

  // REST fallback: fetch clarification session state on page load when job is
  // already in "awaiting_clarification" status but WebSocket hasn't delivered the event yet
  const showClarificationFromStatus = job?.status === "clarification" && !clarificationActive;
  const { data: restClarificationSession } = useQuery({
    queryKey: ["clarification-status", jobId],
    queryFn: () => fetchClarificationStatus(jobId),
    enabled: Boolean(jobId) && showClarificationFromStatus,
    retry: 1,
    staleTime: 30000,
  });

  // Determine the active clarification data: prefer WebSocket, fallback to REST
  const activeClarificationData = clarificationActive
    ? clarificationSession
    : showClarificationFromStatus && restClarificationSession
      ? {
          sessionId: restClarificationSession.session_id,
          submissionId: restClarificationSession.submission_id || jobId,
          questions: restClarificationSession.questions || [],
          roundCount: restClarificationSession.round_count ?? 0,
          maxRounds: restClarificationSession.max_rounds ?? 2,
          expiresAt: restClarificationSession.expires_at,
          revisionToken: restClarificationSession.revision_token,
          intentVersion: restClarificationSession.intent_version,
        }
      : null;
  const shouldShowClarificationPanel = clarificationActive || Boolean(activeClarificationData);

  const retryMutation = useMutation({
    mutationFn: retryJob,
    onSuccess: async () => {
      await refreshJobViews(queryClient, jobId);
    },
  });

  const approveSchemaMutation = useMutation({
    mutationFn: approveSchemaProposal,
    onSuccess: async () => {
      await refreshJobViews(queryClient, jobId);
    },
  });

  const declineSchemaMutation = useMutation({
    mutationFn: declineSchemaProposal,
    onSuccess: async () => {
      await refreshJobViews(queryClient, jobId);
    },
  });

  const confirmExtractionMutation = useMutation({
    mutationFn: () => confirmExtraction(jobId, job?.previewToken),
    onSuccess: async () => {
      await refreshJobViews(queryClient, jobId);
    },
  });

  const schemaProposal = job?.schemaProposal || {};
  const schemaProposalKeys = schemaProposal && typeof schemaProposal === "object" ? Object.keys(schemaProposal) : [];
  const schemaApprovalVisible =
    (
      job?.status === "schema_review"
      || Boolean(schemaProposal.requires_user_approval)
      || schemaProposal.schema_kind === "tabular"
    );
  const extractionPreviewVisible =
    (
      job?.status === "awaiting_confirmation"
      || schemaProposal.schema_kind === "unstructured_extraction_preview"
    );
  const proposedFields = Array.isArray(schemaProposal.proposed_fields)
    ? schemaProposal.proposed_fields.filter(f => f && typeof f === "object")
    : [];
  const validationWarnings = Array.isArray(schemaProposal.validation_warnings)
    ? schemaProposal.validation_warnings.filter(w => w && typeof w === "object")
    : [];
  const rawActions = schemaProposal.action_schema?.actions;
  const actionSchema = Array.isArray(rawActions) ? rawActions.filter(a => a && typeof a === "object") : [];
  const schemaColumns = Array.isArray(job?.columns) ? job.columns : [];
  const schemaRows = Array.isArray(job?.previewRows) ? job.previewRows : [];
  const detectedTypes = job?.detectedTypes && typeof job.detectedTypes === "object"
    ? Object.entries(job.detectedTypes)
    : [];
  const extractionAnchor = String(schemaProposal.anchor_column || "");
  const extractionCompleteCount = Number(schemaProposal.complete_count || 0);
  const extractionPartialCount = Number(schemaProposal.partial_count || 0);
  const extractionInvalidCount = Number(schemaProposal.invalid_count || 0);
  const extractionLlmOnlyCount = Number(schemaProposal.llm_only_count || 0);
  const extractionRecoveredCount = Number(schemaProposal.recovered_count || 0);
  const extractionMergedCount = Number(schemaProposal.merged_count || 0);
  const extractionAmbiguousDateCount = Number(schemaProposal.ambiguous_date_count || 0);
  const extractionAssumedDateConvention = String(schemaProposal.assumed_date_convention || "");

  if (isLoading) {
    return (
      <div className="ff-page-grid">
        <section className="ff-panel">
          <h2>Loading job detail...</h2>
        </section>
      </div>
    );
  }

  if (isError || !job) {
    return (
      <div className="ff-page-grid">
        <section className="ff-panel">
          <h2>We could not load this job.</h2>
          <p className="ff-copy-muted">
            The workflow may have been removed or the backend may be
            unavailable.
          </p>
        </section>
      </div>
    );
  }

  return (
    <div className="ff-page-grid">
      <section className="ff-panel">
        <div className="ff-panel__head">
          <div>
            <p className="ff-eyebrow">Job detail and audit</p>
            <h2>{job.title}</h2>
            <p className="ff-copy-muted">{job.instruction}</p>
          </div>
          <div className="ff-detail-actions">
            <span className={`ff-status ff-status--${job.status}`}>
              {formatJobStatus(job.status)}
            </span>
            {job.status === "complete" && job.outputReady ? (
              <button
                type="button"
                className="ff-secondary-button"
                onClick={() => downloadJobOutput(job.backendId)}
              >
                <FiDownload size={15} />
                Download output
              </button>
            ) : null}
            {job.status === "failed" ? (
              <button
                type="button"
                className="ff-secondary-button"
                onClick={() => retryMutation.mutate(job.backendId)}
                disabled={retryMutation.isPending}
              >
                <FiRefreshCw size={15} />
                {retryMutation.isPending ? "Requeueing..." : "Retry job"}
              </button>
            ) : null}
            {job.status === "schema_review" ? (
              <>
                <button
                  type="button"
                  className="ff-secondary-button"
                  onClick={() => declineSchemaMutation.mutate(job.backendId)}
                  disabled={approveSchemaMutation.isPending || declineSchemaMutation.isPending}
                >
                  <FiX size={15} />
                  {declineSchemaMutation.isPending ? "Declining..." : "Decline schema"}
                </button>
                <button
                  type="button"
                  className="ff-primary-button"
                  onClick={() => approveSchemaMutation.mutate(job.backendId)}
                  disabled={approveSchemaMutation.isPending || declineSchemaMutation.isPending}
                >
                  <FiCheck size={15} />
                  {approveSchemaMutation.isPending ? "Approving..." : "Approve schema"}
                </button>
              </>
            ) : null}
            {extractionPreviewVisible ? (
              <>
                <button
                  type="button"
                  className="ff-primary-button"
                  onClick={() => confirmExtractionMutation.mutate()}
                  disabled={confirmExtractionMutation.isPending}
                >
                  <FiDownload size={15} />
                  {confirmExtractionMutation.isPending ? "Confirming..." : "Confirm & Download Excel"}
                </button>
              </>
            ) : null}
          </div>
        </div>

        <div className="ff-key-metrics" style={{ margin: "2rem 0" }}>
          <div>
            <span>Job ID</span>
            <strong>{job.id}</strong>
          </div>
          <div>
            <span>Submitted by</span>
            <strong>{job.submittedBy}</strong>
          </div>
          <div>
            <span>Submitted at</span>
            <strong>{formatDateTime(job.submittedAt)}</strong>
          </div>
          <div>
            <span>Completed at</span>
            <strong>{formatDateTime(job.completedAt)}</strong>
          </div>
        </div>

        {/* Clarification Panel — rendered when WebSocket delivers clarification events or status is awaiting_clarification */}
        {shouldShowClarificationPanel && activeClarificationData && (
          <ClarificationPanel
            submissionId={activeClarificationData.submissionId}
            sessionId={activeClarificationData.sessionId}
            questions={activeClarificationData.questions}
            roundCount={activeClarificationData.roundCount}
            maxRounds={activeClarificationData.maxRounds}
            expiresAt={activeClarificationData.expiresAt}
            revisionToken={activeClarificationData.revisionToken}
            intentVersion={activeClarificationData.intentVersion}
            onResolved={() => {
              dismissClarification();
              queryClient.invalidateQueries({ queryKey: ["jobs", "detail", jobId] });
              queryClient.invalidateQueries({ queryKey: ["clarification-status", jobId] });
            }}
            onSessionExpired={() => {
              dismissClarification();
              queryClient.invalidateQueries({ queryKey: ["jobs", "detail", jobId] });
              queryClient.invalidateQueries({ queryKey: ["clarification-status", jobId] });
            }}
            onSessionRefresh={refreshClarificationSession}
          />
        )}
        {clarificationResolved && (
          <section className="ff-panel ff-panel--dense" aria-live="polite" aria-label="Clarification resolved">
            <div className="ff-clarification-success">
              <h3>Ambiguities resolved</h3>
              <p className="ff-copy-muted">Your answers have been applied. The job is now running.</p>
            </div>
          </section>
        )}
        {clarificationExpired && (
          <section className="ff-panel ff-panel--dense" aria-live="polite" aria-label="Session expired">
            <div className="ff-clarification-expired">
              <h3>Session expired</h3>
              <p className="ff-copy-muted">
                The clarification session has expired. This job has been moved to quarantine.
              </p>
            </div>
          </section>
        )}
        <section className="ff-panel ff-panel--dense">
          <div className="ff-panel__head">
            <div>
              <p className="ff-eyebrow">Job summary</p>
              <h3>Concise outcome</h3>
            </div>
          </div>
          <p className="ff-job-summary">
            {job.jobSummary || "A concise summary will appear here once the workflow finishes."}
          </p>
        </section>
        {job.status === "quarantined" ? (
          <div className="ff-panel ff-panel--dense">
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Quarantine outcome</p>
                <h3>Part of this workflow requires review</h3>
              </div>
            </div>
            <div className="ff-key-metrics">
              <div>
                <span>Reason</span>
                <strong>{job.reason || "Part of the workflow is not supported by the current agent coverage."}</strong>
              </div>
              <div>
                <span>Next step</span>
                <strong>{job.suggestion || "The unsupported portion is quarantined until coverage is added or the request is adjusted."}</strong>
              </div>
              <div>
                <span>Quarantine status</span>
                <strong>{formatJobStatus(job.quarantineStatus || "queued_for_review")}</strong>
              </div>
              <div>
                <span>Available agents</span>
                <strong>{Array.isArray(job?.availableAgents) && job.availableAgents.length ? job.availableAgents.join(", ") : "None registered for this intent"}</strong>
              </div>
              <div>
                <span>Preferred agent</span>
                <strong>{job.preferredAgentName || "Not assigned"}</strong>
              </div>
            </div>
          </div>
        ) : null}
        {schemaApprovalVisible ? (
          <div className="ff-panel ff-panel--dense">
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Schema approval</p>
                <h3>Review the proposed mapping before processing starts</h3>
              </div>
            </div>
            <div className="ff-key-metrics ff-key-metrics--compact">
              <div>
                <span>Detected structure</span>
                <strong>{schemaProposal.schema_kind || "Tabular preview"}</strong>
              </div>
              <div>
                <span>Suggested next step</span>
                <strong>{job.suggestion || "Approve if the field mapping looks correct."}</strong>
              </div>
              <div>
                <span>Detected fields</span>
                <strong>{schemaColumns.length || proposedFields.length}</strong>
              </div>
              <div>
                <span>Preview rows</span>
                <strong>{schemaRows.length}</strong>
              </div>
              <div>
                <span>Validation warnings</span>
                <strong>{validationWarnings.length}</strong>
              </div>
            </div>
          </div>
        ) : null}
        {extractionPreviewVisible ? (
          <div className="ff-panel ff-panel--dense">
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Extraction preview</p>
                <h3>Review extracted columns and rows before confirmation</h3>
              </div>
            </div>
            <div className="ff-key-metrics ff-key-metrics--compact">
              <div>
                <span>Anchor column</span>
                <strong>{extractionAnchor || "Not inferred"}</strong>
              </div>
              <div>
                <span>Complete rows</span>
                <strong>{extractionCompleteCount}</strong>
              </div>
              <div>
                <span>Partial rows</span>
                <strong>{extractionPartialCount}</strong>
              </div>
              <div>
                <span>Invalid rows</span>
                <strong>{extractionInvalidCount}</strong>
              </div>
              <div>
                <span>Extracted fields</span>
                <strong>{schemaColumns.length}</strong>
              </div>
              <div>
                <span>Recovered rows</span>
                <strong>{extractionRecoveredCount}</strong>
              </div>
              <div>
                <span>Merged rows</span>
                <strong>{extractionMergedCount}</strong>
              </div>
              <div>
                <span>LLM-only rows</span>
                <strong>{extractionLlmOnlyCount}</strong>
              </div>
            </div>
            {extractionAmbiguousDateCount ? (
              <p className="ff-copy-muted" style={{ marginTop: 16 }}>
                {extractionAmbiguousDateCount} row(s) had ambiguous slash dates. The preview assumed{" "}
                {extractionAssumedDateConvention || "DD/MM/YYYY"}.
              </p>
            ) : null}
            {proposedFields.length ? (
              <div className="ff-schema-grid" style={{ marginTop: 18 }}>
                {proposedFields.map((field) => (
                  <div key={`${field.source}-${field.target}`} className="ff-schema-card">
                    <div className="ff-schema-card__head">
                      <strong>{field.target || field.source || "Unmapped field"}</strong>
                      <span>{formatConfidence(field.confidence)}</span>
                    </div>
                    <p>
                      <span>Source:</span> {field.source || "Unknown"}
                    </p>
                    <p>
                      <span>Detected type:</span> {field.detected_type || "Unknown"}
                    </p>
                    <p>
                      <span>Why:</span> {field.reason || "Extracted from the uploaded file."}
                    </p>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="ff-detail-layout">
        {schemaApprovalVisible ? (
          <>
            <article className="ff-panel" style={{ gridRow: "span 2", marginBottom: "2rem" }}>
              <div className="ff-panel__head">
                <div>
                  <p className="ff-eyebrow">Proposed mapping</p>
                  <h3>Source fields and their target structure</h3>
                </div>
              </div>
              <div className="ff-schema-grid">
                {proposedFields.length ? proposedFields.map((field) => (
                  <div key={`${field.source}-${field.target}`} className="ff-schema-card">
                    <div className="ff-schema-card__head">
                      <strong>{field.target || field.source || "Unmapped field"}</strong>
                      <span>{formatConfidence(field.confidence)}</span>
                    </div>
                    <p>
                      <span>Source:</span> {field.source || "Unknown"}
                    </p>
                    <p>
                      <span>Detected type:</span> {field.detected_type || "Unknown"}
                    </p>
                    <p>
                      <span>Why:</span> {field.reason || "Matched from the uploaded structure."}
                    </p>
                  </div>
                )) : (
                  <div className="ff-copy-muted">
                    No proposed mapping fields are available for this upload.
                  </div>
                )}
              </div>
              {detectedTypes.length ? (
                <div className="ff-tag-list" style={{ marginTop: 18 }}>
                  {detectedTypes.map(([field, type]) => (
                    <span key={field}>{field}: {String(type)}</span>
                  ))}
                </div>
              ) : null}
              {validationWarnings.length ? (
                <div className="ff-schema-warning-list">
                  {validationWarnings.map((warning) => (
                    <div
                      key={`${warning.column}-${warning.rule}`}
                      className={`ff-schema-warning ff-schema-warning--${warning.severity || "warning"}`}
                    >
                      <strong>{warning.column}</strong>
                      <p>{warning.reason}</p>
                      <span>
                        {warning.invalid_count} preview row(s) failed `{warning.rule}`
                      </span>
                      {Array.isArray(warning.sample_values) && warning.sample_values.length ? (
                        <small>Examples: {warning.sample_values.join(", ")}</small>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : null}
            </article>

            <article className="ff-panel" style={{ marginBottom: "2rem" }}>
              <div className="ff-panel__head">
                <div>
                  <p className="ff-eyebrow">Execution Actions</p>
                  <h3>Deterministic actions applied to this dataset</h3>
                </div>
              </div>
              <div className="ff-schema-grid">
                {actionSchema.length ? (
                  actionSchema.map((action, idx) => (
                    <div key={idx} className="ff-schema-card">
                      <div className="ff-schema-card__head">
                        <strong>{action?.action || "Unknown Action"}</strong>
                        <span className={`ff-status ff-status--info`}>
                          action
                        </span>
                      </div>
                      <p>
                        <span>Target Roles:</span> {Array.isArray(action?.roles) ? action.roles.join(", ") : "N/A"}
                      </p>
                      {action?.condition_tree && (
                        <p>
                          <span>Condition:</span> {JSON.stringify(action.condition_tree)}
                        </p>
                      )}
                      {action?.mapping && (
                        <p>
                          <span>Mapping:</span> {JSON.stringify(action.mapping)}
                        </p>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="ff-copy-muted">
                    No execution actions detected.
                  </div>
                )}
              </div>
            </article>
          </>
        ) : null}

        <article className="ff-panel">
          <div className="ff-panel__head">
            <div>
              <p className="ff-eyebrow">Execution plan</p>
              <h3>Step-by-step timeline</h3>
            </div>
          </div>
          <div className="ff-timeline">
            {Array.isArray(job?.steps) ? job.steps.map((step, index) => (
              <div key={step.name} className="ff-timeline__item">
                <div className={`ff-timeline__dot is-${step.status}`} />
                <div className="ff-timeline__content">
                  <div className="ff-timeline__head">
                    <div>
                      <strong>
                        {index + 1}. {step.name}
                      </strong>
                      <span>{formatStepStatus(step.status)}</span>
                    </div>
                    <small>{step.time || "Pending"}</small>
                  </div>
                  <p>{step.summary}</p>
                </div>
              </div>
            )) : null}
          </div>
        </article>
        {schemaApprovalVisible || extractionPreviewVisible ? (
          <article className="ff-panel" style={{ gridColumn: "1 / -1" }}>
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Data preview</p>
                <h3>Sample rows captured from the uploaded file</h3>
              </div>
            </div>
            {schemaColumns.length ? (
              <DataTable
                columns={schemaColumns}
                rows={schemaRows}
                pageSize={6}
                title="Uploaded data preview"
              />
            ) : (
              <div className="ff-copy-muted">
                No preview rows are available for this upload yet.
              </div>
            )}
          </article>
        ) : (
          <article className="ff-panel">
            <div className="ff-panel__head">
              <div>
                <p className="ff-eyebrow">Agent summaries</p>
                <h3>Expandable summaries from each agent</h3>
              </div>
            </div>
            <div className="ff-log-stack">
              {Array.isArray(job?.agentSummaries) && job.agentSummaries.length ? job.agentSummaries.map((entry, index) => {
                const open = openIndex === index;
                return (
                  <button
                    key={`${entry.agentId}-${entry.agentName}`}
                    type="button"
                    className={`ff-log-card${open ? " is-open" : ""}`}
                    onClick={() => setOpenIndex(open ? -1 : index)}
                  >
                    <div className="ff-log-card__head">
                      <div>
                        <strong>{entry.agentName}</strong>
                        <span>{formatJobStatus(entry.status)}</span>
                      </div>
                      <FiChevronDown size={16} />
                    </div>
                    {open && (
                      <div className="ff-log-card__body">
                        <p>{entry.summary}</p>
                        {entry.bullets.length ? (
                          <ul className="ff-summary-bullets">
                            {entry.bullets.map((bullet) => (
                              <li key={bullet}>{bullet}</li>
                            ))}
                          </ul>
                        ) : null}
                      </div>
                    )}
                  </button>
                );
              }) : (
                <div className="ff-copy-muted">
                  No agent summaries are available for this workflow yet.
                </div>
              )}
            </div>
          </article>
        )}
      </section>

      <section className="ff-panel">
        <div className="ff-panel__head">
          <div>
            <p className="ff-eyebrow">Audit trail</p>
            <h3>Chronological chain of custody</h3>
          </div>
        </div>
        <div className="ff-audit-list">
          {Array.isArray(job?.audit) ? job.audit.map((entry) => (
            <div key={`${entry.time}-${entry.action}`} className="ff-audit-row">
              <strong>{entry.time}</strong>
              <div>
                <span>{entry.action}</span>
                <p>{entry.detail}</p>
              </div>
            </div>
          )) : null}
          {(!Array.isArray(job?.audit) || !job.audit.length) && (
            <div className="ff-copy-muted">
              No audit entries are available for this workflow yet.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

async function refreshJobViews(queryClient, jobId) {
  await queryClient.invalidateQueries({ queryKey: ["jobs"] });
  await queryClient.invalidateQueries({
    queryKey: ["jobs", "detail", jobId],
  });
  await queryClient.invalidateQueries({ queryKey: ["manager-dashboard"] });
}

function formatConfidence(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return "Unscored";
  return `${Math.round(numeric * 100)}% confidence`;
}
