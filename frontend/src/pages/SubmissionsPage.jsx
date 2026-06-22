import React, { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { FiAlertCircle, FiDownload, FiRefreshCw, FiSearch } from "react-icons/fi";
import { downloadJobOutput, fetchJobs, retryJob } from "../api/finflow.js";
import { useLiveJobRefresh } from "../hooks/useLiveJobRefresh.js";
import {
  formatDateTime,
  formatJobStatus,
  summarizeInstruction,
} from "../utils/finflowFormatters.js";

export default function SubmissionsPage() {
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const queryClient = useQueryClient();
  const {
    data: jobs = [],
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["jobs"],
    queryFn: fetchJobs,
  });
  useLiveJobRefresh();

  const retryMutation = useMutation({
    mutationFn: retryJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["manager-dashboard"] });
    },
  });

  const filteredJobs = useMemo(() => {
    return jobs.filter((job) => {
      const matchesQuery =
        !query.trim() ||
        [job.id, job.fileName, job.instruction].some((value) =>
          String(value || "")
            .toLowerCase()
            .includes(query.trim().toLowerCase()),
        );
      const matchesStatus =
        statusFilter === "all" || job.status === statusFilter;
      return matchesQuery && matchesStatus;
    });
  }, [jobs, query, statusFilter]);

  if (isLoading) {
    return (
      <div className="ff-page-grid">
        <section className="ff-panel">
          <h2>Loading jobs...</h2>
        </section>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="ff-page-grid">
        <section className="ff-panel">
          <h2>We could not load the job list.</h2>
          <p className="ff-copy-muted">
            Check the API connection and try again.
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
            <p className="ff-eyebrow">My jobs</p>
            <h2>Track each workflow from queue to final output.</h2>
          </div>
          <div className="ff-toolbar">
            <label className="ff-search">
              <FiSearch size={15} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search jobs or files"
              />
            </label>
            <select
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value)}
            >
              <option value="all">All statuses</option>
              <option value="running">Running</option>
              <option value="clarification">Needs input</option>
              <option value="quarantined">Quarantined</option>
              <option value="complete">Complete</option>
              <option value="failed">Failed</option>
            </select>
          </div>
        </div>

        <div className="ff-job-list">
          {filteredJobs.map((job) => (
            <article key={job.id} className="ff-job-card">
              <div className="ff-job-card__top">
                <div>
                  <div className="ff-job-card__id">{job.id}</div>
                  <h3>{job.title}</h3>
                </div>
                {job.status === "clarification" ? (
                  <span className="ff-status ff-status--clarification" aria-label="Needs your input">
                    <FiAlertCircle size={14} aria-hidden="true" />
                    Needs your input
                  </span>
                ) : (
                  <span className={`ff-status ff-status--${job.status}`}>
                    {formatJobStatus(job.status)}
                  </span>
                )}
              </div>
              <p>{summarizeInstruction(job.instruction)}</p>
              {job.jobSummary ? (
                <p className="ff-copy-muted">{job.jobSummary}</p>
              ) : null}
              {job.status === "quarantined" ? (
                <p className="ff-copy-muted">
                  {job.suggestion || job.reason || "Part of this workflow is quarantined until supported coverage is available."}
                </p>
              ) : null}
              <div className="ff-job-card__meta">
                <span>{job.fileName}</span>
                <span>{formatDateTime(job.submittedAt)}</span>
                <span>{job.outputFormat}</span>
              </div>
              <div className="ff-job-card__pipeline">
                {job.steps.map((step) => (
                  <div
                    key={step.name}
                    className={`ff-pipeline-step is-${step.status}`}
                  >
                    <strong>{step.name}</strong>
                    <span>{step.status}</span>
                  </div>
                ))}
              </div>
              <div className="ff-job-card__actions">
                <Link to={`/jobs/${job.backendId}`} className="ff-inline-link">
                  Open details
                </Link>
                {job.status === "complete" && job.outputReady ? (
                  <button
                    type="button"
                    className="ff-inline-action"
                    onClick={() => downloadJobOutput(job.backendId)}
                  >
                    <FiDownload size={14} />
                    Download output
                  </button>
                ) : job.status === "failed" ? (
                  <button
                    type="button"
                    className="ff-inline-action"
                    onClick={() => retryMutation.mutate(job.backendId)}
                    disabled={retryMutation.isPending}
                  >
                    <FiRefreshCw size={14} />
                    {retryMutation.isPending ? "Requeueing..." : "Retry job"}
                  </button>
                ) : job.status === "clarification" ? (
                  <Link to={`/jobs/${job.backendId}`} className="ff-inline-action ff-inline-action--clarification">
                    <FiAlertCircle size={14} />
                    Resolve ambiguities
                  </Link>
                ) : job.status === "quarantined" ? (
                  <span className="ff-copy-muted">
                    Quarantined for review
                  </span>
                ) : (
                  <span className="ff-copy-muted">
                    Execution plan updating live
                  </span>
                )}
              </div>
            </article>
          ))}
          {!filteredJobs.length && (
            <div className="ff-copy-muted">
              No jobs matched your current filters.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
