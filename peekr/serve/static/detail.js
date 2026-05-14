// Lazy-load span I/O so a 50 KB prompt only crosses the wire on click.
// One small file, no build step, no framework.
(function () {
  "use strict";

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function findTraceId(el) {
    var n = el;
    while (n) {
      if (n.dataset && n.dataset.traceId) return n.dataset.traceId;
      n = n.parentElement;
    }
    return null;
  }

  function renderIO(target, payload) {
    var parts = [];
    if (payload.system) {
      parts.push(
        '<div class="io-label">system</div><pre>' +
          escapeHtml(payload.system) +
          "</pre>"
      );
    }
    if (payload.input !== null && payload.input !== undefined) {
      parts.push(
        '<div class="io-label">input</div><pre>' +
          escapeHtml(payload.input) +
          "</pre>"
      );
    }
    if (payload.output !== null && payload.output !== undefined) {
      parts.push(
        '<div class="io-label">output</div><pre>' +
          escapeHtml(payload.output) +
          "</pre>"
      );
    }
    if (payload.error) {
      parts.push(
        '<div class="io-label">error</div><div class="io-error"><pre>' +
          escapeHtml(payload.error) +
          "</pre></div>"
      );
    }
    if (payload.eval_scores) {
      var rows = Object.keys(payload.eval_scores)
        .map(function (k) {
          return (
            '<span class="badge badge-eval">' +
            escapeHtml(k) +
            " · " +
            Number(payload.eval_scores[k]).toFixed(2) +
            "</span>"
          );
        })
        .join(" ");
      parts.push('<div class="io-label">eval scores</div><div>' + rows + "</div>");
    }
    if (payload.guardrail_findings && payload.guardrail_findings.length) {
      var items = payload.guardrail_findings
        .map(function (f) {
          return "<li>" + escapeHtml(JSON.stringify(f)) + "</li>";
        })
        .join("");
      parts.push(
        '<div class="io-label">guardrail findings</div><ul>' + items + "</ul>"
      );
    }
    if (parts.length === 0) {
      parts.push('<div class="muted small">No I/O captured for this span.</div>');
    }
    target.innerHTML = parts.join("");
  }

  function loadSpan(details) {
    var body = details.querySelector(".span-body");
    if (!body || body.dataset.loaded === "1") return;
    var io = body.querySelector(".span-io[data-needs-io='1']");
    if (!io) return;
    var traceId = findTraceId(details);
    var spanId = details.dataset.spanId;
    if (!traceId || !spanId) return;
    body.dataset.loaded = "1";
    fetch("/api/trace/" + encodeURIComponent(traceId) + "/span/" + encodeURIComponent(spanId))
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (payload) {
        renderIO(io, payload);
      })
      .catch(function (err) {
        io.innerHTML =
          '<div class="io-error"><pre>Failed to load: ' +
          escapeHtml(err.message || err) +
          "</pre></div>";
      });
  }

  document.addEventListener("toggle", function (e) {
    var t = e.target;
    if (t && t.classList && t.classList.contains("span-row") && t.open) {
      loadSpan(t);
    }
  }, true);
})();
