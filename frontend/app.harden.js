// Set __API_BASE__ safely
(function(){
  const raw = (typeof window.BACKEND_URL === "string" && window.BACKEND_URL.trim()) ? window.BACKEND_URL.trim() : "http://localhost:8000";
  window.__API_BASE__ = raw.replace(/\/$/,''); // strip trailing slash
})();
