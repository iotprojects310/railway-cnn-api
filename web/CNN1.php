<?php
// web/CNN1.php
// Live display page that fetches data directly from Supabase `sensor_data` table
// Usage: place on your PHP-enabled web server. Polls Supabase every ~5s (client polls this page every 5s).
// Configuration: set SUPABASE_HOST and SUPABASE_KEY in environment variables to override defaults.

declare(strict_types=1);

$SUPABASE_HOST = getenv('SUPABASE_HOST') ?: 'kriqlextwdgzawtutzrp.supabase.co';
$SUPABASE_KEY  = getenv('SUPABASE_KEY')  ?: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtyaXFsZXh0d2RnemF3dHV0enJwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA1NzIyMzUsImV4cCI6MjA2NjE0ODIzNX0.B8KO7ivImaUVy-kw1tQIOWdHeimB-yivEtf-ubr6g8g';
$TABLE_NAME    = getenv('SUPABASE_TABLE') ?: 'sensor_data';
$CACHE_TTL     = 4; // seconds server-side cache
$cache_file    = sys_get_temp_dir() . '/supabase_sensor_cache_' . md5("{$SUPABASE_HOST}_{$TABLE_NAME}") . '.json';

function fetch_table_from_supabase(): array {
    global $SUPABASE_HOST, $SUPABASE_KEY, $TABLE_NAME;
    $url = "https://{$SUPABASE_HOST}/rest/v1/{$TABLE_NAME}?select=*";

    $opts = [
        'http' => [
            'method'  => 'GET',
            'header'  => "apikey: {$SUPABASE_KEY}\r\nAuthorization: Bearer {$SUPABASE_KEY}\r\nAccept: application/json\r\n",
            'timeout' => 10,
        ]
    ];
    $ctx = stream_context_create($opts);
    $resp = @file_get_contents($url, false, $ctx);
    if ($resp === false) {
        return ['rows' => [], 'error' => 'fetch_failed'];
    }
    $data = json_decode($resp, true);
    if (!is_array($data)) {
        return ['rows' => [], 'error' => 'invalid_json'];
    }
    return ['rows' => $data];
}

function get_payload_and_json(): array {
    global $cache_file, $CACHE_TTL;

    // If cache fresh, return it
    if (is_file($cache_file) && (time() - filemtime($cache_file) <= $CACHE_TTL)) {
        $cached = @file_get_contents($cache_file);
        if ($cached !== false) {
            $payload = @json_decode($cached, true);
            if (is_array($payload)) {
                return [$payload, $cached];
            }
        }
    }

    $res = fetch_table_from_supabase();
    $payload = [
        'last_updated' => date('c'),
        'rows' => $res['rows'] ?? [],
        'error' => $res['error'] ?? null,
    ];
    $json = json_encode($payload, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);

    $tmp = $cache_file . '.tmp';
    if (@file_put_contents($tmp, $json, LOCK_EX) !== false) {
        @rename($tmp, $cache_file);
    } else {
        @unlink($tmp);
    }
    return [$payload, $json];
}

// JSON mode: return rows fetched directly from Supabase (with ETag/304 support)
if (isset($_GET['json'])) {
    list($payload, $json) = get_payload_and_json();
    $etag = '"' . md5($json) . '"';
    header('Access-Control-Allow-Origin: *');
    header('Content-Type: application/json; charset=utf-8');
    header('ETag: ' . $etag);
    if (isset($_SERVER['HTTP_IF_NONE_MATCH']) && trim($_SERVER['HTTP_IF_NONE_MATCH']) === $etag) {
        http_response_code(304);
        exit;
    }
    echo $json;
    exit;
}

// Normal HTML rendering initial payload
list($payload_init, $json_init) = get_payload_and_json();
$rows = $payload_init['rows'] ?? [];

// find timestamp-like key
function detect_timestamp_key(array $rows): ?string {
    if (empty($rows)) return null;
    $row = $rows[0];
    $cands = ['timestamp','time','ts','created_at','createdAt','datetime','date','t'];
    foreach ($cands as $k) if (array_key_exists($k, $row)) return $k;
    foreach ($row as $k => $_v) if (stripos($k, 'time') !== false || stripos($k, 'date') !== false) return $k;
    return null;
}
$ts_key = detect_timestamp_key($rows);
?>
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Sensor Data (Live)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.15/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 min-h-screen">
  <main class="max-w-6xl mx-auto py-10">
    <h1 class="text-3xl font-bold mb-4 text-center">Live Sensor Data (from sensor_data table)</h1>
    <div class="mb-4 flex items-center justify-between">
      <div class="text-sm text-gray-700">Last sync: <span id="lastSync"><?= htmlspecialchars($payload_init['last_updated'] ?? '') ?></span></div>
      <div class="text-sm text-gray-700">Rows: <span id="rowCount"><?= count($rows) ?></span></div>
    </div>

    <div class="bg-white rounded shadow overflow-auto">
      <table id="dataTable" class="min-w-full text-left">
        <thead class="bg-gray-800 text-white">
          <tr>
            <th class="p-3">#</th>
            <th class="p-3"><?= htmlspecialchars($ts_key ?? 'Timestamp') ?></th>
            <th class="p-3">AccX</th>
            <th class="p-3">AccY</th>
            <th class="p-3">AccZ</th>
            <th class="p-3">Safety Class</th>
          </tr>
        </thead>
        <tbody class="text-sm" id="tableBody"></tbody>
      </table>
    </div>
  </main>

<script>
const API = window.location.pathname + '?json=1';
const POLL_MS = 5000;
let lastEtag = null;
let rows = <?= json_encode(array_values($rows), JSON_HEX_TAG|JSON_HEX_APOS|JSON_HEX_AMP|JSON_HEX_QUOT) ?> || [];
const tsKey = <?= json_encode($ts_key) ?>;

function render() {
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = '';
  rows.forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.className = i % 2 ? 'bg-gray-50' : '';
    const ts = tsKey ? (r[tsKey] ?? '') : (r.timestamp ?? r.time ?? '');
    tr.innerHTML = `
      <td class="p-2 align-top">${i+1}</td>
      <td class="p-2 align-top">${escapeHtml(String(ts ?? ''))}</td>
      <td class="p-2 align-top">${escapeHtml(String(r.AccX ?? r.accx ?? ''))}</td>
      <td class="p-2 align-top">${escapeHtml(String(r.AccY ?? r.accy ?? ''))}</td>
      <td class="p-2 align-top">${escapeHtml(String(r.AccZ ?? r.accz ?? ''))}</td>
      <td class="p-2 align-top font-semibold">${escapeHtml(String(r.safety_class ?? r.safetyClass ?? r.safety ?? ''))}</td>
    `;
    tbody.appendChild(tr);
  });
  document.getElementById('rowCount').textContent = rows.length;
  document.getElementById('lastSync').textContent = new Date().toISOString();
}

function escapeHtml(s) { return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

async function pollOnce() {
  try {
    const headers = {};
    if (lastEtag) headers['If-None-Match'] = lastEtag;
    const res = await fetch(API, { headers, cache: 'no-store' });
    if (res.status === 304) return;
    if (!res.ok) { console.warn('Fetch failed', res.status); return; }
    lastEtag = res.headers.get('ETag') || lastEtag;
    const payload = await res.json();
    const newRows = payload.rows || [];
    if (JSON.stringify(newRows) !== JSON.stringify(rows)) { rows = newRows; render(); }
    if (payload.last_updated) document.getElementById('lastSync').textContent = payload.last_updated;
  } catch (err) { console.error('Polling error', err); }
}

render();
setInterval(pollOnce, POLL_MS);
</script>
</body>
</html>
