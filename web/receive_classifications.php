<?php
// web/receive_classifications.php
// Simple receiver endpoint to accept POST JSON (classification payload)
// - Expects Authorization: Bearer <token> if env RECV_TOKEN is set
// - Stores latest payload in a cache file for the website to read

declare(strict_types=1);
header('Content-Type: application/json; charset=utf-8');
$RECV_TOKEN = getenv('RECV_TOKEN') ?: null;
$cache_file = sys_get_temp_dir() . '/supabase_classification_latest.json';

// check method
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'method_not_allowed']);
    exit;
}

// auth if token configured
$auth_header = $_SERVER['HTTP_AUTHORIZATION'] ?? null;
if ($RECV_TOKEN) {
    if (!$auth_header || stripos($auth_header, 'Bearer ') !== 0) {
        http_response_code(401);
        echo json_encode(['error' => 'missing_authorization']);
        exit;
    }
    $token = substr($auth_header, 7);
    if (hash_equals($RECV_TOKEN, $token) === false) {
        http_response_code(403);
        echo json_encode(['error' => 'invalid_token']);
        exit;
    }
}

$raw = file_get_contents('php://input');
if (!$raw) {
    http_response_code(400);
    echo json_encode(['error' => 'empty_body']);
    exit;
}

// basic validation
$payload = json_decode($raw, true);
if (!is_array($payload) || !isset($payload['rows'])) {
    http_response_code(400);
    echo json_encode(['error' => 'invalid_payload']);
    exit;
}

// atomic write
$tmp = $cache_file . '.tmp';
if (file_put_contents($tmp, json_encode($payload, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE), LOCK_EX) !== false) {
    rename($tmp, $cache_file);
    echo json_encode(['status' => 'ok']);
    exit;
}

http_response_code(500);
echo json_encode(['error' => 'write_failed']);
