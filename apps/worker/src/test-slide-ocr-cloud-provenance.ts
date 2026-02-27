import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';

function runProcess(cmd: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (d) => {
      const s = d.toString();
      stdout += s;
      process.stdout.write(s);
    });
    proc.stderr.on('data', (d) => {
      const s = d.toString();
      stderr += s;
      process.stderr.write(s);
    });
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0) return reject(new Error(`${cmd} exited ${code}. stderr tail: ${stderr.slice(-3000)}`));
      resolve({ stdout, stderr });
    });
  });
}

function resolvePythonBin(): string {
  const explicit = process.env.OCR_PYTHON_BIN || process.env.DIARIZATION_PYTHON_BIN || process.env.FACE_PYTHON_BIN;
  const candidates = [
    explicit ? path.resolve(explicit) : '',
    path.resolve(__dirname, '../venv311/bin/python3'),
    path.resolve(__dirname, '../venv/bin/python3'),
    path.resolve(process.cwd(), 'apps/worker/venv311/bin/python3'),
    path.resolve(process.cwd(), 'apps/worker/venv/bin/python3'),
    'python3',
  ].filter(Boolean);
  return candidates.find((candidate) => candidate === 'python3' || fs.existsSync(candidate)) || 'python3';
}

async function main() {
  const pythonBin = resolvePythonBin();
  const scriptPath = path.resolve(__dirname, 'pipeline/python/slide_ocr_v2.py');

  const py = [
    'import importlib.util, json, os, sys, tempfile',
    'import numpy as np',
    'import cv2',
    `sys.path.insert(0, r"""${path.dirname(scriptPath)}""")`,
    `spec = importlib.util.spec_from_file_location("slide_ocr_v2", r"""${scriptPath}""")`,
    'mod = importlib.util.module_from_spec(spec)',
    'assert spec and spec.loader',
    'sys.modules["slide_ocr_v2"] = mod',
    'spec.loader.exec_module(mod)',
    'mod.init_ocr_engine = lambda *_args, **_kwargs: ("openai_text_detection", {"fake": True}, [])',
    'mod.run_ocr = lambda *_args, **_kwargs: ("Sermon Title", 0.92, {"provider": "openai"})',
    'tmp = tempfile.mkdtemp(prefix="slide-ocr-prov-")',
    'img_path = os.path.join(tmp, "frame.jpg")',
    'img = np.zeros((120, 320, 3), dtype=np.uint8)',
    'cv2.imwrite(img_path, img)',
    'events = [',
    '  {"text": "", "confidence": 0.0, "type": "on_screen_text", "presentation_group_id": "g1", "presentation_is_representative": True, "slide_id": "slide_1"},',
    '  {"text": "", "confidence": 0.0, "type": "on_screen_text", "presentation_group_id": "g1", "presentation_is_representative": False, "slide_id": "slide_2"},',
    ']',
    'extraction = {"output_dir": tmp, "files": [{"event_index": 0, "file": "frame.jpg", "path": img_path}]}',
    'summary = mod.enrich_events_with_cloud_text(',
    '  events=events,',
    '  extraction=extraction,',
    '  lang_hint="es,en",',
    '  enabled=True,',
    '  min_confidence=0.1,',
    '  min_chars=3,',
    '  min_quality=0.0,',
    '  max_images=0,',
    '  min_apply_gain=0.0,',
    '  propagate_duplicates=True,',
    ')',
    'print(json.dumps({"summary": summary, "events": events}))',
  ].join('\n');

  const { stdout } = await runProcess(pythonBin, ['-c', py]);
  const lines = stdout.trim().split('\n').filter(Boolean);
  const last = lines[lines.length - 1] || '{}';
  const parsed = JSON.parse(last) as { summary: any; events: Array<Record<string, any>> };

  const rep = parsed.events?.[0] || {};
  const dup = parsed.events?.[1] || {};
  const repOk = String(rep.ocr_input || '').includes('openai_text_detection:extract');
  const dupOk = String(dup.ocr_input || '').includes('openai_text_detection:group_propagation');

  console.log(`rep_ocr_input=${rep.ocr_input || ''}`);
  console.log(`dup_ocr_input=${dup.ocr_input || ''}`);

  if (!repOk || !dupOk) {
    throw new Error('Cloud OCR provenance tags did not use actual backend label.');
  }
}

main().catch((err) => {
  console.error('test-slide-ocr-cloud-provenance failed:', err);
  process.exit(1);
});
