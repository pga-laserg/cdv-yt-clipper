import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';

type ScenarioResult = {
  name: string;
  ok: boolean;
  detail: string;
};

type Scenario = {
  name: string;
  env: Record<string, string>;
  expectedBackend: string;
  expectedErrorIncludes?: string;
};

function runProcess(cmd: string, args: string[], env: NodeJS.ProcessEnv): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'], env });
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
      if (code !== 0) return reject(new Error(`${cmd} exited ${code}. stderr tail: ${stderr.slice(-2000)}`));
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

async function runScenario(pythonBin: string, scenario: Scenario): Promise<ScenarioResult> {
  const scriptPath = path.resolve(__dirname, 'pipeline/python/ocr_events.py');
  const py = [
    'import importlib.util, json',
    `spec = importlib.util.spec_from_file_location('ocr_events', r'''${scriptPath}''')`,
    'mod = importlib.util.module_from_spec(spec)',
    'assert spec and spec.loader',
    'spec.loader.exec_module(mod)',
    "backend, _engine, errors = mod.init_ocr_engine('gcv_text_detection', 'es,en')",
    "print(json.dumps({'backend': backend, 'errors': errors}))",
  ].join('; ');

  const env: NodeJS.ProcessEnv = {
    ...process.env,
    ...scenario.env,
  };
  const { stdout } = await runProcess(pythonBin, ['-c', py], env);
  const lines = stdout.trim().split('\n').filter(Boolean);
  const last = lines[lines.length - 1] || '{}';
  const parsed = JSON.parse(last) as { backend?: string; errors?: string[] };

  const backendOk = parsed.backend === scenario.expectedBackend;
  const errorOk = scenario.expectedErrorIncludes
    ? Array.isArray(parsed.errors) && parsed.errors.some((e) => e.includes(scenario.expectedErrorIncludes!))
    : true;

  return {
    name: scenario.name,
    ok: backendOk && errorOk,
    detail: `backend=${parsed.backend} errors=${JSON.stringify(parsed.errors || [])}`,
  };
}

async function main() {
  const pythonBin = resolvePythonBin();
  const scenarios: Scenario[] = [
    {
      name: 'gcv_disabled_without_fallback_returns_none',
      env: {
        OCR_DISABLE_GCV: 'true',
        OCR_GCV_FALLBACK: 'false',
        OCR_OPENAI_FALLBACK: 'false',
        OPENAI_API_KEY: '',
        OCR_OPENAI_VISION_MODEL: '',
      },
      expectedBackend: 'none',
      expectedErrorIncludes: 'disabled via OCR_DISABLE_GCV=true',
    },
    {
      name: 'gcv_disabled_with_openai_fallback_returns_openai_backend',
      env: {
        OCR_DISABLE_GCV: 'true',
        OCR_GCV_FALLBACK: 'true',
        OCR_OPENAI_FALLBACK: 'true',
        OPENAI_API_KEY: 'test-key',
        OCR_OPENAI_VISION_MODEL: 'gpt-5-mini',
      },
      expectedBackend: 'openai_text_detection',
    },
  ];

  const results: ScenarioResult[] = [];
  for (const scenario of scenarios) {
    results.push(await runScenario(pythonBin, scenario));
  }

  const failed = results.filter((r) => !r.ok);
  results.forEach((r) => {
    console.log(`[${r.ok ? 'PASS' : 'FAIL'}] ${r.name} ${r.detail}`);
  });

  if (failed.length > 0) {
    throw new Error(`GCV OCR engine test failed: ${failed.map((f) => f.name).join(', ')}`);
  }
}

main().catch((err) => {
  console.error('test-gcv-ocr-engine failed:', err);
  process.exit(1);
});
