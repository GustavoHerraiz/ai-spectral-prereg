// Sonda de procesado de Detector v1 (diseño factorial pareado 2x2).
// Protocolo: l1-manisha2025/PREREGISTRO_V1_PIPELINE_PROBE.md
//
// Pregunta: ¿las dos features congeladas de v1 (jpegPhaseRange, phaseCoherence) responden al
// historial de procesado (JPEG previo, reescalado) con independencia de que la imagen sea IA o real?
//
// Cada imagen produce hasta 4 variantes de 512x512 y se mide con el extractor CONGELADO de v1
// (extractFeatures de bin/evaluate-mobile-v1.mjs, sin modificarlo):
//   C0 recorte central 512, sin JPEG previo, sin resize
//   C1 JPEG previo (calidad Q, 4:2:0) y recorte central 512
//   R0 ventana central 768 -> Lanczos3 (x2/3) -> 512, sin JPEG previo
//   R1 ventana central 768 -> JPEG previo -> Lanczos3 (x2/3) -> 512   (emula "JPEG de origen + reescalado")
// R0/R1 solo existen si el lado menor >= 768. Nunca se amplia ninguna imagen.
//
// Uso:
//   node bin/v1-pipeline-probe.mjs --self-test
//   node bin/v1-pipeline-probe.mjs --inventory --root R --real-dir D --out OUT
//   node bin/v1-pipeline-probe.mjs --run --root R --real-dir D --out OUT [--shard 0/2]
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import winston from 'winston';
import { extractFeatures, loadFrozen } from '../../bin/evaluate-mobile-v1.mjs';

const SELF = fileURLToPath(import.meta.url);
const OUT_SIZE = 512;
const WINDOW_RESIZE = 768;
const ALIGN = 16;
const MIN_POOL_R = 30;
const SUPPORTED = new Set(['.png', '.jpg', '.jpeg', '.tif', '.tiff']);
const FAMILY_OF = { 'stable-diffusion-1-3': 'sd1', 'stable-diffusion-1-4': 'sd1', 'midjourney-v5': 'midjourney' };
const ARMS = ['C0', 'C1', 'R0', 'R1'];
const DEFAULTS = {
  root: '/home/gustavo/dataset_test/synthbuster/synthbuster_data/synthbuster',
  'real-dir': '/home/gustavo/dataset_test/synthbuster/synthbuster_data/synthbuster/Raise-1k/real_RAISE_1k',
  out: 'l1-manisha2025/results/v1-pipeline-probe-run1',
  skip: 'Raise-1k', seed: '20261111', 'n-real': '200', 'n-per-family': '100', 'jpeg-q': '75', shard: '0/1',
};

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(winston.format.timestamp(), winston.format.printf(({ timestamp, level, message }) => `${timestamp} ${level.toUpperCase()} ${message}`)),
  transports: [new winston.transports.Console({ stderrLevels: ['error', 'warn', 'info'] })],
});

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

export function alignedBox(width, height, size) {
  if (Math.min(width, height) < size) throw new RangeError(`lado menor < ${size}`);
  return { left: ALIGN * Math.floor((width - size) / (2 * ALIGN)), top: ALIGN * Math.floor((height - size) / (2 * ALIGN)), width: size, height: size };
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fnv1a(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) { hash ^= text.charCodeAt(i); hash = Math.imul(hash, 0x01000193) >>> 0; }
  return hash >>> 0;
}

export function seededSample(items, count, seed, group) {
  const rng = mulberry32(fnv1a(`${seed}:${group}`));
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [copy[i], copy[j]] = [copy[j], copy[i]]; }
  return copy.slice(0, count).sort((a, b) => a.rel.localeCompare(b.rel));
}

async function walk(dir, base = dir) {
  const found = [];
  for (const entry of (await fs.readdir(dir, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    if (entry.isSymbolicLink() || entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) found.push(...await walk(full, base));
    else if (SUPPORTED.has(path.extname(entry.name).toLowerCase())) found.push({ full, rel: path.relative(base, full) });
  }
  return found;
}

export async function discover(root, realDir, skip = []) {
  const realAbs = path.resolve(realDir);
  const groups = { real: await walk(realAbs) };
  for (const entry of (await fs.readdir(root, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory() || entry.isSymbolicLink() || entry.name.startsWith('.') || skip.includes(entry.name)) continue;
    const full = path.resolve(root, entry.name);
    if (realAbs === full || realAbs.startsWith(full + path.sep)) continue;
    const files = (await walk(full)).map(file => ({ ...file, rel: path.join(entry.name, file.rel) }));
    if (!files.length) continue;
    const family = FAMILY_OF[entry.name] ?? entry.name;
    groups[family] = [...(groups[family] ?? []), ...files];
  }
  return groups;
}

export async function describeFile(file) {
  const metadata = await sharp(file.full, { limitInputPixels: false }).metadata();
  const swap = [5, 6, 7, 8].includes(metadata.orientation);
  const stat = await fs.stat(file.full);
  return { ...file, width: swap ? metadata.height : metadata.width, height: swap ? metadata.width : metadata.height,
    format: metadata.format, depth: metadata.depth, bytes: stat.size };
}

async function describeAll(groups) {
  const described = {};
  for (const [group, files] of Object.entries(groups)) {
    described[group] = [];
    for (const file of files) {
      try { described[group].push(await describeFile(file)); } catch (error) { logger.warn(`metadatos ilegibles ${file.rel}: ${error.message}`); }
    }
  }
  return described;
}

export function summarizeInventory(described) {
  const summary = {};
  for (const [group, files] of Object.entries(described)) {
    const sizes = {}, formats = {}, depths = {};
    for (const file of files) {
      const key = `${file.width}x${file.height}`;
      sizes[key] = (sizes[key] ?? 0) + 1;
      formats[file.format] = (formats[file.format] ?? 0) + 1;
      depths[file.depth] = (depths[file.depth] ?? 0) + 1;
    }
    summary[group] = {
      n: files.length,
      n_crop_eligible: files.filter(f => Math.min(f.width, f.height) >= OUT_SIZE).length,
      n_resize_eligible: files.filter(f => Math.min(f.width, f.height) >= WINDOW_RESIZE).length,
      formats, depths,
      sizes: Object.fromEntries(Object.entries(sizes).sort((a, b) => b[1] - a[1]).slice(0, 12)),
    };
  }
  return summary;
}

export function buildSample(described, options) {
  const sample = [];
  for (const [group, files] of Object.entries(described)) {
    const cropEligible = files.filter(f => Math.min(f.width, f.height) >= OUT_SIZE);
    const resizeEligible = cropEligible.filter(f => Math.min(f.width, f.height) >= WINDOW_RESIZE);
    // Con suficientes imagenes reescalables se muestrea solo entre ellas (diseño pareado); si no, la familia es solo C0/C1.
    const eligible = resizeEligible.length >= MIN_POOL_R ? resizeEligible : cropEligible;
    const count = group === 'real' ? options.nReal : options.nPerFamily;
    for (const file of seededSample(eligible, count, options.seed, group)) {
      sample.push({ key: `${group}/${file.rel}`, group, label: group === 'real' ? 'real' : 'fake', rel: file.rel, full: file.full,
        width: file.width, height: file.height, format: file.format, depth: file.depth, bytes: file.bytes,
        resizeEligible: Math.min(file.width, file.height) >= WINDOW_RESIZE });
    }
  }
  return sample;
}

async function rawWindow(bytes, box) {
  return sharp(bytes, { limitInputPixels: false }).autoOrient().toColourspace('srgb').removeAlpha().extract(box)
    .raw({ depth: 'uchar' }).toBuffer();
}

const rawInput = size => ({ raw: { width: size, height: size, channels: 3 } });

async function jpegRoundTrip(raw, size, quality) {
  const encoded = await sharp(raw, rawInput(size)).jpeg({ quality, chromaSubsampling: '4:2:0' }).toBuffer();
  return sharp(encoded).toColourspace('srgb').removeAlpha().raw({ depth: 'uchar' }).toBuffer();
}

function centerRaw(raw, size, target) {
  const offset = (size - target) / 2, out = Buffer.alloc(target * target * 3);
  for (let y = 0; y < target; y++) raw.copy(out, y * target * 3, ((y + offset) * size + offset) * 3, ((y + offset) * size + offset + target) * 3);
  return out;
}

export async function buildArms(bytes, width, height, quality) {
  const resizable = Math.min(width, height) >= WINDOW_RESIZE;
  const size = resizable ? WINDOW_RESIZE : OUT_SIZE;
  const box = alignedBox(width, height, size);
  const base = await rawWindow(bytes, box);
  assert.equal(base.length, size * size * 3, 'ventana de tamaño inesperado');
  const toPng = (raw, side) => sharp(raw, rawInput(side)).png().toBuffer();
  const jpegBase = await jpegRoundTrip(base, size, quality);
  const arms = { C0: await toPng(resizable ? centerRaw(base, size, OUT_SIZE) : base, OUT_SIZE),
    C1: await toPng(resizable ? centerRaw(jpegBase, size, OUT_SIZE) : jpegBase, OUT_SIZE) };
  if (resizable) {
    const down = raw => sharp(raw, rawInput(size)).resize(OUT_SIZE, OUT_SIZE, { kernel: 'lanczos3', fit: 'fill' }).png().toBuffer();
    arms.R0 = await down(base);
    arms.R1 = await down(jpegBase);
  }
  return { arms, box, window: size };
}

async function processImage(entry, quality) {
  const bytes = await fs.readFile(entry.full);
  const { arms, box, window } = await buildArms(bytes, entry.width, entry.height, quality);
  const measured = {};
  for (const [arm, png] of Object.entries(arms)) {
    const { features } = await extractFeatures(png);
    measured[arm] = { f: features, sha256: sha256(png) };
  }
  return { key: entry.key, group: entry.group, label: entry.label, rel: entry.rel, status: 'ok', sourceSha256: sha256(bytes),
    bytes: entry.bytes, format: entry.format, depth: entry.depth, width: entry.width, height: entry.height,
    window, left: box.left, top: box.top, resizeEligible: entry.resizeEligible, arms: measured };
}

async function readDone(file) {
  try {
    return new Set((await fs.readFile(file, 'utf8')).split('\n').filter(Boolean).map(line => JSON.parse(line).key));
  } catch (error) {
    if (error.code === 'ENOENT') return new Set();
    throw error;
  }
}

async function writeOnce(file, text) {
  try { await fs.writeFile(file, text, { flag: 'wx' }); } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    assert.equal(sha256(await fs.readFile(file)), sha256(text), `${file} existe con contenido distinto: no se mezclan ejecuciones`);
  }
}

export function parseShard(text) {
  const match = /^(\d+)\/(\d+)$/.exec(text);
  if (!match || Number(match[1]) >= Number(match[2])) throw new Error('--shard debe ser k/n con 0<=k<n');
  return { index: Number(match[1]), count: Number(match[2]) };
}

export async function runProbe(options) {
  const frozen = await loadFrozen();
  const described = await describeAll(await discover(options.root, options.realDir, options.skip));
  assert.ok(described.real?.length, 'no hay reales');
  const sample = buildSample(described, options);
  await fs.mkdir(options.out, { recursive: true });
  const sampleText = JSON.stringify(sample.map(({ full, ...rest }) => rest), null, 1) + '\n';
  await writeOnce(path.join(options.out, 'sample.json'), sampleText);
  const meta = { protocol: 'PREREGISTRO_V1_PIPELINE_PROBE.md', scriptSha256: sha256(await fs.readFile(SELF)),
    frozenManifestSha256: frozen.manifestSha256, frozenModelsSha256: frozen.modelsSha256, sampleSha256: sha256(sampleText),
    seed: options.seed, nReal: options.nReal, nPerFamily: options.nPerFamily, jpegQuality: options.quality,
    resizeKernel: 'lanczos3', resizeFactor: 2 / 3, align: ALIGN, node: process.version, sharp: sharp.versions.sharp, vips: sharp.versions.vips,
    inventory: summarizeInventory(described) };
  await writeOnce(path.join(options.out, 'meta.json'), JSON.stringify(meta, null, 2) + '\n');
  sharp.concurrency(1);
  const shard = parseShard(options.shard);
  const file = path.join(options.out, `features.shard${shard.index}of${shard.count}.jsonl`);
  const done = await readDone(file);
  const mine = sample.filter((_, index) => index % shard.count === shard.index);
  logger.info(`shard ${options.shard}: ${mine.length} imagenes, ${done.size} ya hechas, muestra total ${sample.length}`);
  let ok = 0, errors = 0;
  for (const entry of mine) {
    if (done.has(entry.key)) continue;
    let record;
    try { record = await processImage(entry, options.quality); ok++; } catch (error) {
      errors++;
      logger.error(`fallo ${entry.key}: ${error.message}`);
      record = { key: entry.key, group: entry.group, label: entry.label, rel: entry.rel, status: 'error', error: error.message };
    }
    await fs.appendFile(file, JSON.stringify(record) + '\n');
    if ((ok + errors) % 25 === 0) logger.info(`${ok + errors}/${mine.length - done.size} (errores ${errors})`);
  }
  logger.info(`fin shard ${options.shard}: ok=${ok} errores=${errors}`);
  return { ok, errors, total: mine.length };
}

async function runInventory(options) {
  const described = await describeAll(await discover(options.root, options.realDir, options.skip));
  const summary = summarizeInventory(described);
  await fs.mkdir(options.out, { recursive: true });
  await fs.writeFile(path.join(options.out, 'inventory.json'), JSON.stringify(summary, null, 2) + '\n');
  for (const [group, info] of Object.entries(summary)) {
    logger.info(`${group.padEnd(22)} n=${info.n} crop>=512:${info.n_crop_eligible} resize>=768:${info.n_resize_eligible} formatos=${JSON.stringify(info.formats)} prof=${JSON.stringify(info.depths)} tamaños=${JSON.stringify(info.sizes)}`);
  }
}

async function synthImage(file, size, seed) {
  const rng = mulberry32(seed), raw = Buffer.alloc(size * size * 3);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) for (let c = 0; c < 3; c++) {
    const value = 128 + 60 * Math.sin((x + seed) / (9 + c)) * Math.cos(y / 13) + 25 * (rng() - 0.5);
    raw[(y * size + x) * 3 + c] = Math.max(0, Math.min(255, Math.round(value)));
  }
  await sharp(raw, rawInput(size)).png().toFile(file);
  return raw;
}

async function selfTest() {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'v1probe-'));
  const results = [];
  const check = async (name, fn) => {
    try { await fn(); results.push(true); logger.info(`[OK] ${name}`); } catch (error) { results.push(false); logger.error(`[FALLO] ${name}: ${error.message}`); }
  };
  try {
    await check('T1 caja alineada a 16 y centrada', async () => {
      assert.deepEqual(alignedBox(1024, 1024, 512), { left: 256, top: 256, width: 512, height: 512 });
      const box = alignedBox(4928, 3264, WINDOW_RESIZE);
      assert.ok(box.left % ALIGN === 0 && box.top % ALIGN === 0 && Math.abs(box.left - (4928 - WINDOW_RESIZE) / 2) < ALIGN);
      assert.throws(() => alignedBox(500, 900, 512), RangeError);
      const inner = alignedBox(4928, 3264, 512);
      assert.equal(box.left + 128, inner.left);
      assert.equal(box.top + 128, inner.top);
    });
    await check('T2 muestreo determinista y sensible a la semilla', async () => {
      const items = Array.from({ length: 50 }, (_, i) => ({ rel: `f${String(i).padStart(2, '0')}.png` }));
      const a = seededSample(items, 10, 1, 'g'), b = seededSample(items, 10, 1, 'g'), c = seededSample(items, 10, 2, 'g');
      assert.deepEqual(a, b);
      assert.notDeepEqual(a, c);
      assert.equal(new Set(a.map(x => x.rel)).size, 10);
    });
    const root = path.join(tmp, 'root'), realDir = path.join(root, 'Raise-1k', 'real_x');
    await fs.mkdir(realDir, { recursive: true });
    await fs.mkdir(path.join(root, 'fam_big'));
    await fs.mkdir(path.join(root, 'stable-diffusion-1-3'));
    await fs.mkdir(path.join(root, 'stable-diffusion-1-4'));
    await fs.mkdir(path.join(root, 'tiny'));
    for (let i = 0; i < 4; i++) await synthImage(path.join(realDir, `r${i}.png`), i === 3 ? 1100 : 1024, 10 + i);
    for (let i = 0; i < 4; i++) await synthImage(path.join(root, 'fam_big', `b${i}.png`), 1024, 20 + i);
    await synthImage(path.join(root, 'stable-diffusion-1-3', 's0.png'), 512, 30);
    await synthImage(path.join(root, 'stable-diffusion-1-4', 's1.png'), 512, 31);
    await synthImage(path.join(root, 'tiny', 't0.png'), 256, 40);
    let described;
    await check('T3 descubrimiento: SD1.3+1.4 agrupadas, real excluida de familias, <512 fuera de la muestra', async () => {
      described = await describeAll(await discover(root, realDir));
      assert.deepEqual(Object.keys(described).sort(), ['fam_big', 'real', 'sd1', 'tiny']);
      assert.equal(described.sd1.length, 2);
      const sample = buildSample(described, { nReal: 10, nPerFamily: 10, seed: 1 });
      assert.ok(!sample.some(s => s.group === 'tiny'));
      assert.ok(sample.filter(s => s.group === 'sd1').every(s => !s.resizeEligible));
    });
    const source = path.join(realDir, 'r0.png');
    const sourceBytes = await fs.readFile(source);
    let built;
    await check('T4 C0 == recorte central v1 (identico en 1024) y R0 == Lanczos3 de la ventana 768', async () => {
      built = await buildArms(sourceBytes, 1024, 1024, 75);
      assert.deepEqual(Object.keys(built.arms).sort(), ARMS);
      const direct = (await extractFeatures(sourceBytes)).features, viaArm = (await extractFeatures(built.arms.C0)).features;
      assert.deepEqual(viaArm, direct);
      const expected = await sharp(sourceBytes).extract({ left: 128, top: 128, width: 768, height: 768 }).resize(512, 512, { kernel: 'lanczos3', fit: 'fill' }).png().toBuffer();
      const a = await sharp(expected).raw().toBuffer(), b = await sharp(built.arms.R0).raw().toBuffer();
      assert.ok(a.equals(b));
    });
    await check('T5 el JPEG previo cambia pixeles y es determinista; sin ampliar ni reescalar en C', async () => {
      const again = await buildArms(sourceBytes, 1024, 1024, 75);
      for (const arm of ARMS) assert.equal(sha256(again.arms[arm]), sha256(built.arms[arm]));
      assert.notEqual(sha256(built.arms.C0), sha256(built.arms.C1));
      assert.notEqual(sha256(built.arms.R0), sha256(built.arms.R1));
      for (const arm of ARMS) { const m = await sharp(built.arms[arm]).metadata(); assert.deepEqual([m.width, m.height, m.channels], [512, 512, 3]); }
      const c0 = await sharp(built.arms.C0).raw().toBuffer(), full = await sharp(sourceBytes).extract({ left: 256, top: 256, width: 512, height: 512 }).raw().toBuffer();
      assert.ok(c0.equals(full));
    });
    await check('T6 imagenes 512<=lado<768 solo generan C0 y C1', async () => {
      const small = await buildArms(await fs.readFile(path.join(root, 'stable-diffusion-1-3', 's0.png')), 512, 512, 75);
      assert.deepEqual(Object.keys(small.arms).sort(), ['C0', 'C1']);
    });
    await check('T7 ejecucion completa en 2 shards: particion disjunta, reanudable y con metadatos', async () => {
      const out = path.join(tmp, 'out');
      const base = { root, realDir, out, seed: 7, nReal: 3, nPerFamily: 2, quality: 75 };
      const r0 = await runProbe({ ...base, shard: '0/2' }), r1 = await runProbe({ ...base, shard: '1/2' });
      assert.equal(r0.errors + r1.errors, 0);
      const keys = [];
      for (const name of ['features.shard0of2.jsonl', 'features.shard1of2.jsonl']) {
        for (const line of (await fs.readFile(path.join(out, name), 'utf8')).split('\n').filter(Boolean)) keys.push(JSON.parse(line).key);
      }
      assert.equal(new Set(keys).size, keys.length);
      assert.equal(keys.length, JSON.parse(await fs.readFile(path.join(out, 'sample.json'), 'utf8')).length);
      const before = (await fs.readFile(path.join(out, 'features.shard0of2.jsonl'), 'utf8')).length;
      await runProbe({ ...base, shard: '0/2' });
      assert.equal((await fs.readFile(path.join(out, 'features.shard0of2.jsonl'), 'utf8')).length, before);
      const meta = JSON.parse(await fs.readFile(path.join(out, 'meta.json'), 'utf8'));
      assert.match(meta.frozenManifestSha256, /^[0-9a-f]{64}$/);
      await assert.rejects(runProbe({ ...base, seed: 8, shard: '0/2' }));
    });
  } finally {
    await fs.rm(tmp, { recursive: true, force: true });
  }
  logger.info(`autotest: ${results.filter(Boolean).length}/${results.length} correctos`);
  return results.every(Boolean);
}

function parseArgs(argv) {
  const values = { ...DEFAULTS }, flags = new Set();
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (['--self-test', '--inventory', '--run'].includes(arg)) flags.add(arg.slice(2));
    else if (arg.startsWith('--') && arg.slice(2) in DEFAULTS && argv[i + 1] !== undefined) values[arg.slice(2)] = argv[++i];
    else throw new Error(`argumento no valido: ${arg}`);
  }
  const options = { root: path.resolve(values.root), realDir: path.resolve(values['real-dir']), out: path.resolve(values.out),
    seed: Number(values.seed), nReal: Number(values['n-real']), nPerFamily: Number(values['n-per-family']),
    quality: Number(values['jpeg-q']), shard: values.shard, skip: values.skip.split(',').filter(Boolean) };
  for (const key of ['seed', 'nReal', 'nPerFamily', 'quality']) assert.ok(Number.isInteger(options[key]) && options[key] > 0, `${key} invalido`);
  assert.ok(options.quality <= 100);
  return { flags, options };
}

async function main() {
  const { flags, options } = parseArgs(process.argv.slice(2));
  if (flags.has('self-test')) { process.exitCode = (await selfTest()) ? 0 : 1; return; }
  if (flags.has('inventory')) { await runInventory(options); return; }
  if (flags.has('run')) {
    const result = await runProbe(options);
    if (result.errors > 0.01 * result.total) { logger.error('mas del 1% de errores: revisar el log (stderr) antes de analizar'); process.exitCode = 1; }
    return;
  }
  throw new Error('indica --self-test, --inventory o --run');
}

if (process.argv[1] && path.resolve(process.argv[1]) === SELF) {
  try { await main(); } catch (error) { logger.error(`abortado: ${error.message}`); process.exitCode = 2; }
}
