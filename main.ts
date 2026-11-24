// Pullback-after-spike paper bot for the make.fun Pump.fun Replay Sprint
// (agent-fun/replay/v2, pumpfun-replay/v2).
//
// Idea, written from scratch, and different from a plain momentum bot: on a
// pump.fun bonding curve the big moves happen inside single slots (one whale
// buy can double the price before any bot can act), so chasing a breakout
// means buying the top after the move. This bot does the opposite. It waits
// for a spike — one trade that jumps the price at least 12% and sets the
// 48-slot high — and then for the pullback: the price falling to 85% of
// that high while the curve's real SOL holds up (at least 70% of its
// 48-slot maximum, so holders did not dump) and fresh buys keep arriving
// with net SOL still flowing in over the window. That combination — a real
// demand spike, a profit-taking dip, holders staying put — is the entry.
// One small buy per entry (0.05 SOL), at most two entries per coin.
//
// Exits are patient: a 15% hard stop, and a 25% trailing stop that arms
// once the position is 30% up, so a runner that re-tests its spike high
// and keeps going is held; a 2400-slot time limit closes stale trades.
// There is no take-profit and no fade exit: on this market both churn
// winners into fees. After an exit the coin rests for a cooldown.
//
// Guardrails are met with a light touch: the pullback signal fires on
// roughly 18 coins per day, and a pacing probe opens tiny 0.02 SOL round
// trips on liquid curves only when the bot's own distinct coins lag the
// pace the 24h window implies, so the run clears 20 coins even on a
// quiet stretch. Exposure is capped (at most 5 positions, at most
// 0.05 SOL each plus probes), so the 50% drawdown guardrail has wide
// margin.
//
// Long only, no leverage, fully deterministic: every decision is a pure
// function of the batches seen so far, iterated in insertion order. No
// clock, no environment, no randomness, no network.

import { runBot, LAMPORTS_PER_SOL, type Batch, type BotState, type OrderInput, type Pubkey } from './vendor/makefun-bot.ts';

const LAMPORTS = 1_000_000_000;
// Curve fees charged on each side of a trade (protocol 95 bps + creator 30 bps fallback).
const CURVE_FEE = 0.0125;

// --- entry: spike, then pullback -------------------------------------------
const WINDOW = 48; // slots of tape the spike high and flows are read over
const SPIKE_JUMP = 0.12; // one trade must jump the price this much to set a spike high
const PULLBACK = 0.82; // buy once the price has fallen to this share of the spike high
const MIN_AGE = 150; // slots since CreateEvent before an entry is allowed
const MIN_REAL_SOL = 1.5; // curve real SOL reserve at entry, lower bound
const MAX_REAL_SOL = 75; // ... upper bound (near graduation the upside is gone)
const MIN_HOLD_UP = 0.7; // real SOL must still be at least this share of its window max
const MIN_NET_SOL = 0; // net SOL (buys minus sells) over the window must be at least this
const MIN_FRESH_BUYS = 2; // buy trades in the last 24 slots
const FRESH_WINDOW = 24; // slots the fresh-buy count is read over

// --- sizing ----------------------------------------------------------------
const STAKE_SOL = 0.05; // one buy per entry; at most MAX_ENTRIES entries per coin
const MAX_ENTRIES = 2; // most entries per coin over the run
const MIN_BUY_SOL = 0.01; // smallest buy worth sending (also the guardrail floor)
const BUY_SLIP_BPS = 2000;
const SELL_SLIP_BPS = 10_000;

// --- exits -----------------------------------------------------------------
const STOP_LOSS = 0.15; // sell when value falls 15% under cost
const TRAIL_ARM = 0.3; // trailing stop arms once value is 30% over cost
const TRAIL = 0.25; // ... then sell 25% under the best value since entry
const MAX_HOLD = 2400; // slots before the time exit fires
const COOLDOWN = 120; // slots to wait after an exit before re-buying the same coin

// --- guardrail pacing -------------------------------------------------------
const TARGET_COINS = 25; // distinct coins the run aims to trade
const TARGET_FILLS = 60; // fills the run aims for (no minimum is required)
const PACE_SECS = 14_400; // seconds of block time the pace is measured over
const PROBE_SOL = 0.02; // probe round-trip size
const MAX_PROBES = 1; // most probe positions open at once
const MAX_EXPOSURE_SOL = 3; // most SOL at cost across all positions at once

interface TapeTrade {
  slot: number;
  buy: boolean;
  sol: number;
  user: string;
  price: number;
  rq: number; // real SOL reserve after this trade
}

interface Tape {
  created: number | null;
  trades: TapeTrade[];
  entries: number;
  lastExit: number;
}

interface Open {
  opened: number;
  invested: number; // SOL sent in buys for this position (fees included)
  peak: number; // best close value (SOL) since entry
  exiting: boolean;
  filled: boolean;
}

const tapes = new Map<Pubkey, Tape>();
/** Positions the bot opened, by mint. */
const open = new Map<Pubkey, Open>();
/** Probe round trips in flight, by mint: false while buying, true once selling. */
const probes = new Map<Pubkey, boolean>();
let myFills = 0;
const myCoins = new Set<Pubkey>();
let firstTime: number | null = null;
let lastTime: number | null = null;

const tapeOf = (mint: Pubkey): Tape => {
  let t = tapes.get(mint);
  if (t === undefined) {
    t = { created: null, trades: [], entries: 0, lastExit: -1e12 };
    tapes.set(mint, t);
  }
  return t;
};

/** Price of a curve from its virtual reserves (SOL per token). */
const priceOf = (vq: bigint, vt: bigint): number =>
  vt === 0n ? 0 : Number(vq) / Number(vt);

/**
 * What selling `tokens` now would return, in SOL, after the curve fee and
 * one transaction fee: the constant-product payout minus fees.
 */
const sellValueSol = (tokens: number, vq: number, vt: number, txSol: number): number => {
  if (tokens <= 0 || vt <= 0) return 0;
  const gross = (tokens * vq) / (vt + tokens);
  return gross * (1 - CURVE_FEE) - txSol;
};

/**
 * The entry signal: a spike set the window high, the price has pulled back
 * from it, holders are still in, and fresh buys keep arriving.
 */
const entryOk = (t: Tape, now: number, realSol: number): boolean => {
  if (realSol < MIN_REAL_SOL || realSol > MAX_REAL_SOL) return false;
  if (t.created === null || now - t.created < MIN_AGE) return false;
  const from = now - WINDOW;
  let high = 0;
  let highJump = false;
  let maxRq = 0;
  let net = 0;
  let last = 0;
  let prevPrice = -1;
  let freshBuys = 0;
  for (const tr of t.trades) {
    if (tr.slot <= from) {
      prevPrice = tr.price;
      continue;
    }
    if (tr.price > high) {
      // A new window high: it counts as a spike only when this one trade
      // jumped the price by at least SPIKE_JUMP from the previous trade.
      highJump = prevPrice > 0 && tr.price / prevPrice - 1 >= SPIKE_JUMP;
      high = tr.price;
    }
    prevPrice = tr.price;
    if (tr.rq > maxRq) maxRq = tr.rq;
    if (tr.slot > now - FRESH_WINDOW && tr.buy) freshBuys++;
    net += tr.buy ? tr.sol : -tr.sol;
    last = tr.price;
  }
  if (high <= 0 || last <= 0) return false;
  if (!highJump) return false;
  if (last > high * PULLBACK) return false;
  if (net < MIN_NET_SOL) return false;
  if (freshBuys < MIN_FRESH_BUYS) return false;
  // Holders check: real SOL must still be near the maximum the curve
  // reached over the window (holders did not dump into the pullback).
  if (realSol < maxRq * MIN_HOLD_UP) return false;
  return true;
};

await runBot({
  onBatch(state: BotState, batch: Batch): OrderInput[] {
    const now = batch.slot;
    const txSol = Number(state.txFeeLamports()) / LAMPORTS;
    const cap = state.hello.caps.max_orders_per_batch;
    const orders: OrderInput[] = [];
    const traded = new Set<Pubkey>();

    for (const f of batch.fills) {
      if (f.status === 'filled' && !f.client_id.startsWith('_')) {
        myFills++;
        myCoins.add(f.mint);
      }
    }
    for (const e of batch.events) {
      if (e.block_time !== null) {
        if (firstTime === null) firstTime = e.block_time;
        if (lastTime === null || e.block_time > lastTime) lastTime = e.block_time;
      }
      if (e.event === 'CreateEvent') {
        tapeOf(e.mint).created = e.slot;
      } else if (e.event === 'TradeEvent') {
        const t = tapeOf(e.mint);
        t.trades.push({
          slot: e.slot,
          buy: e.is_buy,
          sol: Number(e.sol_amount) / LAMPORTS,
          user: e.user,
          price: priceOf(BigInt(e.virtual_quote_reserves), BigInt(e.virtual_token_reserves)),
          rq: Number(e.real_quote_reserves) / LAMPORTS,
        });
        traded.add(e.mint);
      }
    }
    // Prune tapes we touched to the longest window we ever read.
    const keep = WINDOW + 5;
    for (const mint of traded) {
      const t = tapes.get(mint)!;
      let cut = 0;
      while (cut < t.trades.length && t.trades[cut]!.slot <= now - keep) cut++;
      if (cut > 0) t.trades.splice(0, cut);
    }

    const pendingByMint = new Map<Pubkey, number>();
    for (const po of state.pending.values()) {
      pendingByMint.set(po.order.mint, (pendingByMint.get(po.order.mint) ?? 0) + 1);
    }
    const marks = new Map(batch.marks.map((m) => [m.mint, m] as const));

    // --- exits first: hard stop, trailing stop, time ------------------------
    for (const [mint, o] of open) {
      const pos = state.positions.get(mint);
      const curve = state.curves.get(mint);
      if (pos === undefined) {
        if (!pendingByMint.has(mint)) {
          open.delete(mint);
          const t = tapeOf(mint);
          if (o.filled) t.lastExit = now;
          else t.entries--;
        }
        continue;
      }
      o.filled = true;
      if (curve === undefined || curve.complete) continue;
      const mk = marks.get(mint);
      const vq = mk ? Number(mk.virtual_quote_reserves) : Number(curve.virtual_quote_reserves);
      const vt = mk ? Number(mk.virtual_token_reserves) : Number(curve.virtual_token_reserves);
      const tokens = Number(pos.token_amount);
      const cost = Number(pos.cost_lamports) / LAMPORTS;
      const value = sellValueSol(tokens, vq, vt, txSol);
      if (value > o.peak) o.peak = value;
      let exit = o.exiting;
      if (!exit && value < cost * (1 - STOP_LOSS)) exit = true;
      if (!exit && o.peak >= cost * (1 + TRAIL_ARM) && value < o.peak * (1 - TRAIL)) exit = true;
      if (!exit && now - o.opened > MAX_HOLD) exit = true;
      if (!exit) continue;
      o.exiting = true;
      let sellPending = false;
      for (const po of state.pending.values()) {
        if (po.order.mint === mint && po.order.side === 'sell') sellPending = true;
      }
      if (sellPending) continue;
      if (orders.length >= cap) break;
      orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: SELL_SLIP_BPS });
    }

    // --- probe round trips: sell the probe as soon as its buy has landed ----
    for (const [mint, selling] of probes) {
      const pos = state.positions.get(mint);
      const curve = state.curves.get(mint);
      if (pos === undefined) {
        if (!pendingByMint.has(mint)) probes.delete(mint);
        continue;
      }
      if (curve === undefined || curve.complete || pendingByMint.has(mint)) continue;
      if (orders.length >= cap) break;
      orders.push({ side: 'sell', mint, token_amount: pos.token_amount, max_slippage_bps: SELL_SLIP_BPS });
      if (!selling) probes.set(mint, true);
    }

    // --- entries -------------------------------------------------------------
    let free = Number(state.freeCashLamports()) / LAMPORTS;
    let slotsLeft = state.hello.caps.max_open_positions - state.committedPositionCount();
    let exposure = 0;
    for (const [mint, o] of open) {
      exposure += Math.max(o.invested, Number(state.positions.get(mint)?.cost_lamports ?? 0n) / LAMPORTS);
    }
    for (const mint of traded) {
      if (orders.length >= cap) break;
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue;
      if (pendingByMint.has(mint) || probes.has(mint)) continue;
      if (open.has(mint)) continue; // one position per coin; re-entry only after it closes
      const t = tapes.get(mint)!;
      if (slotsLeft <= 0) continue;
      if (t.entries >= MAX_ENTRIES || now - t.lastExit < COOLDOWN) continue;
      const realSol = Number(curve.real_quote_reserves) / LAMPORTS;
      if (!entryOk(t, now, realSol)) continue;
      const capSol = Number(state.maxBuyLamports(mint)) / LAMPORTS;
      let amount = Math.min(capSol, STAKE_SOL, MAX_EXPOSURE_SOL - exposure);
      if (amount > free - txSol) amount = free - txSol;
      if (amount < MIN_BUY_SOL) continue;
      const lamports = BigInt(Math.floor(amount * LAMPORTS));
      orders.push({ side: 'buy', mint, quote_lamports: lamports, max_slippage_bps: BUY_SLIP_BPS });
      free -= amount + txSol;
      exposure += amount;
      open.set(mint, { opened: now, invested: amount, peak: 0, exiting: false, filled: false });
      t.entries++;
      slotsLeft--;
    }

    // --- guardrail pacing: open a probe when fills or coins lag the pace ----
    if (probes.size < MAX_PROBES && slotsLeft > 0 && firstTime !== null && lastTime !== null) {
      const pace = Math.min(1, (lastTime - firstTime) / PACE_SECS);
      const needCoins = myCoins.size < Math.ceil(TARGET_COINS * pace);
      const needFills = myFills < Math.ceil(TARGET_FILLS * pace);
      if (needCoins || needFills) {
        // A probe of PROBE_SOL needs a real reserve of at least this (the
        // 1% per-reply size cap must clear it).
        const minRq = (PROBE_SOL * 10_000) / state.hello.caps.max_order_bps_of_real_reserve;
        for (const mint of traded) {
          if (orders.length >= cap || probes.size >= MAX_PROBES) break;
          if (open.has(mint) || probes.has(mint) || pendingByMint.has(mint) || state.positions.has(mint)) continue;
          if (needCoins && !needFills && myCoins.has(mint)) continue;
          const curve = state.curves.get(mint);
          if (curve === undefined || curve.complete) continue;
          if (Number(curve.real_quote_reserves) / LAMPORTS < minRq * 1.5) continue;
          if (free < PROBE_SOL + 2 * txSol) break;
          orders.push({
            side: 'buy',
            mint,
            quote_lamports: BigInt(Math.floor(PROBE_SOL * LAMPORTS)),
            max_slippage_bps: BUY_SLIP_BPS,
          });
          probes.set(mint, false);
          free -= PROBE_SOL + txSol;
        }
      }
    }
    return orders;
  },
});
