// Thin bridge so Python can use RailKit's official Node SDK (free tier).
// Usage: node railkit_bridge.mjs <history|live|station> <args...>   (RAILKIT_KEY in env)
import { configure, getTrainHistory, trackTrain, stationByCode } from "railkit";

const [, , fn, ...args] = process.argv;
const calls = { history: getTrainHistory, live: trackTrain, station: stationByCode };
if (!calls[fn]) {
  console.error(`unknown function ${fn}; use one of ${Object.keys(calls).join(", ")}`);
  process.exit(2);
}
try {
  configure(process.env.RAILKIT_KEY || "");
  const res = await calls[fn](...args);
  process.stdout.write(JSON.stringify(res));
} catch (e) {
  process.stdout.write(JSON.stringify({ success: false, error: String(e && e.message || e) }));
}
