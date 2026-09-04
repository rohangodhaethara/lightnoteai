import { chromium } from "playwright";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(__dirname, "..", "docs", "screenshots");
const sampleVideo = path.resolve(__dirname, "..", "samples", "sample_bottle_video.mp4");

const executablePath = process.env.CHROME_EXECUTABLE || undefined;
const browser = await chromium.launch({ executablePath });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

await page.goto("http://localhost:5173", { waitUntil: "networkidle" });
await page.screenshot({ path: path.join(outDir, "01-initial.png") });

await page.getByPlaceholder("e.g. \"Replace the Coca-Cola bottle with Pepsi\"").fill(
  "Replace the Coca-Cola bottle with Pepsi"
);

const fileInput = page.locator('input[type="file"][accept="video/*"]');
await fileInput.setInputFiles(sampleVideo);

await page.screenshot({ path: path.join(outDir, "02-filled-form.png") });

await page.getByRole("button", { name: /Start Processing/i }).click();

await page.screenshot({ path: path.join(outDir, "03-submitting.png") });

// Poll UI until it shows completed or failed, taking a mid-progress shot along the way.
let shotTaken = false;
for (let i = 0; i < 40; i++) {
  await page.waitForTimeout(1500);
  const bodyText = await page.textContent("body");
  if (!shotTaken && bodyText.includes("%")) {
    await page.screenshot({ path: path.join(outDir, "04-processing.png") });
    shotTaken = true;
  }
  if (bodyText.includes("Completed") && bodyText.includes("100%")) break;
  if (bodyText.toLowerCase().includes("error:")) break;
}

await page.waitForTimeout(1000);
await page.screenshot({ path: path.join(outDir, "05-completed.png"), fullPage: true });

await browser.close();
console.log("Screenshots written to", outDir);
