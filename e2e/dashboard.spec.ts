import { expect, test } from "@playwright/test";

test("live dashboard shows a score and the driver state", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "DRIVIO" })).toBeVisible();
  await expect(page.getByRole("img", { name: /Alertness score \d+ out of 100/ })).toBeVisible();
  await expect(page.getByText("Simulated driver. No camera is in use.")).toBeVisible();
});

test("a recorded session appears in analytics with downloadable reports", async ({ page }) => {
  await page.goto("/");
  const sessionLabel = `E2E report validation ${Date.now()}`;
  await page.getByLabel("Session name").fill(sessionLabel);
  await page.getByRole("button", { name: "Start recording" }).click();
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
  await page.waitForTimeout(3000);
  await page.getByRole("button", { name: "Stop recording" }).click();

  await page.getByRole("button", { name: "Analytics" }).click();
  const sessionRow = page.getByRole("row").filter({ hasText: sessionLabel });
  await sessionRow.getByRole("button", { name: "View" }).click();
  await expect(page.getByRole("heading", { name: "Session replay" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Download PDF" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Download CSV" })).toBeVisible();
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: `Delete session ${sessionLabel}` }).click();
  await expect(page.getByRole("button", { name: `Delete session ${sessionLabel}` })).toHaveCount(0);
});

test("experiments: generate a dataset, train and compare", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Experiments" }).click();
  await page.getByRole("button", { name: "Generate demo dataset" }).click();
  await page.getByRole("button", { name: "Train and evaluate" }).click();
  await expect(page.getByRole("heading", { name: "Model comparison" })).toBeVisible();
});

test("collection: guided protocol can start and advance safely", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Collection" }).click();
  await expect(page.getByRole("heading", { name: "Guided protocol" })).toBeVisible({ timeout: 15_000 });
  await page.getByRole("button", { name: "Start guided collection" }).click();
  await expect(page.getByRole("button", { name: "Next phase" })).toBeVisible();
  await expect(page.getByText("Alert baseline:", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Next phase" }).click();
  await expect(page.getByText("Fatigue demonstration:", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Stop collection" }).click();
});
